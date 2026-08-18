import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers import CogVideoXTransformer3DModel, AutoencoderKLCogVideoX, DDPMScheduler
from transformers import AutoTokenizer, T5EncoderModel

from t2v_finetune.dataset import ShareGPT4VideoDataset

# === 配置参数 ===
MODEL_PATH = "/data1/alexqi/model_path/models/CogVideoX-2b"  # 绝对路径
JSONL_PATH = "/data1/alexqi/dataset/sharegpt4video_40k.jsonl"
VIDEO_DIR = "/data1/alexqi/dataset/extracted_videos"
OUTPUT_DIR = "./output/cogvideox-finetuned"  # Checkpoint 保存根目录

BATCH_SIZE = 1
LR = 1e-5
NUM_EPOCHS = 10
NUM_FRAMES = 16  # CogVideoX 常用输入帧数
SAVE_EPOCH_FREQ = 1  # 保存 Checkpoint 的 Epoch 间隔（如每 1 个 Epoch 保存一次）

def main():
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # 1. 获取混合精度类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 2. 加载组件
    accelerator.print("🚀 加载 T2V 组件...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        MODEL_PATH, subfolder="text_encoder", torch_dtype=weight_dtype
    ).to(device)
    vae = AutoencoderKLCogVideoX.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=weight_dtype
    ).to(device)
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", torch_dtype=weight_dtype
    )
    scheduler = DDPMScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    # 冻结 VAE 和 Text Encoder，开启 Transformer 训练
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    transformer.requires_grad_(True)

    # 3. 准备数据集和 DataLoader
    dataset = ShareGPT4VideoDataset(
        jsonl_path=JSONL_PATH,
        video_dir=VIDEO_DIR,
        num_frames=NUM_FRAMES,
        height=480,
        width=720
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=LR)

    # 加速器包装模型与优化器
    transformer, optimizer, dataloader = accelerator.prepare(
        transformer, optimizer, dataloader
    )

    accelerator.print("🔥 开始 T2V 扩散模型微调训练...")
    transformer.train()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(dataloader):
            
            videos = batch["video"]  # (B, T, C, H, W)
            captions = batch["caption"]
            # 打印检查视频像素范围
            # print(f"Video min: {videos.min().item()}, max: {videos.max().item()}")
            with torch.no_grad():
                # A. 提取文本特征
                text_inputs = tokenizer(
                    captions, 
                    padding="max_length", 
                    max_length=226, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device)
                
                prompt_embeds = text_encoder(text_inputs.input_ids)[0].to(dtype=weight_dtype)

                # B. VAE 编码视频至 Latent 隐空间 (B, C, T, H, W)
                videos = videos.permute(0, 2, 1, 3, 4).to(device, dtype=weight_dtype)
                latents = vae.encode(videos).latent_dist.sample()
                
                if hasattr(vae.config, "scaling_factor"):
                    latents = latents * vae.config.scaling_factor
                latents = latents.to(dtype=weight_dtype)

            # C. 前向扩散过程：添加高斯噪声
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
            ).long()
            
            noisy_latents = scheduler.add_noise(latents, noise, timesteps).to(dtype=weight_dtype)

            # 💡 维度调整：(B, C, T, H, W) -> (B, T, C, H, W)
            noisy_latents_input = noisy_latents.permute(0, 2, 1, 3, 4)

            # D. Transformer 预测
            model_pred = transformer(
                hidden_states=noisy_latents_input,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                return_dict=False
            )[0]

            # E. 💡 关键修复：根据 Scheduler 的 prediction_type 自动获取正确的 Target
            if scheduler.config.prediction_type == "epsilon":
                target = noise
            elif scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"不支持的 prediction_type: {scheduler.config.prediction_type}")

            # 转换 target 维度为 (B, T, C, H, W) 与 model_pred 匹配
            target = target.permute(0, 2, 1, 3, 4)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if step % 10 == 0:
                accelerator.print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")

        # ================= 💾 每个 Epoch 保存逻辑 =================
        if (epoch + 1) % SAVE_EPOCH_FREQ == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                checkpoint_dir = os.path.join(OUTPUT_DIR, f"checkpoint-epoch-{epoch+1}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                # 解包分布式/混合精度包装后的模型并保存
                unwrapped_transformer = accelerator.unwrap_model(transformer)
                unwrapped_transformer.save_pretrained(
                    checkpoint_dir,
                    safe_serialization=True
                )
                accelerator.print(f"💾 [Epoch {epoch+1}] Checkpoint 已成功保存至: {checkpoint_dir}")

    # ================= 💾 训练结束保存最终权重 =================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(OUTPUT_DIR, "checkpoint-final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped_transformer = accelerator.unwrap_model(transformer)
        unwrapped_transformer.save_pretrained(
            final_dir,
            safe_serialization=True
        )
        accelerator.print(f"🎉 所有训练完成！最终模型已保存至: {final_dir}")

if __name__ == "__main__":
    main()