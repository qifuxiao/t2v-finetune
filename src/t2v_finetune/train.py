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

BATCH_SIZE = 1
LR = 1e-5
NUM_EPOCHS = 10
NUM_FRAMES = 16  # CogVideoX 常用输入帧数

def main():
    accelerator = Accelerator(mixed_precision="bf16")
    device = accelerator.device

    # 1. 获取混合精度类型
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 2. 加载组件（显式指定 torch_dtype 消除加载警告）
    print("🚀 加载 T2V 组件...")
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

    # 冻结 VAE 和 Text Encoder
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

    print("🔥 开始 T2V 扩散模型微调训练...")
    transformer.train()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(dataloader):
            videos = batch["video"]  # (B, T, C, H, W)
            captions = batch["caption"]

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

                # B. VAE 编码视频至 Latent 隐空间
                # VAE 输入格式为 (B, C, T, H, W)
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

            # 💡 核心修复：调整维度从 (B, C, T, H, W) -> (B, T, C, H, W) 以适配 CogVideoX Transformer
            noisy_latents = noisy_latents.permute(0, 2, 1, 3, 4)

            # D. Transformer 预测噪声
            model_pred = transformer(
                hidden_states=noisy_latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timesteps,
                return_dict=False
            )[0]

            # E. 计算扩散 MSE Loss（将 noise 同步转为 B, T, C, H, W 维度）
            target = noise.permute(0, 2, 1, 3, 4)
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()

            if step % 10 == 0:
                accelerator.print(f"Epoch [{epoch}/{NUM_EPOCHS}] Step [{step}/{len(dataloader)}] Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()