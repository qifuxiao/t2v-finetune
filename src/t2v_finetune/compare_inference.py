import os
import torch
from diffusers import CogVideoXPipeline, CogVideoXTransformer3DModel
from diffusers.utils import export_to_video

# ==================== 1. 配置参数 ====================
BASE_MODEL_PATH = "/data1/alexqi/model_path/models/CogVideoX-2b"

# 请修改为您保存微调权重的路径（如完整 Pipeline 路径或 Transformer 检查点路径）
FINETUNED_PATH = "./output/cogvideox-finetuned/checkpoint-final"  # 或 "./output/transformer"

OUTPUT_DIR = "./eval_results"
SEED = 42                      # 固定随机种子，保证对比公正性
NUM_INFERENCE_STEPS = 50       # 采样步数
GUIDANCE_SCALE = 6.0           # Classifier-Free Guidance 系数
NUM_FRAMES = 16                # 生成帧数

# ==================== 2. 测试提示词列表 ====================
# 包含：1. 密集细节长提示词  2. 运镜控制  3. 物理/动作动态
TEST_PROMPTS = [
    {
        "name": "01_dense_prompt",
        "prompt": "A stylish red panda wearing small round glasses and a blue scarf sits by a cafe window on a rainy day. Raindrops glide down the glass outside. The panda holds a steaming mug of coffee with both paws, sipping slowly while blurred pedestrians walk by in the background."
    },
    {
        "name": "02_camera_motion",
        "prompt": "A low-angle tracking shot of a futuristic neon city street at night. A shiny black sports car accelerates forward through puddles, reflecting vibrant pink and blue neon lights."
    },
    {
        "name": "03_action_physics",
        "prompt": "A clear glass cup falls onto a wooden table in slow motion, water splashing out in fine droplets, creating delicate ripples as it hits the surface."
    }
]

def generate_videos(pipe, prompt_list, model_tag, output_dir):
    """使用指定 Pipeline 生成视频"""
    os.makedirs(output_dir, exist_ok=True)
    
    for item in prompt_list:
        name = item["name"]
        prompt = item["prompt"]
        
        print(f"\n🎬 [{model_tag}] 正在生成测试样本: {name} ...")
        
        # 严格固定 Generator 种子
        generator = torch.Generator(device="cuda").manual_seed(SEED)
        
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=1,
            num_inference_steps=NUM_INFERENCE_STEPS,
            num_frames=NUM_FRAMES,
            guidance_scale=GUIDANCE_SCALE,
            generator=generator,
        ).frames[0]
        
        save_path = os.path.join(output_dir, f"{name}_{model_tag}.mp4")
        export_to_video(video, save_path, fps=8)
        print(f"✅ 已保存: {save_path}")

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ==================== 3. 测试基座模型 ====================
    print("\n📦 [1/2] 正在加载基座模型 (Base Model)...")
    base_pipe = CogVideoXPipeline.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.bfloat16
    )
    base_pipe.enable_model_cpu_offload()  # 开启 CPU Offload 节省显存
    base_pipe.vae.enable_slicing()
    base_pipe.vae.enable_tiling()

    generate_videos(base_pipe, TEST_PROMPTS, "base", OUTPUT_DIR)

    # 清理显存
    del base_pipe
    torch.cuda.empty_cache()

    # ==================== 4. 测试微调模型 ====================
    print("\n🚀 [2/2] 正在加载微调模型 (Fine-tuned Model)...")
    
    # 判断微调权重保存的形式：是整套目录，还是仅保存了 Transformer
    if os.path.exists(os.path.join(FINETUNED_PATH, "model_index.json")):
        # 保存的是完整 Pipeline 目录
        finetuned_pipe = CogVideoXPipeline.from_pretrained(
            FINETUNED_PATH,
            torch_dtype=torch.bfloat16
        )
    else:
        # 仅保存了微调后的 Transformer，其他组件仍加载基座
        print("ℹ️ 检测到单 Transformer 权重，正在合并加载...")
        transformer = CogVideoXTransformer3DModel.from_pretrained(
            FINETUNED_PATH,
            torch_dtype=torch.bfloat16
        )
        finetuned_pipe = CogVideoXPipeline.from_pretrained(
            BASE_MODEL_PATH,
            transformer=transformer,
            torch_dtype=torch.bfloat16
        )

    finetuned_pipe.enable_model_cpu_offload()
    finetuned_pipe.vae.enable_slicing()
    finetuned_pipe.vae.enable_tiling()

    generate_videos(finetuned_pipe, TEST_PROMPTS, "finetuned", OUTPUT_DIR)

    print(f"\n🎉 所有对比视频已生成完毕！请在目录查看: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()