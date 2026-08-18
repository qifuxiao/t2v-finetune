import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from decord import VideoReader, cpu

class ShareGPT4VideoDataset(Dataset):
    """
    用于 T2V 视频生成训练的数据集加载器
    输出: video_tensor: (T, C, H, W) 归一化至 [-1, 1], caption: str
    """
    def __init__(self, jsonl_path: str, video_dir: str, num_frames: int = 16, height: int = 480, width: int = 720):
        self.video_dir = video_dir
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.data_list = []

        local_files = set(os.listdir(video_dir))

        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                v_path = item.get("video_path") or item.get("video") or ""
                v_id = item.get("video_id", "")
                
                candidates = []
                if v_path:
                    candidates.append(os.path.basename(v_path))
                if v_id:
                    candidates.append(f"{v_id}.mp4")
                    candidates.append(v_id)

                matched_filename = None
                for cand in candidates:
                    if cand in local_files:
                        matched_filename = cand
                        break

                if matched_filename:
                    local_video_path = os.path.join(video_dir, matched_filename)
                    caption = ""
                    raw_captions = item.get("captions") or item.get("caption") or item.get("describe") or item.get("text")
                    
                    if isinstance(raw_captions, list):
                        texts = [c.get("content", "") if isinstance(c, dict) else str(c) for c in raw_captions]
                        caption = " ".join([t for t in texts if t])
                    elif isinstance(raw_captions, str):
                        caption = raw_captions

                    if caption:
                        self.data_list.append({
                            "video_path": local_video_path,
                            "caption": caption
                        })

        print(f"✅ T2V 数据集初始化完成！共计 {len(self.data_list)} 条有效视频-文本对。")

    def __len__(self):
        return len(self.data_list)

    def _sample_video_frames(self, video_path: str) -> torch.Tensor:
        try:
            vr = VideoReader(video_path, ctx=cpu(0), width=self.width, height=self.height)
            total_frames = len(vr)
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
            
            # 转为 (T, C, H, W) 缩放到 [-1, 1] (扩散模型标准格式)
            tensor_frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1.0
            return tensor_frames
        except Exception as e:
            print(f"⚠️ 视频读取失败 ({os.path.basename(video_path)}): {e}")
            return torch.zeros((self.num_frames, 3, self.height, self.width))

    def __getitem__(self, idx: int):
        item = self.data_list[idx]
        video_tensor = self._sample_video_frames(item["video_path"])
        return {
            "video": video_tensor,
            "caption": item["caption"]
        }
