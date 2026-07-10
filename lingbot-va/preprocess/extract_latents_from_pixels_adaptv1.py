#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 LeRobot 格式的视频文件（videos/chunk-XXX/camera/episode_XXXXXX.mp4）中读取图像，
用 Wan2.2 VAE (diffusers AutoencoderKLWan) + UMT5 文本编码器抽取 latent，
并以 .pth 形式存到 latents/chunk-XXX/camera/episode_XXXXXX_start_end.pth 中。

同时支持自动为缺少 action_config 的 episodes.jsonl 添加该字段。

输入目录结构示例（--dataset-root）:

your_dataset/
├── videos/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000.mp4
│           └── ...
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
└── meta/
    └── episodes.jsonl

输出目录结构：

your_dataset/
├── latents/
│   └── chunk-000/
│       └── observation.images.cam_high/
│           ├── episode_000000_0_111.pth
│           └── ...
└── ...

每个 .pth 文件包含字典，字段与 README 中 latent_bak 格式一致：
    - latent, latent_num_frames, latent_height, latent_width
    - video_num_frames, video_height, video_width
    - text_emb, text
    - frame_ids, start_frame, end_frame
    - fps, ori_fps

自动添加 action_config 规则（当 episodes.jsonl 缺少该字段时）：
    - start_frame: 0
    - end_frame: episode length
    - action_text: tasks[0]（或指定字段的第一个元素）
    - skill: ""（空字符串）
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import io
import imageio

from torchvision.transforms.functional import resize
from diffusers import AutoencoderKLWan
from transformers import UMT5EncoderModel, T5TokenizerFast

# =================== 模型路径配置 ===================

# 各相机对应的输出视频尺寸 (H, W)
IMAGE_KEYS = {
    'observation.images.cam_high': (256, 320),
    'observation.images.cam_left_wrist': (128, 160),
    'observation.images.cam_right_wrist': (128, 160),
}


def build_wan2_2_components(models_root: Path, device: torch.device):
    """
    加载 diffusers AutoencoderKLWan + transformers UMT5EncoderModel + T5TokenizerFast
    """
    vae_path = models_root / "vae"
    text_encoder_path = models_root / "text_encoder"
    tokenizer_path = models_root / "tokenizer"

    # VAE
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch.bfloat16,
    )
    vae = vae.to(device)
    vae.eval()

    # Text encoder
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch.bfloat16,
    )
    text_encoder = text_encoder.to(device)
    text_encoder.eval()

    # Tokenizer
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path)

    return vae, text_encoder, tokenizer


@torch.no_grad()
def encode_video_with_vae(
    vae, video_tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    用 diffusers AutoencoderKLWan 编码视频。

    inputs:
        video_tensor: [T, 3, H, W], float32, [0,1]
    return:
        latents: [C, T_lat, H_lat, W_lat]
    """
    video_tensor = video_tensor.to(device=device, dtype=torch.bfloat16)
    # diffusers VAE 期望输入 [B, C, T, H, W]
    video_batch = video_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)  # [1, 3, T, H, W]

    latent_dist = vae.encode(video_batch).latent_dist
    latents = latent_dist.sample()

    # 返回 [C, T_lat, H_lat, W_lat]
    return latents.squeeze(0)


@torch.no_grad()
def encode_text(
    text_encoder, tokenizer, text: str, device: torch.device, pad_length: int = 0
) -> torch.Tensor:
    """
    用 UMT5EncoderModel 编码 action_text。

    返回:
        text_emb: [L, D] (bfloat16), 已去除 padding。
        如果 pad_length > 0，则 pad 到固定长度 [pad_length, D]。
    """
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}

    output = text_encoder(**tokens)
    last_hidden = output.last_hidden_state  # [1, L, D]

    # 去除 padding，只保留实际 token
    attention_mask = tokens["attention_mask"][0]
    seq_len = attention_mask.sum().item()
    text_emb = last_hidden[0, :seq_len]  # [L, D]

    text_emb = text_emb.to(torch.bfloat16)

    # 可选：pad 到固定长度
    if pad_length > 0 and text_emb.shape[0] < pad_length:
        pad = torch.zeros(pad_length - text_emb.shape[0], text_emb.shape[1],
                          dtype=text_emb.dtype, device=text_emb.device)
        text_emb = torch.cat([text_emb, pad], dim=0)

    return text_emb


# ============================ 工具函数 ============================


def load_episodes_meta(meta_path: Path) -> Dict[int, Dict[str, Any]]:
    """读取 meta/episodes.jsonl -> {episode_index: meta_dict}"""
    mapping: Dict[int, Dict[str, Any]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            idx = int(obj["episode_index"])
            mapping[idx] = obj
    return mapping


def get_chunk_name(episode_index: int, episodes_per_chunk: int) -> str:
    """根据 episode_index 推 chunk 名"""
    chunk_id = episode_index // episodes_per_chunk
    return f"chunk-{chunk_id:03d}"


def sample_frames(
    total_len: int, start_frame: int, end_frame: int, ori_fps: int, target_fps: int, max_frames: int = 0
) -> List[int]:
    """
    在 [start_frame, end_frame) 范围内按目标 fps 采样帧索引。
    使用 step = ori_fps // target_fps 的均匀间隔采样。
    如果 max_frames > 0，则最多采样 max_frames 个帧。
    """
    start_frame = max(0, start_frame)
    end_frame = min(total_len, end_frame)
    if end_frame <= start_frame:
        return []

    if target_fps <= 0 or ori_fps <= 0:
        return list(range(start_frame, end_frame))

    step = max(1, ori_fps // target_fps)
    ids = list(range(start_frame, end_frame, step))

    if max_frames > 0 and len(ids) > max_frames:
        ids = ids[:max_frames]

    return ids


def read_episode_video_frames(
    video_path: Path,
) -> Tuple[torch.Tensor, int, int, int]:
    """
    从 mp4 视频读取所有帧。

    Returns:
        video: [T, 3, H, W], float32, [0,1]
        T, H, W
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    reader = imageio.get_reader(str(video_path))
    frames = []
    for frame in reader:
        # frame: numpy [H, W, 3], uint8
        frames.append(frame)
    reader.close()

    if len(frames) == 0:
        raise ValueError(f"No frames read from {video_path}")

    # Stack and convert to tensor [T, H, W, 3]
    video_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
    video = torch.from_numpy(video_np)  # [T, H, W, 3]
    # Permute to [T, 3, H, W]
    video = video.permute(0, 3, 1, 2).contiguous()

    T, _, H, W = video.shape
    return video, T, H, W


def add_action_config_to_episodes(
    dataset_root: Path,
    action_text_source: str = "tasks",
    action_text_index: int = 0,
    skill: str = "",
    dry_run: bool = False,
) -> int:
    """
    为 episodes.jsonl 中缺少 action_config 的 episode 添加 action_config 字段。

    规则：
    - 如果 episode 已有 action_config，跳过
    - 否则添加 action_config = [{
        "start_frame": 0,
        "end_frame": length,
        "action_text": tasks[action_text_index] (或 action_text_source 指定字段),
        "skill": skill,
    }]

    Args:
        dataset_root: 数据集根目录（包含 meta/episodes.jsonl）
        action_text_source: 从哪个字段取 action_text，默认 "tasks"
        action_text_index: 取 tasks 列表的第几个元素，默认 0
        skill: skill 字段值，默认空字符串
        dry_run: 如果为 True，只打印不写入

    Returns:
        修改的 episode 数量
    """
    meta_path = dataset_root / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        print(f"[WARN] episodes.jsonl not found at {meta_path}, skip add_action_config.")
        return 0

    lines_out = []
    modified_count = 0

    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                lines_out.append(line)
                continue
            obj = json.loads(line)

            if "action_config" in obj:
                # 已有 action_config，保留原样
                lines_out.append(json.dumps(obj, ensure_ascii=False))
                continue

            length = int(obj.get("length", 0))
            tasks = obj.get("tasks", [])

            # 获取 action_text
            if action_text_source == "tasks" and tasks:
                action_text = tasks[action_text_index] if action_text_index < len(tasks) else tasks[0]
            else:
                action_text = obj.get(action_text_source, "")

            action_config = [{
                "start_frame": 0,
                "end_frame": length,
                "action_text": action_text,
                "skill": skill,
            }]
            obj["action_config"] = action_config
            lines_out.append(json.dumps(obj, ensure_ascii=False))
            modified_count += 1

    if modified_count > 0 and not dry_run:
        # 备份原文件
        backup_path = meta_path.with_suffix(".jsonl.bak")
        if not backup_path.exists():
            meta_path.rename(backup_path)
            print(f"[INFO] backed up original to {backup_path}")

        with meta_path.open("w", encoding="utf-8") as f:
            for line in lines_out:
                f.write(line + "\n")

    if dry_run:
        print(f"[DRY-RUN] Would modify {modified_count} episodes in {meta_path}")
    else:
        print(f"[INFO] modified {modified_count} episodes in {meta_path}")

    return modified_count


def process_dataset(
    dataset_root: Path,
    models_root: Path,
    image_column: str = "observation.images.cam_high",
    episodes_per_chunk: int = 500,
    ori_fps: int = 50,
    target_fps: int = 12,
    max_frames: int = 0,
    text_length: int = 128,
    device_str: str = "cuda:0",
    auto_add_action_config: bool = True,
):
    device = torch.device(device_str)

    # 自动添加 action_config（如果缺失）
    if auto_add_action_config:
        add_action_config_to_episodes(dataset_root)

    # 加载模型
    vae, text_encoder, tokenizer = build_wan2_2_components(models_root, device)

    # 读取 episodes meta
    meta_path = dataset_root / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {meta_path}")
    episodes_meta = load_episodes_meta(meta_path)

    # 视频根目录和输出目录
    videos_root = dataset_root / "videos"
    latents_root = dataset_root / "latents"
    latents_root.mkdir(parents=True, exist_ok=True)

    # 获取当前相机的目标尺寸
    target_size = IMAGE_KEYS.get(image_column)
    if target_size is None:
        raise KeyError(f"Unknown image_column: {image_column}")
    target_h, target_w = target_size

    for episode_index, meta in episodes_meta.items():
        length = int(meta["length"])
        tasks = meta.get("tasks", [])
        action_configs = meta.get("action_config", [])

        if not action_configs:
            # 没有 action_config 时默认整段
            action_configs = [{
                "start_frame": 0,
                "end_frame": length,
                "action_text": tasks[0] if tasks else "",
            }]

        default_text = tasks[0] if tasks else ""
        chunk_name = get_chunk_name(episode_index, episodes_per_chunk)

        # 视频输入路径
        video_path = (
            videos_root / chunk_name / image_column / f"episode_{episode_index:06d}.mp4"
        )
        if not video_path.exists():
            print(f"[WARN] missing episode video: {video_path}, skip.")
            continue

        try:
            video, T, H, W = read_episode_video_frames(video_path)
        except Exception as e:
            print(f"[WARN] failed to read {video_path}: {e}")
            continue

        # Resize 到目标尺寸
        video_resized = resize(video, (target_h, target_w))  # [T, 3, H, W]

        # 输出目录
        out_dir = latents_root / chunk_name / image_column
        out_dir.mkdir(parents=True, exist_ok=True)

        for seg_id, segment in enumerate(action_configs):
            start_frame = int(segment.get("start_frame", 0))
            end_frame = int(segment.get("end_frame", length))
            action_text = segment.get("action_text", default_text)

            frame_ids = sample_frames(
                total_len=length,
                start_frame=start_frame,
                end_frame=end_frame,
                ori_fps=ori_fps,
                target_fps=target_fps,
                max_frames=max_frames,
            )
            if not frame_ids:
                print(
                    f"[WARN] episode {episode_index:06d} seg#{seg_id} "
                    f"[{start_frame},{end_frame}) -> empty frame_ids, skip."
                )
                continue

            # 采样帧
            frames = video_resized[frame_ids]  # [N, 3, H, W]
            N, _, H_res, W_res = frames.shape

            # VAE 编码
            latents = encode_video_with_vae(vae, frames, device=device)
            # latents: [C, T_lat, H_lat, W_lat]

            C_lat, T_lat, H_lat, W_lat = latents.shape

            # 转成 bfloat16 并 flatten 成 [T_lat * H_lat * W_lat, C_lat]
            latents_bf16 = latents.to(dtype=torch.bfloat16)
            latent_flat = latents_bf16.permute(1, 2, 3, 0).reshape(-1, C_lat)

            # 文本编码
            text_emb = encode_text(
                text_encoder, tokenizer, action_text, device=device, pad_length=text_length
            )
            text_emb_bf16 = text_emb.to(dtype=torch.bfloat16)
            text_emb_n, text_emb_d = text_emb_bf16.shape

            if text_emb_d != 4096:
                print(f"[WARN] text_emb_d is {text_emb_d}, expected 4096")

            # 构建输出字典
            row = {
                "latent": latent_flat,
                "latent_num_frames": int(T_lat),
                "latent_height": int(H_lat),
                "latent_width": int(W_lat),

                "video_num_frames": int(N),
                "video_height": int(H_res),
                "video_width": int(W_res),

                "text_emb": text_emb_bf16,
                "text": action_text,

                "frame_ids": [int(x) for x in frame_ids],
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "fps": int(target_fps),
                "ori_fps": int(ori_fps),
            }

            out_path = out_dir / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            torch.save(row, out_path)
            print(f"[OK] saved latents: {out_path} (frames={N}, latent={latent_flat.shape})")

    print("All done.")


def main():
    '''
    示例：
    
    python preprocess/extract_latents_from_pixels0710.py \
        --dataset-root dataset_root \
        --models-root encoder_model_path \
        --max-frames 30 \
        --cameras view1,view2,view3
    '''
    # python preprocess/extract_latents_from_pixels0710.py --dataset-root /data/user/prtroas0003/lishan/dataspace/robotwin_samples/lift_pot-demo_clean_collect_200-50
    # --models-root /data/user/prtroas0003/lishan/modelspace/lingbot-va-base
    # --max-frames 30
    home_path = '~/lishan'
    parser = argparse.ArgumentParser(
        description="Extract Wan2.2 latents from LeRobot video episodes into .pth latent files."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=os.path.join(home_path, 'dataspace/robotwin_samples/lift_pot-demo_clean_collect_200-50'),
        help="Dataset directory (contains videos/, data/, meta/episodes.jsonl)",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default=os.path.join(home_path, 'modelspace/lingbot-va-base'),
        help="Model directory (contains vae/, text_encoder/, tokenizer/)",
    )
    parser.add_argument(
        "--episodes-per-chunk",
        type=int,
        default=500,
        help="Number of episodes per chunk",
    )
    parser.add_argument(
        "--ori-fps",
        type=int,
        default=50,
        help="Original video fps",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=12,
        help="Target sampling fps for latent extraction",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to sample per segment (0=unlimited)",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=128,
        help="Text max length (tokenizer param, mostly unused with T5TokenizerFast)",
    )
    parser.add_argument(
        "--auto-add-action-config",
        action="store_true",
        default=True,
        help="Automatically add action_config to episodes.jsonl if missing (default: True)",
    )
    parser.add_argument(
        "--no-auto-add-action-config",
        action="store_false",
        dest="auto_add_action_config",
        help="Disable automatic addition of action_config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device: cuda:0 / cuda:1 / cpu",
    )
    parser.add_argument(
        "--cameras",
        type=str,
        default=None,
        help="Comma-separated camera names to process (default: all in IMAGE_KEYS)",
    )

    args = parser.parse_args()

    cameras = args.cameras.split(",") if args.cameras else list(IMAGE_KEYS.keys())

    for key in cameras:
        if key not in IMAGE_KEYS:
            print(f"[WARN] Unknown camera '{key}', skipping.")
            continue
        print(f"\n=== Processing camera: {key} ===")
        process_dataset(
            dataset_root=Path(args.dataset_root),
            models_root=Path(args.models_root),
            image_column=key,
            episodes_per_chunk=args.episodes_per_chunk,
            ori_fps=args.ori_fps,
            target_fps=args.fps,
            max_frames=args.max_frames,
            text_length=args.text_length,
            device_str=args.device,
            auto_add_action_config=args.auto_add_action_config,
        )


if __name__ == "__main__":
    main()
