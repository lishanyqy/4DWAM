#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
从 LeRobot 的 episode_XXXXXX.parquet 里读取图像，
用 Wan2.2 VAE + 文本编码器抽取 latent，
并以 parquet 形式存到 latents/chunk-XXX/episode_XXXXXX.parquet 中。

输入目录结构示例（--dataset-root）:

your_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
└── meta/
    └── episodes.jsonl

输出目录结构：

your_dataset/
├── latents/
│   └── chunk-000/
│       ├── episode_000000.parquet   # 一行一个 action segment
│       ├── episode_000001.parquet
│       └── ...
└── ...

每行包含：
    - episode_index, start_frame, end_frame, frame_ids, text
    - latent_bytes, latent_dtype, latent_num_frames, latent_height, latent_width, latent_channels
    - text_emb_bytes, text_emb_dtype, text_emb_n, text_emb_d
    - video_num_frames, video_height, video_width, fps, ori_fps
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
import vae2_2
# from wan.vae import WanTI2V_VAE
# from wan.text_encoder import WanTextEncoder
from transformers import AutoTokenizer
from t5 import T5EncoderModel
import torch
from torchvision.transforms.functional import resize
from torchvision.utils import save_image
# =================== Wan2.2 相关：你需要填的 TODO ===================

# image_keys = {
#     'observation.images.cam_high':[],
#     'observation.images.cam_left_wrist':,
#     'observation.images.cam_right_wrist':,
# }

image_keys = {
    'observation.images.cam_high': (256, 320),
    'observation.images.cam_left_wrist': (128,160),
    'observation.images.cam_right_wrist': (128,160),
}

def build_wan2_2_components(models_root: Path,text_length, device: torch.device):
    """
   
    return vae, text_model

    示意伪代码（不要直接运行）:
    ----------------------------------------------------------------
    """
    # raise NotImplementedError("请在 build_wan2_2_components 里加载 Wan2.2 模型")
    # vae_ckpt = models_root / "Wan2.2-TI2V-5B" / "Wan2.2_VAE.pth"
    vae_ckpt = models_root / "Wan2.2_VAE.pth"
    text_ckpt = models_root / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = models_root / "google" / "umt5-xxl"
    vae = vae2_2.Wan2_2_VAE(vae_pth = vae_ckpt)
    # vae = WanTI2V_VAE.from_checkpoint(vae_ckpt).to(device)
    # vae.eval()

    # text_encoder = WanTextEncoder.from_checkpoint(text_ckpt).to(device)
    # text_encoder.eval()
    text_encoder = T5EncoderModel(
        text_len = text_length,
        checkpoint_path = text_ckpt,
        tokenizer_path = tokenizer_path,
    )
    # text_encoder.model.eval()

    # tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")

    # tokenizer = AutoTokenizer.from_pretrained("google/umt5-xxl")
    return vae, text_encoder

@torch.no_grad()
def encode_video_with_vae(
    vae, video_tensor: torch.Tensor, device: torch.device
) -> torch.Tensor:
    """
    用 Wan2.2 的 VAE 编码视频。

    inputs:
        video_tensor: [T, 3, H, W], float32, [0,1]
    return:
        latents: [T, C, H_lat, W_lat]

    TODO: VAE Model extract the latents

    示意伪代码:

    ----------------------------------------------------------------
    video_tensor = video_tensor.to(device=device, dtype=torch.bfloat16)
    latents = vae.encode([video_tensor])[0]    # [T, C, H_lat, W_lat]
    return latents
    ----------------------------------------------------------------
    """
    return vae.encode(video_tensor)

    # raise NotImplementedError("Please call vae model to extract the latents")


@torch.no_grad()
def encode_text(
    text_encoder, text: str, device: torch.device
) -> torch.Tensor:
    """
    用 Wan2.2 文本编码器编码 action_text。

    返回:
        text_emb: [L, D] 或 [1, D]

    TODO: 换成你自己的文本编码代码。

    ----------------------------------------------------------------
    tokens = tokenizer(
        text,
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=256,
    )
    tokens = {k: v.to(device) for k, v in tokens.items()}
    out = text_encoder(**tokens)
    text_emb = out.last_hidden_state.squeeze(0)  # [L, D]
    return text_emb
    ----------------------------------------------------------------
    """
    text_emb = text_encoder(text,device)
    return text_emb


# ============================ 工具函数 ============================

def tensor_to_bytes(t: torch.Tensor) -> Tuple[bytes, str]:
    """convert tensor to bytes (bytes, dtype_str)。"""
    t_cpu = t.detach().cpu()
    if t_cpu.dtype == torch.bfloat16:
        # convert to 32
        arr = t_cpu.to(torch.float32).numpy()
        return arr.tobytes(), "float32_from_bfloat16"
    else:
        arr = t_cpu.numpy()
        return arr.tobytes(), str(arr.dtype)


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
    """根据 episode_index 推 chunk 名（例如 500 个一块）"""
    chunk_id = episode_index // episodes_per_chunk
    return f"chunk-{chunk_id:03d}"


def sample_frames(
    total_len: int, start_frame: int, end_frame: int, ori_fps: int, target_fps: int
) -> List[int]:
    """
    [start_frame, end_frame) sample target_fps index。
    """
    start_frame = max(0, start_frame)
    end_frame = min(total_len, end_frame)
    if end_frame <= start_frame:
        return []

    if target_fps <= 0 or ori_fps <= 0:
        return list(range(start_frame, end_frame))

    cur_frames = end_frame - start_frame
    # step = float(ori_fps) / float(target_fps)
    # ids: List[int] = []
    # t = float(start_frame)
    # while t < end_frame:
    #     ids.append(int(round(t)))
    #     t += step

    # ids = sorted(set(ids))
    # ids = [i for i in ids if start_frame <= i < end_frame]

    intervals = int(cur_frames / target_fps)

    ids = [i for i in range(start_frame, end_frame, intervals)]

    return ids


def resize_video_tensor(video: torch.Tensor, size: int) -> torch.Tensor:
    """
    [T, 3, H, W] -> resize to H=W=size。
    """
    if size <= 0:
        return video
    video = F.interpolate(video, size=(size, size), mode="bilinear", align_corners=False)
    return video


def read_episode_parquet_frames(
    parquet_path: Path,
    image_column: str,
) -> Tuple[torch.Tensor, int, int, int]:
    """
        Inputs imageio.bytes
        video: [T, 3, H, W], float32, [0,1]
        T, H, W
    """
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    if image_column not in df.columns:
        raise KeyError(f"{image_column!r} not in columns: {df.columns.tolist()}")

    imgs = [v['bytes'] for v in df[image_column]]  # ndarray of objects
    frames = []
    for i, img_bytes in enumerate(imgs):
        arr = np.asarray(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
        if arr.ndim == 3 and arr.shape[-1] == 3:   # [H, W, 3]
            arr = np.transpose(arr, (2, 0, 1))     # -> [3, H, W]
        elif arr.ndim == 3 and arr.shape[0] == 3:  # already [3, H, W]
            pass
        else:
            raise ValueError(
                f"Unexpected image shape for frame {i}: {arr.shape}, "
                f"expect [H,W,3] or [3,H,W]"
            )
        frames.append(torch.from_numpy(arr).float() /255.0)

    # video = torch.from_numpy(np.stack(frames, axis=0)).float() / 255.0  # [T,3,H,W]
    # T, _, H, W = video.shape
    T,_,H,W = len(frames), frames[0].shape[0],frames[0].shape[1],frames[0].shape[2]
    return frames, T, H, W


def process_dataset(
    dataset_root: Path,
    models_root: Path,
    image_column: str = "observation.images.cam_high",
    episodes_per_chunk: int = 500,
    ori_fps: int = 30,
    target_fps: int = 10,
    resize_size: int = 256,
    text_length = 128,
    device_str: str = "cuda:0",
):

    device = torch.device(device_str)

    # load Wan2.2 components
    vae, text_encoder = build_wan2_2_components(models_root, text_length, device)
    # text_encoder()
    # episodes.jsonl
    meta_path = dataset_root / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(f"episodes.jsonl not found at {meta_path}")
    episodes_meta = load_episodes_meta(meta_path)

    data_root = dataset_root / "data"
    latents_root = dataset_root / "latents"
    latents_root.mkdir(parents=True, exist_ok=True)

    for episode_index, meta in episodes_meta.items():
        length = int(meta["length"])
        tasks = meta.get("tasks", [])
        action_configs = meta.get("action_config", [])

        if not action_configs:
            # No action_config
            action_configs = [{
                "start_frame": 0,
                "end_frame": length,
                "action_text": tasks[0] if tasks else "",
            }]

        default_text = tasks[0] if tasks else ""

        chunk_name = get_chunk_name(episode_index, episodes_per_chunk)

        in_path = (
            data_root / chunk_name / f"episode_{episode_index:06d}.parquet"
        )
        if not in_path.exists():
            print(f"[WARN] missing episode parquet: {in_path}, skip.")
            continue

        try:
            video, T, H, W = read_episode_parquet_frames(
                in_path, image_column=image_column
            )
        except Exception as e:
            print(f"[WARN] failed to read {in_path}: {e}")
            continue

        # video_resized = resize_video_tensor(video, resize_size)
        # save_image(video_resized[20],'tmp_pre.png')
        video_resized = resize(torch.stack(video, dim=0), image_keys[image_column])
        # save_image(video_resized[20],'tmp.png')
        # video_resized = video
        episode_ori_fps = len(video_resized)

        # segment_rows: List[Dict[str, Any]] = []
        # one episode write to one parquet
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
                ori_fps=episode_ori_fps,
                target_fps=target_fps,
            )
            if not frame_ids:
                print(
                    f"[WARN] episode {episode_index:06d} seg#{seg_id} "
                    f"[{start_frame},{end_frame}) -> empty frame_ids, skip."
                )
                continue
            
            frames = video_resized[frame_ids]  # [N,3,H,W]
            # frames = torch.stack(video_resized, dim=0)
            N, _, H_res, W_res = frames.shape
            u = frames.permute(1, 0, 2, 3).contiguous()
            u = u.to(device)
            # VAE 编码
            latents = encode_video_with_vae(vae, [u], device=device)[0]
            if latents.dim() != 4:
                raise RuntimeError(
                    f"VAE latent shape expected 4D [N,C,H,W], got {latents.shape}"
                )
            z_dim, T_lat, H_lat, W_lat = latents.shape
            # assert N_lat == N, "latent_num_frames != frame_ids 数量，请检查 VAE encode"
            # latents_bf16 = latents.to(dtype=torch.bfloat16)
            # bfloat16 & flatten 成 [N * H_lat * W_lat, C_lat]
            latents_bf16 = latents.to(dtype=torch.bfloat16)
            latent_flat = (
                latents_bf16.permute(1, 2, 3, 0)
                .reshape(-1, z_dim)
            )  # [N*H_lat*W_lat, C_lat]

            # latent_bytes, latent_dtype_str = tensor_to_bytes(latent_flat)

            # text embedding
            # text_emb = encode_text(
            #     text_encoder, tokenizer, action_text, device=device
            # )
            text_emb = encode_text(
                text_encoder, action_text, device=device
            )[0]
            # if text_emb.shape[-2] < text_length:
            #     text_emb = torch.cat([text_emb,torch.zeros(text_length - text_emb.shape[-2], text_emb.shape[-1])], dim = 0)
            # elif text_emb.shape[-2] > text_length:
            #     raise ValueError(f"Please consider larger text_length, currently is {text_length}")
            text_emb_bf16 = text_emb.to(dtype=torch.bfloat16)
            # text_emb_bytes, text_emb_dtype_str = tensor_to_bytes(text_emb_bf16)
            text_emb_shape = list(text_emb_bf16.shape)
            if len(text_emb_shape) == 1:
                text_emb_n, text_emb_d = 1, int(text_emb_shape[0])
            else:
                text_emb_n, text_emb_d = int(text_emb_shape[0]), int(text_emb_shape[1])
            if text_emb_d != 4096:
                print(f'somthing wrong, the text_emb_d is not 4096,{text_emb_d.shape}')
            row = {
                "episode_index": int(episode_index),
                "segment_index": int(seg_id),
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "frame_ids": [int(x) for x in frame_ids],
                "text": action_text,

                # latents
                "latent": latent_flat,
                # "latent_dtype": latent_dtype_str,
                "latent_num_frames": int(T_lat),
                "latent_height": int(H_lat),
                "latent_width": int(W_lat),
                "latent_channels": int(z_dim),

                # text embedding
                "text_emb": text_emb_bf16,
                # "text_emb_dtype": text_emb_dtype_str,
                "text_emb_n": int(text_emb_n),
                "text_emb_d": int(text_emb_d),

                # video meta data
                "video_num_frames": int(len(frame_ids)),
                "video_height": int(H_res),
                "video_width": int(W_res),
                "fps": int(target_fps),
                "ori_fps": int(episode_ori_fps),
            }
            # segment_rows.append(row)

        # if not segment_rows:
        #     print(f"[INFO] episode {episode_index:06d} has no valid segments, skip saving.")
        #     continue

        
            out_path = out_dir / f"episode_{episode_index:06d}_{row['start_frame']}_{row['end_frame']}.pth"

            torch.save(row,out_path)
            # df_out = pd.DataFrame(segment_rows)
            # df_out.to_parquet(out_path, index=False)
            print(f"[OK] saved latents parquet: {out_path}")

    print("All done.")


def main():
    home_path = '/cpfs01/projects-HDD/cfff-377aad6b032c_HDD/chenshuai/wenxuan/'
    parser = argparse.ArgumentParser(
        description="Extract Wan2.2 latents from LeRobot parquet episodes into parquet latents."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        # required=True,
        default = os.path.join(home_path,'.cache/huggingface/lerobot/robotwin/robotwin_multi_10_tasks'),
        help="LeRobot dataset directory (included data/ and meta/episodes.jsonl)",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        # required=True,
        default = os.path.join(home_path,'.cache/modelscope/Wan-AI/Wan2.2-TI2V-5B/'),
        help="Wan2.2 directory",
    )
    parser.add_argument(
        "--image-column",
        type=str,
        default="observation.images.cam_high",
        help="episode parquet column names",
    )
    parser.add_argument(
        "--episodes-per-chunk",
        type=int,
        default=500,
        help="Each chunk 的 episode 数量，用于从 episode_index 计算 chunk-XXX 名",
    )
    parser.add_argument(
        "--ori-fps",
        type=int,
        default=221,
        help="original episode fps (get from meta)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Sample to training. fps ",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=256,
        help="image resize to (H=W=resize, <=0 represent not resize）",
    )
    parser.add_argument(
        "--text-length",
        type=int,
        default=128,
        help="Text max length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="select device. cuda:0 / cuda:1 / cpu",
    )

    args = parser.parse_args()

    # debug_images_columns(Path(args.dataset_root))
    for key in image_keys:
        process_dataset(
            dataset_root=Path(args.dataset_root),
            models_root=Path(args.models_root),
            # image_column=args.image_column,
            image_column = key,
            episodes_per_chunk=args.episodes_per_chunk,
            ori_fps=args.ori_fps,
            target_fps=args.fps,
            resize_size=args.resize,
            text_length = args.text_length,
            device_str=args.device,
        )


def debug_images_columns(
    dataset_root,
    # models_root,
    # image_column,
):
    """
        Inputs imageio.bytes
        video: [T, 3, H, W], float32, [0,1]
        T, H, W
    """
    parquet_path = dataset_root / "data" / "chunk-000"/ "episode_000001.parquet"
    print(parquet_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    # if image_column not in df.columns:
    #     raise KeyError(f"{image_column!r} not in columns: {df.columns.tolist()}")
    for key in image_keys:
    # imgs = [v['bytes'] for v in df[image_column]]  # ndarray of objects
        # frames = []
        imgs = [v['bytes'] for v in df[key]]
        # for i, img_bytes in enumerate(imgs):
        # arr = np.asarray(
        img = Image.open(io.BytesIO(imgs[122])).convert("RGB")
        # )
        img.save(f'{key}.png')
        # print(len(imgs), arr.shape)
        # if arr.ndim == 3 and arr.shape[-1] == 3:   # [H, W, 3]
        #     arr = np.transpose(arr, (2, 0, 1))     # -> [3, H, W]
        # elif arr.ndim == 3 and arr.shape[0] == 3:  # already [3, H, W]
        #     pass
        # frames.append(torch.from_numpy(arr).float() /255.0)

    # video = torch.from_numpy(np.stack(frames, axis=0)).float() / 255.0  # [T,3,H,W]
    # T, _, H, W = video.shape
    # T,_,H,W = len(frames), frames[0].shape[0],frames[0].shape[1],frames[0].shape[2]
    # return frames, T, H, W

    


if __name__ == "__main__":
    main()