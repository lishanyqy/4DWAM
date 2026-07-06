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
import multiprocessing as mp
import traceback
from pathlib import Path
from typing import Dict, Any, Iterable, List, Sequence, Tuple

import pdb
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
import cv2
import av
import subprocess
from torchvision.transforms.functional import resize
from omegaconf import OmegaConf
from traceanything import TraceAnything
from tqdm import tqdm
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

V = [
    'observation.images.cam_high',
    'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]

def get_video_info(video_path):
    """
    使用 ffprobe 获取视频信息
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get video info: {result.stderr}")
    
    width, height = map(int, result.stdout.strip().split(','))
    return (width, height)

def read_video_frames_ffmpeg_single(video_path, frame_ids, video_size=None):
    """
    使用 ffmpeg 子进程读取指定帧
    
    Args:
        video_path: 视频文件路径
        frame_ids: 要读取的帧索引列表
        video_size: 视频尺寸 (width, height)，如果不提供则自动检测
    
    Returns:
        list of numpy arrays (RGB格式)
    """
    # 1. 获取视频信息（如果未提供）
    if video_size is None:
        video_size = get_video_info(video_path)
    
    width, height = video_size
    frame_size = width * height * 3  # RGB24 每个像素3字节
    
    frames = []
    
    for fid in frame_ids:
        # 为每个帧创建独立的 ffmpeg 进程
        # 使用 select 过滤器提取指定帧
        cmd = [
            "ffmpeg",
            "-hwaccel", "none",      # 禁用硬件加速
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{fid})",  # 选择指定帧
            "-vsync", "vfr",          # 保持原始帧率
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "pipe:1"                  # 输出到 stdout
        ]
        
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # 读取原始帧数据
        raw_data = proc.stdout.read(frame_size)
        proc.wait()
        
        if proc.returncode != 0 or len(raw_data) != frame_size:
            # 如果读取失败，尝试不跳过帧的方式
            raise RuntimeError(
                f"Failed to read frame {fid} from {video_path}. "
                f"FFmpeg error: {proc.stderr.read().decode()}"
            )
        
        # ffmpeg outputs RGB24 here.
        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
        frames.append(frame)
    
    return frames

def read_video_frames_pyav(video_path, frame_ids):
    """使用 PyAV 读取视频帧，对 AV1 有更好支持"""
    container = av.open(str(video_path))
    video_stream = container.streams.video[0]
    
    # 获取视频总帧数
    total_frames = video_stream.frames
    
    frames = []
    for fid in frame_ids:
        if fid >= total_frames:
            raise RuntimeError(f"Frame {fid} out of range")
        
        # 跳转到目标帧（PyAV 内部会处理关键帧优化）
        container.seek(fid, stream=video_stream)
        
        # 读取当前帧
        for packet in container.demux(video_stream):
            for frame in packet.decode():
                # 转换为 numpy 数组 (RGB 格式)
                img = frame.to_ndarray(format='rgb24')
                frames.append(img)
                break
            break
    
    container.close()
    return frames

def discover_task_roots(dataset_root: Path) -> List[Path]:
    """Support either a single task dir or a collection dir containing many task dirs."""
    dataset_root = dataset_root.resolve()
    single_task_markers = ("latents", "videos", "meta")
    if all((dataset_root / marker).exists() for marker in single_task_markers):
        return [dataset_root]

    task_roots = []
    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue
        if all((child / marker).exists() for marker in single_task_markers):
            task_roots.append(child)
    if not task_roots:
        raise FileNotFoundError(
            f"No task directories found under {dataset_root}. "
            "Expected either a task root with latents/videos/meta or a parent dir containing task subdirectories."
        )
    return task_roots


def parse_devices(device: str = "", devices: str = "") -> List[str]:
    if devices.strip():
        parsed = [item.strip() for item in devices.split(",") if item.strip()]
    elif device.strip():
        parsed = [device.strip()]
    else:
        parsed = ["cuda:0"]
    return parsed


def iter_episode_latent_files(view_path: Path) -> Iterable[Path]:
    return sorted(
        path for path in view_path.iterdir()
        if path.is_file() and path.suffix == ".pth"
    )


def build_frame_timesteps(num_frames: int) -> List[float]:
    if num_frames <= 1:
        return [0.0]
    return [t / (num_frames - 1) for t in range(num_frames)]


def process_task(
    task_root: Path,
    ta_model: torch.nn.Module,
    device: torch.device,
    skip_existing: bool = True,
) -> None:
    video_root = task_root / "videos"
    latent_root = task_root / "latents"
    trace_root = task_root / "trace"
    if not latent_root.exists():
        print(f"[WARN] missing latents dir, skip task: {task_root}")
        return
    if not video_root.exists():
        print(f"[WARN] missing videos dir, skip task: {task_root}")
        return

    chunks = sorted(path for path in latent_root.iterdir() if path.is_dir())
    for chunk_dir in tqdm(chunks, desc=f"{task_root.name}: chunks", leave=False):
        chunk_name = chunk_dir.name
        video_chunk_dir = video_root / chunk_name
        for view in V:
            view_path = chunk_dir / view
            if not view_path.exists():
                continue
            trace_view_dir = trace_root / chunk_name / view
            trace_view_dir.mkdir(parents=True, exist_ok=True)

            for eps_path in tqdm(
                iter_episode_latent_files(view_path),
                desc=f"{task_root.name}/{chunk_name}/{view}",
                leave=False,
            ):
                out_path = trace_view_dir / eps_path.name
                if skip_existing and out_path.exists():
                    continue

                eps_video = f"{eps_path.name[:14]}.mp4"
                video_path = video_chunk_dir / view / eps_video
                try:
                    content = torch.load(
                        eps_path,
                        map_location="cpu",
                        weights_only=False,
                    )
                    frame_ids = content["frame_ids"]
                    latent_num_frames = content["latent_num_frames"]
                    frames = read_video_frames_ffmpeg(video_path, frame_ids)
                    if len(frames) != len(frame_ids):
                        raise RuntimeError(
                            f"Expected {len(frame_ids)} frames, got {len(frames)} from {video_path}"
                        )
                    frames = frames_bgr_to_tensor(frames)
                    reframes = resize(frames, image_keys[view]).to(device)

                    frames_size = len(frame_ids)
                    frame_timesteps = build_frame_timesteps(frames_size)
                    frame_list = list(torch.split(reframes, 1, dim=0))
                    dec_output = ta_model.extract_seq_features(frame_list, frame_timesteps)
                    if latent_num_frames and dec_output.shape[0] != latent_num_frames:
                        print(
                            f"[WARN] latent_num_frames mismatch for {eps_path}: "
                            f"expected {latent_num_frames}, got {dec_output.shape[0]}"
                        )
                    torch.save(dec_output.detach().cpu(), out_path)
                except Exception as exc:
                    print(f"[WARN] failed on {eps_path}: {exc}")

    print(f"[OK] finished task: {task_root}")


def process_dataset(
    dataset_root: Path,
    models_root: Path,
    cfg_path: Path,
    device: str = "cuda:0",
    skip_existing: bool = True,
):
    device_obj = torch.device(device)
    model_ckpt = models_root / "trace_anything.pt"
    cfg = _load_cfg(cfg_path)
    ta_model = _build_model_from_cfg(cfg, model_ckpt, device_obj)

    for task_root in discover_task_roots(dataset_root):
        process_task(
            task_root=task_root,
            ta_model=ta_model,
            device=device_obj,
            skip_existing=skip_existing,
        )
    print("All done.")

def _get_state_dict(ckpt: dict) -> dict:
    """Accept either a pure state_dict or a Lightning .ckpt."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt

def _to_dict(x):
    # OmegaConf -> plain dict
    return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x

def _load_cfg(cfg_path: str):
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    return OmegaConf.load(cfg_path)

def _build_model_from_cfg(cfg, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    # net config
    net_cfg = cfg.get("model", {}).get("net", None) or cfg.get("net", None)
    if net_cfg is None:
        raise KeyError("expect cfg.model.net or cfg.net in YAML")

    model = TraceAnything(
        encoder_args=_to_dict(net_cfg["encoder_args"]),
        decoder_args=_to_dict(net_cfg["decoder_args"]),
        head_args=_to_dict(net_cfg["head_args"]),
        targeting_mechanism=net_cfg.get("targeting_mechanism", "bspline_conf"),
        poly_degree=net_cfg.get("poly_degree", 10),
        whether_local=False,
    )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = _get_state_dict(ckpt)

    if all(k.startswith("net.") for k in sd.keys()):
        sd = {k[4:]: v for k, v in sd.items()}

    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def read_video_frames_ffmpeg_batch(video_path, frame_ids):
    """
    批量读取多个帧，只启动一次 ffmpeg 进程
    """
    if not frame_ids:
        return []
    
    # 获取视频信息
    width, height = get_video_info(video_path)
    frame_size = width * height * 3
    
    # 排序 frame_ids 以便顺序读取
    sorted_ids = sorted(frame_ids)
    min_frame = sorted_ids[0]
    max_frame = sorted_ids[-1]
    
    # 启动 ffmpeg 进程输出指定范围的帧
    cmd = [
        "ffmpeg",
        "-hwaccel", "none",
        "-i", str(video_path),
        "-vf", f"select=between(n\\,{min_frame}\\,{max_frame}),setpts=PTS-STARTPTS",
        "-vsync", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "pipe:1"
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_size * 10
    )
    
    # 读取所有帧
    all_frames = []
    expected_frames = max_frame - min_frame + 1
    
    for i in range(expected_frames):
        raw_data = proc.stdout.read(frame_size)
        if len(raw_data) != frame_size:
            break
        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
        all_frames.append(frame)
    
    proc.wait()
    
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()
        raise RuntimeError(f"FFmpeg error: {stderr}")
    
    # 映射到原始的 frame_ids
    frame_map = {min_frame + i: frame for i, frame in enumerate(all_frames)}
    return [frame_map[fid] for fid in frame_ids if fid in frame_map]


def read_video_frames_ffmpeg(video_path, frame_ids):
    """
    智能选择读取方式
    """
    if len(frame_ids) > 10:
        # 批量读取，性能更好
        return read_video_frames_ffmpeg_batch(video_path, frame_ids)
    else:
        # 少量帧，使用原来的逐帧读取
        return read_video_frames_ffmpeg_single(video_path, frame_ids)

# def read_video_frames(video_path, frame_ids):
#     cap = cv2.VideoCapture(str(video_path))
#     if not cap.isOpened():
#         raise RuntimeError(f"Cannot open video: {video_path}")

#     frames = []
#     for fid in frame_ids:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
#         ret, frame = cap.read()
#         if not ret:
#             raise RuntimeError(
#                 f"Failed to read frame {fid} from {video_path}"
#             )
#         frames.append(frame)

#     cap.release()
#     return frames

def frames_bgr_to_tensor(frames_bgr):
    """
    List[np.ndarray(H,W,C,RGB)] -> [T, 3, H, W], RGB float32 [0,1]
    """
    tensors = []
    for frame in frames_bgr:
        x = torch.from_numpy(frame).float() / 255.0   # HWC
        x = x.permute(2, 0, 1)                      # CHW
        tensors.append(x)
    return torch.stack(tensors, dim=0)

def resize_frames(frames_bgr, resize_size=(256, 320)):
    """
    resize_size: (W, H)
    """
    out = []
    for frame in frames_bgr:
        # pdb.set_trace()
        frame = cv2.resize(frame, resize_size, interpolation=cv2.INTER_LINEAR)
        out.append(frame)
    return out


def run_worker(
    worker_id: int,
    task_roots: Sequence[str],
    models_root: str,
    cfg_path: str,
    device: str,
    skip_existing: bool,
):
    if not task_roots:
        print(f"[INFO] worker {worker_id} on {device} has no tasks assigned.")
        return

    try:
        print(
            f"[INFO] worker {worker_id} starting on {device} "
            f"for task(s): {[Path(task_root).name for task_root in task_roots]}"
        )
        torch.set_grad_enabled(False)
        if device.startswith("cuda"):
            cuda_index = torch.device(device).index
            if cuda_index is not None:
                torch.cuda.set_device(cuda_index)

        device_obj = torch.device(device)
        cfg = _load_cfg(cfg_path)
        model_ckpt = Path(models_root) / "trace_anything.pt"
        ta_model = _build_model_from_cfg(cfg, model_ckpt, device_obj)
        for task_root in task_roots:
            process_task(
                task_root=Path(task_root),
                ta_model=ta_model,
                device=device_obj,
                skip_existing=skip_existing,
            )
        print(f"[INFO] worker {worker_id} finished on {device}")
    except Exception:
        print(f"[ERROR] worker {worker_id} failed on {device}")
        print(traceback.format_exc())
        raise


def launch_parallel(
    task_roots: Sequence[Path],
    models_root: Path,
    cfg_path: Path,
    devices: Sequence[str],
    skip_existing: bool,
    max_concurrent_tasks: int,
):
    if not task_roots:
        return

    max_workers = min(len(devices), max(1, max_concurrent_tasks))
    if max_workers == 1:
        run_worker(
            worker_id=0,
            task_roots=[str(path) for path in task_roots],
            models_root=str(models_root),
            cfg_path=str(cfg_path),
            device=devices[0],
            skip_existing=skip_existing,
        )
        return

    ctx = mp.get_context("spawn")
    for wave_start in range(0, len(task_roots), max_workers):
        wave_tasks = task_roots[wave_start: wave_start + max_workers]
        wave_index = wave_start // max_workers + 1
        total_waves = (len(task_roots) + max_workers - 1) // max_workers
        print(
            f"[INFO] starting wave {wave_index}/{total_waves} with "
            f"{len(wave_tasks)} task(s): {[task.name for task in wave_tasks]}"
        )

        processes = []
        for worker_id, task_root in enumerate(wave_tasks):
            device = devices[worker_id % len(devices)]
            proc = ctx.Process(
                target=run_worker,
                args=(
                    worker_id,
                    [str(task_root)],
                    str(models_root),
                    str(cfg_path),
                    device,
                    skip_existing,
                ),
            )
            proc.start()
            processes.append(proc)

        failed = False
        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                failed = True
                print(f"[ERROR] worker pid={proc.pid} exited with code {proc.exitcode}")
        if failed:
            raise RuntimeError(
                f"At least one worker failed during wave {wave_index}/{total_waves}."
            )

def main():
    default_dataset_root = "/data/.cache/datasets/lerobot/robotwin/comp_trace/robotwin_ds/lerobot_robotwin_eef_clean_50/"
    default_model_root = "/root/.cache/models/trace-anything/"
    default_cfg_path = Path(__file__).resolve().parent / "traceanything" / "configs" / "eval.yaml"
    parser = argparse.ArgumentParser(
        description="Extract traceanything latents from LeRobot parquet episodes into parquet latents."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=default_dataset_root,
        help="LeRobot dataset directory (included data/ and meta/episodes.jsonl)",
    )
    parser.add_argument(
        "--ta-model-path",
        type=str,
        default=default_model_root,
        help="TraceAnything model directory containing trace_anything.pt",
    )
    parser.add_argument(
        "--cfg-path",
        type=str,
        default=str(default_cfg_path),
        help="TraceAnything config path",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Single device fallback. Example: cuda:0 / cpu",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="",
        help="Comma separated devices for task-level parallelism, e.g. cuda:0,cuda:1",
    )
    parser.add_argument(
        "--task-names",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of task directory names to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing trace outputs instead of skipping them",
    )
    parser.add_argument(
        "--max-concurrent-tasks",
        type=int,
        default=0,
        help="Maximum number of tasks to run at once. 0 means use all provided devices.",
    )

    args = parser.parse_args()
    dataset_root = Path(args.dataset_root)
    task_roots = discover_task_roots(dataset_root)
    if args.task_names:
        selected = set(args.task_names)
        task_roots = [path for path in task_roots if path.name in selected]
        if not task_roots:
            raise ValueError(f"No tasks matched --task-names: {sorted(selected)}")

    devices = parse_devices(device=args.device, devices=args.devices)
    max_concurrent_tasks = args.max_concurrent_tasks or len(devices)
    print(f"[INFO] found {len(task_roots)} task(s): {[path.name for path in task_roots]}")
    print(f"[INFO] using devices: {devices}")
    print(f"[INFO] max concurrent tasks: {max_concurrent_tasks}")

    launch_parallel(
        task_roots=task_roots,
        models_root=Path(args.ta_model_path),
        cfg_path=Path(args.cfg_path),
        devices=devices,
        skip_existing=not args.overwrite,
        max_concurrent_tasks=max_concurrent_tasks,
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
    for key in image_keys:
        imgs = [v['bytes'] for v in df[key]]
        img = Image.open(io.BytesIO(imgs[122])).convert("RGB")
        img.save(f'{key}.png')

    


if __name__ == "__main__":
    main()
    
