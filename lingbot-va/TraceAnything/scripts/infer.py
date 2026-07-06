# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# scripts/infer.py
"""
Run inference on all scenes and save:
  - <scene>/output.pt with {'preds','views'}
  - <scene>/masks/{i:03d}.png   (binary FG masks)
  - <scene>/images/{i:03d}.png  (RGB frames used for inference)

Masks are computed from ctrl-pt variance + smart Otsu.
"""

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import os
import cv2
import time
import argparse
from typing import List, Dict

import numpy as np
import torch
from PIL import Image
import torchvision.transforms as tvf
from omegaconf import OmegaConf

from trace_anything.trace_anything import TraceAnything


def _pretty(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# allow ${python_eval: ...} in YAML if used
OmegaConf.register_new_resolver("python_eval", lambda code: eval(code))


# ---------------- image I/O ----------------
def _resize_long_side(pil: Image.Image, long: int = 512) -> Image.Image:
    w, h = pil.size
    if w >= h:
        return pil.resize((long, int(h * long / w)), Image.BILINEAR)
    else:
        return pil.resize((int(w * long / h), long), Image.BILINEAR)


def _load_images(input_dir: str, device: torch.device) -> List[Dict]:
    """Read images, rotate portrait->landscape, resize(long=512), crop to 16-multiple, normalize [-1,1]."""
    tfm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5,)*3, (0.5,)*3)])

    fnames = sorted(
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and "_vis" not in f
    )
    if not fnames:
        raise FileNotFoundError(f"No images in {input_dir}")

    views, target = [], None
    for i, f in enumerate(fnames):
        arr = cv2.imread(os.path.join(input_dir, f))
        if arr is None:
            raise FileNotFoundError(f"Failed to read {f}")
        pil = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

        W0, H0 = pil.size
        if H0 > W0:  # portrait -> landscape
            pil = pil.transpose(Image.Transpose.ROTATE_90)

        pil = _resize_long_side(pil, 512)
        if target is None:
            H, W = pil.size[1], pil.size[0]
            target = (H - H % 16, W - W % 16)
            _pretty(f"📐 target size: {target[0]}x{target[1]} (16-multiple)")
        Ht, Wt = target
        pil = pil.crop((0, 0, Wt, Ht))

        tensor = tfm(pil).unsqueeze(0).to(device)  # [1,3,H,W]
        t = i / (len(fnames) - 1) if len(fnames) > 1 else 0.0
        views.append({"img": tensor, "time_step": t})
    return views


# ---------------- ckpt + model ----------------
def _get_state_dict(ckpt: dict) -> dict:
    """Accept either a pure state_dict or a Lightning .ckpt."""
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        return ckpt["state_dict"]
    return ckpt


def _load_cfg(cfg_path: str):
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(cfg_path)
    return OmegaConf.load(cfg_path)


def _to_dict(x):
    # OmegaConf -> plain dict
    return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x


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


# ---------------- smart var threshold ----------------
def _otsu_threshold_from_hist(hist: np.ndarray, bin_edges: np.ndarray) -> float | None:
    total = hist.sum()
    if total <= 0:
        return None
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    w1 = np.cumsum(hist)
    w2 = total - w1
    sum_total = (hist * bin_centers).sum()
    sumB = np.cumsum(hist * bin_centers)
    valid = (w1 > 0) & (w2 > 0)
    if not np.any(valid):
        return None
    m1 = sumB[valid] / w1[valid]
    m2 = (sum_total - sumB[valid]) / w2[valid]
    between = w1[valid] * w2[valid] * (m1 - m2) ** 2
    idx = np.argmax(between)
    return float(bin_centers[valid][idx])


def _smart_var_threshold(var_map_t: torch.Tensor) -> float:
    """
    1) log-transform variance
    2) Otsu on histogram
    3) fallback to 65–80% mid-quantile midpoint
    Returns threshold in original variance domain.
    """
    var_np = var_map_t.detach().float().cpu().numpy()
    v = np.log(var_np + 1e-9)
    hist, bin_edges = np.histogram(v, bins=256)
    thr_log = _otsu_threshold_from_hist(hist, bin_edges)
    if thr_log is None or not np.isfinite(thr_log):
        q65 = float(np.quantile(var_np, 0.65))
        q80 = float(np.quantile(var_np, 0.80))
        return 0.5 * (q65 + q80)
    thr_var = float(np.exp(thr_log))
    q40 = float(np.quantile(var_np, 0.40))
    q95 = float(np.quantile(var_np, 0.95))
    return max(q40, min(q95, thr_var))


# ---------------- main loop ----------------
def run(args):
    base_in = args.input_dir
    base_out = args.output_dir

    if not os.path.isdir(base_in):
        raise FileNotFoundError(base_in)
    os.makedirs(base_out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # config & model
    cfg = _load_cfg(args.config)
    _pretty("🔧 loading model …")
    model = _build_model_from_cfg(cfg, ckpt_path=args.ckpt, device=device)
    _pretty("✅ model ready")

    # iterate scenes
    for scene in sorted(os.listdir(base_in)):
        in_dir = os.path.join(base_in, scene)
        if not os.path.isdir(in_dir):
            continue
        out_dir = os.path.join(base_out, scene)
        masks_dir = os.path.join(out_dir, "masks")
        images_dir = os.path.join(out_dir, "images")
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(masks_dir, exist_ok=True)
        os.makedirs(images_dir, exist_ok=True)

        _pretty(f"\n📂 Scene: {scene}")
        _pretty("🖼️  loading images …")
        views = _load_images(in_dir, device=device)
        if len(views) > 40:
            stride = max(1, len(views) // 39)   # floor division
            views = views[::stride]
        _pretty(f"🧮 {len(views)} views loaded")

        _pretty("🚀 inference …")
        t0 = time.perf_counter()
        with torch.no_grad():
            preds = model.forward(views)
        dt = time.perf_counter() - t0
        ms_per_view = (dt / max(1, len(views))) * 1000.0
        _pretty(f"✅ done | {dt:.2f}s total | {ms_per_view:.1f} ms/view")

        # ---- compute + save FG masks and images ----
        _pretty("🧪 computing FG masks + saving frames …")
        for i, pred in enumerate(preds):
            # variance map over control points (K), mean over xyz -> [H,W]
            ctrl_pts3d = pred["ctrl_pts3d"]
            ctrl_pts3d_t = torch.from_numpy(ctrl_pts3d) if isinstance(ctrl_pts3d, np.ndarray) else ctrl_pts3d
            var_map = torch.var(ctrl_pts3d_t, dim=0, unbiased=False).mean(-1)  # [H,W]
            thr = _smart_var_threshold(var_map)
            fg_mask = (~(var_map <= thr)).detach().cpu().numpy().astype(bool)

            # save mask as binary PNG and stash in preds
            cv2.imwrite(os.path.join(masks_dir, f"{i:03d}.png"), (fg_mask.astype(np.uint8) * 255))
            pred["fg_mask"] = torch.from_numpy(fg_mask)  # CPU bool tensor

            # also save the RGB image we actually ran on
            img = views[i]["img"].detach().cpu().squeeze(0)  # [3,H,W] in [-1,1]
            img_np = (img.permute(1, 2, 0).numpy() + 1.0) * 127.5
            img_uint8 = np.clip(img_np, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(images_dir, f"{i:03d}.png"), cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR))

            # trim heavy intermediates just in case
            pred.pop("track_pts3d", None)
            pred.pop("track_conf", None)

        # persist
        save_path = os.path.join(out_dir, "output.pt")
        torch.save({"preds": preds, "views": views}, save_path)
        _pretty(f"💾 saved: {save_path}")
        _pretty(f"🖼️  masks → {masks_dir} | images → {images_dir}")

def parse_args():
    p = argparse.ArgumentParser("TraceAnything inference")
    p.add_argument("--config", type=str, default="configs/eval.yaml",
                   help="Path to YAML config")
    p.add_argument("--ckpt", type=str, default="checkpoints/trace_anything.pt",
                   help="Path to the checkpoint")
    p.add_argument("--input_dir", type=str, default="./examples/input",
                   help="Directory containing scenes (each subfolder is a scene)")
    p.add_argument("--output_dir", type=str, default="./examples/output",
                   help="Directory to write scene outputs")
    return p.parse_args()

def main():
    args = parse_args()
    run(args)

if __name__ == "__main__":
    main()
