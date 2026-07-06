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

# scripts/view.py
# TraceAnything viewer (viser) – no mask recomputation.
# - Loads FG masks from (priority):
#     1) <scene>/masks/{i:03d}_user.png  or preds[i]["fg_mask_user"]
#     2) <scene>/masks/{i:03d}.png       or preds[i]["fg_mask"]
# - BG mask = ~FG
# - Saves per-frame images next to output.pt (does NOT recompute masks)
# - Initial conf filtering: drop bottom 10% (FG/BG)
# - Downsample everything with --ds (H,W stride)

import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import os
import time
import threading
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import cv2
import viser

# ---- tiny helpers ----
def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def as_float(x):
    a = np.asarray(x)
    return float(a.reshape(-1)[0])

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

# ---- tiny HSV colormap (no heavy deps) ----
import colorsys
def hsv_colormap(vals01: np.ndarray) -> np.ndarray:
    """vals01: [T] in [0,1] -> [T,3] RGB in [0,1]"""
    vals01 = np.clip(vals01, 0.0, 1.0)
    rgb = [colorsys.hsv_to_rgb(v, 1.0, 1.0) for v in vals01]
    return np.asarray(rgb, dtype=np.float32)

# --- repo fn ---
from trace_anything.trace_anything import evaluate_bspline_conf

# ----------------------- I/O -----------------------
def load_output_dict(path_or_dir: str) -> Dict:
    path = path_or_dir
    if os.path.isdir(path):
        path = os.path.join(path, "output.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    out = torch.load(path, map_location="cpu")
    out["_root_dir"] = os.path.dirname(path)
    return out

def save_images(frames: List[Dict], images_dir: str):
    print(f"[viewer] Saving images to: {images_dir}")
    ensure_dir(images_dir)
    for i, fr in enumerate(frames):
        cv2.imwrite(os.path.join(images_dir, f"{i:03d}.png"),
                    cv2.cvtColor(fr["img_rgb_uint8"], cv2.COLOR_RGB2BGR))

# ----------------- mask loading (NO recomputation) -----------------
_REPORTED_MASK_SRC: set[tuple[str, int]] = set()

def _load_fg_mask_for_index(root_dir: str, idx: int, pred: Dict) -> np.ndarray:
    """
    Priority:
      1) masks/{i:03d}_user.png  or pred['fg_mask_user']  (user mask)
      2) masks/{i:03d}.png       or pred['fg_mask']       (raw mask)
    Returns: mask_bool_hw
    """
    def _ret(mask_bool: np.ndarray, src: str) -> np.ndarray:
        key = (root_dir, idx)
        if key not in _REPORTED_MASK_SRC and os.environ.get("TA_SILENCE_MASK_SRC") != "1":
            print(f"[viewer] frame {idx:03d}: using {src}")
            _REPORTED_MASK_SRC.add(key)
        return mask_bool

    # 1) USER
    #   PNG then preds (so external edits override stale preds if any)
    p_png = os.path.join(root_dir, "masks", f"{idx:03d}_user.png")
    if os.path.isfile(p_png):
        arr = cv2.imread(p_png, cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            # p_png_1 = os.path.join(root_dir, "masks", f"{idx:03d}_user_1.png")
            # if os.path.isfile(p_png_1):
            #     arr1 = cv2.imread(p_png_1, cv2.IMREAD_GRAYSCALE)
            #     if arr1 is not None and arr1.shape == arr.shape:
            #         arr = np.maximum(arr, arr1)
            return _ret((arr > 0), "user mask (png)")
    if "fg_mask_user" in pred and pred["fg_mask_user"] is not None:
        m = pred["fg_mask_user"]
        if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
        return _ret(np.asarray(m).astype(bool), "user mask (preds)")

    # 2) RAW
    p_png = os.path.join(root_dir, "masks", f"{idx:03d}.png")
    if os.path.isfile(p_png):
        arr = cv2.imread(p_png, cv2.IMREAD_GRAYSCALE)
        if arr is not None:
            return _ret((arr > 0), "raw mask (png)")
    if "fg_mask" in pred and pred["fg_mask"] is not None:
        m = pred["fg_mask"]
        if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
        return _ret(np.asarray(m).astype(bool), "raw mask (preds)")

    # --- legacy compatibility ---
    for key, fname, label in [
        ("fg_mask_user", f"{idx:03d}_fg_user.png", "user mask (legacy png)"),
        ("fg_mask",      f"{idx:03d}_fg_refined.png", "refined mask (legacy png)"),
        ("fg_mask_raw",  f"{idx:03d}_fg_raw.png", "raw mask (legacy png)"),
    ]:
        if key in pred and pred[key] is not None:
            m = pred[key]
            if isinstance(m, torch.Tensor): m = m.detach().cpu().numpy()
            return _ret(np.asarray(m).astype(bool), f"{label} (preds)")
        path = os.path.join(root_dir, "masks", fname)
        if os.path.isfile(path):
            arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if arr is not None:
                return _ret((arr > 0), label)

    raise FileNotFoundError(
        f"No FG mask for frame {idx}: looked for user/raw (png or preds)."
    )

# -------------- precompute tensors for viewer -------------
def build_precomputes(
    output: Dict,
    t_step: float,
    ds: int,
) -> Tuple[np.ndarray, List[Dict], List[np.ndarray], np.ndarray, np.ndarray]:
    preds = output["preds"]
    views = output["views"]
    n = len(preds)
    assert n == len(views)

    root = output.get("_root_dir", os.getcwd())

    # timeline
    t_vals = np.arange(0.0, 1.0 + 1e-6, t_step, dtype=np.float32)
    if t_vals[-1] >= 1.0:
        t_vals[-1] = 0.99
    T = len(t_vals)
    t_tensor = torch.from_numpy(t_vals)

    frames: List[Dict] = []
    fg_conf_pool_per_t: List[List[np.ndarray]] = [[] for _ in range(T)]
    bg_conf_pool: List[np.ndarray] = []

    stride = slice(None, None, ds)  # ::ds

    for i in range(n):
        pred = preds[i]
        view = views[i]

        # image ([-1,1]) -> uint8 RGB
        img = to_numpy(view["img"].squeeze().permute(1, 2, 0))
        img_uint8 = np.clip((img + 1.0) * 127.5, 0, 255).astype(np.uint8)
        img_uint8 = img_uint8[::ds, ::ds]  # downsample for saving/vis
        H, W = img_uint8.shape[:2]
        HW = H * W
        img_flat = (img_uint8.astype(np.float32) / 255.0).reshape(HW, 3)

        # load FG mask (prefer user, then raw)
        fg_mask = _load_fg_mask_for_index(root, i, pred)

        # match resolution to downsampled view
        # if mask at full-res and we downsampled, stride it; else resize with nearest
        if fg_mask.shape == (H * ds, W * ds) and ds > 1:
            fg_mask = fg_mask[::ds, ::ds]
        elif fg_mask.shape != (H, W):
            fg_mask = cv2.resize(
                (fg_mask.astype(np.uint8) * 255),
                (W, H),
                interpolation=cv2.INTER_NEAREST
            ) > 0

        bg_mask = ~fg_mask
        bg_mask_flat = bg_mask.reshape(-1)
        fg_mask_flat = fg_mask.reshape(-1)

        # control points/conf (K,H,W,[3]) at downsampled stride
        ctrl_pts3d = pred["ctrl_pts3d"][:, stride, stride, :]    # [K,H,W,3]
        ctrl_conf  = pred["ctrl_conf"][:, stride, stride]         # [K,H,W]

        # evaluate curve over T timesteps
        pts3d_t, conf_t = evaluate_bspline_conf(ctrl_pts3d, ctrl_conf, t_tensor)  # [T,H,W,3], [T,H,W]
        pts3d_t = to_numpy(pts3d_t).reshape(T, HW, 3)
        conf_t  = to_numpy(conf_t).reshape(T, HW)

        # FG per t (keep per-t list for later filtering)
        pts_fg_per_t  = [pts3d_t[t][fg_mask_flat] for t in range(T)]
        conf_fg_per_t = [conf_t[t][fg_mask_flat]  for t in range(T)]
        for t in range(T):
            if pts_fg_per_t[t].size > 0:
                fg_conf_pool_per_t[t].append(conf_fg_per_t[t])

        # BG static
        bg_pts = pts3d_t.mean(axis=0)[bg_mask_flat]
        bg_conf_mean = conf_t.mean(axis=0)[bg_mask_flat]
        bg_conf_pool.append(bg_conf_mean)

        frames.append(dict(
            img_rgb_uint8=img_uint8,
            img_rgb_float=img_flat,
            H=H, W=W, HW=HW,
            bg_mask_flat=bg_mask_flat,
            fg_mask_flat=fg_mask_flat,
            pts_fg_per_t=pts_fg_per_t,
            conf_fg_per_t=conf_fg_per_t,
            bg_pts=bg_pts,
            bg_conf_mean=bg_conf_mean,
        ))

    # pools for percentiles
    fg_conf_all_t: List[np.ndarray] = []
    for t in range(T):
        if len(fg_conf_pool_per_t[t]) == 0:
            fg_conf_all_t.append(np.empty((0,), dtype=np.float32))
        else:
            fg_conf_all_t.append(np.concatenate(fg_conf_pool_per_t[t], axis=0).astype(np.float32))

    if len(bg_conf_pool):
        bg_conf_all_flat = np.concatenate(bg_conf_pool, axis=0).astype(np.float32)
    else:
        bg_conf_all_flat = np.empty((0,), dtype=np.float32)

    # frame times (fallback to views' time_step if missing)
    def _get_time(i):
        ti = preds[i].get("time", None)
        if ti is None:
            ti = views[i].get("time_step", float(i / max(1, n - 1)))
        return as_float(ti)
    times = np.array([_get_time(i) for i in range(n)], dtype=np.float64)

    return t_vals, frames, fg_conf_all_t, bg_conf_all_flat, times

def choose_nearest_frame_indices(frame_times: np.ndarray, t_vals: np.ndarray) -> np.ndarray:
    return np.array([int(np.argmin(np.abs(frame_times - tv))) for tv in t_vals], dtype=np.int64)

# ---------------- nodes & state ----------------
class ViewerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.is_updating = False
        self.status_label = None
        self.slider_fg = None
        self.slider_bg = None
        self.slider_time = None
        self.point_size = None
        self.bg_point_size = None
        self.fg_nodes = []  # len T
        self.bg_nodes = []  # len N
        # trajectories
        self.show_traj = None
        self.traj_width = None
        self.traj_frames_text = None
        self.traj_build_btn = None
        self.traj_nodes = []
        self.playing = False
        self.play_btn = None
        self.pause_btn = None
        self.fps_slider = None
        self.loop_checkbox = None

def build_bg_nodes(server: viser.ViserServer, frames: List[Dict], init_percentile: float,
                   bg_conf_all_flat: np.ndarray, state: ViewerState):
    if bg_conf_all_flat.size == 0:
        return
    thr_val = np.percentile(bg_conf_all_flat, init_percentile)
    for i, fr in enumerate(frames):
        keep = fr["bg_conf_mean"] >= thr_val
        pts  = fr["bg_pts"][keep]
        cols = fr["img_rgb_float"][fr["bg_mask_flat"]][keep]
        node = server.scene.add_point_cloud(
            name=f"/bg/frame{i}",
            points=pts,
            colors=cols,
            point_size=state.bg_point_size.value if state.bg_point_size else 0.0002,
            point_shape="rounded",
            visible=True,
        )
        state.bg_nodes.append(node)
        print(f"[viewer] BG frame={i:02d}: add {pts.shape[0]} pts")

def build_fg_nodes(server: viser.ViserServer, frames: List[Dict], nearest_idx: np.ndarray, t_vals: np.ndarray,
                   fg_conf_all_t: List[np.ndarray], init_percentile: float, state: ViewerState):
    print("\n[viewer] Building FG nodes per timeline step …")
    T = len(t_vals)
    for t in range(T):
        fi = int(nearest_idx[t])
        conf_all = fg_conf_all_t[t]
        thr_t = np.percentile(conf_all, init_percentile) if conf_all.size > 0 else np.inf
        conf = frames[fi]["conf_fg_per_t"][t]
        pts  = frames[fi]["pts_fg_per_t"][t]
        keep = conf >= thr_t
        pts_k = pts[keep]
        cols_k = frames[fi]["img_rgb_float"][frames[fi]["fg_mask_flat"]][keep]
        node = server.scene.add_point_cloud(
            name=f"/fg/t{t:02d}",
            points=pts_k,
            colors=cols_k,
            point_size=state.point_size.value if state.point_size else 0.0002,
            point_shape="rounded",
            visible=(t == 0),
        )
        state.fg_nodes.append(node)
        print(f"[viewer] t={t:02d}: add FG node with {pts_k.shape[0]} pts")

def update_conf_filtering(server: viser.ViserServer, state: ViewerState, frames: List[Dict], nearest_idx: np.ndarray,
                          t_vals: np.ndarray, fg_conf_all_t: List[np.ndarray], bg_conf_all_flat: np.ndarray,
                          fg_percentile: float, bg_percentile: float):
    with state.lock:
        if state.is_updating:
            return
        state.is_updating = True
    try:
        if state.status_label:
            state.status_label.value = "⚙️ Filtering… please wait"
        if state.slider_fg: state.slider_fg.disabled = True
        if state.slider_bg: state.slider_bg.disabled = True

        # BG
        thr_bg = np.percentile(bg_conf_all_flat, bg_percentile) if bg_conf_all_flat.size > 0 else np.inf
        print(f"[filter] BG: percentile={bg_percentile:.1f} thr={thr_bg:.6f}")
        for i, node in enumerate(state.bg_nodes):
            conf = frames[i]["bg_conf_mean"]
            keep = conf >= thr_bg
            pts  = frames[i]["bg_pts"][keep]
            cols = frames[i]["img_rgb_float"][frames[i]["bg_mask_flat"]][keep]
            node.points = pts
            node.colors = cols
            print(f"  - frame {i:02d}: keep {pts.shape[0]} pts")
        server.flush()

        # FG
        print(f"[filter] FG: percentile={fg_percentile:.1f}")
        T = len(t_vals)
        for t in range(T):
            fi = int(nearest_idx[t])
            conf_all = fg_conf_all_t[t]
            thr_t = np.percentile(conf_all, fg_percentile) if conf_all.size > 0 else np.inf
            conf = frames[fi]["conf_fg_per_t"][t]
            pts  = frames[fi]["pts_fg_per_t"][t]
            keep = conf >= thr_t
            pts_k = pts[keep]
            cols_k = frames[fi]["img_rgb_float"][frames[fi]["fg_mask_flat"]][keep]
            node = state.fg_nodes[t]
            node.points = pts_k
            node.colors = cols_k
            print(f"  - t {t:02d}: frame {fi:02d}, keep {pts_k.shape[0]} pts")
            if (t % 3) == 0:
                server.flush()
        server.flush()
    finally:
        if state.status_label:
            state.status_label.value = ""
        if state.slider_fg: state.slider_fg.disabled = False
        if state.slider_bg: state.slider_bg.disabled = False
        with state.lock:
            state.is_updating = False

def _update_traj_visibility(state: "ViewerState", server: viser.ViserServer, tidx: int, on: bool):
    if not state.traj_nodes:
        return
    with server.atomic():
        for t, nodes in enumerate(state.traj_nodes):
            vis = on and (t <= tidx)
            for nd in nodes:
                nd.visible = vis
    server.flush()

# ---------------- trajectories ----------------
def build_traj_nodes(
    server: viser.ViserServer,
    output: Dict,
    frames: List[Dict],
    traj_frames: List[int],
    t_vals: np.ndarray,
    max_points: int,
    state: "ViewerState",
):
    # remove old
    for lst in state.traj_nodes:
        for nd in lst:
            nd.remove()
    state.traj_nodes = [[] for _ in range(len(t_vals) - 1)]

    T = len(t_vals)
    vals01 = (np.arange(T - 1, dtype=np.float32)) / max(1, T - 2)
    seg_colors = hsv_colormap(vals01)  # [T-1, 3]

    print("[traj] building …")
    for fi in traj_frames:
        if fi < 0 or fi >= len(frames):
            continue
        fr = frames[fi]
        fg_mask_flat = fr["fg_mask_flat"]
        if not np.any(fg_mask_flat):
            continue

        fg_idx = np.flatnonzero(fg_mask_flat)
        if max_points > 0 and fg_idx.size > max_points:
            sel = np.random.default_rng(42).choice(fg_idx, size=max_points, replace=False)
            inv = {p: j for j, p in enumerate(fg_idx)}
            sel_fg = np.array([inv[p] for p in sel], dtype=np.int64)
        else:
            sel = fg_idx
            sel_fg = np.arange(fg_idx.size, dtype=np.int64)

        arr = np.stack([fr["pts_fg_per_t"][t] for t in range(T)], axis=0)
        if sel_fg.size == 0:
            continue
        arr = arr[:, sel_fg, :]  # [T,N_sel,3]

        for t in range(T - 1):
            p0 = arr[t]
            p1 = arr[t + 1]
            if p0.size == 0:
                continue
            segs = np.stack([p0, p1], axis=1)  # [N,2,3]
            col = np.repeat(seg_colors[t][None, :], segs.shape[0], axis=0)  # [N,3]
            node = server.scene.add_line_segments(
                name=f"/traj/frame{fi}/t{t:02d}",
                points=segs,
                colors=np.repeat(col[:, None, :], 2, axis=1),
                line_width=state.traj_width.value if state.traj_width else 0.075,
                visible=False,
            )
            state.traj_nodes[t].append(node)
    print("[traj] done.")

# ----------------- main viewer -----------------
def serve_view(output: Dict, port: int = 8020, t_step: float = 0.1, ds: int = 2):
    server = viser.ViserServer(port=port)
    server.gui.set_panel_label("TraceAnything Viewer")
    server.gui.configure_theme(control_layout="floating", control_width="medium", show_logo=False)

    # restore camera & scene setup
    server.scene.set_up_direction((0.0, -1.0, 0.0))
    server.scene.world_axes.visible = False

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle):
        with client.atomic():
            client.camera.position = (-0.00141163, -0.01910395, -0.06794288)
            client.camera.look_at  = (-0.00352821, -0.01143425,  0.01549390)
        client.flush()

    root = output.get("_root_dir", os.getcwd())
    images_dir = ensure_dir(os.path.join(root, "images"))
    masks_dir  = ensure_dir(os.path.join(root, "masks"))

    t_vals, frames, fg_conf_all_t, bg_conf_all_flat, times = build_precomputes(output, t_step, ds)
    nearest_idx = choose_nearest_frame_indices(times, t_vals)

    # save images only (masks are assumed precomputed)
    save_images(frames, images_dir)
    print(f"[viewer] Using precomputed FG masks in: {masks_dir} (or preds['fg_mask*'])")

    state = ViewerState()
    with server.gui.add_folder("Point Size", expand_by_default=True):
        state.point_size = server.gui.add_slider("FG Point Size", min=1e-5, max=1e-3, step=1e-4, initial_value=0.0002)
        state.bg_point_size = server.gui.add_slider("BG Point Size", min=1e-5, max=1e-3, step=1e-4, initial_value=0.0002)

    with server.gui.add_folder("Confidence Filtering", expand_by_default=True):
        state.slider_fg = server.gui.add_slider("FG percentile (drop bottom %)", min=0, max=100, step=1, initial_value=10)
        state.slider_bg = server.gui.add_slider("BG percentile (drop bottom %)", min=0, max=100, step=1, initial_value=10)

    with server.gui.add_folder("Playback", expand_by_default=True):
        state.slider_time = server.gui.add_slider("Time", min=0.0, max=1.0, step=t_step, initial_value=0.0)
        state.play_btn = server.gui.add_button("▶ Play")
        state.pause_btn = server.gui.add_button("⏸ Pause")
        state.fps_slider = server.gui.add_slider("FPS", min=1, max=60, step=1, initial_value=10)
        state.loop_checkbox = server.gui.add_checkbox("Loop", True)

    # ---- Trajectories panel ----
    with server.gui.add_folder("Trajectories", expand_by_default=True):
        state.show_traj = server.gui.add_checkbox("Show trajectories", False)
        state.traj_width = server.gui.add_slider("Line width", min=0.01, max=0.2, step=0.005, initial_value=0.075)
        state.traj_frames_text = server.gui.add_text("Frames (e.g. 0,mid,last)", initial_value="0,mid,last")
        state.traj_build_btn = server.gui.add_button("Build / Refresh")

    state.status_label = server.gui.add_markdown("")

    # build nodes
    build_bg_nodes(server, frames, init_percentile=state.slider_bg.value, bg_conf_all_flat=bg_conf_all_flat, state=state)
    build_fg_nodes(server, frames, nearest_idx, t_vals, fg_conf_all_t, init_percentile=state.slider_fg.value, state=state)
    print("\n[viewer] Ready. Open the printed URL and play with sliders!\n")

    # --- playback loop thread ---
    def _playback_loop():
        while True:
            if state.playing:
                try:
                    fps = max(1, int(state.fps_slider.value)) if state.fps_slider else 10
                    dt = 1.0 / float(fps)
                    tv = float(state.slider_time.value)
                    tv_next = tv + t_step
                    if tv_next > 1.0 - 1e-6:
                        if state.loop_checkbox and state.loop_checkbox.value:
                            tv_next = 0.0
                        else:
                            tv_next = 1.0 - 1e-6
                            state.playing = False
                    state.slider_time.value = tv_next
                except Exception:
                    pass
            time.sleep(dt if state.playing else 0.05)

    threading.Thread(target=_playback_loop, daemon=True).start()

    # callbacks
    @state.slider_time.on_update
    def _(_):
        tv = state.slider_time.value
        tidx = int(round(tv / t_step))
        tidx = max(0, min(tidx, len(t_vals) - 1))
        with server.atomic():
            for t in range(len(state.fg_nodes)):
                state.fg_nodes[t].visible = (t == tidx)
        server.flush()
        # NEW: only show trajectories up to current t
        _update_traj_visibility(state, server, tidx, on=(state.show_traj and state.show_traj.value))

    @state.play_btn.on_click
    def _(_):
        state.playing = True

    @state.pause_btn.on_click
    def _(_):
        state.playing = False

    def _rebuild_filter(_):
        update_conf_filtering(
            server=server,
            state=state,
            frames=frames,
            nearest_idx=nearest_idx,
            t_vals=t_vals,
            fg_conf_all_t=fg_conf_all_t,
            bg_conf_all_flat=bg_conf_all_flat,
            fg_percentile=state.slider_fg.value,
            bg_percentile=state.slider_bg.value,
        )

    @state.slider_fg.on_update
    def _(_):
        _rebuild_filter(_)

    @state.slider_bg.on_update
    def _(_):
        _rebuild_filter(_)

    @state.point_size.on_update
    def _(_):
        with server.atomic():
            for n in state.fg_nodes:
                n.point_size = state.point_size.value
        server.flush()

    @state.bg_point_size.on_update
    def _(_):
        with server.atomic():
            for n in state.bg_nodes:
                n.point_size = state.bg_point_size.value
        server.flush()

    # --- trajectories build/refresh and live controls ---
    def _parse_traj_frames():
        txt = (state.traj_frames_text.value or "").strip()
        if not txt:
            return []
        tokens = [t.strip() for t in txt.split(",") if t.strip()]
        n = len(frames)
        out_idx = []
        for tk in tokens:
            if tk == "mid":
                out_idx.append(n // 2)
            elif tk == "last":
                out_idx.append(n - 1)
            else:
                try:
                    out_idx.append(int(tk))
                except Exception:
                    pass
        out_idx = [i for i in sorted(set(out_idx)) if 0 <= i < n]
        return out_idx

    @state.traj_build_btn.on_click
    def _(_):
        sel = _parse_traj_frames()
        if not sel:
            return
        if state.status_label:
            state.status_label.value = "🧵 Building trajectories…"
        build_traj_nodes(
            server=server,
            output=output,
            frames=frames,
            traj_frames=sel,
            t_vals=t_vals,
            max_points=10000,
            state=state,
        )
        tidx = int(round(state.slider_time.value / t_step))
        tidx = max(0, min(tidx, len(t_vals) - 1))
        _update_traj_visibility(state, server, tidx, on=(state.show_traj and state.show_traj.value))
        if state.status_label:
            state.status_label.value = ""
        if state.show_traj.value:
            tidx = int(round(state.slider_time.value / t_step))
            tidx = max(0, min(tidx, len(t_vals) - 1))
            with server.atomic():
                for t, nodes in enumerate(state.traj_nodes):
                    vis = (t <= tidx)
                    for nd in nodes:
                        nd.visible = vis
            server.flush()

    @state.show_traj.on_update
    def _(_):
        tidx = int(round(state.slider_time.value / t_step))
        tidx = max(0, min(tidx, len(t_vals) - 1))
        _update_traj_visibility(state, server, tidx, on=state.show_traj.value)

    @state.traj_width.on_update
    def _(_):
        w = state.traj_width.value
        with server.atomic():
            for nodes in state.traj_nodes:
                for nd in nodes:
                    nd.line_width = w
        server.flush()

    return server

def parse_args():
    p = argparse.ArgumentParser("TraceAnything viewer")
    p.add_argument("--output", type=str, default="./examples/output/elephant/output.pt",
                   help="Path to output.pt or parent directory.")
    p.add_argument("--port", type=int, default=8020)
    p.add_argument("--t_step", type=float, default=0.025)
    p.add_argument("--ds", type=int, default=1, help="downsample stride for H,W (>=1)")
    return p.parse_args()

def main():
    args = parse_args()
    out = load_output_dict(args.output)
    server = serve_view(out, port=args.port, t_step=args.t_step, ds=max(1, args.ds))
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
