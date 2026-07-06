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

#!/usr/bin/env python3
# scripts/user_mask.py
"""
Interactive points on frame 0 -> SAM2 video propagation -> user masks.

Saves per-frame masks to <scene>/masks/{i:03d}_user.png
and stores preds[i]["fg_mask_user"] in <scene>/output.pt.
"""

import sys, pathlib; sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
import os, argparse, shutil, tempfile, inspect
from typing import Dict, List, Optional
import numpy as np, torch, cv2

def dilate_mask(mask: np.ndarray, ksize: int = 3, iterations: int = 2) -> np.ndarray:
    kernel = np.ones((ksize, ksize), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations).astype(bool)


def _pretty(msg:str):
    import time; print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def load_output(scene:str)->Dict:
    p=os.path.join(scene,"output.pt"); 
    if not os.path.isfile(p): raise FileNotFoundError(p)
    out=torch.load(p,map_location="cpu"); out["_root_dir"]=scene; return out

def tensor_to_rgb(img3hw:torch.Tensor)->np.ndarray:
    arr=(img3hw.detach().cpu().permute(1,2,0).numpy()+1.0)*127.5
    return np.clip(arr,0,255).astype(np.uint8)

def build_tmp_jpegs(frames_rgb: List[np.ndarray], scene_dir: str) -> str:
    tmp = os.path.join(scene_dir, "tmp_jpg")
    os.makedirs(tmp, exist_ok=True)
    for i, fr in enumerate(frames_rgb):
        fn = os.path.join(tmp, f"{i:06d}.jpg")
        cv2.imwrite(fn, cv2.cvtColor(fr, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return tmp

def ask_points(frame_bgr:np.ndarray, preview_dir:str)->np.ndarray:
    H,W=frame_bgr.shape[:2]
    while True:
        print("Enter points as: x y   (ENTER empty line to finish)")
        pts=[]
        while True:
            s=input(">>> ").strip()
            if s=="": break
            try:
                x,y=map(float,s.split()); pts.append([x,y,1.0])
            except: print("  ex: 320 180")
        if not pts:
            a=input("No points. [q]=quit, [r]=retry > ").strip().lower()
            if a.startswith("q"): raise SystemExit(0)
            else: continue
        pts=np.array(pts,dtype=np.float32)
        pts[:,0]=np.clip(pts[:,0],0,W-1); pts[:,1]=np.clip(pts[:,1],0,H-1)
        vis=frame_bgr.copy()
        for x,y,_ in pts:
            p=(int(round(x)),int(round(y)))
            cv2.circle(vis,p,4,(0,255,0),-1,cv2.LINE_AA); cv2.circle(vis,p,8,(0,255,0),1,cv2.LINE_AA)
        prev_path = os.path.join(preview_dir, "prompts_preview.png")
        cv2.imwrite(prev_path, vis)
        print(f"Preview: {os.path.abspath(prev_path)}")
        if input("Accept? [y]/n/q: ").strip().lower().startswith("y"): return pts
        if input("Retry or quit? [r]/q: ").strip().lower().startswith("q"): raise SystemExit(0)

from sam2.build_sam import build_sam2_video_predictor

def run_propagation(model_cfg:str, ckpt_path:str, jpg_dir:str, points_xy1:np.ndarray,
                    ann_frame_idx:int=0, ann_obj_id:int=1)->List[np.ndarray]:
    if not os.path.isfile(ckpt_path): raise FileNotFoundError(ckpt_path)
    predictor=build_sam2_video_predictor(model_cfg, ckpt_path)
    coords=points_xy1[:, :2][None,...].astype(np.float32); labels=points_xy1[:, 2][None,...].astype(np.int32)
    use_cuda=torch.cuda.is_available()
    autocast=torch.autocast("cuda",dtype=torch.bfloat16) if use_cuda else torch.cuda.amp.autocast(enabled=False)
    with torch.inference_mode(), autocast:
        state=predictor.init_state(jpg_dir)
        predictor.add_new_points_or_box(inference_state=state, frame_idx=ann_frame_idx,
                                        obj_id=ann_obj_id, points=coords, labels=labels)
        n=len([f for f in os.listdir(jpg_dir) if f.lower().endswith(".jpg")])
        masks:List[Optional[np.ndarray]]=[None]*n
        sig=inspect.signature(predictor.propagate_in_video)
        kwargs={"inference_state":state}
        if "start_frame_idx" in sig.parameters and "end_frame_idx" in sig.parameters:
            kwargs.update(start_frame_idx=ann_frame_idx,end_frame_idx=n-1)
        for yielded in predictor.propagate_in_video(**kwargs):
            if isinstance(yielded,tuple) and len(yielded)==3:
                fi,obj_ids,ms=yielded
            else:
                fi=int(yielded["frame_idx"]); obj_ids=yielded.get("object_ids") or yielded.get("obj_ids"); ms=yielded["masks"]
            pick=None
            for oid,m in zip(obj_ids,ms):
                if int(oid)==ann_obj_id: pick=m; break
            if pick is None:
                pick = masks[0]
            if isinstance(pick,torch.Tensor): pick=pick.detach().cpu().numpy()
            pick=np.asarray(pick); 
            if pick.ndim==3 and pick.shape[0]==1: pick=pick[0]
            masks[int(fi)]=(pick>0.5)
        if hasattr(predictor,"get_frame_masks"):
            for i in range(n):
                if masks[i] is None:
                    oids,ms=predictor.get_frame_masks(state,i)
                    pick=None
                    for oid,m in zip(oids,ms):
                        if int(oid)==ann_obj_id: pick=m; break
                    pick=pick or (ms[0] if ms else None)
                    if isinstance(pick,torch.Tensor): pick=pick.detach().cpu().numpy()
                    if pick is None: continue
                    pick=np.asarray(pick)
                    if pick.ndim==3 and pick.shape[0]==1: pick=pick[0]
                    masks[i]=(pick>0.5)
    H=W=None
    for m in masks:
        if m is not None: H,W=m.shape; break
    if H is None: raise RuntimeError("SAM2 produced no masks.")
    return [dilate_mask(m) if m is not None else np.zeros((H, W), bool) for m in masks]


def save_user_masks(scene:str, masks:List[np.ndarray], preds:List[Dict], views:List[Dict]):
    mdir=os.path.join(scene,"masks"); os.makedirs(mdir,exist_ok=True)
    for i,m in enumerate(masks):
        cv2.imwrite(os.path.join(mdir,f"{i:03d}_user.png"), (m.astype(np.uint8)*255))
        preds[i]["fg_mask_user"]=torch.from_numpy(m.astype(bool))
    torch.save({"preds":preds,"views":views}, os.path.join(scene,"output.pt"))

def parse_args():
    p=argparse.ArgumentParser("User mask via SAM2 video propagation")
    p.add_argument("--scene", type=str, default="./examples/output/breakdance",
                   help="Scene dir containing output.pt")
    p.add_argument("--sam2_cfg", type=str, default="configs/sam2.1/sam2.1_hiera_l.yaml",
                   help="SAM2 config (string or path; passed through)")
    p.add_argument("--sam2_ckpt", type=str, default="../sam2/checkpoints/sam2.1_hiera_large.pt",
                   help="Path to SAM2 checkpoint .pt")
    return p.parse_args()

def main():
    args=parse_args()
    out=load_output(args.scene); preds,views=out["preds"],out["views"]
    if not preds: return _pretty("No frames in output.pt")
    frames=[tensor_to_rgb(views[i]["img"].squeeze(0)) for i in range(len(views))]
    frame0_bgr=cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    tmp_dir = build_tmp_jpegs(frames, args.scene)
    pts_xy1 = ask_points(frame0_bgr, tmp_dir)
    try:
        _pretty("Propagating with SAM2 …")
        masks = run_propagation(
            model_cfg=args.sam2_cfg,
            ckpt_path=args.sam2_ckpt,
            jpg_dir=tmp_dir,
            points_xy1=pts_xy1,
            ann_frame_idx=0,
            ann_obj_id=1,
        )
        save_user_masks(args.scene, masks, preds, views)
        _pretty(f"✅ Saved user masks to {os.path.join(args.scene, 'masks')} and updated output.pt")
        _pretty(f"🗂️  Kept JPEG frames and preview in: {tmp_dir}")
    finally:
        # shutil.rmtree(tmp,ignore_errors=True)
        _pretty("🧹 Not removing temp JPEG folder (preview included)")

if __name__=="__main__": main()
