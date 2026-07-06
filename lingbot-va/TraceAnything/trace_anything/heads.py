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

# trace_anything/heads.py
from typing import List
import torch
import torch.nn as nn
from einops import rearrange
from .layers.dpt_block import DPTOutputAdapter  # 本地化


def _resolve_hw(img_info):
    """Normalize `img_info` into the form (int(H), int(W)).
    Allowed input formats:
      - torch.Tensor of shape (..., 2) 
        (e.g. [num_views, B, 2] or [B, 2])
      - (h, w) tuple/list, where elements can be int or Tensor
    Requirement: H/W must be consistent across the whole batch; 
    otherwise, raise an error 
    (current inference path does not support mixed-size batches).
    """
    if isinstance(img_info, torch.Tensor):
        assert img_info.shape[-1] == 2, f"img_info last dim must be 2, got {img_info.shape}"
        h0 = img_info.reshape(-1, 2)[0, 0].item()
        w0 = img_info.reshape(-1, 2)[0, 1].item()
        if (img_info[..., 0] != img_info[..., 0].reshape(-1)[0]).any() or \
           (img_info[..., 1] != img_info[..., 1].reshape(-1)[0]).any():
            raise AssertionError(f"Mixed H/W in batch not supported: {tuple(img_info.shape)}")
        return int(h0), int(w0)

    if isinstance(img_info, (list, tuple)) and len(img_info) == 2:
        h, w = img_info
        if isinstance(h, torch.Tensor): h = int(h.reshape(-1)[0].item())
        if isinstance(w, torch.Tensor): w = int(w.reshape(-1)[0].item())
        return int(h), int(w)

    raise TypeError(f"Unexpected img_info type: {type(img_info)}")


# ---------- postprocess ----------
def reg_dense_depth(xyz, mode):
    mode, vmin, vmax = mode
    no_bounds = (vmin == -float("inf")) and (vmax == float("inf"))
    assert no_bounds
    if mode == "linear":
        return xyz if no_bounds else xyz.clip(min=vmin, max=vmax)
    d = xyz.norm(dim=-1, keepdim=True)
    xyz = xyz / d.clip(min=1e-8)
    if mode == "square":
        return xyz * d.square()
    if mode == "exp":
        return xyz * torch.expm1(d)
    raise ValueError(f"bad {mode=}")

def reg_dense_conf(x, mode):
    mode, vmin, vmax = mode
    if mode == "exp":
        return vmin + x.exp().clip(max=vmax - vmin)
    if mode == "sigmoid":
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f"bad {mode=}")

def postprocess(out, depth_mode, conf_mode):
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C
    res = dict(pts3d=reg_dense_depth(fmap[..., 0:3], mode=depth_mode))
    if conf_mode is not None:
        res["conf"] = reg_dense_conf(fmap[..., 3], mode=conf_mode)
    return res

def postprocess_multi_point(out, depth_mode, conf_mode):
    fmap = out.permute(0, 2, 3, 1)  # B,H,W,C
    B, H, W, C = fmap.shape
    n_point = C // 4
    pts_3d = fmap[..., :3*n_point].view(B, H, W, 3, n_point).permute(4, 0, 1, 2, 3)  # [K,B,H,W,3]
    conf   = fmap[..., 3*n_point:].view(B, H, W, 1, n_point).squeeze(3).permute(3, 0, 1, 2)  # [K,B,H,W]
    res = dict(pts3d=reg_dense_depth(pts_3d, mode=depth_mode))
    res["conf"] = reg_dense_conf(conf, mode=conf_mode)
    return res

class DPTOutputAdapterFix(DPTOutputAdapter):
    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess
    
    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert (
            self.dim_tokens_enc is not None
        ), "Need to call init(dim_tokens_enc) function first"
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        H, W = int(H), int(W) 

        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        layers = [encoder_tokens[h] for h in self.hooks]          # 4 x [B,N,C]
        layers = [self.adapt_tokens(l) for l in layers]           # 4 x [B,N,C]
        layers = [rearrange(l, "b (nh nw) c -> b c nh nw", nh=N_H, nw=N_W) for l in layers]
        layers = [self.act_postprocess[i](l) for i, l in enumerate(layers)]
        layers = [self.scratch.layer_rn[i](l) for i, l in enumerate(layers)]
        p4 = self.scratch.refinenet4(layers[3])[:, :, : layers[2].shape[2], : layers[2].shape[3]]
        p3 = self.scratch.refinenet3(p4, layers[2])
        p2 = self.scratch.refinenet2(p3, layers[1])
        p1 = self.scratch.refinenet1(p2, layers[0])

        max_chunk = 1 if self.training else 50
        outs = []
        for ch in torch.split(p1, max_chunk, dim=0):
            outs.append(self.head(ch))
        return torch.cat(outs, dim=0)  # [B,C,H,W]

# ---------- Heads ----------
class ScalarHead(nn.Module):
    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, **kwargs):
        super().__init__()
        assert n_cls_token == 0
        dpt_args = dict(output_width_ratio=output_width_ratio, num_channels=num_channels, **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapterFix(**dpt_args)
        dpt_init = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init)
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # B,C,1,1  (C==1)
            nn.Flatten(),             # B,1
            nn.Linear(1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x_list, img_info):
        H, W = _resolve_hw(img_info)                     
        out = self.dpt(x_list, image_size=(H, W))       # [B,1,H,W]
        return self.scalar_head(out)                    # [B,1]

class PixelHead(nn.Module):
    """Output per-pixel (3D point + confidence), supports multiple points (K)."""
    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1,
                 postprocess=postprocess_multi_point,
                 depth_mode=None, conf_mode=None, **kwargs):
        super().__init__()
        assert n_cls_token == 0
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode  = conf_mode
        dpt_args = dict(output_width_ratio=output_width_ratio, num_channels=num_channels, **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapterFix(**dpt_args)
        dpt_init = {} if dim_tokens is None else {"dim_tokens_enc": dim_tokens}
        self.dpt.init(**dpt_init)

    def forward(self, x_list, img_info):
        H, W = _resolve_hw(img_info)                    
        out = self.dpt(x_list, image_size=(H, W))        # [B,C,H,W]
        return self.postprocess(out, self.depth_mode, self.conf_mode)