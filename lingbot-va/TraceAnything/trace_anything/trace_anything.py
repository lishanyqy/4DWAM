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

# trace_anything/trace_anything.py
"""
Minimal inference-only model for this repo.

Assumptions (pruned by asserts):
- encoder_type == 'croco'
- decoder_type == 'transformer'
- head_type == 'dpt'
- targeting_mechanism == 'bspline_conf'
- optional: whether_local (bool)
"""

import math
import time
from copy import deepcopy
from functools import partial
from typing import Dict, List

import torch
import torch.nn as nn
from einops import rearrange
import numpy as np

from .layers.blocks import Block, PositionGetter
from .layers.pos_embed import RoPE2D, get_1d_sincos_pos_embed_from_grid
from .layers.patch_embed import get_patch_embed
from .heads import PixelHead, ScalarHead

from contextlib import contextmanager
import time



# ======== B-spline  ========
PRECOMPUTED_KNOTS = {
    4:  torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    7:  torch.tensor([0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    10: torch.tensor([0.0, 0.0, 0.0, 0.0, 1/3, 1/3, 1/3, 2/3, 2/3, 2/3, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32),
}

def _precompute_knot_differences(n_ctrl_pts, degree, knots):
    denom1 = torch.zeros(n_ctrl_pts, degree + 1, device=knots.device)
    denom2 = torch.zeros(n_ctrl_pts, degree + 1, device=knots.device)
    for k in range(degree + 1):
        for i in range(n_ctrl_pts):
            denom1[i, k] = knots[i + k] - knots[i] if i + k < len(knots) else 0.0
            denom2[i, k] = knots[i + k + 1] - knots[i + 1] if i + k + 1 < len(knots) else 1.0
    return denom1, denom2

PRECOMPUTED_DENOMS = {n: _precompute_knot_differences(n, 3, PRECOMPUTED_KNOTS[n]) for n in [4, 7, 10]}

def _compute_bspline_basis(n_ctrl_pts, degree, t_values, knots, denom1, denom2):
    N = t_values.size(0)
    basis = torch.zeros(N, n_ctrl_pts, degree + 1, device=t_values.device)
    t = t_values
    basis_k0 = torch.zeros(N, n_ctrl_pts, device=t.device)
    for i in range(n_ctrl_pts):
        if i == n_ctrl_pts - 1:
            basis_k0[:, i] = ((knots[i] <= t) & (t <= knots[i + 1])).float()
        else:
            basis_k0[:, i] = ((knots[i] <= t) & (t < knots[i + 1])).float()
    basis[:, :, 0] = basis_k0
    for k in range(1, degree + 1):
        basis_k = torch.zeros(N, n_ctrl_pts, device=t.device)
        for i in range(n_ctrl_pts):
            term1 = ((t - knots[i]) / denom1[i, k]) * basis[:, i, k-1] if denom1[i, k] > 0 else 0.0
            term2 = ((knots[i + k + 1] - t) / denom2[i, k]) * basis[:, i + 1, k-1] if (denom2[i, k] > 0 and i + 1 < n_ctrl_pts) else 0.0
            basis_k[:, i] = term1 + term2
        basis[:, :, k] = basis_k
    return basis[:, :, degree]

def evaluate_bspline_conf(ctrl_pts3d, ctrl_conf, t_values):
    """ctrl_pts3d:[N_ctrl,H,W,3], ctrl_conf:[N_ctrl,H,W], t_values:[T] -> (T,H,W,3),(T,H,W)"""
    n_ctrl_pts, H, W, _ = ctrl_pts3d.shape
    assert n_ctrl_pts in (4, 7, 10), f"unsupported n_ctrl_pts={n_ctrl_pts}"
    degree = 3
    knot_vector = PRECOMPUTED_KNOTS[n_ctrl_pts].to(ctrl_pts3d.device)
    denom1, denom2 = [d.to(ctrl_pts3d.device) for d in PRECOMPUTED_DENOMS[n_ctrl_pts]]
    ctrl_pts3d = ctrl_pts3d.permute(0, 3, 1, 2)         # [N,3,H,W]
    ctrl_conf  = ctrl_conf.unsqueeze(-1).permute(0, 3, 1, 2)  # [N,1,H,W]
    basis = _compute_bspline_basis(n_ctrl_pts, degree, t_values, knot_vector, denom1, denom2)  # [T,N]
    basis = basis.view(-1, n_ctrl_pts, 1, 1, 1)               # [T,N,1,1,1]
    pts3d_t = torch.sum(basis * ctrl_pts3d.unsqueeze(0), dim=1).permute(0, 2, 3, 1)  # [T,H,W,3]
    conf_t  = torch.sum(basis * ctrl_conf.unsqueeze(0),  dim=1).squeeze(1)           # [T,H,W]
    return pts3d_t, conf_t

# ======== Encoders（仅 CroCo） ========
class CroCoEncoder(nn.Module):
    def __init__(
        self,
        img_size=512, patch_size=16, patch_embed_cls="ManyAR_PatchEmbed",
        embed_dim=768, num_heads=12, depth=12, mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        pos_embed="RoPE100", attn_implementation="pytorch_naive",
    ):
        super().__init__()
        assert pos_embed.startswith("RoPE"), f"pos_embed must start with RoPE*, got {pos_embed}"
        self.patch_embed = get_patch_embed(patch_embed_cls, img_size, patch_size, embed_dim)
        freq = float(pos_embed[len("RoPE"):])
        self.rope = RoPE2D(freq=freq)
        self.enc_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, norm_layer=nn.LayerNorm, rope=self.rope,
                  attn_implementation=attn_implementation)
            for _ in range(depth)
        ])
        self.enc_norm = norm_layer(embed_dim)

    def forward(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        for blk in self.enc_blocks:
            x = blk(x, pos)
        x = self.enc_norm(x)
        return x, pos


# ======== Decoder ========
class TraceDecoder(nn.Module):
    def __init__(
        self,
        random_image_idx_embedding: bool,
        enc_embed_dim: int,
        embed_dim: int = 768,
        num_heads: int = 12,
        depth: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        attn_implementation: str = "pytorch_naive",
        attn_bias_for_inference_enabled=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.decoder_embed = nn.Linear(enc_embed_dim, embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop, attn_drop=attn_drop,
                  norm_layer=nn.LayerNorm, attn_implementation=attn_implementation,
                  attn_bias_for_inference_enabled=attn_bias_for_inference_enabled)
            for _ in range(depth)
        ])
        self.random_image_idx_embedding = random_image_idx_embedding
        self.register_buffer(
            "image_idx_emb",
            torch.from_numpy(get_1d_sincos_pos_embed_from_grid(embed_dim, np.arange(1000))).float(),
            persistent=False,
        )
        self.dec_norm = norm_layer(embed_dim)

    def _get_random_image_pos(self, encoded_feats, batch_size, num_views, max_image_idx, device):
        image_ids = torch.zeros(batch_size, num_views, dtype=torch.long)
        image_ids[:, 0] = 0
        per_forward_pass_seed = torch.randint(0, 2 ** 32, (1,)).item()
        per_rank_generator = torch.Generator().manual_seed(per_forward_pass_seed)
        for b in range(batch_size):
            random_ids = torch.randperm(max_image_idx, generator=per_rank_generator)[:num_views - 1] + 1
            image_ids[b, 1:] = random_ids
        image_ids = image_ids.to(device)
        image_pos_list = []
        for i in range(num_views):
            num_patches = encoded_feats[i].shape[1]
            pos_for_view = self.image_idx_emb[image_ids[:, i]].unsqueeze(1).repeat(1, num_patches, 1)
            image_pos_list.append(pos_for_view)
        return torch.cat(image_pos_list, dim=1)

    def forward(self, encoded_feats: List[torch.Tensor], positions: List[torch.Tensor],
                image_ids: torch.Tensor, image_timesteps: torch.Tensor):
        x = torch.cat(encoded_feats, dim=1)
        pos = torch.cat(positions, dim=1)
        outputs = [x]
        x = self.decoder_embed(x)

        if self.random_image_idx_embedding:
            image_pos = self._get_random_image_pos(
                encoded_feats=encoded_feats,
                batch_size=encoded_feats[0].shape[0],
                num_views=len(encoded_feats),
                max_image_idx=self.image_idx_emb.shape[0] - 1,
                device=x.device,
            )
        else:
            num_embeddings = self.image_idx_emb.shape[0]
            indices = (image_timesteps * (num_embeddings - 1)).long().view(-1)
            image_pos = torch.index_select(self.image_idx_emb, dim=0, index=indices)
            image_pos = image_pos.view(1, image_timesteps.shape[1], self.image_idx_emb.shape[1])

        x += image_pos
        for blk in self.dec_blocks:
            x = blk(x, pos)
            outputs.append(x)

        x = self.dec_norm(x)
        outputs[-1] = x
        return outputs, image_pos


# ======== main model ========
class TraceAnything(nn.Module):
    def __init__(self, *, encoder_args: Dict, decoder_args: Dict, head_args: Dict,
                 targeting_mechanism: str = "bspline_conf",
                 poly_degree: int = 10, whether_local: bool = False):
        super().__init__()

        assert targeting_mechanism == "bspline_conf", f"Only bspline_conf is supported now, got {targeting_mechanism}"
        assert encoder_args.get("encoder_type") == "croco"
        assert decoder_args.get("decoder_type", "transformer") in ("transformer", "fast3r")
        assert head_args.get("head_type") == "dpt"

        self.targeting_mechanism = targeting_mechanism
        self.poly_degree = int(poly_degree)
        self.whether_local = bool(whether_local or head_args.get("with_local_head", False))

        # build encoder / decoder
        enc_args = deepcopy(encoder_args); enc_args.pop("encoder_type", None)
        self.encoder = CroCoEncoder(**enc_args)

        dec_args = deepcopy(decoder_args); dec_args.pop("decoder_type", None)
        self.decoder = TraceDecoder(**dec_args)

        # build heads
        feature_dim = 256
        last_dim = feature_dim // 2
        ed = encoder_args["embed_dim"]; dd = decoder_args["embed_dim"]
        hooks = [0, decoder_args["depth"] * 2 // 4, decoder_args["depth"] * 3 // 4, decoder_args["depth"]]
        self.ds_head_time = ScalarHead(
            num_channels=1, feature_dim=feature_dim, last_dim=last_dim,
            hooks_idx=hooks, dim_tokens=[ed, dd, dd, dd], head_type="regression",
            patch_size=head_args["patch_size"],
        )
        out_nchan = (3 + bool(head_args["conf_mode"])) * self.poly_degree
        self.ds_head_track = PixelHead(
            num_channels=out_nchan, feature_dim=feature_dim, last_dim=last_dim,
            hooks_idx=hooks, dim_tokens=[ed, dd, dd, dd], head_type="regression",
            patch_size=head_args["patch_size"],
            depth_mode=head_args["depth_mode"], conf_mode=head_args["conf_mode"],
        )
        if self.whether_local:
            self.ds_head_local = PixelHead(
                num_channels=out_nchan, feature_dim=feature_dim, last_dim=last_dim,
                hooks_idx=hooks, dim_tokens=[ed, dd, dd, dd], head_type="regression",
                patch_size=head_args["patch_size"],
                depth_mode=head_args["depth_mode"], conf_mode=head_args["conf_mode"],
            )

        self.time_head  = self.ds_head_time
        self.track_head = self.ds_head_track
        if self.whether_local:
            self.local_head = self.ds_head_local

        self.max_parallel_views_for_head = 25

    def _encode_images(self, views, chunk_size=400):
        B = views[0]["img"].shape[0]
        same_shape = all(v["img"].shape == views[0]["img"].shape for v in views)
        if same_shape:
            imgs = torch.cat([v["img"] for v in views], dim=0)
            true_shapes = torch.cat([v.get("true_shape", torch.tensor(v["img"].shape[-2:])[None].repeat(B, 1)) for v in views], dim=0)
            feats_chunks, pos_chunks = [], []
            for s in range(0, imgs.shape[0], chunk_size):
                e = min(s + chunk_size, imgs.shape[0])
                f, p = self.encoder(imgs[s:e], true_shapes[s:e])
                feats_chunks.append(f); pos_chunks.append(p)
            feats = torch.cat(feats_chunks, dim=0); pos = torch.cat(pos_chunks, dim=0)
            encoded_feats = torch.split(feats, B, dim=0)
            positions    = torch.split(pos,   B, dim=0)
            shapes       = torch.split(true_shapes, B, dim=0)
        else:
            encoded_feats, positions, shapes = [], [], []
            for v in views:
                img = v["img"]
                true_shape = v.get("true_shape", torch.tensor(img.shape[-2:])[None].repeat(B, 1))
                f, p = self.encoder(img, true_shape)
                encoded_feats.append(f); positions.append(p); shapes.append(true_shape)
        return encoded_feats, positions, shapes

    @torch.no_grad()
    def forward(self, views, profiling: bool = False):
        # 1) encode
        encoded_feats, positions, shapes = self._encode_images(views)

        # 2) build time embedding
        num_images = len(views)
        B, P, D = encoded_feats[0].shape
        image_ids, image_times = [], []
        for i, ef in enumerate(encoded_feats):
            num_patches = ef.shape[1]
            image_ids.extend([i] * num_patches)
            image_times.extend([views[i]["time_step"]] * num_patches)
        image_ids  = torch.tensor(image_ids * B,  device=encoded_feats[0].device).reshape(B, -1)
        image_times = torch.tensor(image_times * B, device=encoded_feats[0].device).reshape(B, -1)

        # 3) decode
        dec_output, _ = self.decoder(encoded_feats, positions, image_ids, image_times)

        # 4) gather outputs per view
        P_patches = P
        gathered_outputs_list = []
        for layer_output in dec_output:
            layer_output = rearrange(layer_output, 'B (n P) D -> (n B) P D', n=num_images, P=P_patches)
            gathered_outputs_list.append(layer_output)

        # 5) heads
        time_step = self.time_head(gathered_outputs_list, torch.stack(shapes))          # [N, 1] per view
        track_tmp = self.track_head(gathered_outputs_list, torch.stack(shapes))        # dict with pts3d/conf after postprocess
        ctrl_pts3d, ctrl_conf = track_tmp['pts3d'], track_tmp['conf']                  # [K,N,H,W,3], [K,N,H,W]

        if self.whether_local:
            local_tmp = self.local_head(gathered_outputs_list, torch.stack(shapes))
            ctrl_pts3d_local, ctrl_conf_local = local_tmp['pts3d'], local_tmp['conf']

        # 6) evaluate bspline track over all reference times
        results = [{} for _ in range(num_images)]
        t_values = torch.stack([time_step[i].squeeze() for i in range(num_images)])    # [N]
        for img_id in range(num_images):
            pts3d_t, conf_t = evaluate_bspline_conf(ctrl_pts3d[:, img_id], ctrl_conf[:, img_id], t_values.detach())
            res = {
                'time': time_step[img_id],
                'ctrl_pts3d': ctrl_pts3d[:, img_id],
                'ctrl_conf':  ctrl_conf[:,  img_id],
                'track_pts3d': [pts3d_t[[t]] for t in range(num_images)],
                'track_conf':  [conf_t[[t]]  for t in range(num_images)],
            }
            if self.whether_local:
                pts3d_t_l, conf_t_l = evaluate_bspline_conf(ctrl_pts3d_local[:, img_id], ctrl_conf_local[:, img_id], t_values.detach())
                res.update({
                    'ctrl_pts3d_local': ctrl_pts3d_local[:, img_id],
                    'ctrl_conf_local':  ctrl_conf_local[:,  img_id],
                    'track_pts3d_local': [pts3d_t_l[[t]] for t in range(num_images)],
                    'track_conf_local':  [conf_t_l[[t]]  for t in range(num_images)],
                })
            results[img_id] = res

        return results


    
