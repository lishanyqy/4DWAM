from pathlib import Path
import sys
from typing import Any, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

from fastwam.utils.logging_config import get_logger

from .action_dit import ActionDiT
from .alignment import unified_dest_and_motion
from .helpers.loader import load_wan22_ti2v_5b_components
from .mot import MoT
from .schedulers.scheduler_continuous import WanContinuousFlowMatchScheduler

logger = get_logger(__name__)


class FastWAM(torch.nn.Module):
    """MoT world model with video/action experts."""

    def __init__(
        self,
        video_expert,
        action_expert: ActionDiT,
        mot: MoT,
        vae,
        text_encoder=None,
        tokenizer=None,
        text_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        traceanything_config: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.video_expert = video_expert
        self.action_expert = action_expert
        self.mot = mot
        # Keep trainer compatibility: optimizer and freeze logic use `model.dit`.
        self.dit = self.mot

        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        if text_dim is None:
            if self.text_encoder is None:
                raise ValueError("`text_dim` is required when `text_encoder` is not loaded.")
            text_dim = int(self.text_encoder.dim)
        self.text_dim = int(text_dim)
        self.proprio_dim = None if proprio_dim is None else int(proprio_dim)
        if self.proprio_dim is not None:
            self.proprio_encoder = nn.Linear(self.proprio_dim, self.text_dim).to(torch_dtype)
        else:
            self.proprio_encoder = None
        self.traceanything = None
        self.traceanything_camera_sizes = None
        self.traceanything_camera_order = None
        self.traceanything_patch_size = None
        self.traceanything_spatial_layout = "auto"
        self.traceanything_align_to_video_patch = True
        self.traceanything_max_frames = None
        self.traceanything_dtype = None
        self.trace_align_layer = None
        self.trace_loss_weight = 0.0
        self.trace_dest_loss_enabled = True
        self.trace_motion_loss_enabled = True
        self.trace_dest_loss_weight = 1.0
        self.trace_motion_loss_weight = 1.0
        self.trace_proj = None
        self.align_repr_proj = None
        if traceanything_config is not None and bool(traceanything_config.get("enabled", False)):
            self._init_traceanything(traceanything_config, torch_dtype=torch_dtype)

        self.train_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_train_shift,
        )
        self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=video_num_train_timesteps,
            shift=video_infer_shift,
        )
        self.train_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_train_shift,
        )
        self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
            num_train_timesteps=action_num_train_timesteps,
            shift=action_infer_shift,
        )
        # Optional aliases for consistency with Wan22Core naming.
        self.train_scheduler = self.train_video_scheduler
        self.infer_scheduler = self.infer_video_scheduler

        self.device = torch.device(device)
        self.torch_dtype = torch_dtype
        self.loss_lambda_video = float(loss_lambda_video)
        self.loss_lambda_action = float(loss_lambda_action)

        self.to(self.device)
        
        self.is_checked = False

    def _init_traceanything(self, cfg: dict[str, Any], torch_dtype: torch.dtype):
        repo_root = Path(__file__).resolve().parents[4]
        trace_root = Path(cfg.get("root", repo_root / "src" / "TraceAnything")).expanduser()
        config_path = Path(cfg.get("config", trace_root / "configs" / "eval.yaml")).expanduser()
        ckpt_path = Path(
            cfg.get(
                "ckpt",
                "/root/.cache/models/trace-anything/trace_anything.pt",
            )
        ).expanduser()
        if not ckpt_path.exists():
            ckpt_path = repo_root / "checkpoints" / "trace_anything.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"TraceAnything checkpoint not found: {ckpt_path}")
        if not config_path.exists():
            raise FileNotFoundError(f"TraceAnything config not found: {config_path}")

        sys.path.insert(0, str(trace_root))
        from trace_anything.trace_anything import TraceAnything

        ta_cfg = OmegaConf.load(config_path)
        net_cfg = ta_cfg.get("model", {}).get("net", None) or ta_cfg.get("net", None)
        if net_cfg is None:
            raise KeyError("Expected TraceAnything config to contain model.net or net.")

        def _to_dict(x):
            return OmegaConf.to_container(x, resolve=True) if not isinstance(x, dict) else x

        model = TraceAnything(
            encoder_args=_to_dict(net_cfg["encoder_args"]),
            decoder_args=_to_dict(net_cfg["decoder_args"]),
            head_args=_to_dict(net_cfg["head_args"]),
            targeting_mechanism=net_cfg.get("targeting_mechanism", "bspline_conf"),
            poly_degree=net_cfg.get("poly_degree", 10),
            whether_local=False,
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and isinstance(ckpt.get("state_dict"), dict) else ckpt
        if all(k.startswith("net.") for k in state_dict.keys()):
            state_dict = {k[4:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        model.to(dtype=torch_dtype)
        model.eval().requires_grad_(False)

        trace_dim = int(net_cfg["decoder_args"]["embed_dim"])
        self.traceanything_patch_size = int(net_cfg["encoder_args"].get("patch_size", 16))
        self.traceanything = model
        self.traceanything_dtype = torch_dtype
        self.traceanything_feature_dim = trace_dim
        target_dim = int(cfg.get("alignment_target_dim", 2048))
        video_hidden_dim = int(getattr(self.video_expert, "hidden_dim"))
        self.trace_proj = nn.Sequential(
            nn.Linear(trace_dim, video_hidden_dim, dtype=torch_dtype),
            nn.GELU(),
            nn.Linear(video_hidden_dim, target_dim, dtype=torch_dtype),
        )
        self.align_repr_proj = nn.Sequential(
            nn.Linear(video_hidden_dim, target_dim, dtype=torch_dtype),
            nn.GELU(),
            nn.Linear(target_dim, target_dim, dtype=torch_dtype),
        )
        self.trace_align_layer = int(cfg.get("align_layer", 16))
        self.trace_loss_weight = float(cfg.get("loss_weight", cfg.get("trace_coef", 0.0)))
        self.trace_dest_loss_enabled = bool(cfg.get("enable_dest_loss", True))
        self.trace_motion_loss_enabled = bool(cfg.get("enable_motion_loss", True))
        self.trace_dest_loss_weight = float(cfg.get("dest_loss_weight", 1.0))
        self.trace_motion_loss_weight = float(cfg.get("motion_loss_weight", 1.0))
        self.traceanything_camera_sizes = {
            "cam_high": (256, 320),
            "cam_left_wrist": (128, 160),
            "cam_right_wrist": (128, 160),
            "observation.images.cam_high": (256, 320),
            "observation.images.cam_left_wrist": (128, 160),
            "observation.images.cam_right_wrist": (128, 160),
        }
        custom_sizes = cfg.get("camera_sizes", None)
        if custom_sizes is not None:
            self.traceanything_camera_sizes.update({k: tuple(v) for k, v in custom_sizes.items()})
        self.traceanything_camera_order = list(cfg.get("camera_order", [
            "cam_high",
            "cam_left_wrist",
            "cam_right_wrist",
        ]))
        self.traceanything_spatial_layout = str(cfg.get("spatial_layout", "auto"))
        self.traceanything_align_to_video_patch = bool(cfg.get("align_to_video_patch", True))
        self.traceanything_max_frames = None if cfg.get("max_frames") is None else int(cfg.get("max_frames"))
        logger.info(
            "Loaded frozen TraceAnything from %s with config %s; trace_dim=%d align_layer=%d "
            "trace_loss_weight=%s enable_dest_loss=%s enable_motion_loss=%s spatial_layout=%s "
            "align_to_video_patch=%s",
            ckpt_path,
            config_path,
            trace_dim,
            self.trace_align_layer,
            self.trace_loss_weight,
            self.trace_dest_loss_enabled,
            self.trace_motion_loss_enabled,
            self.traceanything_spatial_layout,
            self.traceanything_align_to_video_patch,
        )

    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        vae_path: str | None = None,
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        traceanything_config: Optional[dict[str, Any]] = None,
    ):
        if video_dit_config is None:
            raise ValueError("`video_dit_config` is required for FastWAM.from_wan22_pretrained().")
        if "text_dim" not in video_dit_config:
            raise ValueError("`video_dit_config['text_dim']` is required for FastWAM.")

        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            vae_path=vae_path,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.num_heads) != int(video_expert.num_heads):
            raise ValueError("ActionDiT `num_heads` must match video expert for MoT mixed attention.")
        if int(action_expert.attn_head_dim) != int(video_expert.attn_head_dim):
            raise ValueError("ActionDiT `attn_head_dim` must match video expert for MoT mixed attention.")
        if int(len(action_expert.blocks)) != int(len(video_expert.blocks)):
            raise ValueError("ActionDiT `num_layers` must match video expert.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            traceanything_config=traceanything_config,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.mot.to(*args, **kwargs)
        if self.text_encoder is not None:
            self.text_encoder.to(*args, **kwargs)
        self.vae.to(*args, **kwargs)
        if self.traceanything is not None:
            self.traceanything.to(*args, **kwargs)
            self.traceanything_dtype = next(self.traceanything.parameters()).dtype
        return self

    def _get_traceanything_dtype(self) -> torch.dtype:
        if self.traceanything is None:
            raise RuntimeError("TraceAnything is not enabled.")
        try:
            return next(self.traceanything.parameters()).dtype
        except StopIteration:
            if self.traceanything_dtype is None:
                return self.torch_dtype
            return self.traceanything_dtype

    @staticmethod
    def _check_resize_height_width(height, width, num_frames):
        if height % 16 != 0:
            height = (height + 15) // 16 * 16
        if width % 16 != 0:
            width = (width + 15) // 16 * 16
        if num_frames % 4 != 1:
            num_frames = (num_frames + 3) // 4 * 4 + 1
        return height, width, num_frames

    @torch.no_grad()
    def encode_prompt(self, prompt: Union[str, Sequence[str]]):
        if self.text_encoder is None or self.tokenizer is None:
            raise ValueError(
                "Prompt encoding requires loaded text encoder/tokenizer. "
                "Set `load_text_encoder=true` or provide precomputed `context/context_mask`."
            )
        ids, mask = self.tokenizer(prompt, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device, dtype=torch.bool)
        prompt_emb = self.text_encoder(ids, mask)
        # FIXME: original implementation's zero padding is visible in cross-attn.
        seq_lens = mask.gt(0).sum(dim=1).long()
        for i, v in enumerate(seq_lens):
            prompt_emb[i, v:] = 0
        mask = torch.ones_like(mask)
        return prompt_emb.to(device=self.device), mask

    def _append_proprio_to_context(
        self,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        proprio: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.proprio_encoder is None or proprio is None:
            return context, context_mask
        if proprio.ndim != 2:
            raise ValueError(f"`proprio` must be 2D [B, D], got shape {tuple(proprio.shape)}")
        if self.proprio_dim is None or proprio.shape[1] != self.proprio_dim:
            raise ValueError(
                f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}"
            )
        proprio_token = self.proprio_encoder(
            proprio.to(device=self.device, dtype=context.dtype).unsqueeze(1)
        ).to(dtype=context.dtype) # [B, 1, D]
        proprio_mask = torch.ones((context_mask.shape[0], 1), dtype=torch.bool, device=context_mask.device)
        return (
            torch.cat([context, proprio_token], dim=1),
            torch.cat([context_mask, proprio_mask], dim=1),
        )

    @staticmethod
    def _build_trace_timesteps(num_frames: int) -> list[float]:
        if num_frames <= 1:
            return [0.0]
        return [i / (num_frames - 1) for i in range(num_frames)]

    def _resolve_trace_spatial_layout(self, num_cameras: int) -> str:
        layout = str(self.traceanything_spatial_layout or "auto").lower()
        if layout == "auto":
            if num_cameras == 3:
                return "robotwin"
            if num_cameras == 2:
                return "horizontal"
            return "sequential"
        aliases = {
            "none": "sequential",
            "concat": "sequential",
            "token_concat": "sequential",
        }
        return aliases.get(layout, layout)

    def _pool_trace_grid_to_video_patch(self, trace_grid: torch.Tensor) -> torch.Tensor:
        if not self.traceanything_align_to_video_patch:
            return trace_grid.flatten(2, 3)

        patch_size = getattr(self.video_expert, "patch_size", (1, 1, 1))
        patch_h = int(patch_size[1])
        patch_w = int(patch_size[2])
        if patch_h <= 0 or patch_w <= 0:
            raise ValueError(f"Invalid video patch size for trace alignment: {patch_size}")

        batch_size, num_frames, grid_h, grid_w, dim = trace_grid.shape
        if grid_h % patch_h != 0 or grid_w % patch_w != 0:
            raise ValueError(
                "TraceAnything stitched grid must be divisible by video DiT patch size, "
                f"got grid=({grid_h}, {grid_w}), patch=({patch_h}, {patch_w}). "
                "Check traceanything.camera_sizes and spatial_layout."
            )

        trace_grid = trace_grid.view(
            batch_size,
            num_frames,
            grid_h // patch_h,
            patch_h,
            grid_w // patch_w,
            patch_w,
            dim,
        )
        trace_grid = trace_grid.mean(dim=(3, 5))
        return trace_grid.reshape(batch_size, num_frames, -1, dim)

    def _stitch_traceanything_tokens(
        self,
        camera_tokens: list[torch.Tensor],
        camera_grids: list[tuple[int, int]],
    ) -> torch.Tensor:
        if len(camera_tokens) == 0:
            raise ValueError("No TraceAnything camera tokens were extracted.")

        layout = self._resolve_trace_spatial_layout(len(camera_tokens))
        if layout == "sequential":
            return torch.cat(camera_tokens, dim=2)

        camera_maps = []
        for tokens, (grid_h, grid_w) in zip(camera_tokens, camera_grids):
            expected_tokens = grid_h * grid_w
            if tokens.shape[2] != expected_tokens:
                raise ValueError(
                    "TraceAnything token count does not match the configured camera grid: "
                    f"tokens={tokens.shape[2]}, grid=({grid_h}, {grid_w})."
                )
            camera_maps.append(tokens.view(tokens.shape[0], tokens.shape[1], grid_h, grid_w, tokens.shape[3]))

        if layout == "robotwin":
            if len(camera_maps) != 3:
                raise ValueError(f"`spatial_layout='robotwin'` requires 3 cameras, got {len(camera_maps)}.")
            top, left, right = camera_maps
            if left.shape[2] != right.shape[2] or left.shape[3] != right.shape[3]:
                raise ValueError(
                    "RoboTwin wrist TraceAnything grids must have identical shapes, "
                    f"got left={tuple(left.shape[2:4])}, right={tuple(right.shape[2:4])}."
                )
            if top.shape[3] != left.shape[3] + right.shape[3]:
                raise ValueError(
                    "RoboTwin top TraceAnything grid width must equal left+right wrist widths, "
                    f"got top_w={top.shape[3]}, left_w={left.shape[3]}, right_w={right.shape[3]}."
                )
            bottom = torch.cat([left, right], dim=3)
            stitched = torch.cat([top, bottom], dim=2)
        elif layout == "horizontal":
            grid_h = camera_maps[0].shape[2]
            if any(cam.shape[2] != grid_h for cam in camera_maps):
                raise ValueError(
                    "`spatial_layout='horizontal'` requires all TraceAnything camera grids to "
                    f"have the same height, got {[tuple(cam.shape[2:4]) for cam in camera_maps]}."
                )
            stitched = torch.cat(camera_maps, dim=3)
        elif layout == "vertical":
            grid_w = camera_maps[0].shape[3]
            if any(cam.shape[3] != grid_w for cam in camera_maps):
                raise ValueError(
                    "`spatial_layout='vertical'` requires all TraceAnything camera grids to "
                    f"have the same width, got {[tuple(cam.shape[2:4]) for cam in camera_maps]}."
                )
            stitched = torch.cat(camera_maps, dim=2)
        else:
            raise ValueError(
                f"Unsupported traceanything.spatial_layout={self.traceanything_spatial_layout!r}. "
                "Expected one of: auto, sequential, robotwin, horizontal, vertical."
            )

        return self._pool_trace_grid_to_video_patch(stitched)

    @torch.no_grad()
    def _extract_traceanything_tokens(self, trace_video: torch.Tensor) -> torch.Tensor:
        if self.traceanything is None:
            raise RuntimeError("TraceAnything is not enabled.")
        if trace_video.ndim != 6:
            raise ValueError(
                "`sample['trace_video']` must be 6D [B,num_cameras,T,3,H,W], "
                f"got shape {tuple(trace_video.shape)}"
            )
        if trace_video.shape[3] != 3:
            raise ValueError(f"`trace_video` channel dimension must be 3, got {tuple(trace_video.shape)}")

        batch_size, num_cameras, num_frames, _, _, _ = trace_video.shape
        trace_dtype = self._get_traceanything_dtype()
        trace_video = trace_video.to(device=self.device, dtype=torch.float32, non_blocking=True)

        camera_tokens = []
        camera_grids = []
        for cam_idx in range(num_cameras):
            camera_name = (
                self.traceanything_camera_order[cam_idx]
                if cam_idx < len(self.traceanything_camera_order)
                else str(cam_idx)
            )
            resize_hw = self.traceanything_camera_sizes.get(camera_name)
            if resize_hw is None:
                resize_hw = tuple(trace_video.shape[-2:])
            resize_hw = tuple(int(v) for v in resize_hw)
            patch_size = int(self.traceanything_patch_size or 16)
            if resize_hw[0] % patch_size != 0 or resize_hw[1] % patch_size != 0:
                raise ValueError(
                    "TraceAnything camera resize must be divisible by its patch size, "
                    f"got camera={camera_name}, resize={resize_hw}, patch_size={patch_size}."
                )

            frames = trace_video[:, cam_idx]  # [B,T,3,H,W], expected [0,1]
            frames = frames.flatten(0, 1)  # [B*T,3,H,W]
            frames = F.interpolate(frames, size=resize_hw, mode="bilinear", align_corners=False)
            frames = frames * 2.0 - 1.0
            frames = frames.to(dtype=trace_dtype)
            frames = frames.view(batch_size, num_frames, 3, resize_hw[0], resize_hw[1])

            frame_indices = list(range(num_frames))
            if self.traceanything_max_frames is not None and num_frames > self.traceanything_max_frames:
                frame_indices = torch.linspace(
                    0,
                    num_frames - 1,
                    self.traceanything_max_frames,
                    device="cpu",
                ).round().long().unique().tolist()
                frames = frames[:, frame_indices]

            frame_list = [frames[:, t] for t in range(frames.shape[1])]
            timesteps = self._build_trace_timesteps(len(frame_list))
            features = self.traceanything.extract_seq_features(frame_list, timesteps)
            features = features.view(len(frame_list), batch_size, features.shape[1], features.shape[2])
            features = features.permute(1, 0, 2, 3).contiguous()  # [B,T,P,D]
            camera_tokens.append(features)
            camera_grids.append((resize_hw[0] // patch_size, resize_hw[1] // patch_size))

        return self._stitch_traceanything_tokens(camera_tokens, camera_grids).to(dtype=self.torch_dtype)

    def _maybe_extract_traceanything_features(
        self,
        trace_video: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.traceanything is None:
            return None
        if trace_video is None:
            raise ValueError("TraceAnything is enabled, but sample does not contain `trace_video`.")
        return self._extract_traceanything_tokens(trace_video)

    @staticmethod
    def _resample_trace_features(
        trace_features: torch.Tensor,
        target_frames: int,
        target_tokens_per_frame: int,
    ) -> tuple[torch.Tensor, int]:
        if trace_features.ndim != 4:
            raise ValueError(
                f"`trace_features` must be [B,F,T,D], got shape {tuple(trace_features.shape)}"
            )
        batch_size, num_frames, num_tokens, dim = trace_features.shape
        work_dtype = trace_features.dtype
        trace = trace_features.float()

        if num_frames != target_frames:
            trace = trace.permute(0, 2, 3, 1).reshape(batch_size * num_tokens, dim, num_frames)
            trace = F.interpolate(trace, size=target_frames, mode="linear", align_corners=False)
            trace = trace.reshape(batch_size, num_tokens, dim, target_frames).permute(0, 3, 1, 2)

        if num_tokens != target_tokens_per_frame:
            trace = trace.reshape(batch_size * target_frames, num_tokens, dim).permute(0, 2, 1)
            trace = F.interpolate(trace, size=target_tokens_per_frame, mode="linear", align_corners=False)
            trace = trace.permute(0, 2, 1).reshape(batch_size, target_frames, target_tokens_per_frame, dim)

        trace = trace.reshape(batch_size, target_frames * target_tokens_per_frame, dim)
        return trace.to(dtype=work_dtype), target_tokens_per_frame

    def _compute_trace_alignment_loss(
        self,
        video_hidden_states: torch.Tensor,
        trace_features: torch.Tensor,
        video_pre: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        if self.trace_proj is None or self.align_repr_proj is None:
            raise RuntimeError("Trace alignment projections are not initialized.")
        grid_f, _, _ = video_pre["meta"]["grid_size"]
        tokens_per_frame = int(video_pre["meta"]["tokens_per_frame"])
        trace_hidden_states, tokens_per_frame = self._resample_trace_features(
            trace_features,
            target_frames=int(grid_f),
            target_tokens_per_frame=tokens_per_frame,
        )
        align_hidden_states = self.align_repr_proj(video_hidden_states)
        trace_hidden_states = self.trace_proj(trace_hidden_states.to(
            device=video_hidden_states.device,
            dtype=video_hidden_states.dtype,
        ))
        return unified_dest_and_motion(
            align_hidden_states,
            trace_hidden_states,
            tokens_per_frame=tokens_per_frame,
        )

    @torch.no_grad()
    def _encode_video_latents(self, video_tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        z = self.vae.encode(
            video_tensor,
            device=self.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        return z

    @torch.no_grad()
    def _encode_input_image_latents_tensor(self, input_image: torch.Tensor, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        image = input_image.to(device=self.device)[0].unsqueeze(1)
        z = self.vae.encode([image], device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        if isinstance(z, list):
            z = z[0].unsqueeze(0)
        return z

    def _decode_latents(self, latents, tiled=False, tile_size=(30, 52), tile_stride=(15, 26)):
        video_tensor = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        video_tensor = video_tensor.squeeze(0).detach().float().clamp(-1, 1)
        video_tensor = ((video_tensor + 1.0) * 127.5).to(torch.uint8).cpu()
        frames = []
        for t in range(video_tensor.shape[1]):
            frame = video_tensor[:, t].permute(1, 2, 0).numpy()
            frames.append(Image.fromarray(frame))
        return frames

    def build_inputs(self, sample, tiled: bool = False):
        video = sample["video"]
        if "context" not in sample or "context_mask" not in sample:
            raise ValueError(
                "FastWAM training requires `sample['context']` and `sample['context_mask']`."
            )
        context = sample["context"]
        context_mask = sample["context_mask"]
        proprio = sample.get("proprio", None)
        if video.ndim != 5:
            raise ValueError(f"`sample['video']` must be 5D [B, 3, T, H, W], got shape {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"`sample['video']` channel dimension must be 3, got shape {tuple(video.shape)}")

        batch_size, _, num_frames, height, width = video.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"Video spatial dims must be multiples of 16, got H={height}, W={width}"
            )
            
        if num_frames % 4 != 1:
            raise ValueError(f"Video T must satisfy T % 4 == 1, got T={num_frames}")
        if num_frames <= 1:
            raise ValueError(f"Video T must be > 1 for action-conditioned training, got T={num_frames}")

        if "action" not in sample:
            raise ValueError("`sample['action']` is required for FastWAM training.")

        action = sample["action"]
        if action.ndim != 3:
            raise ValueError(f"`sample['action']` must be 3D [B, T, a_dim], got shape {tuple(action.shape)}")
        action_horizon = int(action.shape[1])
        if action_horizon % (num_frames - 1) != 0:
            raise ValueError(
                f"`sample['action']` temporal dimension must be divisible by video transitions ({num_frames - 1}), got {action_horizon}"
            )

        action_is_pad = sample.get("action_is_pad", None)
        if action_is_pad is not None:
            if action_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['action_is_pad']` must be 2D [B, T], got shape {tuple(action_is_pad.shape)}"
                )
            if action_is_pad.shape[0] != batch_size or action_is_pad.shape[1] != action_horizon:
                raise ValueError(
                    "`sample['action_is_pad']` shape mismatch: "
                    f"got {tuple(action_is_pad.shape)} vs expected ({batch_size}, {action_horizon})"
                )

        image_is_pad = sample.get("image_is_pad", None)
        if image_is_pad is not None:
            if image_is_pad.ndim != 2:
                raise ValueError(
                    f"`sample['image_is_pad']` must be 2D [B, T], got shape {tuple(image_is_pad.shape)}"
                )
            if image_is_pad.shape[0] != batch_size or image_is_pad.shape[1] != num_frames:
                raise ValueError(
                    "`sample['image_is_pad']` shape mismatch: "
                    f"got {tuple(image_is_pad.shape)} vs expected ({batch_size}, {num_frames})"
                )
        
        input_video = video.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        input_latents = self._encode_video_latents(input_video, tiled=tiled)

        first_frame_latents = None
        fuse_flag = False
        if getattr(self.video_expert, "fuse_vae_embedding_in_latents", False):
            first_frame_latents = input_latents[:, :, 0:1]
            fuse_flag = True

        if context.ndim != 3 or context_mask.ndim != 2:
            raise ValueError(
                f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
            )
        context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
        context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        trace_features = self._maybe_extract_traceanything_features(sample.get("trace_video", None))
        if self.proprio_encoder is not None:
            if proprio is None:
                raise ValueError("`sample['proprio']` is required when `proprio_dim` is enabled.")
            if proprio.ndim != 3:
                raise ValueError(f"`sample['proprio']` must be 3D [B, T, d], got shape {tuple(proprio.shape)}")
            if proprio.shape[2] != self.proprio_dim:
                raise ValueError(
                    f"`sample['proprio']` last dim must be {self.proprio_dim}, got {proprio.shape[2]}"
                )
            proprio = proprio[:, 0, :] # [B, D]
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio.to(device=self.device, dtype=self.torch_dtype),
            )
        action = action.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)

        if action_is_pad is not None:
            action_is_pad = action_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if image_is_pad is not None:
            image_is_pad = image_is_pad.to(device=self.device, dtype=torch.bool, non_blocking=True)

        return {
            "context": context,
            "context_mask": context_mask,
            "input_latents": input_latents,
            "first_frame_latents": first_frame_latents,
            "fuse_vae_embedding_in_latents": fuse_flag,
            "action": action,
            "action_is_pad": action_is_pad,
            "image_is_pad": image_is_pad,
            "trace_features": trace_features,
        }

    @torch.no_grad()
    def _build_mot_attention_mask(
        self,
        video_seq_len: int,
        action_seq_len: int,
        video_tokens_per_frame: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_seq_len = video_seq_len + action_seq_len
        mask = torch.zeros((total_seq_len, total_seq_len), dtype=torch.bool, device=device)

        # video -> video
        mask[:video_seq_len, :video_seq_len] = self.video_expert.build_video_to_video_mask(
            video_seq_len=video_seq_len,
            video_tokens_per_frame=video_tokens_per_frame,
            device=device,
        )
        # action -> action
        mask[video_seq_len:, video_seq_len:] = True
        # action -> first-frame video only
        first_frame_tokens = min(video_tokens_per_frame, video_seq_len)
        mask[video_seq_len:, :first_frame_tokens] = True
        return mask

    def _compute_video_loss_per_sample(
        self,
        pred_video: torch.Tensor,
        target_video: torch.Tensor,
        image_is_pad: Optional[torch.Tensor],
        include_initial_video_step: bool,
    ) -> torch.Tensor:
        video_loss_token = F.mse_loss(pred_video.float(), target_video.float(), reduction="none").mean(dim=(1, 3, 4))
        if image_is_pad is None:
            return video_loss_token.mean(dim=1)

        temporal_factor = int(self.vae.temporal_downsample_factor)
        if temporal_factor <= 0:
            raise ValueError(f"`vae.temporal_downsample_factor` must be positive, got {temporal_factor}.")
        if image_is_pad.shape[1] < 1:
            raise ValueError("`image_is_pad` must contain at least one frame.")
        if (image_is_pad.shape[1] - 1) % temporal_factor != 0:
            raise ValueError(
                "Cannot align `image_is_pad` with video latent steps: "
                f"num_frames={image_is_pad.shape[1]}, temporal_downsample_factor={temporal_factor}."
            )

        tail_is_pad = image_is_pad[:, 1:]
        latent_tail_is_pad = tail_is_pad.view(image_is_pad.shape[0], -1, temporal_factor).all(dim=2)
        if include_initial_video_step:
            video_is_pad = torch.cat([image_is_pad[:, :1], latent_tail_is_pad], dim=1)
        else:
            video_is_pad = latent_tail_is_pad

        if video_is_pad.shape[1] != video_loss_token.shape[1]:
            raise ValueError(
                "Video-loss mask shape mismatch: "
                f"mask steps={video_is_pad.shape[1]}, loss steps={video_loss_token.shape[1]}."
            )

        valid = (~video_is_pad).to(device=video_loss_token.device, dtype=video_loss_token.dtype)
        valid_sum = valid.sum(dim=1).clamp(min=1.0)
        return (video_loss_token * valid).sum(dim=1) / valid_sum

    def check_input(self, inputs):
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f'{key}, shape: {value.shape}')
            else:
                print(f'{key}, type: {type(value)}')

    def training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
        
        # if not self.is_checked:
        #     self.check_input(inputs)
        #     self.is_checked = True
        
        input_latents = inputs["input_latents"]
        batch_size = input_latents.shape[0]
        context = inputs["context"]
        context_mask = inputs["context_mask"]
        action = inputs["action"]
        action_is_pad = inputs["action_is_pad"]
        image_is_pad = inputs["image_is_pad"]
        trace_features = inputs.get("trace_features", None)

        noise_video = torch.randn_like(input_latents)
        timestep_video = self.train_video_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=input_latents.dtype,
        )
        latents = self.train_video_scheduler.add_noise(input_latents, noise_video, timestep_video)
        target_video = self.train_video_scheduler.training_target(input_latents, noise_video, timestep_video)

        if inputs["first_frame_latents"] is not None:
            latents[:, :, 0:1] = inputs["first_frame_latents"]

        noise_action = torch.randn_like(action)
        timestep_action = self.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device=self.device,
            dtype=action.dtype,
        )
        noisy_action = self.train_action_scheduler.add_noise(action, noise_action, timestep_action)
        target_action = self.train_action_scheduler.training_target(action, noise_action, timestep_action)

        video_pre = self.video_expert.pre_dit(
            x=latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=action,
            fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
        )

        action_pre = self.action_expert.pre_dit(
            action_tokens=noisy_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        video_tokens = video_pre["tokens"]
        action_tokens = action_pre["tokens"]

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_tokens.shape[1],
            action_seq_len=action_tokens.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_tokens.device,
        )
        capture_trace_alignment = (
            trace_features is not None
            and self.trace_align_layer is not None
            and self.trace_loss_weight > 0.0
            and (self.trace_dest_loss_enabled or self.trace_motion_loss_enabled)
        )
        mot_out = self.mot(
            embeds_all={
                "video": video_tokens,
                "action": action_tokens,
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
            capture_layer_idx=(self.trace_align_layer if capture_trace_alignment else None),
            capture_experts=("video",),
        )
        if capture_trace_alignment:
            tokens_out, captured_tokens = mot_out
        else:
            tokens_out = mot_out
            captured_tokens = {}

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)

        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)

        include_initial_video_step = inputs["first_frame_latents"] is None
        if inputs["first_frame_latents"] is not None:
            pred_video = pred_video[:, :, 1:]
            target_video = target_video[:, :, 1:]

        loss_video_per_sample = self._compute_video_loss_per_sample(
            pred_video=pred_video,
            target_video=target_video,
            image_is_pad=image_is_pad,
            include_initial_video_step=include_initial_video_step,
        )
        video_weight = self.train_video_scheduler.training_weight(timestep_video).to(
            loss_video_per_sample.device, dtype=loss_video_per_sample.dtype
        )
        loss_video = (loss_video_per_sample * video_weight).mean()

        action_loss_token = F.mse_loss(pred_action.float(), target_action.float(), reduction="none").mean(dim=2) # [B, T]
        if action_is_pad is not None:
            valid = (~action_is_pad).to(device=action_loss_token.device, dtype=action_loss_token.dtype)
            valid_sum = valid.sum(dim=1).clamp(min=1.0)
            action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
        else:
            action_loss_per_sample = action_loss_token.mean(dim=1)

        action_weight = self.train_action_scheduler.training_weight(timestep_action).to(
            action_loss_per_sample.device, dtype=action_loss_per_sample.dtype
        )
        loss_action = (action_loss_per_sample * action_weight).mean()

        trace_loss_total = loss_video.new_zeros(())
        trace_loss_dict = {}
        if capture_trace_alignment:
            trace_loss_dict = self._compute_trace_alignment_loss(
                video_hidden_states=captured_tokens["video"],
                trace_features=trace_features,
                video_pre=video_pre,
            )
            trace_loss_total = (
                (
                    self.trace_dest_loss_weight * trace_loss_dict["dest_loss"]
                    if self.trace_dest_loss_enabled
                    else loss_video.new_zeros(())
                )
                + (
                    self.trace_motion_loss_weight * trace_loss_dict["motion_loss"]
                    if self.trace_motion_loss_enabled
                    else loss_video.new_zeros(())
                )
            )

        loss_total = (
            self.loss_lambda_video * loss_video
            + self.loss_lambda_action * loss_action
            + self.trace_loss_weight * trace_loss_total
        )
        loss_dict = {
            "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
            "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
        }
        if capture_trace_alignment:
            loss_dict["loss_trace"] = self.trace_loss_weight * float(trace_loss_total.detach().item())
            if self.trace_dest_loss_enabled:
                loss_dict["loss_trace_dest"] = (
                    self.trace_loss_weight
                    * self.trace_dest_loss_weight
                    * float(trace_loss_dict["dest_loss"].detach().item())
                )
            if self.trace_motion_loss_enabled:
                loss_dict["loss_trace_motion"] = (
                    self.trace_loss_weight
                    * self.trace_motion_loss_weight
                    * float(trace_loss_dict["motion_loss"].detach().item())
                )
        return loss_total, loss_dict

    @torch.no_grad()
    def _predict_joint_noise(
        self,
        latents_video: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_video: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
        gt_action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_pre = self.video_expert.pre_dit(
            x=latents_video,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=gt_action,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )

        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )

        pred_video = self.video_expert.post_dit(tokens_out["video"], video_pre)
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_video, pred_action

    @torch.no_grad()
    def _predict_action_noise(
        self,
        first_frame_latents: torch.Tensor,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        fuse_vae_embedding_in_latents: bool,
    ) -> torch.Tensor:
        timestep_video = torch.zeros_like(timestep_action, dtype=first_frame_latents.dtype, device=self.device)
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
        )
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )

        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        tokens_out = self.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_action = self.action_expert.post_dit(tokens_out["action"], action_pre)
        return pred_action

    @torch.no_grad()
    def _predict_action_noise_with_cache(
        self,
        latents_action: torch.Tensor,
        timestep_action: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        video_kv_cache: list[dict[str, torch.Tensor]],
        attention_mask: torch.Tensor,
        video_seq_len: int,
    ) -> torch.Tensor:
        action_pre = self.action_expert.pre_dit(
            action_tokens=latents_action,
            timestep=timestep_action,
            context=context,
            context_mask=context_mask,
        )
        action_tokens = self.mot.forward_action_with_video_cache(
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            video_kv_cache=video_kv_cache,
            attention_mask=attention_mask,
            video_seq_len=video_seq_len,
        )
        return self.action_expert.post_dit(action_tokens, action_pre)

    @torch.no_grad()
    def infer_joint(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_video_frames: int,
        action_horizon: int,
        action: Optional[torch.Tensor] = None, # NOTE: this is gt action for conditioning videos, not for action expert
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
        test_action_with_infer_action: bool = True,
    ) -> dict[str, Any]:
        self.eval()
        if test_action_with_infer_action:
            if seed is None:
                raise ValueError("`test_action_with_infer_action=True` requires non-null `seed`.")
            action_only_out = self.infer_action(
                prompt=prompt,
                input_image=input_image.clone(),
                action_horizon=action_horizon,
                context=context.clone() if context is not None else None,
                context_mask=context_mask.clone() if context_mask is not None else None,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                seed=seed,
                rand_device=rand_device,
                tiled=tiled,
                proprio=proprio.clone() if proprio is not None else None,
            )["action"]
        
        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        checked_h, checked_w, checked_t = self._check_resize_height_width(height, width, num_video_frames)
        if (checked_h, checked_w) != (height, width):
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if checked_t != num_video_frames:
            raise ValueError(
                f"`num_video_frames` must satisfy T % 4 == 1, got {num_video_frames}"
            )
        if action is not None:
            if action.ndim == 2:
                action = action.unsqueeze(0)
            if action.ndim != 3 or action.shape[0] != 1 or action.shape[1] != action_horizon:
                # NOTE: This enforces action condition to have the same shape as action horizon to predict, which may be unnecessary
                raise ValueError(
                    f"`action` must have shape [1, T, a_dim] or [T, a_dim], got {tuple(action.shape)} with action_horizon={action_horizon}"
                )
            action = action.to(device=self.device, dtype=self.torch_dtype)
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        latent_t = (num_video_frames - 1) // self.vae.temporal_downsample_factor + 1
        latent_h = height // self.vae.upsampling_factor
        latent_w = width // self.vae.upsampling_factor

        video_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        action_generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_video = torch.randn(
            (1, self.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=video_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=action_generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        latents_video[:, :, 0:1] = first_frame_latents.clone()
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        infer_timesteps_video, infer_deltas_video = self.infer_video_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_video.dtype,
            shift_override=sigma_shift,
        )
        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_video, step_delta_video, step_t_action, step_delta_action in zip(
            infer_timesteps_video,
            infer_deltas_video,
            infer_timesteps_action,
            infer_deltas_action,
        ):
            timestep_video = step_t_video.unsqueeze(0).to(dtype=latents_video.dtype, device=self.device)
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_video_posi, pred_action_posi = self._predict_joint_noise(
                latents_video=latents_video,
                latents_action=latents_action,
                timestep_video=timestep_video,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                fuse_vae_embedding_in_latents=fuse_flag,
                gt_action=action,
            )
            pred_video = pred_video_posi
            pred_action = pred_action_posi

            latents_video = self.infer_video_scheduler.step(pred_video, step_delta_video, latents_video)
            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)
            latents_video[:, :, 0:1] = first_frame_latents.clone()

        action_out = latents_action[0].detach().to(device="cpu", dtype=torch.float32)
        if test_action_with_infer_action:
            if not torch.allclose(action_out, action_only_out, atol=1e-2, rtol=1e-2):
                max_abs_diff = (action_out - action_only_out).abs().max().item()
                logger.warning(
                    f"Action from infer_joint and infer_action differ with max abs diff {max_abs_diff:.6f}. "
                )

        return {
            "video": self._decode_latents(latents_video, tiled=tiled),
            "action": action_out,
        }

    @torch.no_grad()
    def infer_action(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        action_horizon: int,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ) -> dict[str, Any]:
        self.eval()
        if str(getattr(self.video_expert, "video_attention_mask_mode", "")) != "first_frame_causal":
            raise ValueError(
                "`infer_action` requires `video_attention_mask_mode='first_frame_causal'`."
            )

        if input_image.ndim == 3:
            input_image = input_image.unsqueeze(0)
        if input_image.ndim != 4 or input_image.shape[0] != 1 or input_image.shape[1] != 3:
            raise ValueError(
                f"`input_image` must have shape [1,3,H,W] or [3,H,W], got {tuple(input_image.shape)}"
            )
        _, _, height, width = input_image.shape
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(
                f"`input_image` must be resized before infer, expected multiples of 16 but got HxW=({height},{width})"
            )
        if proprio is not None:
            if self.proprio_dim is None:
                raise ValueError("`proprio` was provided but `proprio_dim=None` so `proprio_encoder` is disabled.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            elif proprio.ndim == 2 and proprio.shape[0] == 1:
                pass
            else:
                raise ValueError(f"`proprio` must be [D] or [1,D], got shape {tuple(proprio.shape)}")
            if proprio.shape[1] != self.proprio_dim:
                raise ValueError(f"`proprio` last dim must be {self.proprio_dim}, got {proprio.shape[1]}")
            proprio = proprio.to(device=self.device, dtype=self.torch_dtype)

        generator = None if seed is None else torch.Generator(device=rand_device).manual_seed(seed)
        latents_action = torch.randn(
            (1, action_horizon, self.action_expert.action_dim),
            generator=generator,
            device=rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.torch_dtype)

        input_image = input_image.to(device=self.device, dtype=self.torch_dtype)
        first_frame_latents = self._encode_input_image_latents_tensor(input_image=input_image, tiled=tiled)
        fuse_flag = bool(getattr(self.video_expert, "fuse_vae_embedding_in_latents", False))

        use_prompt = prompt is not None
        use_context = context is not None or context_mask is not None
        if use_prompt and use_context:
            raise ValueError("`prompt` and `context/context_mask` are mutually exclusive.")
        if not use_prompt and not use_context:
            raise ValueError("Either `prompt` or both `context/context_mask` must be provided.")

        if use_prompt:
            context, context_mask = self.encode_prompt(prompt)
        else:
            if context is None or context_mask is None:
                raise ValueError("`context` and `context_mask` must be both provided together.")
            if context.ndim == 2:
                context = context.unsqueeze(0)
            if context_mask.ndim == 1:
                context_mask = context_mask.unsqueeze(0)
            if context.ndim != 3 or context_mask.ndim != 2:
                raise ValueError(
                    f"`context/context_mask` must be [B,L,D]/[B,L], got {tuple(context.shape)} and {tuple(context_mask.shape)}"
                )
            context = context.to(device=self.device, dtype=self.torch_dtype, non_blocking=True)
            context_mask = context_mask.to(device=self.device, dtype=torch.bool, non_blocking=True)
        if proprio is not None:
            context, context_mask = self._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        timestep_video = torch.zeros(
            (first_frame_latents.shape[0],),
            dtype=first_frame_latents.dtype,
            device=self.device,
        )
        video_pre = self.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        video_seq_len = int(video_pre["tokens"].shape[1])
        attention_mask = self._build_mot_attention_mask(
            video_seq_len=video_seq_len,
            action_seq_len=latents_action.shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=video_pre["tokens"].device,
        )
        video_kv_cache = self.mot.prefill_video_cache(
            video_tokens=video_pre["tokens"],
            video_freqs=video_pre["freqs"],
            video_t_mod=video_pre["t_mod"],
            video_context_payload={
                "context": video_pre["context"],
                "mask": video_pre["context_mask"],
            },
            video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
        )

        infer_timesteps_action, infer_deltas_action = self.infer_action_scheduler.build_inference_schedule(
            num_inference_steps=num_inference_steps,
            device=self.device,
            dtype=latents_action.dtype,
            shift_override=sigma_shift,
        )
        for step_t_action, step_delta_action in zip(infer_timesteps_action, infer_deltas_action):
            timestep_action = step_t_action.unsqueeze(0).to(dtype=latents_action.dtype, device=self.device)

            pred_action_posi = self._predict_action_noise_with_cache(
                latents_action=latents_action,
                timestep_action=timestep_action,
                context=context,
                context_mask=context_mask,
                video_kv_cache=video_kv_cache,
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            pred_action = pred_action_posi

            latents_action = self.infer_action_scheduler.step(pred_action, step_delta_action, latents_action)

        return {
            "action": latents_action[0].detach().to(device="cpu", dtype=torch.float32),
        }

    @torch.no_grad()
    def infer(
        self,
        prompt: Optional[str],
        input_image: torch.Tensor,
        num_frames: int,
        action: Optional[torch.Tensor] = None,
        action_horizon: Optional[int] = None,
        proprio: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        negative_prompt: Optional[str] = None,
        text_cfg_scale: float = 5.0,
        action_cfg_scale: float = 1.0,
        num_inference_steps: int = 20,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        rand_device: str = "cpu",
        tiled: bool = False,
    ):
        return self.infer_joint(
            prompt=prompt,
            input_image=input_image,
            num_video_frames=num_frames,
            action_horizon=action_horizon,
            action=action,
            proprio=proprio,
            context=context,
            context_mask=context_mask,
            negative_prompt=negative_prompt,
            text_cfg_scale=text_cfg_scale,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            seed=seed,
            rand_device=rand_device,
            tiled=tiled,
        )

    def save_checkpoint(self, path, optimizer=None, step=None):
        payload = {
            "mot": self.mot.state_dict(),
            "step": step,
            "torch_dtype": str(self.torch_dtype),
        }
        if self.proprio_encoder is not None:
            payload["proprio_encoder"] = self.proprio_encoder.state_dict()
        if self.trace_proj is not None:
            payload["trace_proj"] = self.trace_proj.state_dict()
        if self.align_repr_proj is not None:
            payload["align_repr_proj"] = self.align_repr_proj.state_dict()
        if optimizer is not None:
            payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    def load_checkpoint(self, path, optimizer=None):
        payload = torch.load(path, map_location="cpu")
        if "module" in payload and isinstance(payload["module"], dict):
            missing, unexpected = self.load_state_dict(payload["module"], strict=False)
            logger.info(
                "Loaded DeepSpeed model state_dict with strict=False. Missing=%d Unexpected=%d",
                len(missing),
                len(unexpected),
            )
            return payload

        if "mot" in payload:
            self.mot.load_state_dict(payload["mot"], strict=False)
        elif "dit" in payload:
            logger.warning("Loading legacy `dit` checkpoint into video expert only.")
            self.video_expert.load_state_dict(payload["dit"], strict=False)
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {path}")
        if self.proprio_encoder is not None:
            if "proprio_encoder" in payload:
                self.proprio_encoder.load_state_dict(payload["proprio_encoder"], strict=True)
            else:
                logger.warning("Checkpoint has no `proprio_encoder` weights; keeping current `proprio_encoder` params.")
        elif "proprio_encoder" in payload:
            logger.warning("Checkpoint contains `proprio_encoder` weights but current model has `proprio_dim=None`; ignoring.")

        if self.trace_proj is not None:
            if "trace_proj" in payload:
                self.trace_proj.load_state_dict(payload["trace_proj"], strict=True)
            else:
                logger.warning("Checkpoint has no `trace_proj` weights; keeping current `trace_proj` params.")
        elif "trace_proj" in payload:
            logger.warning("Checkpoint contains `trace_proj` weights but current model has trace alignment disabled; ignoring.")

        if self.align_repr_proj is not None:
            if "align_repr_proj" in payload:
                self.align_repr_proj.load_state_dict(payload["align_repr_proj"], strict=True)
            else:
                logger.warning("Checkpoint has no `align_repr_proj` weights; keeping current `align_repr_proj` params.")
        elif "align_repr_proj" in payload:
            logger.warning("Checkpoint contains `align_repr_proj` weights but current model has trace alignment disabled; ignoring.")

        if optimizer is not None and "optimizer" in payload:
            optimizer.load_state_dict(payload["optimizer"])
        return payload

    def load_trace_alignment_checkpoint(self, path):
        payload = torch.load(path, map_location="cpu")
        source = payload["module"] if isinstance(payload, dict) and isinstance(payload.get("module"), dict) else payload
        if not isinstance(source, dict):
            raise ValueError(f"Trace alignment checkpoint must contain a state dict, got: {type(source)}")

        def extract_module_state(prefix: str):
            if prefix in source and isinstance(source[prefix], dict):
                return source[prefix]
            prefix_with_dot = f"{prefix}."
            return {
                key[len(prefix_with_dot):]: value
                for key, value in source.items()
                if isinstance(key, str) and key.startswith(prefix_with_dot)
            }

        trace_state = extract_module_state("trace_proj")
        align_state = extract_module_state("align_repr_proj")

        if self.trace_proj is None:
            if trace_state:
                logger.warning("Trace checkpoint contains `trace_proj`, but current model has trace alignment disabled; ignoring.")
        elif trace_state:
            self.trace_proj.load_state_dict(trace_state, strict=True)
            logger.info("Loaded `trace_proj` alignment weights from %s.", path)
        else:
            logger.warning("Trace checkpoint has no `trace_proj` weights: %s", path)

        if self.align_repr_proj is None:
            if align_state:
                logger.warning("Trace checkpoint contains `align_repr_proj`, but current model has trace alignment disabled; ignoring.")
        elif align_state:
            self.align_repr_proj.load_state_dict(align_state, strict=True)
            logger.info("Loaded `align_repr_proj` alignment weights from %s.", path)
        else:
            logger.warning("Trace checkpoint has no `align_repr_proj` weights: %s", path)

        return payload

    def forward(self, *args, **kwargs):
        return self.training_loss(*args, **kwargs)
