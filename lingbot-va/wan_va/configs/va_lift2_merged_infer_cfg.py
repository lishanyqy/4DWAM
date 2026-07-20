# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Thin inference config for LIFT2 merged LingBot-VA checkpoints.

Server loads transformer from ``wan22_pretrained_model_name_or_path/transformer``.
Point that path at a training checkpoint directory (which contains ``transformer/``),
and keep ``vae_path`` / ``text_model_path`` on the base model root.
"""
import json
import os

from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg

_CACHE_PATH = os.environ.get('CACHE_PATH', '/soft/wangxi/.cache')
_BASE_MODEL_ROOT = os.environ.get(
    'LINGBOT_VA_BASE_MODEL',
    os.path.join(
        _CACHE_PATH,
        'huggingface/hub/models--robbyant--lingbot-va-base/'
        'snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c',
    ),
)
_DEFAULT_CHECKPOINT = os.environ.get(
    'LIFT2_VA_CHECKPOINT',
    os.path.join(
        '/soft/wangxi/4DWAM/lingbot-va/checkpoints',
        'lift2_merged_lingbot_va',
        'train_8_8_5000_resume_from_2000',
        'checkpoints',
        'checkpoint_step_1000',
    ),
)
_DATASET_PATH = os.environ.get(
    'LIFT2_VA_DATASET_PATH',
    '/soft/wangxi/4DWAM/datasets_converted/lift2_merged_step3_compatible_60hz',
)

va_lift2_merged_infer_cfg = EasyDict(__name__='Config: VA Lift2 merged infer server')
va_lift2_merged_infer_cfg.update(va_robotwin_cfg)

va_lift2_merged_infer_cfg.infer_mode = 'server'
va_lift2_merged_infer_cfg.host = '0.0.0.0'
va_lift2_merged_infer_cfg.port = 7777

va_lift2_merged_infer_cfg.env_type = 'robotwin_tshape'
va_lift2_merged_infer_cfg.height = 256
va_lift2_merged_infer_cfg.width = 320
va_lift2_merged_infer_cfg.frame_chunk_size = 2
va_lift2_merged_infer_cfg.action_per_frame = 16
va_lift2_merged_infer_cfg.action_dim = 30

va_lift2_merged_infer_cfg.vae_path = _BASE_MODEL_ROOT
va_lift2_merged_infer_cfg.text_model_path = _BASE_MODEL_ROOT
# VA_Server loads ``.../transformer`` under this path.
va_lift2_merged_infer_cfg.wan22_pretrained_model_name_or_path = _DEFAULT_CHECKPOINT

va_lift2_merged_infer_cfg.action_norm_stats_path = os.path.join(
    _DATASET_PATH,
    'meta',
    'action_norm_stats.json',
)
with open(
        va_lift2_merged_infer_cfg.action_norm_stats_path,
        'r',
        encoding='utf-8',
) as action_norm_stats_file:
    action_norm_statistics = json.load(action_norm_stats_file)

va_lift2_merged_infer_cfg.norm_stat = action_norm_statistics['norm_stat']
if (
        len(va_lift2_merged_infer_cfg.norm_stat['q01'])
        != va_lift2_merged_infer_cfg.action_dim
        or len(va_lift2_merged_infer_cfg.norm_stat['q99'])
        != va_lift2_merged_infer_cfg.action_dim):
    raise ValueError(
        'Action normalization statistics must match action_dim='
        f'{va_lift2_merged_infer_cfg.action_dim}: '
        f'q01={len(va_lift2_merged_infer_cfg.norm_stat["q01"])}, '
        f'q99={len(va_lift2_merged_infer_cfg.norm_stat["q99"])}')

va_lift2_merged_infer_cfg.save_root = os.environ.get(
    'LIFT2_VA_SAVE_ROOT',
    './real_robot_server_out',
)
va_lift2_merged_infer_cfg.enable_trace = False
va_lift2_merged_infer_cfg.trace_coef = 0.0
