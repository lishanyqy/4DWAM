# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Thin inference config for LIFT2 4DWAM checkpoints.

Same server protocol as ``lift2_merged_infer``; only the default transformer
checkpoint path differs. Override with ``LIFT2_4DWAM_CHECKPOINT``.
"""
import os

from easydict import EasyDict

from .va_lift2_merged_infer_cfg import va_lift2_merged_infer_cfg

_DEFAULT_4DWAM_CHECKPOINT = os.environ.get(
    'LIFT2_4DWAM_CHECKPOINT',
    os.path.join(
        '/soft/wangxi/4DWAM/lingbot-va/checkpoints',
        '4DWAM_20',
        'train_8_8_20000',
        'checkpoints',
        'checkpoint_step_1000',
    ),
)

va_4dwam_lift2_infer_cfg = EasyDict(__name__='Config: VA 4DWAM Lift2 infer server')
va_4dwam_lift2_infer_cfg.update(va_lift2_merged_infer_cfg)
va_4dwam_lift2_infer_cfg.wan22_pretrained_model_name_or_path = _DEFAULT_4DWAM_CHECKPOINT
va_4dwam_lift2_infer_cfg.port = 7777
va_4dwam_lift2_infer_cfg.infer_mode = 'server'
