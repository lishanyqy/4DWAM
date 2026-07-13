# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os
from copy import deepcopy

from .train4dwam_test import lingbot4dwam_train_cfg as test_train_cfg


lingbot4dwam_train_cfg = deepcopy(test_train_cfg)
lingbot4dwam_train_cfg.__name__ = (
    'Config: VA Lift2 mix F=285+F=235 long-episode test train'
)

lingbot4dwam_train_cfg.dataset_path = os.path.join(
    lingbot4dwam_train_cfg.data_path,
    'lift2_mix285_235_test_step3_compatible_60hz',
)
lingbot4dwam_train_cfg.action_norm_stats_path = os.path.join(
    lingbot4dwam_train_cfg.dataset_path,
    'meta',
    'action_norm_stats.json',
)
lingbot4dwam_train_cfg.empty_emb_path = os.path.join(
    lingbot4dwam_train_cfg.dataset_path,
    'empty_emb.pt',
)

# Enough steps to alternate the two long shapes under DistributedSampler
# padding and to pass the first optimizer-state initialization.
lingbot4dwam_train_cfg.num_steps = 4
lingbot4dwam_train_cfg.enable_wandb = False
lingbot4dwam_train_cfg.keyword = (
    f'lift2_mix285_235_test_4DWAM_{lingbot4dwam_train_cfg.align_layer}'
)
lingbot4dwam_train_cfg.save_root = os.path.join(
    '/soft/wangxi/4DWAM/lingbot-va/checkpoints',
    lingbot4dwam_train_cfg.keyword,
    'smoke_4_steps',
)
