# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os
from copy import deepcopy

from .train4dwam_test import lingbot4dwam_train_cfg as test_train_cfg


lingbot4dwam_train_cfg = deepcopy(test_train_cfg)
lingbot4dwam_train_cfg.__name__ = (
    'Config: VA Lift2 globally longest 50 episodes test train'
)

lingbot4dwam_train_cfg.dataset_path = os.path.join(
    lingbot4dwam_train_cfg.data_path,
    'lift2_longest50_step3_compatible_60hz',
)
lingbot4dwam_train_cfg.action_norm_stats_path = os.path.join(
    lingbot4dwam_train_cfg.dataset_path,
    'meta',
    'action_norm_stats.json',
)
with open(
    lingbot4dwam_train_cfg.action_norm_stats_path,
    'r',
    encoding='utf-8',
) as action_norm_stats_file:
    action_norm_statistics = json.load(action_norm_stats_file)

lingbot4dwam_train_cfg.norm_stat = action_norm_statistics['norm_stat']
if (
    len(lingbot4dwam_train_cfg.norm_stat['q01'])
    != lingbot4dwam_train_cfg.action_dim
    or len(lingbot4dwam_train_cfg.norm_stat['q99'])
    != lingbot4dwam_train_cfg.action_dim
):
    raise ValueError(
        'Action normalization statistics must match action_dim='
        f'{lingbot4dwam_train_cfg.action_dim}: '
        f'q01={len(lingbot4dwam_train_cfg.norm_stat["q01"])}, '
        f'q99={len(lingbot4dwam_train_cfg.norm_stat["q99"])}'
    )

lingbot4dwam_train_cfg.empty_emb_path = os.path.join(
    lingbot4dwam_train_cfg.dataset_path,
    'empty_emb.pt',
)

# Fifty optimizer steps exercise repeated sampling across the longest episodes
# while preserving the full-training batch size and accumulation settings.
lingbot4dwam_train_cfg.num_steps = 50
lingbot4dwam_train_cfg.enable_wandb = False
lingbot4dwam_train_cfg.keyword = (
    f'lift2_longest50_4DWAM_{lingbot4dwam_train_cfg.align_layer}'
)
lingbot4dwam_train_cfg.save_root = os.path.join(
    '/soft/wangxi/4DWAM/lingbot-va/checkpoints',
    lingbot4dwam_train_cfg.keyword,
    'smoke_50_steps',
)
