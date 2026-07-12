# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os

from easydict import EasyDict

from ..va_robotwin_train_cfg import va_robotwin_cfg


lingbot4dwam_train_cfg = EasyDict(__name__='Config: VA Lift2 merged train')
lingbot4dwam_train_cfg.update(va_robotwin_cfg)

# Merged 60 Hz Step3-compatible LeRobot dataset.
lingbot4dwam_train_cfg.data_path = '/soft/wangxi/4DWAM/datasets_converted'
lingbot4dwam_train_cfg.dataset_path = os.path.join(
    lingbot4dwam_train_cfg.data_path,
    'lift2_merged_step3_compatible_60hz',
)

# Use action quantiles computed after the same Euler-to-quaternion,
# segment-relative, temporal-alignment, and 30D channel mapping used by the
# training dataset loader.
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
    lingbot4dwam_train_cfg.data_path,
    'empty_emb.pt',
)
lingbot4dwam_train_cfg.enable_wandb = True
lingbot4dwam_train_cfg.load_worker = 16
lingbot4dwam_train_cfg.save_interval = 1000
lingbot4dwam_train_cfg.gc_interval = 50
lingbot4dwam_train_cfg.cfg_prob = 0.1

lingbot4dwam_train_cfg.cache_path = os.environ.get(
    'CACHE_PATH',
    '/soft/wangxi/.cache',
)
lingbot4dwam_train_cfg.wan22_pretrained_model_name_or_path = os.path.join(
    lingbot4dwam_train_cfg.cache_path,
    'huggingface/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c',
)

# To resume training, set this to a checkpoint directory.
# lingbot4dwam_train_cfg.resume_from = '/path/to/checkpoint_step_XXXX/'

# Training parameters.
lingbot4dwam_train_cfg.learning_rate = 1e-5
lingbot4dwam_train_cfg.beta1 = 0.9
lingbot4dwam_train_cfg.beta2 = 0.95
lingbot4dwam_train_cfg.weight_decay = 0.1
lingbot4dwam_train_cfg.warmup_steps = 10
lingbot4dwam_train_cfg.batch_size = 1
lingbot4dwam_train_cfg.gradient_accumulation_steps = 8
lingbot4dwam_train_cfg.num_steps = 20000
lingbot4dwam_train_cfg.align_layer = 20
lingbot4dwam_train_cfg.keyword = (
    f'lift2_merged_4DWAM_{lingbot4dwam_train_cfg.align_layer}'
)
lingbot4dwam_train_cfg.save_root = os.path.join(
    '/soft/wangxi/4DWAM/lingbot-va/checkpoints',
    lingbot4dwam_train_cfg.keyword,
    (
        f'train_{int(64 / lingbot4dwam_train_cfg.gradient_accumulation_steps)}_'
        f'{lingbot4dwam_train_cfg.gradient_accumulation_steps}_'
        f'{lingbot4dwam_train_cfg.num_steps}'
    ),
)
lingbot4dwam_train_cfg.max_tokens = 512

# 4D WAM hyperparameters.
lingbot4dwam_train_cfg.enable_trace = True
lingbot4dwam_train_cfg.trace_coef = 0.01

lingbot4dwam_train_cfg.loss_weights = {
    'dest_loss': 0.01,
    'motion_loss': 0.01,
    'trace_loss': 0.01,
}
