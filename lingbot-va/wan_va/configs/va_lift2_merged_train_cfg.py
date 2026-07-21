# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os

from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg


va_lift2_merged_train_cfg = EasyDict(
    __name__='Config: VA Lift2 merged train without trace'
)
va_lift2_merged_train_cfg.update(va_robotwin_cfg)

# Reuse the converted merged dataset while training the ordinary LingBot-VA
# objectives without the additional 4DWAM trace branch.
va_lift2_merged_train_cfg.data_path = '/soft/wangxi/4DWAM/datasets_converted'
va_lift2_merged_train_cfg.dataset_path = os.path.join(
    va_lift2_merged_train_cfg.data_path,
    'lift2_merged_step3_compatible_60hz',
)

# These statistics match the 30D action mapping performed by the dataset
# conversion and loader, so they must remain paired with this dataset.
va_lift2_merged_train_cfg.action_norm_stats_path = os.path.join(
    va_lift2_merged_train_cfg.dataset_path,
    'meta',
    'action_norm_stats.json',
)
with open(
    va_lift2_merged_train_cfg.action_norm_stats_path,
    'r',
    encoding='utf-8',
) as action_norm_stats_file:
    action_norm_statistics = json.load(action_norm_stats_file)

action_representation = action_norm_statistics.get('representation', {})
expected_action_representation = {
    'pose_mode': 'segment-relative',
    'quaternion_order': 'xyzw',
    'euler_axes': 'xyz',
}
for representation_key, expected_value in expected_action_representation.items():
    actual_value = action_representation.get(representation_key)
    if actual_value != expected_value:
        raise ValueError(
            'Action normalization statistics use an incompatible '
            f'{representation_key}: expected {expected_value!r}, '
            f'got {actual_value!r} in '
            f'{va_lift2_merged_train_cfg.action_norm_stats_path}'
        )

if (
    action_representation.get('used_action_channel_ids')
    != va_lift2_merged_train_cfg.used_action_channel_ids
):
    raise ValueError(
        'Action normalization statistics use incompatible action channels: '
        f'expected {va_lift2_merged_train_cfg.used_action_channel_ids}, '
        f'got {action_representation.get("used_action_channel_ids")}'
    )

va_lift2_merged_train_cfg.norm_stat = action_norm_statistics['norm_stat']
if (
    len(va_lift2_merged_train_cfg.norm_stat['q01'])
    != va_lift2_merged_train_cfg.action_dim
    or len(va_lift2_merged_train_cfg.norm_stat['q99'])
    != va_lift2_merged_train_cfg.action_dim
):
    raise ValueError(
        'Action normalization statistics must match action_dim='
        f'{va_lift2_merged_train_cfg.action_dim}: '
        f'q01={len(va_lift2_merged_train_cfg.norm_stat["q01"])}, '
        f'q99={len(va_lift2_merged_train_cfg.norm_stat["q99"])}'
    )

va_lift2_merged_train_cfg.empty_emb_path = os.path.join(
    va_lift2_merged_train_cfg.data_path,
    'empty_emb_128.pt',
)
va_lift2_merged_train_cfg.max_tokens = 128

va_lift2_merged_train_cfg.cache_path = os.environ.get(
    'CACHE_PATH',
    '/soft/wangxi/.cache',
)
va_lift2_merged_train_cfg.wan22_pretrained_model_name_or_path = os.path.join(
    va_lift2_merged_train_cfg.cache_path,
    'huggingface/hub/models--robbyant--lingbot-va-base/'
    'snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c',
)

va_lift2_merged_train_cfg.enable_wandb = True
va_lift2_merged_train_cfg.load_worker = 16
va_lift2_merged_train_cfg.save_interval = 200
va_lift2_merged_train_cfg.gc_interval = 50
va_lift2_merged_train_cfg.cfg_prob = 0.1

# The existing LIFT2 checkpoints were trained with the legacy quaternion
# representation. Start this XYZW run from the base LingBot-VA transformer.
va_lift2_merged_train_cfg.resume_from = None

va_lift2_merged_train_cfg.learning_rate = 1e-5
va_lift2_merged_train_cfg.beta1 = 0.9
va_lift2_merged_train_cfg.beta2 = 0.95
va_lift2_merged_train_cfg.weight_decay = 0.1
va_lift2_merged_train_cfg.warmup_steps = 10
va_lift2_merged_train_cfg.batch_size = 1
va_lift2_merged_train_cfg.gradient_accumulation_steps = 8
va_lift2_merged_train_cfg.num_steps = 5000
va_lift2_merged_train_cfg.align_layer = 20

va_lift2_merged_train_cfg.keyword = 'lift2_merged_lingbot_va_xyzw_v2'
va_lift2_merged_train_cfg.save_root = os.path.join(
    '/soft/wangxi/4DWAM/lingbot-va/checkpoints',
    va_lift2_merged_train_cfg.keyword,
    (
        f'train_{int(64 / va_lift2_merged_train_cfg.gradient_accumulation_steps)}_'
        f'{va_lift2_merged_train_cfg.gradient_accumulation_steps}_'
        f'{va_lift2_merged_train_cfg.num_steps}'
    ),
)

# Disable the 4DWAM-only trace objective. Existing trace files may remain in
# the dataset; the loader/model will not use them for this training run.
va_lift2_merged_train_cfg.enable_trace = False
va_lift2_merged_train_cfg.trace_coef = 0.0
va_lift2_merged_train_cfg.loss_weights = {
    'dest_loss': 0.03,
    'motion_loss': 0.01,
    'trace_loss': 0.0,
}
