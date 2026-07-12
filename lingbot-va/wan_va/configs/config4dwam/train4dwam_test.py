# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from ..va_robotwin_train_cfg import va_robotwin_cfg

lingbot4dwam_train_cfg = EasyDict(__name__='Config: VA test train')
lingbot4dwam_train_cfg.update(va_robotwin_cfg)

# Please fill dataset_path
lingbot4dwam_train_cfg.data_path = '/soft/wangxi/4DWAM/datasets_converted'
lingbot4dwam_train_cfg.dataset_path = os.path.join(lingbot4dwam_train_cfg.data_path, 'lift2_test_step3_compatible_60hz')

lingbot4dwam_train_cfg.empty_emb_path = os.path.join(lingbot4dwam_train_cfg.data_path, 'empty_emb.pt')
lingbot4dwam_train_cfg.enable_wandb = True
lingbot4dwam_train_cfg.load_worker = 16
lingbot4dwam_train_cfg.save_interval = 1000
lingbot4dwam_train_cfg.gc_interval = 50
lingbot4dwam_train_cfg.cfg_prob = 0.1
# Please fill lingbot-va base path
lingbot4dwam_train_cfg.cache_path = os.environ.get(
    "CACHE_PATH", "/soft/wangxi/.cache"
)
lingbot4dwam_train_cfg.wan22_pretrained_model_name_or_path = os.path.join('/soft/wangxi/.cache/', 'huggingface/hub/models--robbyant--lingbot-va-base/snapshots/68b7bc1b35da6ddc67ea94c4ceb58d768fbb3f9c')
# if resume
# lingbot4dwam_train_cfg.resume_from = os.path.join('/data/lingbot/ckpt/ta_4_16_50000/checkpoints/checkpoint_step_2000/')

# Training parameters
lingbot4dwam_train_cfg.learning_rate = 1e-5
lingbot4dwam_train_cfg.beta1 = 0.9
lingbot4dwam_train_cfg.beta2 = 0.95
lingbot4dwam_train_cfg.weight_decay = 0.1
lingbot4dwam_train_cfg.warmup_steps = 10
lingbot4dwam_train_cfg.batch_size = 1
lingbot4dwam_train_cfg.gradient_accumulation_steps = 8
lingbot4dwam_train_cfg.num_steps = 20000
lingbot4dwam_train_cfg.align_layer = 20
lingbot4dwam_train_cfg.keyword = f'4DWAM_{lingbot4dwam_train_cfg.align_layer}'
# Change your save_root
lingbot4dwam_train_cfg.save_root = f"/soft/wangxi/4DWAM/lingbot-va/checkpoints/{lingbot4dwam_train_cfg.keyword}/train_{int(64/lingbot4dwam_train_cfg.gradient_accumulation_steps)}_{lingbot4dwam_train_cfg.gradient_accumulation_steps}_{lingbot4dwam_train_cfg.num_steps}"
lingbot4dwam_train_cfg.max_tokens = 512

# 4D WAM Hyper-Parameter
lingbot4dwam_train_cfg.enable_trace = True
lingbot4dwam_train_cfg.trace_coef = 0.01

lingbot4dwam_train_cfg.loss_weights = {
    'dest_loss':0.01,
    'motion_loss':0.01,
    'trace_loss':0.01,
}