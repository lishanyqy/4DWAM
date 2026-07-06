# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from ..va_robotwin_cfg_task10 import va_robotwin_cfg
import os

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin task10 train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

# va_robotwin_train_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'
# va_robotwin_train_cfg.home_path = '/cpfs01/projects-HDD/cfff-377aad6b032c_HDD/chenshuai/wenxuan/'
va_robotwin_train_cfg.cache_path = '/root/.cache'
va_robotwin_train_cfg.data_path = '/data/.cache/datasets/lerobot/robotwin/'
va_robotwin_train_cfg.dataset_path = os.path.join(va_robotwin_train_cfg.data_path, 'clean_tiny')
va_robotwin_train_cfg.empty_emb_path = os.path.join(va_robotwin_train_cfg.data_path, 'empty_emb.pt')
va_robotwin_train_cfg.enable_wandb = True
# va_robotwin_train_cfg.load_worker = 16 # for multiprocesses
va_robotwin_train_cfg.load_worker = 16
va_robotwin_train_cfg.save_interval = 500
va_robotwin_train_cfg.gc_interval = 50
va_robotwin_train_cfg.cfg_prob = 0.1
va_robotwin_train_cfg.wan22_pretrained_model_name_or_path = os.path.join(va_robotwin_train_cfg.cache_path, 'modelscope/hub/models/robbyant/lingbot-va-base/')
va_robotwin_train_cfg.resume_from = os.path.join('/sharedata/lsy/lingbot_ckpt_archives/ckpt/va_4_16_50000/checkpoints/checkpoint_step_2000/')

# Training parameters
va_robotwin_train_cfg.learning_rate = 1e-5
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = 10
va_robotwin_train_cfg.batch_size = 1
va_robotwin_train_cfg.gradient_accumulation_steps = 8
va_robotwin_train_cfg.num_steps = 520
va_robotwin_train_cfg.keyword = 'va_from_va2000'
va_robotwin_train_cfg.save_root = f'/data/lingbot/{va_robotwin_train_cfg.keyword}/train_{int(64/va_robotwin_train_cfg.gradient_accumulation_steps)}_{va_robotwin_train_cfg.gradient_accumulation_steps}_{va_robotwin_train_cfg.num_steps}'
# va_robotwin_train_cfg.save_root = f'/data/lingbot/ckpt_resume2000/ta_{int(64/va_robotwin_train_cfg.gradient_accumulation_steps)}_{va_robotwin_train_cfg.gradient_accumulation_steps}_50000'
# others
va_robotwin_train_cfg.max_tokens = 512
va_robotwin_train_cfg.enable_trace = False
va_robotwin_train_cfg.trace_coef = 0.01

va_robotwin_train_cfg.loss_weights = {
    'dest_loss':0.01,
    'motion_loss':0.01,
    'trace_loss':0.01,
}