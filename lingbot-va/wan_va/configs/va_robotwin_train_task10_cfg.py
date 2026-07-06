# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg_task10 import va_robotwin_cfg
import os

va_robotwin_train_cfg = EasyDict(__name__='Config: VA robotwin task10 train')
va_robotwin_train_cfg.update(va_robotwin_cfg)

# va_robotwin_train_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'
va_robotwin_train_cfg.home_path = '/cpfs01/projects-HDD/cfff-377aad6b032c_HDD/chenshuai/wenxuan/'
va_robotwin_train_cfg.dataset_path = os.path.join(va_robotwin_train_cfg.home_path, '.cache/huggingface/lerobot/robotwin/robotwin_multi_10_tasks')
va_robotwin_train_cfg.empty_emb_path = os.path.join(va_robotwin_train_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_train_cfg.enable_wandb = False
# va_robotwin_train_cfg.load_worker = 16 # for multiprocesses
va_robotwin_train_cfg.load_worker = 0 
va_robotwin_train_cfg.save_interval = 3000
va_robotwin_train_cfg.gc_interval = 50
va_robotwin_train_cfg.cfg_prob = 0.1
va_robotwin_train_cfg.wan22_pretrained_model_name_or_path = os.path.join(va_robotwin_train_cfg.home_path, '.cache/modelscope/Robbyant/lingbot-va-base')

# Training parameters
va_robotwin_train_cfg.learning_rate = 1e-5
va_robotwin_train_cfg.beta1 = 0.9
va_robotwin_train_cfg.beta2 = 0.95
va_robotwin_train_cfg.weight_decay = 0.1
va_robotwin_train_cfg.warmup_steps = 10
va_robotwin_train_cfg.batch_size = 1
va_robotwin_train_cfg.gradient_accumulation_steps = 32
va_robotwin_train_cfg.num_steps = 40000

# others
va_robotwin_train_cfg.max_tokens = 128