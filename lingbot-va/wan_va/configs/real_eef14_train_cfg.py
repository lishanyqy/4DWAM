# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .real_eef14_cfg import real_eef14_cfg

real_eef14_train_cfg = EasyDict(__name__='Config: VA robotwin eef14 raw train')
real_eef14_train_cfg.update(real_eef14_cfg)

real_eef14_train_cfg.dataset_path = '/path/to/your/dataset'
real_eef14_train_cfg.empty_emb_path = os.path.join(real_eef14_train_cfg.dataset_path, 'empty_emb.pt')
real_eef14_train_cfg.enable_wandb = True
real_eef14_train_cfg.load_worker = 16
real_eef14_train_cfg.save_interval = 1000
real_eef14_train_cfg.gc_interval = 50
real_eef14_train_cfg.cfg_prob = 0.1

real_eef14_train_cfg.learning_rate = 1e-5
real_eef14_train_cfg.beta1 = 0.9
real_eef14_train_cfg.beta2 = 0.95
real_eef14_train_cfg.weight_decay = 0.1
real_eef14_train_cfg.warmup_steps = 10
real_eef14_train_cfg.batch_size = 1
real_eef14_train_cfg.gradient_accumulation_steps = 1
real_eef14_train_cfg.num_steps = 50000
