# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_robotwin_infer_cfg import va_robotwin_infer_cfg
from .va_robotwin_train_task10_cfg import va_robotwin_train_cfg as va_robotwin_train_task10_cfg
from .va_robotwin_infer_task10_cfg import va_robotwin_train_cfg as va_robotwin_infer_task10_cfg
from .va_robotwin_train_ls_cfg import va_robotwin_train_cfg as va_robotwin_train_ls_cfg
from .va_robotwin_infer_ls_cfg import va_robotwin_train_cfg as va_robotwin_infer_ls_cfg
from .tiny.va_robotwin_train_cfg import va_robotwin_train_cfg as train_tiny_cfg
from .tiny.va_robotwin_infer_cfg import va_robotwin_train_cfg as infer_tiny_cfg
from .clean.robotwin_ta_train import va_robotwin_train_cfg as clean_ta_train
from .clean.robotwin_ta_infer import va_robotwin_train_cfg as clean_ta_infer
from .clean.robotwin_va_train import va_robotwin_train_cfg as clean_va_train
from .clean.robotwin_va_infer import va_robotwin_train_cfg as clean_va_infer
from .clean.robotwin_ta_train_fromva import va_robotwin_train_cfg as ta_fromva
from .step2.ta_train_coef import va_robotwin_train_cfg as coef_train
# from .step2.ta_train_motion import va_robotwin_train_cfg as motion_train
from .step2.ta_infer_motion import va_robotwin_train_cfg as motion_infer

from .motion.dest_ta_train import va_robotwin_train_cfg as dest_train
from .motion.motion_ta_train import va_robotwin_train_cfg as motion_train
from .motion.unified_dest_motion_train import va_robotwin_train_cfg as unified_dest_motion_train
from .motion.unified_dest_motion_train_full import va_robotwin_train_cfg as unified_clean_train

from .va.va_train_from_va2000 import va_robotwin_train_cfg as va_train
from .va.va_clean10k import va_robotwin_train_cfg as va_clean10k

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'robotwin_infer': va_robotwin_infer_cfg,
    'robotwin_train_task10': va_robotwin_train_task10_cfg,
    'robotwin_infer_task10': va_robotwin_infer_task10_cfg,
    'robotwin_train_ls': va_robotwin_train_ls_cfg,
    'robotwin_infer_ls': va_robotwin_infer_ls_cfg,
    'tiny_train': train_tiny_cfg,
    'tiny_infer': infer_tiny_cfg,
    'clean_ta_train': clean_ta_train,
    'clean_ta_infer':clean_ta_infer,
    'clean_va_train': clean_va_train,
    'clean_va_infer':clean_va_infer,
    'ta_fromva':ta_fromva,
    'coef_train':coef_train,
    'motion_train':motion_train,
    'motion_infer':motion_infer,
    
    'dest_train':dest_train,
    'unified_dest_motion_train':unified_dest_motion_train,
    'unified_clean_train': unified_clean_train,
    
    'va_train': va_train,
    'va_clean10k': va_clean10k,
}