# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .real_eef14_cfg import real_eef14_cfg
from .real_eef14_train_cfg import real_eef14_train_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_lift2_merged_train_cfg import va_lift2_merged_train_cfg
from .va_lift2_merged_infer_cfg import va_lift2_merged_infer_cfg
from .va_4dwam_lift2_infer_cfg import va_4dwam_lift2_infer_cfg

from .config4dwam.train4dwam import lingbot4dwam_train_cfg as train_4dwam
from .config4dwam.train4dwam_long185_test import lingbot4dwam_train_cfg as train_4dwam_long185_test
from .config4dwam.train4dwam_long285_test import lingbot4dwam_train_cfg as train_4dwam_long285_test
from .config4dwam.train4dwam_mix285_235_test import (
    lingbot4dwam_train_cfg as train_4dwam_mix285_235_test,
)
from .config4dwam.train4dwam_longest50_test import (
    lingbot4dwam_train_cfg as train_4dwam_longest50_test,
)
from .config4dwam.train4dwam_lift2_merged import lingbot4dwam_train_cfg as train_4dwam_lift2_merged
from .config4dwam.train4dwam_test import lingbot4dwam_train_cfg as train_4dwam_test

# from .step2.ta_train_motion import va_robotwin_train_cfg as motion_train

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'lift2_merged_va': va_lift2_merged_train_cfg,
    'lift2_merged_infer': va_lift2_merged_infer_cfg,
    '4dwam_lift2_infer': va_4dwam_lift2_infer_cfg,
    '4dwam': train_4dwam,
    '4dwam_long185_test': train_4dwam_long185_test,
    '4dwam_long285_test': train_4dwam_long285_test,
    '4dwam_mix285_235_test': train_4dwam_mix285_235_test,
    '4dwam_longest50_test': train_4dwam_longest50_test,
    '4dwam_lift2_merged': train_4dwam_lift2_merged,
    '4dwam_test': train_4dwam_test,
    'real_eef14_train': real_eef14_train_cfg,
}
