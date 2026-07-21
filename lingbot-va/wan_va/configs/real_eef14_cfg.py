# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_cfg import va_robotwin_cfg

real_eef14_cfg = EasyDict(__name__='Config: VA robotwin eef14 raw')
real_eef14_cfg.update(va_robotwin_cfg)

real_eef14_cfg.action_dim = 30
real_eef14_cfg.action_per_frame = 16

real_eef14_cfg.used_action_channel_ids = list(range(14))
inverse_used_action_channel_ids = [
    len(real_eef14_cfg.used_action_channel_ids)
] * real_eef14_cfg.action_dim
for i, j in enumerate(real_eef14_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
real_eef14_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

real_eef14_cfg.action_norm_method = 'quantiles'
real_eef14_cfg.norm_stat = {
    "q01": [
        -0.0002833375665068161,
        -0.28079317684751004,
        -0.11028339755372145,
        -0.11142503925268016,
        -0.12982331481543005,
        -0.4227886394147341,
        0.6335943209262695,
        -2.2998872736934572e-05,
        -0.17982885117352226,
        -0.10840431956690737,
        -0.4723544908859897,
        -0.09583946621146586,
        -0.3737651453011277,
        0.644342883967759,
    ] + [0.] * 16,
    "q99": [
        0.33629929071146775,
        0.1527795589306129,
        0.15961637323373024,
        0.4623737583755882,
        0.4501497876991899,
        0.20025978061730534,
        1.0,
        0.3657023078989005,
        0.2858931361158285,
        0.1977524097496644,
        0.19507470151996412,
        0.6071689802981095,
        0.3115309436284244,
        1.0,
    ] + [0.] * 16,
}
