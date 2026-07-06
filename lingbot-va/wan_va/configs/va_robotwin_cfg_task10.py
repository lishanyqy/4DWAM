# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .shared_config import va_shared_cfg

va_robotwin_cfg = EasyDict(__name__='Config: VA robotwin')
va_robotwin_cfg.update(va_shared_cfg)

va_robotwin_cfg.wan22_pretrained_model_name_or_path = "/path/to/pretrained/model"

va_robotwin_cfg.attn_window = 72
va_robotwin_cfg.frame_chunk_size = 4
va_robotwin_cfg.env_type = 'robotwin_tshape'

va_robotwin_cfg.height = 256
va_robotwin_cfg.width = 320
va_robotwin_cfg.action_dim = 30
va_robotwin_cfg.action_per_frame = 16
va_robotwin_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_robotwin_cfg.guidance_scale = 5
va_robotwin_cfg.action_guidance_scale = 1

va_robotwin_cfg.num_inference_steps = 25
va_robotwin_cfg.video_exec_step = -1
va_robotwin_cfg.action_num_inference_steps = 50

va_robotwin_cfg.snr_shift = 5.0
va_robotwin_cfg.action_snr_shift = 1.0

# va_robotwin_cfg.used_action_channel_ids = list(range(14))
va_robotwin_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))

inverse_used_action_channel_ids = [
    len(va_robotwin_cfg.used_action_channel_ids)
] * va_robotwin_cfg.action_dim

for i, j in enumerate(va_robotwin_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids
# [0,1,2,3,4,...,13,14,14,14...14]

va_robotwin_cfg.action_norm_method = 'quantiles'
# va_robotwin_cfg.action_norm_method = 'stats'
# quantiles
va_robotwin_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}

# "action": {"min": [-0.6221486330032349, 0.0, 0.0, -0.9532037973403931, -0.05479000136256218, -0.4001050889492035, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3511052429676056, -0.06377764791250229, 0.0], "max": [0.0, 2.016253709793091, 1.284923791885376, 0.9645599722862244, 0.42375120520591736, 0.017589999362826347, 1.0, 0.6228851675987244, 1.6095983982086182, 1.0734100341796875, 0.5172135829925537, 0.0, 0.0, 1.0], "mean": [-0.49123117327690125, 1.573000431060791, 0.8129812479019165, 0.34724944829940796, 0.1145293116569519, -0.12377040833234787, 0.38009050488471985, 0.2347407341003418, 0.596382737159729, 0.3894250690937042, 0.1975121945142746, -0.10537786036729813, -0.02419017069041729, 0.774392306804657], "std": [0.1541290581226349, 0.39733535051345825, 0.26328393816947937, 0.7446917295455933, 0.12363357096910477, 0.16553480923175812, 0.46438995003700256, 0.2834515869617462, 0.7290891408920288, 0.4784833490848541, 0.23907622694969177, 0.12706927955150604, 0.028559422120451927, 0.40586569905281067], "count": [221]}
