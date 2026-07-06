# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_robotwin_cfg import va_robotwin_cfg
import os

va_robotwin_infer_cfg = EasyDict(__name__='Config: VA robotwin task10 train')
va_robotwin_infer_cfg.update(va_robotwin_cfg)
va_robotwin_infer_cfg.infer_mode = 'server'
# va_robotwin_infer_cfg.resume_from = '/robby/share/Robotics/lilin1/code/Wan_VA_Release/train_out/checkpoints/checkpoint_step_10'
va_robotwin_infer_cfg.home_path = '/cpfs01/projects-HDD/cfff-377aad6b032c_HDD/chenshuai/wenxuan/'
va_robotwin_infer_cfg.dataset_path = os.path.join(va_robotwin_infer_cfg.home_path, '.cache/huggingface/lerobot/robotwin/robotwin_multi_10_tasks')
va_robotwin_infer_cfg.empty_emb_path = os.path.join(va_robotwin_infer_cfg.dataset_path, 'empty_emb.pt')
va_robotwin_infer_cfg.enable_wandb = False
# va_robotwin_infer_cfg.load_worker = 16
va_robotwin_infer_cfg.load_worker = 0 # for multiprocesses
va_robotwin_infer_cfg.save_interval = 2000
va_robotwin_infer_cfg.gc_interval = 50
va_robotwin_infer_cfg.cfg_prob = 0.1
unified_path = '.cache/modelscope/Robbyant/lingbot-va-posttrain-robotwin'
va_robotwin_infer_cfg.vae_path = os.path.join(va_robotwin_infer_cfg.home_path, unified_path)
va_robotwin_infer_cfg.text_model_path = os.path.join(va_robotwin_infer_cfg.home_path, unified_path)
# va_robotwin_infer_cfg.wan22_pretrained_model_name_or_path = os.path.join(va_robotwin_infer_cfg.home_path, 'mta/lingbot-va/train_out/checkpoints/checkpoint_step_20000')
va_robotwin_infer_cfg.wan22_pretrained_model_name_or_path = os.path.join(va_robotwin_infer_cfg.home_path, unified_path)

# Training parameters
va_robotwin_infer_cfg.learning_rate = 1e-5
va_robotwin_infer_cfg.beta1 = 0.9
va_robotwin_infer_cfg.beta2 = 0.95
va_robotwin_infer_cfg.weight_decay = 0.1
va_robotwin_infer_cfg.warmup_steps = 10
va_robotwin_infer_cfg.batch_size = 1
va_robotwin_infer_cfg.gradient_accumulation_steps = 1
va_robotwin_infer_cfg.num_steps = 20000

# others
va_robotwin_infer_cfg.max_tokens = 128
# va_robotwin_infer_cfg.action_norm_method = 'quantiles'

va_robotwin_infer_cfg.action_max = [5.68151522, 3.45788741, 3.66244769, 1.78976679, 1.56326973,
       1.77522206, 1.        , 1.26083982, 3.24260497, 3.6588552 ,
       1.89039254, 1.28473699, 3.15973711, 1.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.        , 0.        ]

va_robotwin_infer_cfg.action_min = [-6.37791872e+00, -3.16402118e-04, -2.96088438e-02, -1.92681313e+00,
       -1.42168283e+00, -6.23234034e+00,  0.00000000e+00, -6.31657696e+00,
       -4.66045767e-01, -1.64696306e-03, -1.98937869e+00, -1.37959743e+00,
       -6.26952934e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
        0.00000000e+00,  0.00000000e+00]
