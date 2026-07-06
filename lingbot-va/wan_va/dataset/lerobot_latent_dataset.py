# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    get_episode_data_index,
    get_safe_version,
    hf_transform_to_torch
)
import packaging.version
from lerobot.datasets.compute_stats import (
    # aggregate_stats, 
    compute_episode_stats
)
import numpy as np
from pathlib import Path
from collections.abc import Callable
import os
from tqdm import tqdm
from multiprocessing import Pool
from functools import partial
import torch
from einops import rearrange
from torch.utils.data import DataLoader
from scipy.spatial.transform import Rotation as R
from lerobot.constants import HF_LEROBOT_HOME
import pdb
import datasets
from datasets import concatenate_datasets, load_dataset
import pandas as pd
# from evaluation.robotwin.geometry import euler2quat
from utils.geometry import euler2quat
# from datasets.utils import hf_transform_to_torch
# from remote_pdb import RemotePdb

ORIGINS = {
    'action_range':16,
    'left' : (0,7),
    'right': (8,15)
}

MINE_TEST = {
    'action_range':14,
    'left' : (0,6),
    'left_gripper':(6,7),
    'right': (7,13),
    'right_gripper':(13,14),
}

def recursive_find_file(directory, filename='info.json'):
    result = []
    try:
        for root, dirs, files in os.walk(directory):
            if filename in files:
                full_path = os.path.join(root, filename)
                result.append(full_path)
    except PermissionError:
        print(f"Error: can not access {directory}")
    except Exception as e:
        print(f"Error: {e}")
    return result

def construct_lerobot(
    repo_id,
    config,
):
    return LatentLeRobotDataset(
        repo_id=repo_id,
        config=config,
    )

def construct_lerobot_multi_processor(config, 
                                      num_init_worker=8,
                                      ):
    datasets_out_lst = []
    construct_func = partial(
        construct_lerobot,
        config=config,
    )
    repo_list = recursive_find_file(config.dataset_path, 'info.json')
    repo_list = [v.split('/meta/info.json')[0] for v in repo_list]
    print(f'length of repo_list: {len(repo_list)}')
    # pdb.set_trace()
    if num_init_worker == 1:
        datasets_out_lst = []
        for repo in repo_list:
            result = construct_func(repo)
            datasets_out_lst.append(result)
    else:
        with Pool(num_init_worker) as pool:
            datasets_out_lst = pool.map(construct_func, repo_list)

    return datasets_out_lst

def get_relative_pose(pose, action_config = None):
    if torch.is_tensor(pose):
        pose = pose.detach().cpu().numpy()
    
    rot = R.from_quat(pose[:, 3:7])
    first_rot = R.from_quat(np.tile(pose[:1, 3:7], (pose.shape[0], 1)))
    trans = pose[:, :3]
    relative_trans = trans - trans[0:1]

    relative_rot = first_rot.inv() * rot
    relative_quat = relative_rot.as_quat()

    relative_pose = np.concatenate([relative_trans, relative_quat], axis=1)
    return torch.from_numpy(relative_pose)

def get_relative_joint_action(action):
    if torch.is_tensor(action):
        action = action.detach().cpu().numpy()
    relative_action = action - action[0:1]
    return torch.from_numpy(relative_action).float()
    

class MultiLatentLeRobotDataset(torch.utils.data.Dataset):
    ''''
    This dataset is used for multiple datasets concatenation, such as [task1, task2,...]
    local idx == global_idx in tasks.
    '''
    def __init__(
        self,
        config,
        num_init_worker=128,
    ):
        self._datasets = construct_lerobot_multi_processor(config, 
                                                           num_init_worker, 
                                                           )
        self.item_id_to_dataset_id, self.acc_dset_num = (
            self._get_item_id_to_dataset_id()
        )

    def __len__(
        self,
    ):
        return sum(len(v) for v in self._datasets)

    def _get_item_id_to_dataset_id(self):
        item_id_to_dataset_id = {}
        acc_dset_num = {}
        acc_nums = [0]
        id = 0
        for dset_id, dset in enumerate(self._datasets):
            acc_nums.append(acc_nums[-1] + len(dset))
            for _ in range(len(dset)):
                item_id_to_dataset_id[id] = dset_id
                id += 1
        for did in range(len(self._datasets)):
            acc_dset_num[did] = acc_nums[did]
        return item_id_to_dataset_id, acc_dset_num

    def __getitem__(self, idx) -> dict:
        assert idx < len(self)
        cur_dset = self._datasets[self.item_id_to_dataset_id[idx]]
        local_idx = idx - self.acc_dset_num[self.item_id_to_dataset_id[idx]]
        return cur_dset[local_idx]


def aggregate_stats(stats_list: dict[str,dict[str, dict]]) -> dict[str, dict[str, np.ndarray]]:
    """Aggregate stats from multiple compute_stats outputs into a single set of stats.

    The final stats will have the union of all data keys from each of the stats dicts.

    For instance:
    - new_min = min(min_dataset_0, min_dataset_1, ...)
    - new_max = max(max_dataset_0, max_dataset_1, ...)
    - new_mean = (mean of all data, weighted by counts)
    - new_std = (std of all data)
    """

    # _assert_type_and_shape(stats_list)

    # data_keys = {key for stats in stats_list for key in stats}
    data_keys = {
        key for key in stats_list[0]
    }
    # aggregated_stats = {key: {} for key in data_keys}
    aggregated_stats = {}
    stats_list = [
        stats_list[key] for key in stats_list
    ]
    for stats in stats_list:
        for key, v in stats.items():
            if key not in aggregated_stats:
                aggregated_stats[key] = {
                    "min": np.array(v["min"]),
                    "max": np.array(v["max"]),
                    "sum": np.array(v["mean"]) * v["count"],
                    "sq_sum": (np.array(v["std"])**2 + np.array(v["mean"])**2) * v["count"],
                    "count": v["count"]
                }
            else:
                aggregated_stats[key]["min"] = np.minimum(aggregated_stats[key]["min"], v["min"])
                aggregated_stats[key]["max"] = np.maximum(aggregated_stats[key]["max"], v["max"])
                aggregated_stats[key]["sum"] += np.array(v["mean"]) * v["count"]
                aggregated_stats[key]["sq_sum"] += (np.array(v["std"])**2 + np.array(v["mean"])**2) * v["count"]
                aggregated_stats[key]["count"] += v["count"]

    return aggregated_stats

class LatentLeRobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id,
        config=None,
        download_videos = None,
    ):
        # This is for temporary.
        self.action_conf = MINE_TEST

        self.repo_id = repo_id
        self.root = HF_LEROBOT_HOME / repo_id
        self.image_transforms = None
        self.delta_timestamps = None
        self.episodes = None
        self.tolerance_s = 1e-4
        self.revision = "v2.1"
        self.video_backend = 'pyav'
        self.delta_indices = None
        self.batch_encoding_size = 1
        self.episodes_since_last_encoding = 0
        self.image_writer = None
        self.episode_buffer = None
        self.root.mkdir(exist_ok=True, parents=True)
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync
            =False
        )
        # pdb.set_trace()
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)
        
        # if self.meta._version >= packaging.version.parse("v2.1"):
        #     # episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
        #     self.stats = aggregate_stats(self.meta.episodes_stats)
        #     self.action_max = np.pad(self.stats['action']['max'], (0, 30 - len(self.stats['action']['max'])), mode='constant')
        #     self.action_min = np.pad(self.stats['action']['min'], (0, 30 - len(self.stats['action']['max'])), mode='constant')

        try:
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset_tmp()
            # self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()
        # pdb.set_trace()
        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)
        
        self.latent_path = Path(repo_id) / 'latents'
        self.trace_path = Path(repo_id) / 'trace'
        self.empty_emb = torch.load(config.empty_emb_path, weights_only=False)
        self.config = config
        self.cfg_prob = config.cfg_prob
        self.used_video_keys = config.obs_cam_keys
        self.q01 = np.array(config.norm_stat['q01'], dtype='float')[None]
        self.q99 = np.array(config.norm_stat['q99'], dtype='float')[None]
        # pdb.set_trace()
        self._hf_torch_view = self.hf_dataset.with_format(
                type='torch',
                columns=['action'],
                output_all_columns=False
            )
        self.parse_meta()
        self.enable_trace = config.enable_trace

    def load_hf_dataset_tmp(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data" / "chunk-000")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def parse_meta(self):
        out = []
        for key, value in self.meta.episodes.items():
            episode_index = value["episode_index"]
            tasks = value["tasks"]
            # rank = int(os.environ.get("RANK", 0))
            # if rank == 0:  # 只在 rank0 停住
            #     print(f"[Debug] Waiting for remote-PDB on port {4444} ...", flush=True)
            #     RemotePdb("0.0.0.0", 4444).set_trace()
            # # pdb.set_trace()
            # print(value.keys())
            action_config = value["action_config"]
            for acfg in action_config:
                cur_meta = {
                    "episode_index": episode_index,
                    "tasks": tasks,
                }
                cur_meta.update(acfg)

                check_statu = self._check_meta(
                    cur_meta["start_frame"],
                    cur_meta["end_frame"],
                    cur_meta["episode_index"],
                )

                if check_statu:
                    out.append(cur_meta)
        # pdb.set_trace()
        self.new_metas = out

    def _check_meta(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"

        for key in self.used_video_keys:
            cur_path = latent_path / key
            # if 
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            if not os.path.exists(latent_file):
                return False
        return True

    def _get_global_idx(self, episode_index: int, local_index: int):
        ep_start = self.episode_data_index["from"][episode_index]
        return local_index + ep_start

    def _get_range_hf_data(self, start_frame, end_frame):
        batch = self._hf_torch_view[start_frame:end_frame]
        return batch

    def _flatten_latent_dict(self, latent_dict):
        out = {}
        for key, value in latent_dict.items():
            for inner_key, inner_value in value.items():
                new_key = f"{key}.{inner_key}"
                out[new_key] = inner_value
        return out

    def _get_range_latent_data(self, start_frame, end_frame, episode_index):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        latent_path = Path(self.latent_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        for key in self.used_video_keys:
            cur_path = latent_path / key
            latent_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            assert os.path.exists(latent_file)
            # df = pd.read_parquet(latent_file)
            # latent_data = df.to_dict(orient="records")
            # latent_data = df.to_dict()
            latent_data = torch.load(latent_file, weights_only=False)
            out[key] = latent_data
        
        return self._flatten_latent_dict(out)
    
        
    def _cat_video_latents(self,
                           data_dict
                           ):
        latent_lst = []
        for key in self.used_video_keys:
            latent= data_dict[f"{key}.latent"]
            
            latent_num_frames = data_dict[f"{key}.latent_num_frames"]
            latent_height = data_dict[f"{key}.latent_height"]
            latent_width = data_dict[f"{key}.latent_width"]
            latent = rearrange(latent, 
                                 '(f h w) c -> f h w c', 
                                #  f=, 
                                 h=latent_height, 
                                 w=latent_width)
            # print('#'*20,latent.shape)
            latent_lst.append(latent)
        wrist_latent = torch.cat(latent_lst[1:], dim=2)
        cat_latent = torch.cat([wrist_latent, latent_lst[0]], dim=1)

        text_emb = data_dict[f"{self.used_video_keys[0]}.text_emb"]
        if torch.rand(1).item() < self.cfg_prob:
            text_emb = self.empty_emb

        out_dict = dict(
            latents = cat_latent,
            text_emb = text_emb,
        )
        return out_dict
    
    def _action_post_process14(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        # left_action = get_relative_pose(action[:, :self.action_conf['left'][1]])
        left_action = get_relative_joint_action(action[:, :self.action_conf['left'][1]])
        # right_action = get_relative_pose(action[:, self.action_conf['right'][0]:self.action_conf['right'][1]])
        right_action = get_relative_joint_action(action[:, self.action_conf['right'][0]:self.action_conf['right'][1]])
        action = np.concatenate([left_action, action[:, self.action_conf['left_gripper'][0]:self.action_conf['left_gripper'][1]], right_action, action[:, self.action_conf['right_gripper'][0]:self.action_conf['right_gripper'][1]]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids] # here repeat to dim=30, len(inverse_used_action_channel_ids) = 30
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        # action_aligned = action_aligned - self.meta.episodes_stats
        # action_aligned = (action_aligned - self.q01) / (
        #         self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = (action_aligned - self.action_min) / (
            self.action_max - self.action_min + 1e-6
        ) * 2. - 1.

        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def _action_post_process_abandon(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def action14_to_action16(self, action):
        """
        action: (F, 14)
        return: (F, 16)

        output order:
        [left_xyz(3), left_quat(4), left_grip(1),
        right_xyz(3), right_quat(4), right_grip(1)]
        """
        action = np.asarray(action)
        assert action.ndim == 2 and action.shape[1] == 14

        out = []
        for a in action:
            lq = euler2quat(a[3], a[4], a[5])      # (4,)
            rq = euler2quat(a[10], a[11], a[12])   # (4,)

            a16 = np.concatenate([
                a[:3],      # left xyz
                lq,         # left quat
                a[6:7],     # left grip
                a[7:10],    # right xyz
                rq,         # right quat
                a[13:14],   # right grip
            ], axis=0)

            out.append(a16)

        return np.stack(out, axis=0)

    def _action_post_process(self, local_start_frame, local_end_frame, latent_frame_ids, action):
        if action.shape[1] == 14:
            action = self.action14_to_action16(action)
        act_shift = int(latent_frame_ids[0] - local_start_frame)
        frame_stride = latent_frame_ids[1] - latent_frame_ids[0]
        action = action[act_shift:]
        if self.config.env_type == 'robotwin_tshape': ## TODO support get_relative_pose for other dataset, currently only support robotwin 
            left_action = get_relative_pose(action[:, :7])
            right_action = get_relative_pose(action[:, 8:15])
            action = np.concatenate([left_action, action[:, 7:8], right_action, action[:, 15:16]], axis=1)
        action = np.pad(action, pad_width=((frame_stride * 4, 0), (0, 0)), mode='constant', constant_values=0)

        latent_frame_num = (len(latent_frame_ids) - 1) // 4 + 1
        required_action_num = latent_frame_num * frame_stride * 4

        action = action[:required_action_num]
        action_mask = np.ones_like(action, dtype='bool')
        assert action.shape[0] == required_action_num


        action_paded = np.pad(action, ((0, 0), (0, 1)), mode='constant', constant_values=0)
        action_mask_padded = np.pad(action_mask, ((0, 0), (0, 1)), mode='constant', constant_values=0)

        action_aligned = action_paded[:, self.config.inverse_used_action_channel_ids]
        action_mask_aligned = action_mask_padded[:, self.config.inverse_used_action_channel_ids]
        action_aligned = (action_aligned - self.q01) / (
                self.q99 - self.q01 + 1e-6) * 2. - 1.
        action_aligned = rearrange(action_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_mask_aligned = rearrange(action_mask_aligned, "(f n) c -> c f n 1", f=latent_frame_num)
        action_aligned *= action_mask_aligned
        return torch.from_numpy(action_aligned).float(), torch.from_numpy(action_mask_aligned).bool()

    def _trace_data(self, start_frame, end_frame, episode_index, latent_data_dict):
        episode_chunk = self.meta.get_episode_chunk(episode_index)
        trace_path = Path(self.trace_path) / f"chunk-{episode_chunk:03d}"
        out = {}
        trace_list = []
        for key in self.used_video_keys:
            cur_path = trace_path / key
            trace_file = (
                cur_path / f"episode_{episode_index:06d}_{start_frame}_{end_frame}.pth"
            )
            # print(trace_file)
            assert os.path.exists(trace_file)
            # df = pd.read_parquet(latent_file)
            # latent_data = df.to_dict(orient="records")
            # latent_data = df.to_dict()
            trace_data = torch.load(trace_file, weights_only=False)
            height, width = latent_data_dict[f"{key}.latent_height"], latent_data_dict[f"{key}.latent_width"]
            # (F, T, D) (85, 320, 1024)
            # trace_data = arrange()
            trace_data = rearrange(trace_data, 
                                 'f (h w) c -> f h w c', 
                                #  f=, 
                                 h=height, 
                                 w=width)
            trace_list.append(trace_data)
        wrist_trace = torch.cat(trace_list[1:], dim=2)
        cat_trace = torch.cat([wrist_trace, trace_list[0]], dim=1)
        # return self._flatten_latent_dict(out)
        return cat_trace

    def __getitem__(self, idx) -> dict:
        idx = idx % len(self.new_metas)
        cur_meta = self.new_metas[idx]
        episode_index = cur_meta["episode_index"]
        start_frame = cur_meta["start_frame"]
        end_frame = cur_meta["end_frame"]
        local_start_frame = start_frame
        local_end_frame = end_frame

        ori_data_dict = self._get_range_latent_data(start_frame, end_frame, episode_index)
        if self.enable_trace:
            trace_data = self._trace_data(start_frame, end_frame, episode_index, ori_data_dict)

        latent_frame_ids = ori_data_dict[f"{self.used_video_keys[0]}.frame_ids"]
        start_frame = self._get_global_idx(episode_index, start_frame)
        end_frame = self._get_global_idx(episode_index, end_frame)
        hf_data_frames = self._get_range_hf_data(start_frame, end_frame)
        ori_data_dict.update(hf_data_frames)
        # pdb.set_trace()
        out_dict = self._cat_video_latents(ori_data_dict)

        out_dict['actions'], out_dict['actions_mask'] = self._action_post_process(local_start_frame, local_end_frame, latent_frame_ids, ori_data_dict['action'])

        out_dict['latents'] = out_dict['latents'].permute(3, 0, 1, 2)
        if self.enable_trace:
            out_dict['trace'] = trace_data
        # print(f'index: {idx}, {out_dict.keys()}')
        return out_dict

    def __len__(self):
        return len(self.new_metas)

if __name__ == '__main__':
    from wan_va.configs import VA_CONFIGS
    from tqdm import tqdm
    dset = MultiLatentLeRobotDataset(
        VA_CONFIGS['robotwin_train']
    )
    for key, value in dset[0].items():
        if isinstance(value, torch.Tensor):
            print(f'{key}: {value.shape} tensor')
        elif isinstance(value, np.ndarray):
            print(f'{key}: {value.shape} np')
        else:
            print(f'{key}: {value}')
    print(len(dset))
    dloader = DataLoader(
            dset,
            batch_size=1,
            shuffle=True,
            num_workers=32,
        )
    max_l = 0
    action_list = []
    for data in tqdm(dloader):
        _, _, F, H, W = data['latents'].shape
        max_l = max(max_l, F*H*W)
        action_list.append(data['actions'].flatten(2).permute(0, 2, 1).flatten(0, 1))
    action_all = torch.cat(action_list, dim=0)
    print(max_l)
    print(action_all.shape, action_all.mean(dim=0), action_all.min(dim=0)[0], action_all.max(dim=0)[0])
    
