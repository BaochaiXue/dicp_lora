# All Code from diffusion_policy

## reference_material/diffusion_policy_code/diffusion_policy/dataset/base_dataset.py
```python
from typing import Dict

import torch
import torch.nn
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseLowdimDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseLowdimDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: T, Do
            action: T, Da
        """
        raise NotImplementedError()


class BaseImageDataset(torch.utils.data.Dataset):
    def get_validation_dataset(self) -> 'BaseLowdimDataset':
        # return an empty dataset by default
        return BaseImageDataset()

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        raise NotImplementedError()

    def get_all_actions(self) -> torch.Tensor:
        raise NotImplementedError()
    
    def __len__(self) -> int:
        return 0
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        output:
            obs: 
                key: T, *
            action: T, Da
        """
        raise NotImplementedError()

```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/pusht_image_dataset.py
```python
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class PushTImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': sample['action'].astype(np.float32) # T, 2
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/pusht_dataset.py
```python
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class PushTLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='keypoint',
            state_key='state',
            action_key='action',
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, state_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask
            )
        self.obs_key = obs_key
        self.state_key = state_key
        self.action_key = action_key
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer[self.action_key])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        keypoint = sample[self.obs_key]
        state = sample[self.state_key]
        agent_pos = state[:,:2]
        obs = np.concatenate([
            keypoint.reshape(keypoint.shape[0], -1), 
            agent_pos], axis=-1)

        data = {
            'obs': obs, # T, D_o
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/robomimic_replay_image_dataset.py
```python
from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import zarr
import os
import shutil
import copy
import json
import hashlib
from filelock import FileLock
from threadpoolctl import threadpool_limits
import concurrent.futures
import multiprocessing
from omegaconf import OmegaConf
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs, Jpeg2k
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)
register_codecs()

class RobomimicReplayImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            abs_action=False,
            rotation_rep='rotation_6d', # ignored when abs_action=False
            use_legacy_normalizer=False,
            use_cache=False,
            seed=42,
            val_ratio=0.0
        ):
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = None
        if use_cache:
            cache_zarr_path = dataset_path + '.zarr.zip'
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        # store = zarr.DirectoryStore(cache_zarr_path)
                        replay_buffer = _convert_robomimic_to_replay(
                            store=zarr.MemoryStore(), 
                            shape_meta=shape_meta, 
                            dataset_path=dataset_path, 
                            abs_action=abs_action, 
                            rotation_transformer=rotation_transformer)
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _convert_robomimic_to_replay(
                store=zarr.MemoryStore(), 
                shape_meta=shape_meta, 
                dataset_path=dataset_path, 
                abs_action=abs_action, 
                rotation_transformer=rotation_transformer)

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        # for key in rgb_keys:
        #     replay_buffer[key].compressor.numthreads=1

        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.abs_action = abs_action
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer

        # obs
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])

            if key.endswith('pos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith('quat'):
                # quaternion is in [-1,1] already
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith('qpos'):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError('unsupported')
            normalizer[key] = this_normalizer

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(data['action'].astype(np.float32))
        }
        return torch_data


def _convert_actions(raw_actions, abs_action, rotation_transformer):
    actions = raw_actions
    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
        actions = raw_actions
    return actions


def _convert_robomimic_to_replay(store, shape_meta, dataset_path, abs_action, rotation_transformer, 
        n_workers=None, max_inflight_tasks=None):
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    if max_inflight_tasks is None:
        max_inflight_tasks = n_workers * 5

    # parse shape_meta
    rgb_keys = list()
    lowdim_keys = list()
    # construct compressors and chunks
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        shape = attr['shape']
        type = attr.get('type', 'low_dim')
        if type == 'rgb':
            rgb_keys.append(key)
        elif type == 'low_dim':
            lowdim_keys.append(key)
    
    root = zarr.group(store)
    data_group = root.require_group('data', overwrite=True)
    meta_group = root.require_group('meta', overwrite=True)

    with h5py.File(dataset_path) as file:
        # count total steps
        demos = file['data']
        episode_ends = list()
        prev_end = 0
        for i in range(len(demos)):
            demo = demos[f'demo_{i}']
            episode_length = demo['actions'].shape[0]
            episode_end = prev_end + episode_length
            prev_end = episode_end
            episode_ends.append(episode_end)
        n_steps = episode_ends[-1]
        episode_starts = [0] + episode_ends[:-1]
        _ = meta_group.array('episode_ends', episode_ends, 
            dtype=np.int64, compressor=None, overwrite=True)

        # save lowdim data
        for key in tqdm(lowdim_keys + ['action'], desc="Loading lowdim data"):
            data_key = 'obs/' + key
            if key == 'action':
                data_key = 'actions'
            this_data = list()
            for i in range(len(demos)):
                demo = demos[f'demo_{i}']
                this_data.append(demo[data_key][:].astype(np.float32))
            this_data = np.concatenate(this_data, axis=0)
            if key == 'action':
                this_data = _convert_actions(
                    raw_actions=this_data,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer
                )
                assert this_data.shape == (n_steps,) + tuple(shape_meta['action']['shape'])
            else:
                assert this_data.shape == (n_steps,) + tuple(shape_meta['obs'][key]['shape'])
            _ = data_group.array(
                name=key,
                data=this_data,
                shape=this_data.shape,
                chunks=this_data.shape,
                compressor=None,
                dtype=this_data.dtype
            )
        
        def img_copy(zarr_arr, zarr_idx, hdf5_arr, hdf5_idx):
            try:
                zarr_arr[zarr_idx] = hdf5_arr[hdf5_idx]
                # make sure we can successfully decode
                _ = zarr_arr[zarr_idx]
                return True
            except Exception as e:
                return False
        
        with tqdm(total=n_steps*len(rgb_keys), desc="Loading image data", mininterval=1.0) as pbar:
            # one chunk per thread, therefore no synchronization needed
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = set()
                for key in rgb_keys:
                    data_key = 'obs/' + key
                    shape = tuple(shape_meta['obs'][key]['shape'])
                    c,h,w = shape
                    this_compressor = Jpeg2k(level=50)
                    img_arr = data_group.require_dataset(
                        name=key,
                        shape=(n_steps,h,w,c),
                        chunks=(1,h,w,c),
                        compressor=this_compressor,
                        dtype=np.uint8
                    )
                    for episode_idx in range(len(demos)):
                        demo = demos[f'demo_{episode_idx}']
                        hdf5_arr = demo['obs'][key]
                        for hdf5_idx in range(hdf5_arr.shape[0]):
                            if len(futures) >= max_inflight_tasks:
                                # limit number of inflight tasks
                                completed, futures = concurrent.futures.wait(futures, 
                                    return_when=concurrent.futures.FIRST_COMPLETED)
                                for f in completed:
                                    if not f.result():
                                        raise RuntimeError('Failed to encode image!')
                                pbar.update(len(completed))

                            zarr_idx = episode_starts[episode_idx] + hdf5_idx
                            futures.add(
                                executor.submit(img_copy, 
                                    img_arr, zarr_idx, hdf5_arr, hdf5_idx))
                completed, futures = concurrent.futures.wait(futures)
                for f in completed:
                    if not f.result():
                        raise RuntimeError('Failed to encode image!')
                pbar.update(len(completed))

    replay_buffer = ReplayBuffer(root)
    return replay_buffer

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/mujoco_image_dataset.py
```python
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class MujocoImageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
            ):
        
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            # zarr_path, keys=['img', 'state', 'action'])
            zarr_path, keys=['robot_0_camera_images', 'robot_0_tcp_xyz_wxyz', 'robot_0_gripper_width', 'action_0_tcp_xyz_wxyz', 'action_0_gripper_width'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': np.concatenate([self.replay_buffer['action_0_tcp_xyz_wxyz'], self.replay_buffer['action_0_gripper_width']], axis=-1),
            'agent_pos': np.concatenate([self.replay_buffer['robot_0_tcp_xyz_wxyz'], self.replay_buffer['robot_0_tcp_xyz_wxyz']], axis=-1)
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        # agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        agent_pos = np.concatenate([sample['robot_0_tcp_xyz_wxyz'], sample['robot_0_gripper_width']], axis=-1).astype(np.float32)
        agent_action = np.concatenate([sample['action_0_tcp_xyz_wxyz'], sample['action_0_gripper_width']], axis=-1).astype(np.float32)
        # image = np.moveaxis(sample['img'],-1,1)/255
        image = np.moveaxis(sample['robot_0_camera_images'].astype(np.float32).squeeze(1),-1,1)/255

        data = {
            'obs': {
                'image': image, # T, 3, 224, 224
                'agent_pos': agent_pos, # T, 8 (x,y,z,qx,qy,qz,qw,gripper_width)
            },
            'action': agent_action # T, 8 (x,y,z,qx,qy,qz,qw,gripper_width)
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('/home/yihuai/robotics/repositories/mujoco/mujoco-env/data/collect_heuristic_data/2024-12-24_11-36-15_100episodes/merged_data.zarr')
    dataset = MujocoImageDataset(zarr_path, horizon=16)
    print(dataset[0])
    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)

if __name__ == '__main__':
    test()
```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/robomimic_replay_lowdim_dataset.py
```python
from typing import Dict, List
import torch
import numpy as np
import h5py
from tqdm import tqdm
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset, LinearNormalizer
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.common.normalize_util import (
    robomimic_abs_action_only_normalizer_from_stat,
    robomimic_abs_action_only_dual_arm_normalizer_from_stat,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RobomimicReplayLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_keys: List[str]=[
                'object', 
                'robot0_eef_pos', 
                'robot0_eef_quat', 
                'robot0_gripper_qpos'],
            abs_action=False,
            rotation_rep='rotation_6d',
            use_legacy_normalizer=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None
        ):
        obs_keys = list(obs_keys)
        rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep=rotation_rep)

        replay_buffer = ReplayBuffer.create_empty_numpy()
        with h5py.File(dataset_path) as file:
            demos = file['data']
            for i in tqdm(range(len(demos)), desc="Loading hdf5 to ReplayBuffer"):
                demo = demos[f'demo_{i}']
                episode = _data_to_obs(
                    raw_obs=demo['obs'],
                    raw_actions=demo['actions'][:].astype(np.float32),
                    obs_keys=obs_keys,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer)
                replay_buffer.add_episode(episode)

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.use_legacy_normalizer = use_legacy_normalizer
    
    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        stat = array_to_stats(self.replay_buffer['action'])
        if self.abs_action:
            if stat['mean'].shape[-1] > 10:
                # dual arm
                this_normalizer = robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat)
            else:
                this_normalizer = robomimic_abs_action_only_normalizer_from_stat(stat)
            
            if self.use_legacy_normalizer:
                this_normalizer = normalizer_from_stat(stat)
        else:
            # already normalized
            this_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer['action'] = this_normalizer
        
        # aggregate obs stats
        obs_stat = array_to_stats(self.replay_buffer['obs'])


        normalizer['obs'] = normalizer_from_stat(obs_stat)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])
    
    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

def normalizer_from_stat(stat):
    max_abs = np.maximum(stat['max'].max(), np.abs(stat['min']).max())
    scale = np.full_like(stat['max'], fill_value=1/max_abs)
    offset = np.zeros_like(stat['max'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )
    
def _data_to_obs(raw_obs, raw_actions, obs_keys, abs_action, rotation_transformer):
    obs = np.concatenate([
        raw_obs[key] for key in obs_keys
    ], axis=-1).astype(np.float32)

    if abs_action:
        is_dual_arm = False
        if raw_actions.shape[-1] == 14:
            # dual arm
            raw_actions = raw_actions.reshape(-1,2,7)
            is_dual_arm = True

        pos = raw_actions[...,:3]
        rot = raw_actions[...,3:6]
        gripper = raw_actions[...,6:]
        rot = rotation_transformer.forward(rot)
        raw_actions = np.concatenate([
            pos, rot, gripper
        ], axis=-1).astype(np.float32)
    
        if is_dual_arm:
            raw_actions = raw_actions.reshape(-1,20)
    
    data = {
        'obs': obs,
        'action': raw_actions
    }
    return data

```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/real_pusht_image_dataset.py
```python
from typing import Dict, List
import torch
import numpy as np
import zarr
import os
import shutil
from filelock import FileLock
from threadpoolctl import threadpool_limits
from omegaconf import OmegaConf
import cv2
import json
import hashlib
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.common.normalize_util import (
    get_range_normalizer_from_stat,
    get_image_range_normalizer,
    get_identity_normalizer_from_stat,
    array_to_stats
)

class RealPushTImageDataset(BaseImageDataset):
    def __init__(self,
            shape_meta: dict,
            dataset_path: str,
            horizon=1,
            pad_before=0,
            pad_after=0,
            n_obs_steps=None,
            n_latency_steps=0,
            use_cache=False,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            delta_action=False,
        ):
        assert os.path.isdir(dataset_path)
        
        replay_buffer = None
        if use_cache:
            # fingerprint shape_meta
            shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
            shape_meta_hash = hashlib.md5(shape_meta_json.encode('utf-8')).hexdigest()
            cache_zarr_path = os.path.join(dataset_path, shape_meta_hash + '.zarr.zip')
            cache_lock_path = cache_zarr_path + '.lock'
            print('Acquiring lock on cache.')
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    # cache does not exists
                    try:
                        print('Cache does not exist. Creating!')
                        replay_buffer = _get_replay_buffer(
                            dataset_path=dataset_path,
                            shape_meta=shape_meta,
                            store=zarr.MemoryStore()
                        )
                        print('Saving cache to disk.')
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(
                                store=zip_store
                            )
                    except Exception as e:
                        shutil.rmtree(cache_zarr_path)
                        raise e
                else:
                    print('Loading cached ReplayBuffer from Disk.')
                    with zarr.ZipStore(cache_zarr_path, mode='r') as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore())
                    print('Loaded!')
        else:
            replay_buffer = _get_replay_buffer(
                dataset_path=dataset_path,
                shape_meta=shape_meta,
                store=zarr.MemoryStore()
            )
        
        if delta_action:
            # replace action as relative to previous frame
            actions = replay_buffer['action'][:]
            # support positions only at this time
            assert actions.shape[1] <= 3
            actions_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                # delta action is the difference between previous desired position and the current
                # it should be scheduled at the previous timestep for the current timestep
                # to ensure consistency with positional mode
                actions_diff[start+1:end] = np.diff(actions[start:end], axis=0)
            replay_buffer['action'][:] = actions_diff

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                rgb_keys.append(key)
            elif type == 'low_dim':
                lowdim_keys.append(key)
        
        key_first_k = dict()
        if n_obs_steps is not None:
            # only take first k obs from images
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        sampler = SequenceSampler(
            replay_buffer=replay_buffer, 
            sequence_length=horizon+n_latency_steps,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        
        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon+self.n_latency_steps,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=self.val_mask
            )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer['action'])
        
        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key])
        
        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        # to save RAM, only return first n_obs_steps of OBS
        # since the rest will be discarded anyway.
        # when self.n_obs_steps is None
        # this slice does nothing (takes all)
        T_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.rgb_keys:
            # move channel last to channel first
            # T,H,W,C
            # convert uint8 image to float32
            obs_dict[key] = np.moveaxis(data[key][T_slice],-1,1
                ).astype(np.float32) / 255.
            # T,C,H,W
            # save ram
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            # save ram
            del data[key]
        
        action = data['action'].astype(np.float32)
        # handle latency by dropping first n_latency_steps action
        # observations are already taken care of by T_slice
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps:]

        torch_data = {
            'obs': dict_apply(obs_dict, torch.from_numpy),
            'action': torch.from_numpy(action)
        }
        return torch_data

def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[...,idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr

def _get_replay_buffer(dataset_path, shape_meta, store):
    # parse shape meta
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta['obs']
    for key, attr in obs_shape_meta.items():
        type = attr.get('type', 'low_dim')
        shape = tuple(attr.get('shape'))
        if type == 'rgb':
            rgb_keys.append(key)
            c,h,w = shape
            out_resolutions[key] = (w,h)
        elif type == 'low_dim':
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if 'pose' in key:
                assert tuple(shape) in [(2,),(6,)]
    
    action_shape = tuple(shape_meta['action']['shape'])
    assert action_shape in [(2,),(6,)]

    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ['action'],
            image_keys=rgb_keys
        )

    # transform lowdim dimensions
    if action_shape == (2,):
        # 2D action space, only controls X and Y
        zarr_arr = replay_buffer['action']
        zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])
    
    for key, shape in lowdim_shapes.items():
        if 'pose' in key and shape == (2,):
            # only take X and Y
            zarr_arr = replay_buffer[key]
            zarr_resize_index_last_dim(zarr_arr, idxs=[0,1])

    return replay_buffer


def test():
    import hydra
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize('../diffusion_policy/config'):
        cfg = hydra.compose('train_robomimic_real_image_workspace')
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt
    normalizer = dataset.get_normalizer()
    nactions = normalizer['action'].normalize(dataset.replay_buffer['action'][:])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    _ = plt.hist(dists, bins=100); plt.title('real action velocity')

```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/kitchen_mjl_lowdim_dataset.py
```python
from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from tqdm import tqdm
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.env.kitchen.kitchen_util import parse_mjl_logs

class KitchenMjlLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            abs_action=True,
            robot_noise_ratio=0.0,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        if not abs_action:
            raise NotImplementedError()

        robot_pos_noise_amp = np.array([0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   , 0.1   ,
            0.1   , 0.005 , 0.005 , 0.0005, 0.0005, 0.0005, 0.0005, 0.0005,
            0.0005, 0.005 , 0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ,
            0.005 , 0.005 , 0.1   , 0.1   , 0.1   , 0.005 ], dtype=np.float32)
        rng = np.random.default_rng(seed=seed)

        data_directory = pathlib.Path(dataset_dir)
        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i, mjl_path in enumerate(tqdm(list(data_directory.glob('*/*.mjl')))):
            try:
                data = parse_mjl_logs(str(mjl_path.absolute()), skipamount=40)
                qpos = data['qpos'].astype(np.float32)
                obs = np.concatenate([
                    qpos[:,:9],
                    qpos[:,-21:],
                    np.zeros((len(qpos),30),dtype=np.float32)
                ], axis=-1)
                if robot_noise_ratio > 0:
                    # add observation noise to match real robot
                    noise = robot_noise_ratio * robot_pos_noise_amp * rng.uniform(
                        low=-1., high=1., size=(obs.shape[0], 30))
                    obs[:,:30] += noise
                episode = {
                    'obs': obs,
                    'action': data['ctrl'].astype(np.float32)
                }
                self.replay_buffer.add_episode(episode)
            except Exception as e:
                print(i, e)

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/blockpush_lowdim_dataset.py
```python
from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class BlockPushLowdimDataset(BaseLowdimDataset):
    def __init__(self, 
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            obs_key='obs',
            action_key='action',
            obs_eef_target=True,
            use_manual_normalizer=False,
            seed=42,
            val_ratio=0.0
            ):
        super().__init__()
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=[obs_key, action_key])

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)
        self.obs_key = obs_key
        self.action_key = action_key
        self.obs_eef_target = obs_eef_target
        self.use_manual_normalizer = use_manual_normalizer
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = self._sample_to_data(self.replay_buffer)

        normalizer = LinearNormalizer()
        if not self.use_manual_normalizer:
            normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        else:
            x = data['obs']
            stat = {
                'max': np.max(x, axis=0),
                'min': np.min(x, axis=0),
                'mean': np.mean(x, axis=0),
                'std': np.std(x, axis=0)
            }

            is_x = np.zeros(stat['max'].shape, dtype=bool)
            is_y = np.zeros_like(is_x)
            is_x[[0,3,6,8,10,13]] = True
            is_y[[1,4,7,9,11,14]] = True
            is_rot = ~(is_x|is_y)

            def normalizer_with_masks(stat, masks):
                global_scale = np.ones_like(stat['max'])
                global_offset = np.zeros_like(stat['max'])
                for mask in masks:
                    output_max = 1
                    output_min = -1
                    input_max = stat['max'][mask].max()
                    input_min = stat['min'][mask].min()
                    input_range = input_max - input_min
                    scale = (output_max - output_min) / input_range
                    offset = output_min - scale * input_min
                    global_scale[mask] = scale
                    global_offset[mask] = offset
                return SingleFieldLinearNormalizer.create_manual(
                    scale=global_scale,
                    offset=global_offset,
                    input_stats_dict=stat
                )

            normalizer['obs'] = normalizer_with_masks(stat, [is_x, is_y, is_rot])
            normalizer['action'] = SingleFieldLinearNormalizer.create_fit(
                data['action'], last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        obs = sample[self.obs_key] # T, D_o
        if not self.obs_eef_target:
            obs[:,8:10] = 0
        data = {
            'obs': obs,
            'action': sample[self.action_key], # T, D_a
        }
        return data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

```

## reference_material/diffusion_policy_code/diffusion_policy/dataset/kitchen_lowdim_dataset.py
```python
from typing import Dict
import torch
import numpy as np
import copy
import pathlib
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, get_val_mask
from diffusion_policy.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseLowdimDataset

class KitchenLowdimDataset(BaseLowdimDataset):
    def __init__(self,
            dataset_dir,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0
        ):
        super().__init__()

        data_directory = pathlib.Path(dataset_dir)
        observations = np.load(data_directory / "observations_seq.npy")
        actions = np.load(data_directory / "actions_seq.npy")
        masks = np.load(data_directory / "existence_mask.npy")

        self.replay_buffer = ReplayBuffer.create_empty_numpy()
        for i in range(len(masks)):
            eps_len = int(masks[i].sum())
            obs = observations[i,:eps_len].astype(np.float32)
            action = actions[i,:eps_len].astype(np.float32)
            data = {                              
                'obs': obs,
                'action': action
            }
            self.replay_buffer.add_episode(data)
        
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask)

        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'obs': self.replay_buffer['obs'],
            'action': self.replay_buffer['action']
        }
        if 'range_eps' not in kwargs:
            # to prevent blowing up dims that barely change
            kwargs['range_eps'] = 5e-2
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer['action'])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = sample

        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data

```

## reference_material/diffusion_policy_code/diffusion_policy/gym_util/sync_vector_env.py
```python
import numpy as np
from copy import deepcopy

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.vector.utils import concatenate, create_empty_array

__all__ = ["SyncVectorEnv"]


class SyncVectorEnv(VectorEnv):
    """Vectorized environment that serially runs multiple environments.
    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.
    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.
    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.
    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    """

    def __init__(self, env_fns, observation_space=None, action_space=None, copy=True):
        self.env_fns = env_fns
        self.envs = [env_fn() for env_fn in env_fns]
        self.copy = copy
        self.metadata = self.envs[0].metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or self.envs[0].observation_space
            action_space = action_space or self.envs[0].action_space
        super(SyncVectorEnv, self).__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        self._check_observation_spaces()
        self.observations = create_empty_array(
            self.single_observation_space, n=self.num_envs, fn=np.zeros
        )
        self._rewards = np.zeros((self.num_envs,), dtype=np.float64)
        self._dones = np.zeros((self.num_envs,), dtype=np.bool_)
        # self._rewards = [0] * self.num_envs
        # self._dones = [False] * self.num_envs
        self._actions = None

    def seed(self, seeds=None):
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def reset_wait(self):
        self._dones[:] = False
        observations = []
        for env in self.envs:
            observation = env.reset()
            observations.append(observation)
        self.observations = concatenate(
            observations, self.observations, self.single_observation_space
        )

        return deepcopy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        self._actions = actions

    def step_wait(self):
        observations, infos = [], []
        for i, (env, action) in enumerate(zip(self.envs, self._actions)):
            observation, self._rewards[i], self._dones[i], info = env.step(action)
            # if self._dones[i]:
            #     observation = env.reset()
            observations.append(observation)
            infos.append(info)
        self.observations = concatenate(
            observations, self.observations, self.single_observation_space
        )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.copy(self._rewards),
            np.copy(self._dones),
            infos,
        )

    def close_extras(self, **kwargs):
        [env.close() for env in self.envs]

    def _check_observation_spaces(self):
        for env in self.envs:
            if not (env.observation_space == self.single_observation_space):
                break
        else:
            return True
        raise RuntimeError(
            "Some environments have an observation space "
            "different from `{0}`. In order to batch observations, the "
            "observation spaces from all environments must be "
            "equal.".format(self.single_observation_space)
        )
    
    def call(self, name, *args, **kwargs) -> tuple:
        """Calls the method with name and applies args and kwargs.

        Args:
            name: The method name
            *args: The method args
            **kwargs: The method kwargs

        Returns:
            Tuple of results
        """
        results = []
        for env in self.envs:
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args, **kwargs))
            else:
                results.append(function)

        return tuple(results)

    def call_each(self, name: str, 
            args_list: list=None, 
            kwargs_list: list=None):
        n_envs = len(self.envs)
        if args_list is None:
            args_list = [[]] * n_envs
        assert len(args_list) == n_envs

        if kwargs_list is None:
            kwargs_list = [dict()] * n_envs
        assert len(kwargs_list) == n_envs

        results = []
        for i, env in enumerate(self.envs):
            function = getattr(env, name)
            if callable(function):
                results.append(function(*args_list[i], **kwargs_list[i]))
            else:
                results.append(function)

        return tuple(results)


    def render(self, *args, **kwargs):
        return self.call('render', *args, **kwargs)
    
    def set_attr(self, name: str, values):
        """Sets an attribute of the sub-environments.

        Args:
            name: The property name to change
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise, a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
        """
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        for env, value in zip(self.envs, values):
            setattr(env, name, value)
```

## reference_material/diffusion_policy_code/diffusion_policy/gym_util/multistep_wrapper.py
```python
import gym
from gym import spaces
import numpy as np
from collections import defaultdict, deque
import dill

def stack_repeated(x, n):
    return np.repeat(np.expand_dims(x,axis=0),n,axis=0)

def repeated_box(box_space, n):
    return spaces.Box(
        low=stack_repeated(box_space.low, n),
        high=stack_repeated(box_space.high, n),
        shape=(n,) + box_space.shape,
        dtype=box_space.dtype
    )

def repeated_space(space, n):
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    elif isinstance(space, spaces.Dict):
        result_space = spaces.Dict()
        for key, value in space.items():
            result_space[key] = repeated_space(value, n)
        return result_space
    else:
        raise RuntimeError(f'Unsupported space type {type(space)}')

def take_last_n(x, n):
    x = list(x)
    n = min(len(x), n)
    return np.array(x[-n:])

def dict_take_last_n(x, n):
    result = dict()
    for key, value in x.items():
        result[key] = take_last_n(value, n)
    return result

def aggregate(data, method='max'):
    if method == 'max':
        # equivalent to any
        return np.max(data)
    elif method == 'min':
        # equivalent to all
        return np.min(data)
    elif method == 'mean':
        return np.mean(data)
    elif method == 'sum':
        return np.sum(data)
    else:
        raise NotImplementedError()

def stack_last_n_obs(all_obs, n_steps):
    assert(len(all_obs) > 0)
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, 
        dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result


class MultiStepWrapper(gym.Wrapper):
    def __init__(self, 
            env, 
            n_obs_steps, 
            n_action_steps, 
            max_episode_steps=None,
            reward_agg_method='max'
        ):
        super().__init__(env)
        self._action_space = repeated_space(env.action_space, n_action_steps)
        self._observation_space = repeated_space(env.observation_space, n_obs_steps)
        self.max_episode_steps = max_episode_steps
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.reward_agg_method = reward_agg_method
        self.n_obs_steps = n_obs_steps

        self.obs = deque(maxlen=n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=n_obs_steps+1))
    
    def reset(self):
        """Resets the environment using kwargs."""
        obs = super().reset()

        self.obs = deque([obs], maxlen=self.n_obs_steps+1)
        self.reward = list()
        self.done = list()
        self.info = defaultdict(lambda : deque(maxlen=self.n_obs_steps+1))

        obs = self._get_obs(self.n_obs_steps)
        return obs

    def step(self, action):
        """
        actions: (n_action_steps,) + action_shape
        """
        for act in action:
            if len(self.done) > 0 and self.done[-1]:
                # termination
                break
            observation, reward, done, info = super().step(act)

            self.obs.append(observation)
            self.reward.append(reward)
            if (self.max_episode_steps is not None) \
                and (len(self.reward) >= self.max_episode_steps):
                # truncation
                done = True
            self.done.append(done)
            self._add_info(info)

        observation = self._get_obs(self.n_obs_steps)
        reward = aggregate(self.reward, self.reward_agg_method)
        done = aggregate(self.done, 'max')
        info = dict_take_last_n(self.info, self.n_obs_steps)
        return observation, reward, done, info

    def _get_obs(self, n_steps=1):
        """
        Output (n_steps,) + obs_shape
        """
        assert(len(self.obs) > 0)
        if isinstance(self.observation_space, spaces.Box):
            return stack_last_n_obs(self.obs, n_steps)
        elif isinstance(self.observation_space, spaces.Dict):
            result = dict()
            for key in self.observation_space.keys():
                result[key] = stack_last_n_obs(
                    [obs[key] for obs in self.obs],
                    n_steps
                )
            return result
        else:
            raise RuntimeError('Unsupported space type')

    def _add_info(self, info):
        for key, value in info.items():
            self.info[key].append(value)
    
    def get_rewards(self):
        return self.reward
    
    def get_attr(self, name):
        return getattr(self, name)

    def run_dill_function(self, dill_fn):
        fn = dill.loads(dill_fn)
        return fn(self)
    
    def get_infos(self):
        result = dict()
        for k, v in self.info.items():
            result[k] = list(v)
        return result

```

## reference_material/diffusion_policy_code/diffusion_policy/gym_util/async_vector_env.py
```python
"""
Back ported methods: call, set_attr from v0.26
Disabled auto-reset after done
Added render method.
"""


import numpy as np
import multiprocessing as mp
import time
import sys
from enum import Enum
from copy import deepcopy

from gym import logger
from gym.vector.vector_env import VectorEnv
from gym.error import (
    AlreadyPendingCallError,
    NoAsyncCallError,
    ClosedEnvironmentError,
    CustomSpaceError,
)
from gym.vector.utils import (
    create_shared_memory,
    create_empty_array,
    write_to_shared_memory,
    read_from_shared_memory,
    concatenate,
    CloudpickleWrapper,
    clear_mpi_env_vars,
)

__all__ = ["AsyncVectorEnv"]


class AsyncState(Enum):
    DEFAULT = "default"
    WAITING_RESET = "reset"
    WAITING_STEP = "step"
    WAITING_CALL = "call"


class AsyncVectorEnv(VectorEnv):
    """Vectorized environment that runs multiple environments in parallel. It
    uses `multiprocessing` processes, and pipes for communication.
    Parameters
    ----------
    env_fns : iterable of callable
        Functions that create the environments.
    observation_space : `gym.spaces.Space` instance, optional
        Observation space of a single environment. If `None`, then the
        observation space of the first environment is taken.
    action_space : `gym.spaces.Space` instance, optional
        Action space of a single environment. If `None`, then the action space
        of the first environment is taken.
    shared_memory : bool (default: `True`)
        If `True`, then the observations from the worker processes are
        communicated back through shared variables. This can improve the
        efficiency if the observations are large (e.g. images).
    copy : bool (default: `True`)
        If `True`, then the `reset` and `step` methods return a copy of the
        observations.
    context : str, optional
        Context for multiprocessing. If `None`, then the default context is used.
        Only available in Python 3.
    daemon : bool (default: `True`)
        If `True`, then subprocesses have `daemon` flag turned on; that is, they
        will quit if the head process quits. However, `daemon=True` prevents
        subprocesses to spawn children, so for some environments you may want
        to have it set to `False`
    worker : function, optional
        WARNING - advanced mode option! If set, then use that worker in a subprocess
        instead of a default one. Can be useful to override some inner vector env
        logic, for instance, how resets on done are handled. Provides high
        degree of flexibility and a high chance to shoot yourself in the foot; thus,
        if you are writing your own worker, it is recommended to start from the code
        for `_worker` (or `_worker_shared_memory`) method below, and add changes
    """

    def __init__(
        self,
        env_fns,
        dummy_env_fn=None,
        observation_space=None,
        action_space=None,
        shared_memory=True,
        copy=True,
        context=None,
        daemon=True,
        worker=None,
    ):
        ctx = mp.get_context(context)
        self.env_fns = env_fns
        self.shared_memory = shared_memory
        self.copy = copy

        # Added dummy_env_fn to fix OpenGL error in Mujoco
        # disable any OpenGL rendering in dummy_env_fn, since it
        # will conflict with OpenGL context in the forked child process
        if dummy_env_fn is None:
            dummy_env_fn = env_fns[0]
        dummy_env = dummy_env_fn()
        self.metadata = dummy_env.metadata

        if (observation_space is None) or (action_space is None):
            observation_space = observation_space or dummy_env.observation_space
            action_space = action_space or dummy_env.action_space
        dummy_env.close()
        del dummy_env
        super(AsyncVectorEnv, self).__init__(
            num_envs=len(env_fns),
            observation_space=observation_space,
            action_space=action_space,
        )

        if self.shared_memory:
            try:
                _obs_buffer = create_shared_memory(
                    self.single_observation_space, n=self.num_envs, ctx=ctx
                )
                self.observations = read_from_shared_memory(
                    _obs_buffer, self.single_observation_space, n=self.num_envs
                )
            except CustomSpaceError:
                raise ValueError(
                    "Using `shared_memory=True` in `AsyncVectorEnv` "
                    "is incompatible with non-standard Gym observation spaces "
                    "(i.e. custom spaces inheriting from `gym.Space`), and is "
                    "only compatible with default Gym spaces (e.g. `Box`, "
                    "`Tuple`, `Dict`) for batching. Set `shared_memory=False` "
                    "if you use custom observation spaces."
                )
        else:
            _obs_buffer = None
            self.observations = create_empty_array(
                self.single_observation_space, n=self.num_envs, fn=np.zeros
            )

        self.parent_pipes, self.processes = [], []
        self.error_queue = ctx.Queue()
        target = _worker_shared_memory if self.shared_memory else _worker
        target = worker or target
        with clear_mpi_env_vars():
            for idx, env_fn in enumerate(self.env_fns):
                parent_pipe, child_pipe = ctx.Pipe()
                process = ctx.Process(
                    target=target,
                    name="Worker<{0}>-{1}".format(type(self).__name__, idx),
                    args=(
                        idx,
                        CloudpickleWrapper(env_fn),
                        child_pipe,
                        parent_pipe,
                        _obs_buffer,
                        self.error_queue,
                    ),
                )

                self.parent_pipes.append(parent_pipe)
                self.processes.append(process)

                process.daemon = daemon
                process.start()
                child_pipe.close()

        self._state = AsyncState.DEFAULT
        self._check_observation_spaces()

    def seed(self, seeds=None):
        self._assert_is_running()
        if seeds is None:
            seeds = [None for _ in range(self.num_envs)]
        if isinstance(seeds, int):
            seeds = [seeds + i for i in range(self.num_envs)]
        assert len(seeds) == self.num_envs

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `seed` while waiting "
                "for a pending call to `{0}` to complete.".format(self._state.value),
                self._state.value,
            )

        for pipe, seed in zip(self.parent_pipes, seeds):
            pipe.send(("seed", seed))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def reset_async(self):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `reset_async` while waiting "
                "for a pending call to `{0}` to complete".format(self._state.value),
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("reset", None))
        self._state = AsyncState.WAITING_RESET

    def reset_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `reset_wait` times out. If
            `None`, the call to `reset_wait` never times out.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_RESET:
            raise NoAsyncCallError(
                "Calling `reset_wait` without any prior " "call to `reset_async`.",
                AsyncState.WAITING_RESET.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `reset_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        if not self.shared_memory:
            self.observations = concatenate(
                results, self.observations, self.single_observation_space
            )

        return deepcopy(self.observations) if self.copy else self.observations

    def step_async(self, actions):
        """
        Parameters
        ----------
        actions : iterable of samples from `action_space`
            List of actions.
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `step_async` while waiting "
                "for a pending call to `{0}` to complete.".format(self._state.value),
                self._state.value,
            )

        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", action))
        self._state = AsyncState.WAITING_STEP

    def step_wait(self, timeout=None):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `step_wait` times out. If
            `None`, the call to `step_wait` never times out.
        Returns
        -------
        observations : sample from `observation_space`
            A batch of observations from the vectorized environment.
        rewards : `np.ndarray` instance (dtype `np.float_`)
            A vector of rewards from the vectorized environment.
        dones : `np.ndarray` instance (dtype `np.bool_`)
            A vector whose entries indicate whether the episode has ended.
        infos : list of dict
            A list of auxiliary diagnostic information.
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_STEP:
            raise NoAsyncCallError(
                "Calling `step_wait` without any prior call " "to `step_async`.",
                AsyncState.WAITING_STEP.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                "The call to `step_wait` has timed out after "
                "{0} second{1}.".format(timeout, "s" if timeout > 1 else "")
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT
        observations_list, rewards, dones, infos = zip(*results)

        if not self.shared_memory:
            self.observations = concatenate(
                observations_list, self.observations, self.single_observation_space
            )

        return (
            deepcopy(self.observations) if self.copy else self.observations,
            np.array(rewards),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def close_extras(self, timeout=None, terminate=False):
        """
        Parameters
        ----------
        timeout : int or float, optional
            Number of seconds before the call to `close` times out. If `None`,
            the call to `close` never times out. If the call to `close` times
            out, then all processes are terminated.
        terminate : bool (default: `False`)
            If `True`, then the `close` operation is forced and all processes
            are terminated.
        """
        timeout = 0 if terminate else timeout
        try:
            if self._state != AsyncState.DEFAULT:
                logger.warn(
                    "Calling `close` while waiting for a pending "
                    "call to `{0}` to complete.".format(self._state.value)
                )
                function = getattr(self, "{0}_wait".format(self._state.value))
                function(timeout)
        except mp.TimeoutError:
            terminate = True

        if terminate:
            for process in self.processes:
                if process.is_alive():
                    process.terminate()
        else:
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.send(("close", None))
            for pipe in self.parent_pipes:
                if (pipe is not None) and (not pipe.closed):
                    pipe.recv()

        for pipe in self.parent_pipes:
            if pipe is not None:
                pipe.close()
        for process in self.processes:
            process.join()

    def _poll(self, timeout=None):
        self._assert_is_running()
        if timeout is None:
            return True
        end_time = time.perf_counter() + timeout
        delta = None
        for pipe in self.parent_pipes:
            delta = max(end_time - time.perf_counter(), 0)
            if pipe is None:
                return False
            if pipe.closed or (not pipe.poll(delta)):
                return False
        return True

    def _check_observation_spaces(self):
        self._assert_is_running()
        for pipe in self.parent_pipes:
            pipe.send(("_check_observation_space", self.single_observation_space))
        same_spaces, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        if not all(same_spaces):
            raise RuntimeError(
                "Some environments have an observation space "
                "different from `{0}`. In order to batch observations, the "
                "observation spaces from all environments must be "
                "equal.".format(self.single_observation_space)
            )

    def _assert_is_running(self):
        if self.closed:
            raise ClosedEnvironmentError(
                "Trying to operate on `{0}`, after a "
                "call to `close()`.".format(type(self).__name__)
            )

    def _raise_if_errors(self, successes):
        if all(successes):
            return

        num_errors = self.num_envs - sum(successes)
        assert num_errors > 0
        for _ in range(num_errors):
            index, exctype, value = self.error_queue.get()
            logger.error(
                "Received the following error from Worker-{0}: "
                "{1}: {2}".format(index, exctype.__name__, value)
            )
            logger.error("Shutting down Worker-{0}.".format(index))
            self.parent_pipes[index].close()
            self.parent_pipes[index] = None

        logger.error("Raising the last exception back to the main process.")
        raise exctype(value)
    
    def call_async(self, name: str, *args, **kwargs):
        """Calls the method with name asynchronously and apply args and kwargs to the method.

        Args:
            name: Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Raises:
            ClosedEnvironmentError: If the environment was closed (if :meth:`close` was previously called).
            AlreadyPendingCallError: Calling `call_async` while waiting for a pending call to complete
        """
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe in self.parent_pipes:
            pipe.send(("_call", (name, args, kwargs)))
        self._state = AsyncState.WAITING_CALL

    def call_wait(self, timeout = None) -> list:
        """Calls all parent pipes and waits for the results.

        Args:
            timeout: Number of seconds before the call to `step_wait` times out.
                If `None` (default), the call to `step_wait` never times out.

        Returns:
            List of the results of the individual calls to the method or property for each environment.

        Raises:
            NoAsyncCallError: Calling `call_wait` without any prior call to `call_async`.
            TimeoutError: The call to `call_wait` has timed out after timeout second(s).
        """
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results

    def call(self, name: str, *args, **kwargs):
        """Call a method, or get a property, from each parallel environment.

        Args:
            name (str): Name of the method or property to call.
            *args: Arguments to apply to the method call.
            **kwargs: Keyword arguments to apply to the method call.

        Returns:
            List of the results of the individual calls to the method or property for each environment.
        """
        self.call_async(name, *args, **kwargs)
        return self.call_wait()
    

    def call_each(self, name: str, 
            args_list: list=None, 
            kwargs_list: list=None, 
            timeout = None):
        n_envs = len(self.parent_pipes)
        if args_list is None:
            args_list = [[]] * n_envs
        assert len(args_list) == n_envs

        if kwargs_list is None:
            kwargs_list = [dict()] * n_envs
        assert len(kwargs_list) == n_envs

        # send
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `call_async` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for i, pipe in enumerate(self.parent_pipes):
            pipe.send(("_call", (name, args_list[i], kwargs_list[i])))
        self._state = AsyncState.WAITING_CALL

        # receive
        self._assert_is_running()
        if self._state != AsyncState.WAITING_CALL:
            raise NoAsyncCallError(
                "Calling `call_wait` without any prior call to `call_async`.",
                AsyncState.WAITING_CALL.value,
            )

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError(
                f"The call to `call_wait` has timed out after {timeout} second(s)."
            )

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        return results


    def set_attr(self, name: str, values):
        """Sets an attribute of the sub-environments.

        Args:
            name: Name of the property to be set in each individual environment.
            values: Values of the property to be set to. If ``values`` is a list or
                tuple, then it corresponds to the values for each individual
                environment, otherwise a single value is set for all environments.

        Raises:
            ValueError: Values must be a list or tuple with length equal to the number of environments.
            AlreadyPendingCallError: Calling `set_attr` while waiting for a pending call to complete.
        """
        self._assert_is_running()
        if not isinstance(values, (list, tuple)):
            values = [values for _ in range(self.num_envs)]
        if len(values) != self.num_envs:
            raise ValueError(
                "Values must be a list or tuple with length equal to the "
                f"number of environments. Got `{len(values)}` values for "
                f"{self.num_envs} environments."
            )

        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError(
                "Calling `set_attr` while waiting "
                f"for a pending call to `{self._state.value}` to complete.",
                self._state.value,
            )

        for pipe, value in zip(self.parent_pipes, values):
            pipe.send(("_setattr", (name, value)))
        _, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)

    def render(self, *args, **kwargs):
        return self.call('render', *args, **kwargs)



def _worker(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is None
    env = env_fn()
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                pipe.send((observation, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     observation = env.reset()
                pipe.send(((observation, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))

            elif command == "_check_observation_space":
                pipe.send((data == env.observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == "reset":
                observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send((None, True))
            elif command == "step":
                observation, reward, done, info = env.step(data)
                # if done:
                #     observation = env.reset()
                write_to_shared_memory(
                    index, observation, shared_memory, observation_space
                )
                pipe.send(((None, reward, done, info), True))
            elif command == "seed":
                env.seed(data)
                pipe.send((None, True))
            elif command == "close":
                pipe.send((None, True))
                break
            elif command == "_call":
                name, args, kwargs = data
                if name in ["reset", "step", "seed", "close"]:
                    raise ValueError(
                        f"Trying to call function `{name}` with "
                        f"`_call`. Use `{name}` directly instead."
                    )
                function = getattr(env, name)
                if callable(function):
                    pipe.send((function(*args, **kwargs), True))
                else:
                    pipe.send((function, True))
            elif command == "_setattr":
                name, value = data
                setattr(env, name, value)
                pipe.send((None, True))
            elif command == "_check_observation_space":
                pipe.send((data == observation_space, True))
            else:
                raise RuntimeError(
                    "Received unknown command `{0}`. Must "
                    "be one of {`reset`, `step`, `seed`, `close`, "
                    "`_check_observation_space`}.".format(command)
                )
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()
```

## reference_material/diffusion_policy_code/diffusion_policy/policy/base_image_policy.py
```python
from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseImagePolicy(ModuleAttrMixin):
    # init accepts keyword argument shape_meta, see config/task/*_image.yaml

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py
```python
from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionUnetHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            crop_shape=(76, 76),
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        print("Diffusion params: %e" % sum(p.numel() for p in self.model.parameters()))
        print("Vision params: %e" % sum(p.numel() for p in self.obs_encoder.parameters()))
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/diffusion_unet_image_policy.py
```python
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply

class DiffusionUnetImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            obs_encoder: MultiImageObsEncoder,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shapes
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        # create diffusion model
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim * n_obs_steps

        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, Do
            global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/diffusion_transformer_hybrid_image_policy.py
```python
from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class DiffusionTransformerHybridImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            # task params
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            # image
            crop_shape=(76, 76),
            obs_encoder_group_norm=False,
            eval_fixed_crop=False,
            # arch
            n_layer=8,
            n_cond_layers=0,
            n_head=4,
            n_emb=256,
            p_drop_emb=0.0,
            p_drop_attn=0.3,
            causal_attn=True,
            time_as_cond=True,
            obs_as_cond=True,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        # create diffusion model
        obs_feature_dim = obs_encoder.output_shape()[0]
        input_dim = action_dim if obs_as_cond else (obs_feature_dim + action_dim)
        output_dim = input_dim
        cond_dim = obs_feature_dim if obs_as_cond else 0

        model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=output_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=cond_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            p_drop_emb=p_drop_emb,
            p_drop_attn=p_drop_attn,
            causal_attn=causal_attn,
            time_as_cond=time_as_cond,
            obs_as_cond=obs_as_cond,
            n_cond_layers=n_cond_layers
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            cond = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, To, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            transformer_weight_decay: float, 
            obs_encoder_weight_decay: float,
            learning_rate: float, 
            betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=transformer_weight_decay)
        optim_groups.append({
            "params": self.obs_encoder.parameters(),
            "weight_decay": obs_encoder_weight_decay
        })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]
        To = self.n_obs_steps

        # handle different ways of passing observation
        cond = None
        trajectory = nactions
        if self.obs_as_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            cond = nobs_features.reshape(batch_size, To, -1)
            if self.pred_action_steps_only:
                start = To - 1
                end = start + self.n_action_steps
                trajectory = nactions[:,start:end]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            trajectory = torch.cat([nactions, nobs_features], dim=-1).detach()

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/ibc_dfo_lowdim_policy.py
```python
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class IbcDfoLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            dropout=0.1,
            train_n_neg=128,
            pred_n_iter=5,
            pred_n_samples=16384,
            kevin_inference=False,
            andy_train=False
        ):
        super().__init__()

        in_action_channels = action_dim * n_action_steps
        in_obs_channels = obs_dim * n_obs_steps
        in_channels = in_action_channels + in_obs_channels
        mid_channels = 1024
        out_channels = 1

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        self.normalizer = LinearNormalizer()

        self.train_n_neg = train_n_neg
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.kevin_inference = kevin_inference
        self.andy_train = andy_train
    
    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape
        s = obs.reshape(B,1,-1).expand(-1,N,-1)
        x = torch.cat([s, action.reshape(B,N,-1)], dim=-1).reshape(B*N,-1)
        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        x = x.reshape(B,N)
        return x

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim
        Ta = self.n_action_steps

        # only take necessary obs
        this_obs = nobs[:,:To]
        naction_stats = self.get_naction_stats()

        # first sample
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=this_obs.dtype)
        # (B, N, Ta, Da)

        if self.kevin_inference:
            # kevin's implementation
            noise_scale = 3e-2
            for i in range(self.pred_n_iter):
                # Compute energies.
                logits = self.forward(this_obs, samples)
                probs = F.softmax(logits, dim=-1)

                # Resample with replacement.
                idxs = torch.multinomial(probs, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                samples = samples + torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=naction_stats['min'], max=naction_stats['max'])

            # Return target with highest probability.
            logits = self.forward(this_obs, samples)
            probs = F.softmax(logits, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            acts_n = samples[torch.arange(samples.size(0)), best_idxs, :]
        else:
            # andy's implementation
            zero = torch.tensor(0, device=self.device)
            resample_std = torch.tensor(3e-2, device=self.device)
            for i in range(self.pred_n_iter):
                # Forward pass.
                logits = self.forward(this_obs, samples) # (B, N)
                prob = torch.softmax(logits, dim=-1)

                if i < (self.pred_n_iter - 1):
                    idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                    samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                    samples += torch.normal(zero, resample_std, size=samples.shape, device=self.device)

            # Return one sample per x in batch.
            idxs = torch.multinomial(prob, num_samples=1, replacement=True)
            acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)

        action = self.normalizer['action'].unnormalize(acts_n)
        result = {
            'action': action
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        nobs = nbatch['obs']
        naction = nbatch['action']

        # shapes
        Do = self.obs_dim
        Da = self.action_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps
        T = self.horizon
        B = naction.shape[0]

        this_obs = nobs[:,:To]
        start = To - 1
        end = start + Ta
        this_action = naction[:,start:end]

        # Small additive noise to true positives.
        this_action += torch.normal(mean=0, std=1e-4,
            size=this_action.shape,
            dtype=this_action.dtype,
            device=this_action.device)

        # Sample negatives: (B, train_n_neg, Ta, Da)
        naction_stats = self.get_naction_stats()
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.train_n_neg, Ta)).to(
            dtype=this_action.dtype)
        action_samples = torch.cat([
            this_action.unsqueeze(1), samples], dim=1)
        # (B, train_n_neg+1, Ta, Da)

        if self.andy_train:
            # Get onehot labels
            labels = torch.zeros(action_samples.shape[:2], 
                dtype=this_action.dtype, device=this_action.device)
            labels[:,0] = 1
            logits = self.forward(this_obs, action_samples)
            # (B, N)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(logits * labels, axis=-1))
        else:
            labels = torch.zeros((B,),dtype=torch.int64, device=this_action.device)
            # training
            logits = self.forward(this_obs, action_samples)
            loss = F.cross_entropy(logits, labels)
        return loss


    def get_naction_stats(self):
        Da = self.action_dim
        naction_stats = self.normalizer['action'].get_output_stats()
        repeated_stats = dict()
        for key, value in naction_stats.items():
            assert len(value.shape) == 1
            n_repeats = Da // value.shape[0]
            assert value.shape[0] * n_repeats == Da
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/base_lowdim_policy.py
```python
from typing import Dict
import torch
import torch.nn as nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.model.common.normalizer import LinearNormalizer

class BaseLowdimPolicy(ModuleAttrMixin):  
    # ========= inference  ============
    # also as self.device and self.dtype for inference device transfer
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            obs: B,To,Do
        return: 
            action: B,Ta,Da
        To = 3
        Ta = 4
        T = 6
        |o|o|o|
        | | |a|a|a|a|
        |o|o|
        | |a|a|a|a|a|
        | | | | |a|a|
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError()

    
```

## reference_material/diffusion_policy_code/diffusion_policy/policy/robomimic_image_policy.py
```python
from typing import Dict
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

class RobomimicImagePolicy(BaseImagePolicy):
    def __init__(self, 
            shape_meta: dict,
            algo_name='bc_rnn',
            obs_type='image',
            task_name='square',
            dataset_type='ph',
            crop_shape=(76,76)
        ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type)

        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.config = config

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        nobs_dict = self.normalizer(obs_dict)
        robomimic_obs_dict = dict_apply(nobs_dict, lambda x: x[:,0,...])
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,:] # (B, 1, Da)
        }
        return result

    def reset(self):
        self.model.reset()

    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def train_on_batch(self, batch, epoch, validate=False):
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        robomimic_batch = {
            'obs': nobs,
            'actions': nactions
        }
        input_batch = self.model.process_batch_for_training(
            robomimic_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate)
        # keys: losses, predictions
        return info
    
    def on_epoch_end(self, epoch):
        self.model.on_epoch_end(epoch)

    def get_optimizer(self):
        return self.model.optimizers['policy']


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg.shape_meta

    policy = RobomimicImagePolicy(shape_meta=shape_meta)


```

## reference_material/diffusion_policy_code/diffusion_policy/policy/ibc_dfo_hybrid_image_policy.py
```python
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.robomimic_config_util import get_robomimic_config
from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
import robomimic.models.base_nets as rmbn
import diffusion_policy.model.vision.crop_randomizer as dmvc
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class IbcDfoHybridImagePolicy(BaseImagePolicy):
    def __init__(self,
            shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            dropout=0.1,
            train_n_neg=128,
            pred_n_iter=5,
            pred_n_samples=16384,
            kevin_inference=False,
            andy_train=False,
            obs_encoder_group_norm=True,
            eval_fixed_crop=True,
            crop_shape=(76, 76),
        ):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta['obs']
        obs_config = {
            'low_dim': [],
            'rgb': [],
            'depth': [],
            'scan': []
        }
        obs_key_shapes = dict()
        for key, attr in obs_shape_meta.items():
            shape = attr['shape']
            obs_key_shapes[key] = list(shape)

            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                obs_config['rgb'].append(key)
            elif type == 'low_dim':
                obs_config['low_dim'].append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")

        # get raw robomimic config
        config = get_robomimic_config(
            algo_name='bc_rnn',
            hdf5_type='image',
            task_name='square',
            dataset_type='ph')
        
        with config.unlocked():
            # set config with shape_meta
            config.observation.modalities.obs = obs_config

            if crop_shape is None:
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality['obs_randomizer_class'] = None
            else:
                # set random crop parameter
                ch, cw = crop_shape
                for key, modality in config.observation.encoder.items():
                    if modality.obs_randomizer_class == 'CropRandomizer':
                        modality.obs_randomizer_kwargs.crop_height = ch
                        modality.obs_randomizer_kwargs.crop_width = cw

        # init global state
        ObsUtils.initialize_obs_utils_with_config(config)

        # load model
        policy: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes=obs_key_shapes,
                ac_dim=action_dim,
                device='cpu',
            )

        self.obs_encoder = obs_encoder = policy.nets['policy'].nets['encoder'].nets['obs']
        
        if obs_encoder_group_norm:
            # replace batch norm with group norm
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features//16, 
                    num_channels=x.num_features)
            )
            # obs_encoder.obs_nets['agentview_image'].nets[0].nets
        
        # obs_encoder.obs_randomizers['agentview_image']
        if eval_fixed_crop:
            replace_submodules(
                root_module=obs_encoder,
                predicate=lambda x: isinstance(x, rmbn.CropRandomizer),
                func=lambda x: dmvc.CropRandomizer(
                    input_shape=x.input_shape,
                    crop_height=x.crop_height,
                    crop_width=x.crop_width,
                    num_crops=x.num_crops,
                    pos_enc=x.pos_enc
                )
            )

        obs_feature_dim = obs_encoder.output_shape()[0]
        in_action_channels = action_dim * n_action_steps
        in_obs_channels = obs_feature_dim * n_obs_steps
        in_channels = in_action_channels + in_obs_channels
        mid_channels = 1024
        out_channels = 1

        self.dense0 = nn.Linear(in_features=in_channels, out_features=mid_channels)
        self.drop0 = nn.Dropout(dropout)
        self.dense1 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop1 = nn.Dropout(dropout)
        self.dense2 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop2 = nn.Dropout(dropout)
        self.dense3 = nn.Linear(in_features=mid_channels, out_features=mid_channels)
        self.drop3 = nn.Dropout(dropout)
        self.dense4 = nn.Linear(in_features=mid_channels, out_features=out_channels)

        self.normalizer = LinearNormalizer()

        self.train_n_neg = train_n_neg
        self.pred_n_iter = pred_n_iter
        self.pred_n_samples = pred_n_samples
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.horizon = horizon
        self.kevin_inference = kevin_inference
        self.andy_train = andy_train
    
    def forward(self, obs, action):
        B, N, Ta, Da = action.shape
        B, To, Do = obs.shape
        s = obs.reshape(B,1,-1).expand(-1,N,-1)
        x = torch.cat([s, action.reshape(B,N,-1)], dim=-1).reshape(B*N,-1)
        x = self.drop0(torch.relu(self.dense0(x)))
        x = self.drop1(torch.relu(self.dense1(x)))
        x = self.drop2(torch.relu(self.dense2(x)))
        x = self.drop3(torch.relu(self.dense3(x)))
        x = self.dense4(x)
        x = x.reshape(B,N)
        return x

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'past_action' not in obs_dict # not implemented yet
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Ta = self.n_action_steps
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # encode obs
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        nobs_features = nobs_features.reshape(B,To,-1)

        # only take necessary obs
        naction_stats = self.get_naction_stats()

        # first sample
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.pred_n_samples, Ta)).to(
            dtype=dtype)
        # (B, N, Ta, Da)

        if self.kevin_inference:
            # kevin's implementation
            noise_scale = 3e-2
            for i in range(self.pred_n_iter):
                # Compute energies.
                logits = self.forward(nobs_features, samples)
                probs = F.softmax(logits, dim=-1)

                # Resample with replacement.
                idxs = torch.multinomial(probs, self.pred_n_samples, replacement=True)
                samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]

                # Add noise and clip to target bounds.
                samples = samples + torch.randn_like(samples) * noise_scale
                samples = samples.clamp(min=naction_stats['min'], max=naction_stats['max'])

            # Return target with highest probability.
            logits = self.forward(nobs_features, samples)
            probs = F.softmax(logits, dim=-1)
            best_idxs = probs.argmax(dim=-1)
            acts_n = samples[torch.arange(samples.size(0)), best_idxs, :]
        else:
            # andy's implementation
            zero = torch.tensor(0, device=self.device)
            resample_std = torch.tensor(3e-2, device=self.device)
            for i in range(self.pred_n_iter):
                # Forward pass.
                logits = self.forward(nobs_features, samples) # (B, N)
                prob = torch.softmax(logits, dim=-1)

                if i < (self.pred_n_iter - 1):
                    idxs = torch.multinomial(prob, self.pred_n_samples, replacement=True)
                    samples = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs]
                    samples += torch.normal(zero, resample_std, size=samples.shape, device=self.device)

            # Return one sample per x in batch.
            idxs = torch.multinomial(prob, num_samples=1, replacement=True)
            acts_n = samples[torch.arange(samples.size(0)).unsqueeze(-1), idxs].squeeze(1)

        action = self.normalizer['action'].unnormalize(acts_n)
        result = {
            'action': action
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        naction = self.normalizer['action'].normalize(batch['action'])

        # shapes
        Do = self.obs_feature_dim
        Da = self.action_dim
        To = self.n_obs_steps
        Ta = self.n_action_steps
        T = self.horizon
        B = naction.shape[0]

        # encode obs
        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs, 
            lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        nobs_features = self.obs_encoder(this_nobs)
        # reshape back to B, To, Do
        nobs_features = nobs_features.reshape(B,To,-1)

        start = To - 1
        end = start + Ta
        this_action = naction[:,start:end]

        # Small additive noise to true positives.
        this_action += torch.normal(mean=0, std=1e-4,
            size=this_action.shape,
            dtype=this_action.dtype,
            device=this_action.device)

        # Sample negatives: (B, train_n_neg, Ta, Da)
        naction_stats = self.get_naction_stats()
        action_dist = torch.distributions.Uniform(
            low=naction_stats['min'],
            high=naction_stats['max']
        )
        samples = action_dist.sample((B, self.train_n_neg, Ta)).to(
            dtype=this_action.dtype)
        action_samples = torch.cat([
            this_action.unsqueeze(1), samples], dim=1)
        # (B, train_n_neg+1, Ta, Da)

        if self.andy_train:
            # Get onehot labels
            labels = torch.zeros(action_samples.shape[:2], 
                dtype=this_action.dtype, device=this_action.device)
            labels[:,0] = 1
            logits = self.forward(nobs_features, action_samples)
            # (B, N)
            logits = torch.log_softmax(logits, dim=-1)
            loss = -torch.mean(torch.sum(logits * labels, axis=-1))
        else:
            labels = torch.zeros((B,),dtype=torch.int64, device=this_action.device)
            # training
            logits = self.forward(nobs_features, action_samples)
            loss = F.cross_entropy(logits, labels)
        return loss

    def get_naction_stats(self):
        Da = self.action_dim
        naction_stats = self.normalizer['action'].get_output_stats()
        repeated_stats = dict()
        for key, value in naction_stats.items():
            assert len(value.shape) == 1
            n_repeats = Da // value.shape[0]
            assert value.shape[0] * n_repeats == Da
            repeated_stats[key] = value.repeat(n_repeats)
        return repeated_stats

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/diffusion_unet_lowdim_policy.py
```python
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/bet_lowdim_policy.py
```python
from typing import Dict, Tuple
import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.bet.action_ae.discretizers.k_means import KMeansDiscretizer
from diffusion_policy.model.bet.latent_generators.mingpt import MinGPT
from diffusion_policy.model.bet.utils import eval_mode

class BETLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            action_ae: KMeansDiscretizer, 
            obs_encoding_net: nn.Module, 
            state_prior: MinGPT,
            horizon,
            n_action_steps,
            n_obs_steps):
        super().__init__()
    
        self.normalizer = LinearNormalizer()
        self.action_ae = action_ae
        self.obs_encoding_net = obs_encoding_net
        self.state_prior = state_prior
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        T = self.horizon

        # pad To to T
        obs = torch.full((B,T,Do), -2, dtype=nobs.dtype, device=nobs.device)
        obs[:,:To,:] = nobs[:,:To,:]

        # (B,T,Do)
        enc_obs = self.obs_encoding_net(obs)

        # Sample latents from the prior
        latents, offsets = self.state_prior.generate_latents(enc_obs)

        # un-descritize
        naction_pred = self.action_ae.decode_actions(
            latent_action_batch=(latents, offsets)
        )
        # (B,T,Da)

        # un-normalize
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        result = {
            'action': action,
            'action_pred': action_pred
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def fit_action_ae(self, input_actions: torch.Tensor):
        self.action_ae.fit_discretizer(input_actions=input_actions)
    
    def get_latents(self, latent_collection_loader):
        training_latents = list()
        with eval_mode(self.action_ae, self.obs_encoding_net, no_grad=True):
            for observations, action, mask in latent_collection_loader:
                obs, act = observations.to(self.device, non_blocking=True), action.to(self.device, non_blocking=True)
                enc_obs = self.obs_encoding_net(obs)
                latent = self.action_ae.encode_into_latent(act, enc_obs)
                reconstructed_action = self.action_ae.decode_actions(
                    latent,
                    enc_obs,
                )
                total_mse_loss += F.mse_loss(act, reconstructed_action, reduction="sum")
                if type(latent) == tuple:
                    # serialize into tensor; assumes last dim is latent dim
                    detached_latents = tuple(x.detach() for x in latent)
                    training_latents.append(torch.cat(detached_latents, dim=-1))
                else:
                    training_latents.append(latent.detach())
        training_latents_tensor = torch.cat(training_latents, dim=0)
        return training_latents_tensor

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.state_prior.get_optimizer(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))
    
    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # mask out observations after n_obs_steps
        To = self.n_obs_steps
        obs[:,To:,:] = -2 # (normal obs range [-1,1])

        enc_obs = self.obs_encoding_net(obs)
        latent = self.action_ae.encode_into_latent(action, enc_obs)
        _, loss, loss_components = self.state_prior.get_latent_and_loss(
            obs_rep=enc_obs,
            target_latents=latent,
            return_loss_components=True,
        )
        return loss, loss_components

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py
```python
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator

class DiffusionTransformerLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: TransformerForDiffusion,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_cond=False,
            pred_action_steps_only=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        if pred_action_steps_only:
            assert obs_as_cond

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_cond = obs_as_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            cond=None, generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        cond = None
        cond_data = None
        cond_mask = None
        if self.obs_as_cond:
            cond = nobs[:,:To]
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            cond=cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not self.obs_as_cond:
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
        ) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(
                weight_decay=weight_decay, 
                learning_rate=learning_rate, 
                betas=tuple(betas))

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        cond = None
        trajectory = action
        if self.obs_as_cond:
            cond = obs[:,:self.n_obs_steps,:]
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)
        
        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

```

## reference_material/diffusion_policy_code/diffusion_policy/policy/robomimic_lowdim_policy.py
```python
from typing import Dict
import torch
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

from robomimic.algo import algo_factory
from robomimic.algo.algo import PolicyAlgo
import robomimic.utils.obs_utils as ObsUtils
from diffusion_policy.common.robomimic_config_util import get_robomimic_config

class RobomimicLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            action_dim, 
            obs_dim,
            algo_name='bc_rnn',
            obs_type='low_dim',
            task_name='square',
            dataset_type='ph',
        ):
        super().__init__()
        # key for robomimic obs input
        # previously this is 'object', 'robot0_eef_pos' etc
        obs_key = 'obs'

        config = get_robomimic_config(
            algo_name=algo_name,
            hdf5_type=obs_type,
            task_name=task_name,
            dataset_type=dataset_type)
        with config.unlocked():
            config.observation.modalities.obs.low_dim = [obs_key]
        
        ObsUtils.initialize_obs_utils_with_config(config)
        model: PolicyAlgo = algo_factory(
                algo_name=config.algo_name,
                config=config,
                obs_key_shapes={obs_key: [obs_dim]},
                ac_dim=action_dim,
                device='cpu',
            )
        self.model = model
        self.nets = model.nets
        self.normalizer = LinearNormalizer()
        self.obs_key = obs_key
        self.config = config

    def to(self,*args,**kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.model.device = device
        super().to(*args,**kwargs)
    
    # =========== inference =============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs = self.normalizer['obs'].normalize(obs_dict['obs'])
        assert obs.shape[1] == 1
        robomimic_obs_dict = {self.obs_key: obs[:,0,:]}
        naction = self.model.get_action(robomimic_obs_dict)
        action = self.normalizer['action'].unnormalize(naction)
        # (B, Da)
        result = {
            'action': action[:,None,:] # (B, 1, Da)
        }
        return result
    
    def reset(self):
        self.model.reset()
        
    # =========== training ==============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
    
    def train_on_batch(self, batch, epoch, validate=False):
        nbatch = self.normalizer.normalize(batch)
        robomimic_batch = {
            'obs': {self.obs_key: nbatch['obs']},
            'actions': nbatch['action']
        }
        input_batch = self.model.process_batch_for_training(
            robomimic_batch)
        info = self.model.train_on_batch(
            batch=input_batch, epoch=epoch, validate=validate)
        # keys: losses, predictions
        return info

    def get_optimizer(self):
        return self.model.optimizers['policy']

```

## reference_material/diffusion_policy_code/diffusion_policy/common/env_util.py
```python
import cv2
import numpy as np


def render_env_video(env, states, actions=None):
    observations = states
    imgs = list()
    for i in range(len(observations)):
        state = observations[i]
        env.set_state(state)
        if i == 0:
            env.set_state(state)
        img = env.render()
        # draw action
        if actions is not None:
            action = actions[i]
            coord = (action / 512 * 96).astype(np.int32)
            cv2.drawMarker(img, coord, 
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=8, thickness=1)
        imgs.append(img)
    imgs = np.array(imgs)
    return imgs

```

## reference_material/diffusion_policy_code/diffusion_policy/common/normalize_util.py
```python
from diffusion_policy.model.common.normalizer import SingleFieldLinearNormalizer
from diffusion_policy.common.pytorch_util import dict_apply, dict_apply_reduce, dict_apply_split
import numpy as np


def get_range_normalizer_from_stat(stat, output_max=1, output_min=-1, range_eps=1e-7):
    # -1, 1 normalization
    input_max = stat['max']
    input_min = stat['min']
    input_range = input_max - input_min
    ignore_dim = input_range < range_eps
    input_range[ignore_dim] = output_max - output_min
    scale = (output_max - output_min) / input_range
    offset = output_min - scale * input_min
    offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_image_range_normalizer():
    scale = np.array([2], dtype=np.float32)
    offset = np.array([-1], dtype=np.float32)
    stat = {
        'min': np.array([0], dtype=np.float32),
        'max': np.array([1], dtype=np.float32),
        'mean': np.array([0.5], dtype=np.float32),
        'std': np.array([np.sqrt(1/12)], dtype=np.float32)
    }
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def get_identity_normalizer_from_stat(stat):
    scale = np.ones_like(stat['min'])
    offset = np.zeros_like(stat['min'])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale,
        offset=offset,
        input_stats_dict=stat
    )

def robomimic_abs_action_normalizer_from_stat(stat, rotation_transformer):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'rot': x[...,3:6],
            'gripper': x[...,6:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    def get_rot_param_info(stat):
        example = rotation_transformer.forward(stat['mean'])
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info
    
    def get_gripper_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    rot_param, rot_info = get_rot_param_info(result['rot'])
    gripper_param, gripper_info = get_gripper_param_info(result['gripper'])

    param = dict_apply_reduce(
        [pos_param, rot_param, gripper_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, rot_info, gripper_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_normalizer_from_stat(stat):
    result = dict_apply_split(
        stat, lambda x: {
            'pos': x[...,:3],
            'other': x[...,3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos_param, pos_info = get_pos_param_info(result['pos'])
    other_param, other_info = get_other_param_info(result['other'])

    param = dict_apply_reduce(
        [pos_param, other_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos_info, other_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def robomimic_abs_action_only_dual_arm_normalizer_from_stat(stat):
    Da = stat['max'].shape[-1]
    Dah = Da // 2
    result = dict_apply_split(
        stat, lambda x: {
            'pos0': x[...,:3],
            'other0': x[...,3:Dah],
            'pos1': x[...,Dah:Dah+3],
            'other1': x[...,Dah+3:]
    })

    def get_pos_param_info(stat, output_max=1, output_min=-1, range_eps=1e-7):
        # -1, 1 normalization
        input_max = stat['max']
        input_min = stat['min']
        input_range = input_max - input_min
        ignore_dim = input_range < range_eps
        input_range[ignore_dim] = output_max - output_min
        scale = (output_max - output_min) / input_range
        offset = output_min - scale * input_min
        offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]

        return {'scale': scale, 'offset': offset}, stat

    
    def get_other_param_info(stat):
        example = stat['max']
        scale = np.ones_like(example)
        offset = np.zeros_like(example)
        info = {
            'max': np.ones_like(example),
            'min': np.full_like(example, -1),
            'mean': np.zeros_like(example),
            'std': np.ones_like(example)
        }
        return {'scale': scale, 'offset': offset}, info

    pos0_param, pos0_info = get_pos_param_info(result['pos0'])
    pos1_param, pos1_info = get_pos_param_info(result['pos1'])
    other0_param, other0_info = get_other_param_info(result['other0'])
    other1_param, other1_info = get_other_param_info(result['other1'])

    param = dict_apply_reduce(
        [pos0_param, other0_param, pos1_param, other1_param], 
        lambda x: np.concatenate(x,axis=-1))
    info = dict_apply_reduce(
        [pos0_info, other0_info, pos1_info, other1_info], 
        lambda x: np.concatenate(x,axis=-1))

    return SingleFieldLinearNormalizer.create_manual(
        scale=param['scale'],
        offset=param['offset'],
        input_stats_dict=info
    )


def array_to_stats(arr: np.ndarray):
    stat = {
        'min': np.min(arr, axis=0),
        'max': np.max(arr, axis=0),
        'mean': np.mean(arr, axis=0),
        'std': np.std(arr, axis=0)
    }
    return stat

```

## reference_material/diffusion_policy_code/diffusion_policy/common/json_logger.py
```python
from typing import Optional, Callable, Any, Sequence
import os
import copy
import json
import numbers
import pandas as pd


def read_json_log(path: str, 
        required_keys: Sequence[str]=tuple(), 
        **kwargs) -> pd.DataFrame:
    """
    Read json-per-line file, with potentially incomplete lines.
    kwargs passed to pd.read_json
    """
    lines = list()
    with open(path, 'r') as f:
        while True:
            # one json per line
            line = f.readline()
            if len(line) == 0:
                # EOF
                break
            elif not line.endswith('\n'):
                # incomplete line
                break
            is_relevant = False
            for k in required_keys:
                if k in line:
                    is_relevant = True
                    break
            if is_relevant:
                lines.append(line)
    if len(lines) < 1:
        return pd.DataFrame()  
    json_buf = f'[{",".join([line for line in (line.strip() for line in lines) if line])}]'
    df = pd.read_json(json_buf, **kwargs)
    return df

class JsonLogger:
    def __init__(self, path: str, 
            filter_fn: Optional[Callable[[str,Any],bool]]=None):
        if filter_fn is None:
            filter_fn = lambda k,v: isinstance(v, numbers.Number)

        # default to append mode
        self.path = path
        self.filter_fn = filter_fn
        self.file = None
        self.last_log = None
    
    def start(self):
        # use line buffering
        try:
            self.file = file = open(self.path, 'r+', buffering=1)
        except FileNotFoundError:
            self.file = file = open(self.path, 'w+', buffering=1)

        # Move the pointer (similar to a cursor in a text editor) to the end of the file
        pos = file.seek(0, os.SEEK_END)

        # Read each character in the file one at a time from the last
        # character going backwards, searching for a newline character
        # If we find a new line, exit the search
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        # now the file pointer is at one past the last '\n'
        # and pos is at the last '\n'.
        last_line_end = file.tell()
        
        # find the start of second last line
        pos = max(0, pos-1)
        file.seek(pos, os.SEEK_SET)
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        # now the file pointer is at one past the second last '\n'
        last_line_start = file.tell()

        if last_line_start < last_line_end:
            # has last line of json
            last_line = file.readline()
            self.last_log = json.loads(last_line)
        
        # remove the last incomplete line
        file.seek(last_line_end)
        file.truncate()
    
    def stop(self):
        self.file.close()
        self.file = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def log(self, data: dict):
        filtered_data = dict(
            filter(lambda x: self.filter_fn(*x), data.items()))
        # save current as last log
        self.last_log = filtered_data
        for k, v in filtered_data.items():
            if isinstance(v, numbers.Integral):
                filtered_data[k] = int(v)
            elif isinstance(v, numbers.Number):
                filtered_data[k] = float(v)
        buf = json.dumps(filtered_data)
        # ensure one line per json
        buf = buf.replace('\n','') + '\n'
        self.file.write(buf)
    
    def get_last_log(self):
        return copy.deepcopy(self.last_log)

```

## reference_material/diffusion_policy_code/diffusion_policy/common/sampler.py
```python
from typing import Optional
import numpy as np
import numba
from diffusion_policy.common.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends:np.ndarray, sequence_length:int, 
    episode_mask: np.ndarray,
    pad_before: int=0, pad_after: int=0,
    debug:bool=True) -> np.ndarray:
    episode_mask.shape == episode_ends.shape        
    pad_before = min(max(pad_before, 0), sequence_length-1)
    pad_after = min(max(pad_after, 0), sequence_length-1)

    indices = list()
    for i in range(len(episode_ends)):
        if not episode_mask[i]:
            # skip episode
            continue
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx
        
        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after
        
        # range stops one idx before end
        for idx in range(min_start, max_start+1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx+sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx+start_idx)
            end_offset = (idx+sequence_length+start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert(start_offset >= 0)
                assert(end_offset >= 0)
                assert (sample_end_idx - sample_start_idx) == (buffer_end_idx - buffer_start_idx)
            indices.append([
                buffer_start_idx, buffer_end_idx, 
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def get_val_mask(n_episodes, val_ratio, seed=0):
    val_mask = np.zeros(n_episodes, dtype=bool)
    if val_ratio <= 0:
        return val_mask

    # have at least 1 episode for validation, and at least 1 episode for train
    n_val = min(max(1, round(n_episodes * val_ratio)), n_episodes-1)
    rng = np.random.default_rng(seed=seed)
    val_idxs = rng.choice(n_episodes, size=n_val, replace=False)
    val_mask[val_idxs] = True
    return val_mask


def downsample_mask(mask, max_n, seed=0):
    # subsample training data
    train_mask = mask
    if (max_n is not None) and (np.sum(train_mask) > max_n):
        n_train = int(max_n)
        curr_train_idxs = np.nonzero(train_mask)[0]
        rng = np.random.default_rng(seed=seed)
        train_idxs_idx = rng.choice(len(curr_train_idxs), size=n_train, replace=False)
        train_idxs = curr_train_idxs[train_idxs_idx]
        train_mask = np.zeros_like(train_mask)
        train_mask[train_idxs] = True
        assert np.sum(train_mask) == n_train
    return train_mask

class SequenceSampler:
    def __init__(self, 
        replay_buffer: ReplayBuffer, 
        sequence_length:int,
        pad_before:int=0,
        pad_after:int=0,
        keys=None,
        key_first_k=dict(),
        episode_mask: Optional[np.ndarray]=None,
        ):
        """
        key_first_k: dict str: int
            Only take first k data from these keys (to improve perf)
        """

        super().__init__()
        assert(sequence_length >= 1)
        if keys is None:
            keys = list(replay_buffer.keys())
        
        episode_ends = replay_buffer.episode_ends[:]
        if episode_mask is None:
            episode_mask = np.ones(episode_ends.shape, dtype=bool)

        if np.any(episode_mask):
            indices = create_indices(episode_ends, 
                sequence_length=sequence_length, 
                pad_before=pad_before, 
                pad_after=pad_after,
                episode_mask=episode_mask
                )
        else:
            indices = np.zeros((0,4), dtype=np.int64)

        # (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        self.indices = indices 
        self.keys = list(keys) # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.key_first_k = key_first_k
    
    def __len__(self):
        return len(self.indices)
        
    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx \
            = self.indices[idx]
        result = dict()
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full((n_data,) + input_arr.shape[1:], 
                    fill_value=np.nan, dtype=input_arr.dtype)
                try:
                    sample[:k_data] = input_arr[buffer_start_idx:buffer_start_idx+k_data]
                except Exception as e:
                    import pdb; pdb.set_trace()
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype)
                if sample_start_idx > 0:
                    data[:sample_start_idx] = sample[0]
                if sample_end_idx < self.sequence_length:
                    data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result

```

## reference_material/diffusion_policy_code/diffusion_policy/common/robomimic_config_util.py
```python
from omegaconf import OmegaConf
from robomimic.config import config_factory
import robomimic.scripts.generate_paper_configs as gpc
from robomimic.scripts.generate_paper_configs import (
    modify_config_for_default_image_exp,
    modify_config_for_default_low_dim_exp,
    modify_config_for_dataset,
)

def get_robomimic_config(
        algo_name='bc_rnn', 
        hdf5_type='low_dim', 
        task_name='square', 
        dataset_type='ph'
    ):
    base_dataset_dir = '/tmp/null'
    filter_key = None

    # decide whether to use low-dim or image training defaults
    modifier_for_obs = modify_config_for_default_image_exp
    if hdf5_type in ["low_dim", "low_dim_sparse", "low_dim_dense"]:
        modifier_for_obs = modify_config_for_default_low_dim_exp

    algo_config_name = "bc" if algo_name == "bc_rnn" else algo_name
    config = config_factory(algo_name=algo_config_name)
    # turn into default config for observation modalities (e.g.: low-dim or rgb)
    config = modifier_for_obs(config)
    # add in config based on the dataset
    config = modify_config_for_dataset(
        config=config, 
        task_name=task_name, 
        dataset_type=dataset_type, 
        hdf5_type=hdf5_type, 
        base_dataset_dir=base_dataset_dir,
        filter_key=filter_key,
    )
    # add in algo hypers based on dataset
    algo_config_modifier = getattr(gpc, f'modify_{algo_name}_config_for_dataset')
    config = algo_config_modifier(
        config=config, 
        task_name=task_name, 
        dataset_type=dataset_type, 
        hdf5_type=hdf5_type,
    )
    return config
    


```

## reference_material/diffusion_policy_code/diffusion_policy/common/checkpoint_util.py
```python
from typing import Optional, Dict
import os

class TopKCheckpointManager:
    def __init__(self,
            save_dir,
            monitor_key: str,
            mode='min',
            k=1,
            format_str='epoch={epoch:03d}-train_loss={train_loss:.3f}.ckpt'
        ):
        assert mode in ['max', 'min']
        assert k >= 0

        self.save_dir = save_dir
        self.monitor_key = monitor_key
        self.mode = mode
        self.k = k
        self.format_str = format_str
        self.path_value_map = dict()
    
    def get_ckpt_path(self, data: Dict[str, float]) -> Optional[str]:
        if self.k == 0:
            return None

        value = data[self.monitor_key]
        ckpt_path = os.path.join(
            self.save_dir, self.format_str.format(**data))
        
        if len(self.path_value_map) < self.k:
            # under-capacity
            self.path_value_map[ckpt_path] = value
            return ckpt_path
        
        # at capacity
        sorted_map = sorted(self.path_value_map.items(), key=lambda x: x[1])
        min_path, min_value = sorted_map[0]
        max_path, max_value = sorted_map[-1]

        delete_path = None
        if self.mode == 'max':
            if value > min_value:
                delete_path = min_path
        else:
            if value < max_value:
                delete_path = max_path

        if delete_path is None:
            return None
        else:
            del self.path_value_map[delete_path]
            self.path_value_map[ckpt_path] = value

            if not os.path.exists(self.save_dir):
                os.mkdir(self.save_dir)

            if os.path.exists(delete_path):
                os.remove(delete_path)
            return ckpt_path

```

## reference_material/diffusion_policy_code/diffusion_policy/common/replay_buffer.py
```python
from typing import Union, Dict, Optional
import os
import math
import numbers
import zarr
import numcodecs
import numpy as np
from functools import cached_property

def check_chunks_compatible(chunks: tuple, shape: tuple):
    assert len(shape) == len(chunks)
    for c in chunks:
        assert isinstance(c, numbers.Integral)
        assert c > 0

def rechunk_recompress_array(group, name, 
        chunks=None, chunk_length=None,
        compressor=None, tmp_key='_temp'):
    old_arr = group[name]
    if chunks is None:
        if chunk_length is not None:
            chunks = (chunk_length,) + old_arr.chunks[1:]
        else:
            chunks = old_arr.chunks
    check_chunks_compatible(chunks, old_arr.shape)
    
    if compressor is None:
        compressor = old_arr.compressor
    
    if (chunks == old_arr.chunks) and (compressor == old_arr.compressor):
        # no change
        return old_arr

    # rechunk recompress
    group.move(name, tmp_key)
    old_arr = group[tmp_key]
    n_copied, n_skipped, n_bytes_copied = zarr.copy(
        source=old_arr,
        dest=group,
        name=name,
        chunks=chunks,
        compressor=compressor,
    )
    del group[tmp_key]
    arr = group[name]
    return arr

def get_optimal_chunks(shape, dtype, 
        target_chunk_bytes=2e6, 
        max_chunk_length=None):
    """
    Common shapes
    T,D
    T,N,D
    T,H,W,C
    T,N,H,W,C
    """
    itemsize = np.dtype(dtype).itemsize
    # reversed
    rshape = list(shape[::-1])
    if max_chunk_length is not None:
        rshape[-1] = int(max_chunk_length)
    split_idx = len(shape)-1
    for i in range(len(shape)-1):
        this_chunk_bytes = itemsize * np.prod(rshape[:i])
        next_chunk_bytes = itemsize * np.prod(rshape[:i+1])
        if this_chunk_bytes <= target_chunk_bytes \
            and next_chunk_bytes > target_chunk_bytes:
            split_idx = i

    rchunks = rshape[:split_idx]
    item_chunk_bytes = itemsize * np.prod(rshape[:split_idx])
    this_max_chunk_length = rshape[split_idx]
    next_chunk_length = min(this_max_chunk_length, math.ceil(
            target_chunk_bytes / item_chunk_bytes))
    rchunks.append(next_chunk_length)
    len_diff = len(shape) - len(rchunks)
    rchunks.extend([1] * len_diff)
    chunks = tuple(rchunks[::-1])
    # print(np.prod(chunks) * itemsize / target_chunk_bytes)
    return chunks


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self, 
            root: Union[zarr.Group, 
            Dict[str,dict]]):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        assert('data' in root)
        assert('meta' in root)
        assert('episode_ends' in root['meta'])
        for key, value in root['data'].items():
            assert(value.shape[0] == root['meta']['episode_ends'][-1])
        self.root = root
    
    # ============= create constructors ===============
    @classmethod
    def create_empty_zarr(cls, storage=None, root=None):
        if root is None:
            if storage is None:
                storage = zarr.MemoryStore()
            root = zarr.group(store=storage)
        data = root.require_group('data', overwrite=False)
        meta = root.require_group('meta', overwrite=False)
        if 'episode_ends' not in meta:
            episode_ends = meta.zeros('episode_ends', shape=(0,), dtype=np.int64,
                compressor=None, overwrite=False)
        return cls(root=root)
    
    @classmethod
    def create_empty_numpy(cls):
        root = {
            'data': dict(),
            'meta': {
                'episode_ends': np.zeros((0,), dtype=np.int64)
            }
        }
        return cls(root=root)
    
    @classmethod
    def create_from_group(cls, group, **kwargs):
        if 'data' not in group:
            # create from stratch
            buffer = cls.create_empty_zarr(root=group, **kwargs)
        else:
            # already exist
            buffer = cls(root=group, **kwargs)
        return buffer

    @classmethod
    def create_from_path(cls, zarr_path, mode='r', **kwargs):
        """
        Open a on-disk zarr directly (for dataset larger than memory).
        Slower.
        """
        group = zarr.open(os.path.expanduser(zarr_path), mode)
        return cls.create_from_group(group, **kwargs)
    
    # ============= copy constructors ===============
    @classmethod
    def copy_from_store(cls, src_store, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Load to memory.
        """
        src_root = zarr.group(src_store)
        root = None
        if store is None:
            # numpy backend
            meta = dict()
            for key, value in src_root['meta'].items():
                if isinstance(value, zarr.Group):
                    continue
                if len(value.shape) == 0:
                    meta[key] = np.array(value)
                else:
                    meta[key] = value[:]

            if keys is None:
                keys = src_root['data'].keys()
            data = dict()
            for key in keys:
                arr = src_root['data'][key]
                data[key] = arr[:]

            root = {
                'meta': meta,
                'data': data
            }
        else:
            root = zarr.group(store=store)
            # copy without recompression
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(source=src_store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
            data_group = root.create_group('data', overwrite=True)
            if keys is None:
                keys = src_root['data'].keys()
            for key in keys:
                value = src_root['data'][key]
                cks = cls._resolve_array_chunks(
                    chunks=chunks, key=key, array=value)
                cpr = cls._resolve_array_compressor(
                    compressors=compressors, key=key, array=value)
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=src_store, dest=store,
                        source_path=this_path, dest_path=this_path,
                        if_exists=if_exists
                    )
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
        buffer = cls(root=root)
        return buffer
    
    @classmethod
    def copy_from_path(cls, zarr_path, backend=None, store=None, keys=None, 
            chunks: Dict[str,tuple]=dict(), 
            compressors: Union[dict, str, numcodecs.abc.Codec]=dict(), 
            if_exists='replace',
            **kwargs):
        """
        Copy a on-disk zarr to in-memory compressed.
        Recommended
        """
        if backend == 'numpy':
            print('backend argument is deprecated!')
            store = None
        group = zarr.open(os.path.expanduser(zarr_path), 'r')
        return cls.copy_from_store(src_store=group.store, store=store, 
            keys=keys, chunks=chunks, compressors=compressors, 
            if_exists=if_exists, **kwargs)

    # ============= save methods ===============
    def save_to_store(self, store, 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(),
            if_exists='replace', 
            **kwargs):
        
        root = zarr.group(store)
        if self.backend == 'zarr':
            # recompression free copy
            n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                source=self.root.store, dest=store,
                source_path='/meta', dest_path='/meta', if_exists=if_exists)
        else:
            meta_group = root.create_group('meta', overwrite=True)
            # save meta, no chunking
            for key, value in self.root['meta'].items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape)
        
        # save data, chunk
        data_group = root.create_group('data', overwrite=True)
        for key, value in self.root['data'].items():
            cks = self._resolve_array_chunks(
                chunks=chunks, key=key, array=value)
            cpr = self._resolve_array_compressor(
                compressors=compressors, key=key, array=value)
            if isinstance(value, zarr.Array):
                if cks == value.chunks and cpr == value.compressor:
                    # copy without recompression
                    this_path = '/data/' + key
                    n_copied, n_skipped, n_bytes_copied = zarr.copy_store(
                        source=self.root.store, dest=store,
                        source_path=this_path, dest_path=this_path, if_exists=if_exists)
                else:
                    # copy with recompression
                    n_copied, n_skipped, n_bytes_copied = zarr.copy(
                        source=value, dest=data_group, name=key,
                        chunks=cks, compressor=cpr, if_exists=if_exists
                    )
            else:
                # numpy
                _ = data_group.array(
                    name=key,
                    data=value,
                    chunks=cks,
                    compressor=cpr
                )
        return store

    def save_to_path(self, zarr_path,             
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict(), 
            if_exists='replace', 
            **kwargs):
        store = zarr.DirectoryStore(os.path.expanduser(zarr_path))
        return self.save_to_store(store, chunks=chunks, 
            compressors=compressors, if_exists=if_exists, **kwargs)

    @staticmethod
    def resolve_compressor(compressor='default'):
        if compressor == 'default':
            compressor = numcodecs.Blosc(cname='lz4', clevel=5, 
                shuffle=numcodecs.Blosc.NOSHUFFLE)
        elif compressor == 'disk':
            compressor = numcodecs.Blosc('zstd', clevel=5, 
                shuffle=numcodecs.Blosc.BITSHUFFLE)
        return compressor

    @classmethod
    def _resolve_array_compressor(cls, 
            compressors: Union[dict, str, numcodecs.abc.Codec], key, array):
        # allows compressor to be explicitly set to None
        cpr = 'nil'
        if isinstance(compressors, dict):
            if key in compressors:
                cpr = cls.resolve_compressor(compressors[key])
            elif isinstance(array, zarr.Array):
                cpr = array.compressor
        else:
            cpr = cls.resolve_compressor(compressors)
        # backup default
        if cpr == 'nil':
            cpr = cls.resolve_compressor('default')
        return cpr
    
    @classmethod
    def _resolve_array_chunks(cls,
            chunks: Union[dict, tuple], key, array):
        cks = None
        if isinstance(chunks, dict):
            if key in chunks:
                cks = chunks[key]
            elif isinstance(array, zarr.Array):
                cks = array.chunks
        elif isinstance(chunks, tuple):
            cks = chunks
        else:
            raise TypeError(f"Unsupported chunks type {type(chunks)}")
        # backup default
        if cks is None:
            cks = get_optimal_chunks(shape=array.shape, dtype=array.dtype)
        # check
        check_chunks_compatible(chunks=cks, shape=array.shape)
        return cks
    
    # ============= properties =================
    @cached_property
    def data(self):
        return self.root['data']
    
    @cached_property
    def meta(self):
        return self.root['meta']

    def update_meta(self, data):
        # sanitize data
        np_data = dict()
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np_data[key] = value
            else:
                arr = np.array(value)
                if arr.dtype == object:
                    raise TypeError(f"Invalid value type {type(value)}")
                np_data[key] = arr

        meta_group = self.meta
        if self.backend == 'zarr':
            for key, value in np_data.items():
                _ = meta_group.array(
                    name=key,
                    data=value, 
                    shape=value.shape, 
                    chunks=value.shape,
                    overwrite=True)
        else:
            meta_group.update(np_data)
        
        return meta_group
    
    @property
    def episode_ends(self):
        return self.meta['episode_ends']
    
    def get_episode_idxs(self):
        import numba
        numba.jit(nopython=True)
        def _get_episode_idxs(episode_ends):
            result = np.zeros((episode_ends[-1],), dtype=np.int64)
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i-1]
                end = episode_ends[i]
                for idx in range(start, end):
                    result[idx] = i
            return result
        return _get_episode_idxs(self.episode_ends)
        
    
    @property
    def backend(self):
        backend = 'numpy'
        if isinstance(self.root, zarr.Group):
            backend = 'zarr'
        return backend
    
    # =========== dict-like API ==============
    def __repr__(self) -> str:
        if self.backend == 'zarr':
            return str(self.root.tree())
        else:
            return super().__repr__()

    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data

    # =========== our API ==============
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)

    @property
    def chunk_size(self):
        if self.backend == 'zarr':
            return next(iter(self.data.arrays()))[-1].chunks[0]
        return None

    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths

    def add_episode(self, 
            data: Dict[str, np.ndarray], 
            chunks: Optional[Dict[str,tuple]]=dict(),
            compressors: Union[str, numcodecs.abc.Codec, dict]=dict()):
        assert(len(data) > 0)
        is_zarr = (self.backend == 'zarr')

        curr_len = self.n_steps
        episode_length = None
        for key, value in data.items():
            assert(len(value.shape) >= 1)
            if episode_length is None:
                episode_length = len(value)
            else:
                assert(episode_length == len(value))
        new_len = curr_len + episode_length

        for key, value in data.items():
            new_shape = (new_len,) + value.shape[1:]
            # create array
            if key not in self.data:
                if is_zarr:
                    cks = self._resolve_array_chunks(
                        chunks=chunks, key=key, array=value)
                    cpr = self._resolve_array_compressor(
                        compressors=compressors, key=key, array=value)
                    arr = self.data.zeros(name=key, 
                        shape=new_shape, 
                        chunks=cks,
                        dtype=value.dtype,
                        compressor=cpr)
                else:
                    # copy data to prevent modify
                    arr = np.zeros(shape=new_shape, dtype=value.dtype)
                    self.data[key] = arr
            else:
                arr = self.data[key]
                assert(value.shape[1:] == arr.shape[1:])
                # same method for both zarr and numpy
                if is_zarr:
                    arr.resize(new_shape)
                else:
                    arr.resize(new_shape, refcheck=False)
            # copy data
            arr[-value.shape[0]:] = value
        
        # append to episode ends
        episode_ends = self.episode_ends
        if is_zarr:
            episode_ends.resize(episode_ends.shape[0] + 1)
        else:
            episode_ends.resize(episode_ends.shape[0] + 1, refcheck=False)
        episode_ends[-1] = new_len

        # rechunk
        if is_zarr:
            if episode_ends.chunks[0] < episode_ends.shape[0]:
                rechunk_recompress_array(self.meta, 'episode_ends', 
                    chunk_length=int(episode_ends.shape[0] * 1.5))
    
    def drop_episode(self):
        is_zarr = (self.backend == 'zarr')
        episode_ends = self.episode_ends[:].copy()
        assert(len(episode_ends) > 0)
        start_idx = 0
        if len(episode_ends) > 1:
            start_idx = episode_ends[-2]
        for key, value in self.data.items():
            new_shape = (start_idx,) + value.shape[1:]
            if is_zarr:
                value.resize(new_shape)
            else:
                value.resize(new_shape, refcheck=False)
        if is_zarr:
            self.episode_ends.resize(len(episode_ends)-1)
        else:
            self.episode_ends.resize(len(episode_ends)-1, refcheck=False)
    
    def pop_episode(self):
        assert(self.n_episodes > 0)
        episode = self.get_episode(self.n_episodes-1, copy=True)
        self.drop_episode()
        return episode

    def extend(self, data):
        self.add_episode(data)

    def get_episode(self, idx, copy=False):
        idx = list(range(len(self.episode_ends)))[idx]
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        result = self.get_steps_slice(start_idx, end_idx, copy=copy)
        return result
    
    def get_episode_slice(self, idx):
        start_idx = 0
        if idx > 0:
            start_idx = self.episode_ends[idx-1]
        end_idx = self.episode_ends[idx]
        return slice(start_idx, end_idx)

    def get_steps_slice(self, start, stop, step=None, copy=False):
        _slice = slice(start, stop, step)

        result = dict()
        for key, value in self.data.items():
            x = value[_slice]
            if copy and isinstance(value, np.ndarray):
                x = x.copy()
            result[key] = x
        return result
    
    # =========== chunking =============
    def get_chunks(self) -> dict:
        assert self.backend == 'zarr'
        chunks = dict()
        for key, value in self.data.items():
            chunks[key] = value.chunks
        return chunks
    
    def set_chunks(self, chunks: dict):
        assert self.backend == 'zarr'
        for key, value in chunks.items():
            if key in self.data:
                arr = self.data[key]
                if value != arr.chunks:
                    check_chunks_compatible(chunks=value, shape=arr.shape)
                    rechunk_recompress_array(self.data, key, chunks=value)

    def get_compressors(self) -> dict:
        assert self.backend == 'zarr'
        compressors = dict()
        for key, value in self.data.items():
            compressors[key] = value.compressor
        return compressors

    def set_compressors(self, compressors: dict):
        assert self.backend == 'zarr'
        for key, value in compressors.items():
            if key in self.data:
                arr = self.data[key]
                compressor = self.resolve_compressor(value)
                if compressor != arr.compressor:
                    rechunk_recompress_array(self.data, key, compressor=compressor)

```

## reference_material/diffusion_policy_code/diffusion_policy/common/pytorch_util.py
```python
from typing import Dict, Callable, List
import collections
import torch
import torch.nn as nn

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def pad_remaining_dims(x, target):
    assert x.shape == target.shape[:len(x.shape)]
    return x.reshape(x.shape + (1,)*(len(target.shape) - len(x.shape)))

def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result


def replace_submodules(
        root_module: nn.Module, 
        predicate: Callable[[nn.Module], bool], 
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    bn_list = [k.split('.') for k, m 
        in root_module.named_modules(remove_duplicate=True) 
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def optimizer_to(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device)
    return optimizer

```

## reference_material/diffusion_policy_code/diffusion_policy/common/robomimic_util.py
```python
import numpy as np
import copy

import h5py
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from scipy.spatial.transform import Rotation

from robomimic.config import config_factory


class RobomimicAbsoluteActionConverter:
    def __init__(self, dataset_path, algo_name='bc'):
        # default BC config
        config = config_factory(algo_name=algo_name)

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        # must ran before create dataset
        ObsUtils.initialize_obs_utils_with_config(config)

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        abs_env_meta = copy.deepcopy(env_meta)
        abs_env_meta['env_kwargs']['controller_configs']['control_delta'] = False

        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert len(env.env.robots) in (1, 2)

        abs_env = EnvUtils.create_env_from_metadata(
            env_meta=abs_env_meta,
            render=False, 
            render_offscreen=False,
            use_image_obs=False, 
        )
        assert not abs_env.env.robots[0].controller.use_delta

        self.env = env
        self.abs_env = abs_env
        self.file = h5py.File(dataset_path, 'r')
    
    def __len__(self):
        return len(self.file['data'])

    def convert_actions(self, 
            states: np.ndarray, 
            actions: np.ndarray) -> np.ndarray:
        """
        Given state and delta action sequence
        generate equivalent goal position and orientation for each step
        keep the original gripper action intact.
        """
        # in case of multi robot
        # reshape (N,14) to (N,2,7)
        # or (N,7) to (N,1,7)
        stacked_actions = actions.reshape(*actions.shape[:-1],-1,7)

        env = self.env
        # generate abs actions
        action_goal_pos = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_goal_ori = np.zeros(
            stacked_actions.shape[:-1]+(3,), 
            dtype=stacked_actions.dtype)
        action_gripper = stacked_actions[...,[-1]]
        for i in range(len(states)):
            _ = env.reset_to({'states': states[i]})

            # taken from robot_env.py L#454
            for idx, robot in enumerate(env.env.robots):
                # run controller goal generator
                robot.control(stacked_actions[i,idx], policy_step=True)
            
                # read pos and ori from robots
                controller = robot.controller
                action_goal_pos[i,idx] = controller.goal_pos
                action_goal_ori[i,idx] = Rotation.from_matrix(
                    controller.goal_ori).as_rotvec()

        stacked_abs_actions = np.concatenate([
            action_goal_pos,
            action_goal_ori,
            action_gripper
        ], axis=-1)
        abs_actions = stacked_abs_actions.reshape(actions.shape)
        return abs_actions

    def convert_idx(self, idx):
        file = self.file
        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)
        return abs_actions

    def convert_and_eval_idx(self, idx):
        env = self.env
        abs_env = self.abs_env
        file = self.file
        # first step have high error for some reason, not representative
        eval_skip_steps = 1

        demo = file[f'data/demo_{idx}']
        # input
        states = demo['states'][:]
        actions = demo['actions'][:]

        # generate abs actions
        abs_actions = self.convert_actions(states, actions)

        # verify
        robot0_eef_pos = demo['obs']['robot0_eef_pos'][:]
        robot0_eef_quat = demo['obs']['robot0_eef_quat'][:]

        delta_error_info = self.evaluate_rollout_error(
            env, states, actions, robot0_eef_pos, robot0_eef_quat, 
            metric_skip_steps=eval_skip_steps)
        abs_error_info = self.evaluate_rollout_error(
            abs_env, states, abs_actions, robot0_eef_pos, robot0_eef_quat,
            metric_skip_steps=eval_skip_steps)

        info = {
            'delta_max_error': delta_error_info,
            'abs_max_error': abs_error_info
        }
        return abs_actions, info

    @staticmethod
    def evaluate_rollout_error(env, 
            states, actions, 
            robot0_eef_pos, 
            robot0_eef_quat, 
            metric_skip_steps=1):
        # first step have high error for some reason, not representative

        # evaluate abs actions
        rollout_next_states = list()
        rollout_next_eef_pos = list()
        rollout_next_eef_quat = list()
        obs = env.reset_to({'states': states[0]})
        for i in range(len(states)):
            obs = env.reset_to({'states': states[i]})
            obs, reward, done, info = env.step(actions[i])
            obs = env.get_observation()
            rollout_next_states.append(env.get_state()['states'])
            rollout_next_eef_pos.append(obs['robot0_eef_pos'])
            rollout_next_eef_quat.append(obs['robot0_eef_quat'])
        rollout_next_states = np.array(rollout_next_states)
        rollout_next_eef_pos = np.array(rollout_next_eef_pos)
        rollout_next_eef_quat = np.array(rollout_next_eef_quat)

        next_state_diff = states[1:] - rollout_next_states[:-1]
        max_next_state_diff = np.max(np.abs(next_state_diff[metric_skip_steps:]))

        next_eef_pos_diff = robot0_eef_pos[1:] - rollout_next_eef_pos[:-1]
        next_eef_pos_dist = np.linalg.norm(next_eef_pos_diff, axis=-1)
        max_next_eef_pos_dist = next_eef_pos_dist[metric_skip_steps:].max()

        next_eef_rot_diff = Rotation.from_quat(robot0_eef_quat[1:]) \
            * Rotation.from_quat(rollout_next_eef_quat[:-1]).inv()
        next_eef_rot_dist = next_eef_rot_diff.magnitude()
        max_next_eef_rot_dist = next_eef_rot_dist[metric_skip_steps:].max()

        info = {
            'state': max_next_state_diff,
            'pos': max_next_eef_pos_dist,
            'rot': max_next_eef_rot_dist
        }
        return info

```

## reference_material/diffusion_policy_code/diffusion_policy/common/nested_dict_util.py
```python
import functools

def nested_dict_map(f, x):
    """
    Map f over all leaf of nested dict x
    """

    if not isinstance(x, dict):
        return f(x)
    y = dict()
    for key, value in x.items():
        y[key] = nested_dict_map(f, value)
    return y

def nested_dict_reduce(f, x):
    """
    Map f over all values of nested dict x, and reduce to a single value
    """
    if not isinstance(x, dict):
        return x

    reduced_values = list()
    for value in x.values():
        reduced_values.append(nested_dict_reduce(f, value))
    y = functools.reduce(f, reduced_values)
    return y


def nested_dict_check(f, x):
    bool_dict = nested_dict_map(f, x)
    result = nested_dict_reduce(lambda x, y: x and y, bool_dict)
    return result

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/robomimic_image_runner.py
```python
import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicImageWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=enable_render,
        use_image_obs=enable_render, 
    )
    return env


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta, 
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
        
        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    RobomimicImageWrapper(
                        env=robomimic_env,
                        shape_meta=shape_meta,
                        init_state=None,
                        render_obs_key=render_obs_key
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)


        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()
        
        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video
        
        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/base_image_runner.py
```python
from typing import Dict
from diffusion_policy.policy.base_image_policy import BaseImagePolicy

class BaseImageRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseImagePolicy) -> Dict:
        raise NotImplementedError()

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/blockpush_lowdim_runner.py
```python
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.block_pushing.block_pushing_multimodal import BlockPushMultimodal
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from gym.wrappers import FlattenObservation

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class BlockPushLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=5,
            crf=22,
            past_action=False,
            abs_action=False,
            obs_eef_target=True,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        task_fps = 10
        steps_per_render = max(10 // fps, 1)

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    FlattenObservation(
                        BlockPushMultimodal(
                            control_frequency=task_fps,
                            shared_memory=False,
                            seed=seed,
                            abs_action=abs_action
                        )
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
        self.obs_eef_target = obs_eef_target


    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        last_info = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval BlockPushLowdimRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                if not self.obs_eef_target:
                    obs[...,8:10] = 0
                np_obs_dict = {
                    'obs': obs.astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            last_info[this_global_slice] = [dict((k,v[-1]) for k, v in x.items()) for x in info][this_local_slice]

        # log
        total_rewards = collections.defaultdict(list)
        total_p1 = collections.defaultdict(list)
        total_p2 = collections.defaultdict(list)
        prefix_event_counts = collections.defaultdict(lambda :collections.defaultdict(lambda : 0))
        prefix_counts = collections.defaultdict(lambda : 0)

        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            this_rewards = all_rewards[i]
            total_reward = np.unique(this_rewards).sum() # (0, 0.49, 0.51)
            p1 = total_reward > 0.4
            p2 = total_reward > 0.9

            total_rewards[prefix].append(total_reward)
            total_p1[prefix].append(p1)
            total_p2[prefix].append(p2)
            log_data[prefix+f'sim_max_reward_{seed}'] = total_reward

            # aggregate event counts
            prefix_counts[prefix] += 1
            for key, value in last_info[i].items():
                delta_count = 1 if value > 0 else 0
                prefix_event_counts[prefix][key] += delta_count

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in total_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in total_p1.items():
            name = prefix+'p1'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in total_p2.items():
            name = prefix+'p2'
            value = np.mean(value)
            log_data[name] = value
        
        # summarize probabilities
        for prefix, events in prefix_event_counts.items():
            prefix_count = prefix_counts[prefix]
            for event, count in events.items():
                prob = count / prefix_count
                key = prefix + event
                log_data[key] = prob

        return log_data

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/robomimic_lowdim_runner.py
```python
import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.env.robomimic.robomimic_lowdim_wrapper import RobomimicLowdimWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, obs_keys):
    ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        # only way to not show collision geometry
        # is to enable render_offscreen
        # which uses a lot of RAM.
        render_offscreen=False,
        use_image_obs=False, 
    )
    return env


class RobomimicLowdimRunner(BaseLowdimRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self, 
            output_dir,
            dataset_path,
            obs_keys,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            n_latency_steps=0,
            render_hw=(256,256),
            render_camera_name='agentview',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        """
        Assuming:
        n_obs_steps=2
        n_latency_steps=3
        n_action_steps=4
        o: obs
        i: inference
        a: action
        Batch t:
        |o|o| | | | | | |
        | |i|i|i| | | | |
        | | | | |a|a|a|a|
        Batch t+1
        | | | | |o|o| | | | | | |
        | | | | | |i|i|i| | | | |
        | | | | | | | | |a|a|a|a|
        """

        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta, 
                    obs_keys=obs_keys
                )
            # hard reset doesn't influence lowdim env
            # robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                    VideoRecordingWrapper(
                        RobomimicLowdimWrapper(
                            env=robomimic_env,
                            obs_keys=obs_keys,
                            init_state=None,
                            render_hw=render_hw,
                            render_camera_name=render_camera_name
                        ),
                        video_recoder=VideoRecorder.create_h264(
                            fps=fps,
                            codec='h264',
                            input_pix_fmt='rgb24',
                            crf=crf,
                            thread_type='FRAME',
                            thread_count=1
                        ),
                        file_path=None,
                        steps_per_render=steps_per_render
                    ),
                    n_obs_steps=env_n_obs_steps,
                    n_action_steps=env_n_action_steps,
                    max_episode_steps=max_steps
                )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state, 
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicLowdimWrapper)
                    env.env.env.init_state = init_state

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))
        
        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, 
                enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicLowdimWrapper)
                env.env.env.init_state = None
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        env = AsyncVectorEnv(env_fns)
        # env = SyncVectorEnv(env_fns)

        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.env_n_obs_steps = env_n_obs_steps
        self.env_n_action_steps = env_n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec

    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env
        
        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Lowdim {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[:,:self.n_obs_steps].astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")
                
                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/kitchen_lowdim_runner.py
```python
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import logging
import wandb.sdk.data_types.video as wv
import gym
import gym.spaces
import multiprocessing as mp
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

module_logger = logging.getLogger(__name__)

class KitchenLowdimRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            dataset_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            max_steps=280,
            n_obs_steps=2,
            n_action_steps=8,
            render_hw=(240,360),
            fps=12.5,
            crf=22,
            past_action=False,
            tqdm_interval_sec=5.0,
            abs_action=False,
            robot_noise_ratio=0.1,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        task_fps = 12.5
        steps_per_render = int(max(task_fps // fps, 1))

        def env_fn():
            from diffusion_policy.env.kitchen.v0 import KitchenAllV0
            from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
            env = KitchenAllV0(use_abs_action=abs_action)
            env.robot_noise_ratio = robot_noise_ratio
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    KitchenLowdimWrapper(
                        env=env,
                        init_qpos=None,
                        init_qvel=None,
                        render_hw=tuple(render_hw)
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        all_init_qpos = np.load(pathlib.Path(dataset_dir) / "all_init_qpos.npy")
        all_init_qvel = np.load(pathlib.Path(dataset_dir) / "all_init_qvel.npy")
        module_logger.info(f'Loaded {len(all_init_qpos)} known initial conditions.')

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis
            init_qpos = None
            init_qvel = None
            if i < len(all_init_qpos):
                init_qpos = all_init_qpos[i]
                init_qvel = all_init_qvel[i]

            def init_fn(env, init_qpos=init_qpos, init_qvel=init_qvel, enable_render=enable_render):
                from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set initial condition
                assert isinstance(env.env.env, KitchenLowdimWrapper)
                env.env.env.init_qpos = init_qpos
                env.env.env.init_qvel = init_qvel
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                from diffusion_policy.env.kitchen.kitchen_lowdim_wrapper import KitchenLowdimWrapper
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set initial condition
                assert isinstance(env.env.env, KitchenLowdimWrapper)
                env.env.env.init_qpos = None
                env.env.env.init_qvel = None

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))
        
        def dummy_env_fn():
            # Avoid importing or using env in the main process
            # to prevent OpenGL context issue with fork.
            # Create a fake env whose sole purpos is to provide 
            # obs/action spaces and metadata.
            env = gym.Env()
            env.observation_space = gym.spaces.Box(
                -8, 8, shape=(60,), dtype=np.float32)
            env.action_space = gym.spaces.Box(
                -8, 8, shape=(9,), dtype=np.float32)
            env.metadata = {
                'render.modes': ['human', 'rgb_array', 'depth_array'],
                'video.frames_per_second': 12
            }
            env = MultiStepWrapper(
                env=env,
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )
            return env
        
        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec


    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        last_info = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval BlockPushLowdimRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = {
                    'obs': obs.astype(np.float32)
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            last_info[this_global_slice] = [dict((k,v[-1]) for k, v in x.items()) for x in info][this_local_slice]

        # reward is number of tasks completed, max 7
        # use info to record the order of task completion?
        # also report the probably to completing n tasks (different aggregation of reward).

        # log
        log_data = dict()
        prefix_total_reward_map = collections.defaultdict(list)
        prefix_n_completed_map = collections.defaultdict(list)
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            this_rewards = all_rewards[i]
            total_reward = np.sum(this_rewards) / 7
            prefix_total_reward_map[prefix].append(total_reward)

            n_completed_tasks = len(last_info[i]['completed_tasks'])
            prefix_n_completed_map[prefix].append(n_completed_tasks)

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in prefix_total_reward_map.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in prefix_n_completed_map.items():
            n_completed = np.array(value)
            for i in range(7):
                n = i + 1
                p_n = np.mean(n_completed >= n)
                name = prefix + f'p_{n}'
                log_data[name] = p_n

        return log_data

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/pusht_keypoints_runner.py
```python
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner

class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        if n_envs is None:
            n_envs = n_train + n_test

        # handle latency step
        # to mimic latency, we request n_latency_steps additional steps 
        # of past observations, and the discard the last n_latency_steps
        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        # assert n_obs_steps <= n_action_steps
        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseLowdimPolicy):
        device = policy.device
        dtype = policy.dtype

        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                Do = obs.shape[-1] // 2
                # create obs dict
                np_obs_dict = {
                    # handle n_latency_steps by discarding the last n_latency_steps
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                # handle latency_steps, we discard the first n_latency_steps actions
                # to simulate latency
                action = np_action_dict['action'][:,self.n_latency_steps:]

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # import pdb; pdb.set_trace()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/pusht_image_runner.py
```python
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
from diffusion_policy.env.pusht.pusht_image_env import PushTImageEnv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class PushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            fps=10,
            crf=22,
            render_size=96,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)
        if n_envs is None:
            n_envs = n_train + n_test

        steps_per_render = max(10 // fps, 1)
        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTImageEnv(
                        legacy=legacy_test,
                        render_size=render_size
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()
        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # set seed
                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)
            
            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        env = AsyncVectorEnv(env_fns)

        # test env
        # env.reset(seed=env_seeds)
        # x = env.step(env.action_space.sample())
        # imgs = env.call('render')
        # import pdb; pdb.set_trace()

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec
    
    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)
            
            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function', 
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval PushtImageRunner {chunk_idx+1}/{n_chunks}", 
                leave=False, mininterval=self.tqdm_interval_sec)
            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)
                
                # device transfer
                obs_dict = dict_apply(np_obs_dict, 
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']

                # step env
                obs, reward, done, info = env.step(action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        return log_data

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/base_lowdim_runner.py
```python
from typing import Dict
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy

class BaseLowdimRunner:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def run(self, policy: BaseLowdimPolicy) -> Dict:
        raise NotImplementedError()

```

## reference_material/diffusion_policy_code/diffusion_policy/env_runner/real_pusht_image_runner.py
```python
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class RealPushTImageRunner(BaseImageRunner):
    def __init__(self,
            output_dir):
        super().__init__(output_dir)
    
    def run(self, policy: BaseImagePolicy):
        return dict()

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/action_ae/__init__.py
```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import abc

from typing import Optional, Union

import diffusion_policy.model.bet.utils as utils


class AbstractActionAE(utils.SaveModule, abc.ABC):
    @abc.abstractmethod
    def fit_model(
        self,
        input_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        obs_encoding_net: Optional[nn.Module] = None,
    ) -> None:
        pass

    @abc.abstractmethod
    def encode_into_latent(
        self,
        input_action: torch.Tensor,
        input_rep: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """
        Given the input action, discretize it.

        Inputs:
        input_action (shape: ... x action_dim): The input action to discretize. This can be in a batch,
        and is generally assumed that the last dimnesion is the action dimension.

        Outputs:
        discretized_action (shape: ... x num_tokens): The discretized action.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def decode_actions(
        self,
        latent_action_batch: Optional[torch.Tensor],
        input_rep_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given a discretized action, convert it to a continuous action.

        Inputs:
        latent_action_batch (shape: ... x num_tokens): The discretized action
        generated by the discretizer.

        Outputs:
        continuous_action (shape: ... x action_dim): The continuous action.
        """
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def num_latents(self) -> Union[int, float]:
        """
        Number of possible latents for this generator, useful for state priors that use softmax.
        """
        return float("inf")

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/action_ae/discretizers/k_means.py
```python
import torch
import numpy as np

import tqdm

from typing import Optional, Tuple, Union
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin


class KMeansDiscretizer(DictOfTensorMixin):
    """
    Simplified and modified version of KMeans algorithm  from sklearn.
    """

    def __init__(
        self,
        action_dim: int,
        num_bins: int = 100,
        predict_offsets: bool = False,
    ):
        super().__init__()
        self.n_bins = num_bins
        self.action_dim = action_dim
        self.predict_offsets = predict_offsets

    def fit_discretizer(self, input_actions: torch.Tensor) -> None:
        assert (
            self.action_dim == input_actions.shape[-1]
        ), f"Input action dimension {self.action_dim} does not match fitted model {input_actions.shape[-1]}"

        flattened_actions = input_actions.view(-1, self.action_dim)
        cluster_centers = KMeansDiscretizer._kmeans(
            flattened_actions, ncluster=self.n_bins
        )
        self.params_dict['bin_centers'] = cluster_centers

    @property
    def suggested_actions(self) -> torch.Tensor:
        return self.params_dict['bin_centers']

    @classmethod
    def _kmeans(cls, x: torch.Tensor, ncluster: int = 512, niter: int = 50):
        """
        Simple k-means clustering algorithm adapted from Karpathy's minGPT library
        https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
        """
        N, D = x.size()
        c = x[torch.randperm(N)[:ncluster]]  # init clusters at random

        pbar = tqdm.trange(niter)
        pbar.set_description("K-means clustering")
        for i in pbar:
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            if ndead:
                tqdm.tqdm.write(
                    "done step %d/%d, re-initialized %d dead clusters"
                    % (i + 1, niter, ndead)
                )
            c[nanix] = x[torch.randperm(N)[:ndead]]  # re-init dead clusters
        return c

    def encode_into_latent(
        self, input_action: torch.Tensor, input_rep: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Given the input action, discretize it using the k-Means clustering algorithm.

        Inputs:
        input_action (shape: ... x action_dim): The input action to discretize. This can be in a batch,
        and is generally assumed that the last dimnesion is the action dimension.

        Outputs:
        discretized_action (shape: ... x num_tokens): The discretized action.
        If self.predict_offsets is True, then the offsets are also returned.
        """
        assert (
            input_action.shape[-1] == self.action_dim
        ), "Input action dimension does not match fitted model"

        # flatten the input action
        flattened_actions = input_action.view(-1, self.action_dim)

        # get the closest cluster center
        closest_cluster_center = torch.argmin(
            torch.sum(
                (flattened_actions[:, None, :] - self.params_dict['bin_centers'][None, :, :]) ** 2,
                dim=2,
            ),
            dim=1,
        )
        # Reshape to the original shape
        discretized_action = closest_cluster_center.view(input_action.shape[:-1] + (1,))

        if self.predict_offsets:
            # decode from latent and get the difference
            reconstructed_action = self.decode_actions(discretized_action)
            offsets = input_action - reconstructed_action
            return (discretized_action, offsets)
        else:
            # return the one-hot vector
            return discretized_action

    def decode_actions(
        self,
        latent_action_batch: torch.Tensor,
        input_rep_batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Given the latent action, reconstruct the original action.

        Inputs:
        latent_action (shape: ... x 1): The latent action to reconstruct. This can be in a batch,
        and is generally assumed that the last dimension is the action dimension. If the latent_action_batch
        is a tuple, then it is assumed to be (discretized_action, offsets).

        Outputs:
        reconstructed_action (shape: ... x action_dim): The reconstructed action.
        """
        offsets = None
        if type(latent_action_batch) == tuple:
            latent_action_batch, offsets = latent_action_batch
        # get the closest cluster center
        closest_cluster_center = self.params_dict['bin_centers'][latent_action_batch]
        # Reshape to the original shape
        reconstructed_action = closest_cluster_center.view(
            latent_action_batch.shape[:-1] + (self.action_dim,)
        )
        if offsets is not None:
            reconstructed_action += offsets
        return reconstructed_action

    @property
    def discretized_space(self) -> int:
        return self.n_bins

    @property
    def latent_dim(self) -> int:
        return 1

    @property
    def num_latents(self) -> int:
        return self.n_bins

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/latent_generators/mingpt.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import diffusion_policy.model.bet.latent_generators.latent_generator as latent_generator

import diffusion_policy.model.bet.libraries.mingpt.model as mingpt_model
import diffusion_policy.model.bet.libraries.mingpt.trainer as mingpt_trainer
from diffusion_policy.model.bet.libraries.loss_fn import FocalLoss, soft_cross_entropy

from typing import Optional, Tuple


class MinGPT(latent_generator.AbstractLatentGenerator):
    def __init__(
        self,
        input_dim: int,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        embd_pdrop: float = 0.1,
        resid_pdrop: float = 0.1,
        attn_pdrop: float = 0.1,
        block_size: int = 128,
        vocab_size: int = 50257,
        latent_dim: int = 768,  # Ignore, used for compatibility with other models.
        action_dim: int = 0,
        discrete_input: bool = False,
        predict_offsets: bool = False,
        offset_loss_scale: float = 1.0,
        focal_loss_gamma: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.input_size = input_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.embd_pdrop = embd_pdrop
        self.resid_pdrop = resid_pdrop
        self.attn_pdrop = attn_pdrop
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.action_dim = action_dim
        self.predict_offsets = predict_offsets
        self.offset_loss_scale = offset_loss_scale
        self.focal_loss_gamma = focal_loss_gamma
        for k, v in kwargs.items():
            setattr(self, k, v)

        gpt_config = mingpt_model.GPTConfig(
            input_size=self.input_size,
            vocab_size=self.vocab_size * (1 + self.action_dim)
            if self.predict_offsets
            else self.vocab_size,
            block_size=self.block_size,
            n_layer=n_layer,
            n_head=n_head,
            n_embd=n_embd,
            discrete_input=discrete_input,
            embd_pdrop=embd_pdrop,
            resid_pdrop=resid_pdrop,
            attn_pdrop=attn_pdrop,
        )

        self.model = mingpt_model.GPT(gpt_config)

    def get_latent_and_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
        seq_masks: Optional[torch.Tensor] = None,
        return_loss_components: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Unlike torch.transformers, GPT takes in batch x seq_len x embd_dim
        # obs_rep = einops.rearrange(obs_rep, "seq batch embed -> batch seq embed")
        # target_latents = einops.rearrange(
        #     target_latents, "seq batch embed -> batch seq embed"
        # )
        # While this has been trained autoregressively,
        # there is no reason why it needs to be so.
        # We can just use the observation as the input and the next latent as the target.
        if self.predict_offsets:
            target_latents, target_offsets = target_latents
        is_soft_target = (target_latents.shape[-1] == self.vocab_size) and (
            self.vocab_size != 1
        )
        if is_soft_target:
            target_latents = target_latents.view(-1, target_latents.size(-1))
            criterion = soft_cross_entropy
        else:
            target_latents = target_latents.view(-1)
            if self.vocab_size == 1:
                # unify k-means (target_class == 0) and GMM (target_prob == 1)
                target_latents = torch.zeros_like(target_latents)
            criterion = FocalLoss(gamma=self.focal_loss_gamma)
        if self.predict_offsets:
            output, _ = self.model(obs_rep)
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            batch = logits.shape[0]
            seq = logits.shape[1]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
            # calculate (optionally soft) cross entropy and offset losses
            class_loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
            # offset loss is only calculated on the target class
            # if soft targets, argmax is considered the target class
            selected_offsets = offsets[
                torch.arange(offsets.size(0)),
                target_latents.argmax(dim=-1).view(-1)
                if is_soft_target
                else target_latents.view(-1),
            ]
            offset_loss = self.offset_loss_scale * F.mse_loss(
                selected_offsets, target_offsets.view(-1, self.action_dim)
            )
            loss = offset_loss + class_loss
            logits = einops.rearrange(logits, "batch seq classes -> seq batch classes")
            offsets = einops.rearrange(
                offsets,
                "(N T) V A -> T N V A",  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
                N=batch,
                T=seq,
            )
            if return_loss_components:
                return (
                    (logits, offsets),
                    loss,
                    {"offset": offset_loss, "class": class_loss, "total": loss},
                )
            else:
                return (logits, offsets), loss
        else:
            logits, _ = self.model(obs_rep)
            loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
            logits = einops.rearrange(
                logits, "batch seq classes -> seq batch classes"
            )  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            if return_loss_components:
                return logits, loss, {"class": loss, "total": loss}
            else:
                return logits, loss

    def generate_latents(
        self, obs_rep: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, embed = obs_rep.shape

        output, _ = self.model(obs_rep, None)
        if self.predict_offsets:
            logits = output[:, :, : self.vocab_size]
            offsets = output[:, :, self.vocab_size :]
            offsets = einops.rearrange(
                offsets,
                "N T (V A) -> (N T) V A",  # N = batch, T = seq
                V=self.vocab_size,
                A=self.action_dim,
            )
        else:
            logits = output
        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        if self.predict_offsets:
            sampled_offsets = offsets[
                torch.arange(offsets.shape[0]), sampled_data.flatten()
            ].view(batch, seq, self.action_dim)

            return (sampled_data, sampled_offsets)
        else:
            return sampled_data

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        trainer_cfg = mingpt_trainer.TrainerConfig(
            weight_decay=weight_decay, learning_rate=learning_rate, betas=betas
        )
        return self.model.configure_optimizers(trainer_cfg)

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/latent_generators/transformer.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import diffusion_policy.model.bet.latent_generators.latent_generator as latent_generator

from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.model.bet.libraries.loss_fn import FocalLoss, soft_cross_entropy

from typing import Optional, Tuple

class Transformer(latent_generator.AbstractLatentGenerator):
    def __init__(
        self,
        input_dim: int,
        num_bins: int,
        action_dim: int,
        horizon: int,
        focal_loss_gamma: float,
        offset_loss_scale: float,
        **kwargs
    ):
        super().__init__()
        self.model = TransformerForDiffusion(
            input_dim=input_dim,
            output_dim=num_bins * (1 + action_dim),
            horizon=horizon,
            **kwargs
        )
        self.vocab_size = num_bins
        self.focal_loss_gamma = focal_loss_gamma
        self.offset_loss_scale = offset_loss_scale
        self.action_dim = action_dim
    
    def get_optimizer(self, **kwargs) -> torch.optim.Optimizer:
        return self.model.configure_optimizers(**kwargs)
    
    def get_latent_and_loss(self, 
            obs_rep: torch.Tensor, 
            target_latents: torch.Tensor, 
            return_loss_components=True,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
        target_latents, target_offsets = target_latents
        target_latents = target_latents.view(-1)
        criterion = FocalLoss(gamma=self.focal_loss_gamma)

        t = torch.tensor(0, device=self.model.device)
        output = self.model(obs_rep, t)
        logits = output[:, :, : self.vocab_size]
        offsets = output[:, :, self.vocab_size :]
        batch = logits.shape[0]
        seq = logits.shape[1]
        offsets = einops.rearrange(
            offsets,
            "N T (V A) -> (N T) V A",  # N = batch, T = seq
            V=self.vocab_size,
            A=self.action_dim,
        )
        # calculate (optionally soft) cross entropy and offset losses
        class_loss = criterion(logits.view(-1, logits.size(-1)), target_latents)
        # offset loss is only calculated on the target class
        # if soft targets, argmax is considered the target class
        selected_offsets = offsets[
            torch.arange(offsets.size(0)),
            target_latents.view(-1),
        ]
        offset_loss = self.offset_loss_scale * F.mse_loss(
            selected_offsets, target_offsets.view(-1, self.action_dim)
        )
        loss = offset_loss + class_loss
        logits = einops.rearrange(logits, "batch seq classes -> seq batch classes")
        offsets = einops.rearrange(
            offsets,
            "(N T) V A -> T N V A",  # ? N, T order? Anyway does not affect loss and training (might affect visualization)
            N=batch,
            T=seq,
        )
        return (
            (logits, offsets),
            loss,
            {"offset": offset_loss, "class": class_loss, "total": loss},
        )

    def generate_latents(
        self, obs_rep: torch.Tensor
    ) -> torch.Tensor:
        t = torch.tensor(0, device=self.model.device)
        output = self.model(obs_rep, t)
        logits = output[:, :, : self.vocab_size]
        offsets = output[:, :, self.vocab_size :]
        offsets = einops.rearrange(
            offsets,
            "N T (V A) -> (N T) V A",  # N = batch, T = seq
            V=self.vocab_size,
            A=self.action_dim,
        )

        probs = F.softmax(logits, dim=-1)
        batch, seq, choices = probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_data = torch.multinomial(probs.view(-1, choices), num_samples=1)
        sampled_data = einops.rearrange(
            sampled_data, "(batch seq) 1 -> batch seq 1", batch=batch, seq=seq
        )
        sampled_offsets = offsets[
            torch.arange(offsets.shape[0]), sampled_data.flatten()
        ].view(batch, seq, self.action_dim)
        return (sampled_data, sampled_offsets)

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/latent_generators/latent_generator.py
```python
import abc
import torch
from typing import Tuple, Optional

import diffusion_policy.model.bet.utils as utils


class AbstractLatentGenerator(abc.ABC, utils.SaveModule):
    """
    Abstract class for a generative model that can generate latents given observation representations.

    In the probabilisitc sense, this model fits and samples from P(latent|observation) given some observation.
    """

    @abc.abstractmethod
    def get_latent_and_loss(
        self,
        obs_rep: torch.Tensor,
        target_latents: torch.Tensor,
        seq_masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a set of observation representation and generated latents, get the encoded latent and the loss.

        Inputs:
        input_action: Batch of the actions taken in the multimodal demonstrations.
        target_latents: Batch of the latents that the generator should learn to generate the actions from.
        seq_masks: Batch of masks that indicate which timesteps are valid.

        Outputs:
        latent: The sampled latent from the observation.
        loss: The loss of the latent generator.
        """
        pass

    @abc.abstractmethod
    def generate_latents(
        self, seq_obses: torch.Tensor, seq_masks: torch.Tensor
    ) -> torch.Tensor:
        """
        Given a batch of sequences of observations, generate a batch of sequences of latents.

        Inputs:
        seq_obses: Batch of sequences of observations, of shape seq x batch x dim, following the transformer convention.
        seq_masks: Batch of sequences of masks, of shape seq x batch, following the transformer convention.

        Outputs:
        seq_latents: Batch of sequences of latents of shape seq x batch x latent_dim.
        """
        pass

    def get_optimizer(
        self, weight_decay: float, learning_rate: float, betas: Tuple[float, float]
    ) -> torch.optim.Optimizer:
        """
        Default optimizer class. Override this if you want to use a different optimizer.
        """
        return torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas
        )


class LatentGeneratorDataParallel(torch.nn.DataParallel):
    def get_latent_and_loss(self, *args, **kwargs):
        return self.module.get_latent_and_loss(*args, **kwargs)  # type: ignore

    def generate_latents(self, *args, **kwargs):
        return self.module.generate_latents(*args, **kwargs)  # type: ignore

    def get_optimizer(self, *args, **kwargs):
        return self.module.get_optimizer(*args, **kwargs)  # type: ignore

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/utils.py
```python
import os
import random
from collections import OrderedDict
from typing import List, Optional

import einops
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import random_split
import wandb


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


class eval_mode:
    def __init__(self, *models, no_grad=False):
        self.models = models
        self.no_grad = no_grad
        self.no_grad_context = torch.no_grad()

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)
        if self.no_grad:
            self.no_grad_context.__enter__()

    def __exit__(self, *args):
        if self.no_grad:
            self.no_grad_context.__exit__(*args)
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def freeze_module(module: nn.Module) -> nn.Module:
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    return module


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def shuffle_along_axis(a, axis):
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a, idx, axis=axis)


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)


class TrainWithLogger:
    def reset_log(self):
        self.log_components = OrderedDict()

    def log_append(self, log_key, length, loss_components):
        for key, value in loss_components.items():
            key_name = f"{log_key}/{key}"
            count, sum = self.log_components.get(key_name, (0, 0.0))
            self.log_components[key_name] = (
                count + length,
                sum + (length * value.detach().cpu().item()),
            )

    def flush_log(self, epoch, iterator=None):
        log_components = OrderedDict()
        iterator_log_component = OrderedDict()
        for key, value in self.log_components.items():
            count, sum = value
            to_log = sum / count
            log_components[key] = to_log
            # Set the iterator status
            log_key, name_key = key.split("/")
            iterator_log_name = f"{log_key[0]}{name_key[0]}".upper()
            iterator_log_component[iterator_log_name] = to_log
        postfix = ",".join(
            "{}:{:.2e}".format(key, iterator_log_component[key])
            for key in iterator_log_component.keys()
        )
        if iterator is not None:
            iterator.set_postfix_str(postfix)
        wandb.log(log_components, step=epoch)
        self.log_components = OrderedDict()


class SaveModule(nn.Module):
    def set_snapshot_path(self, path):
        self.snapshot_path = path
        print(f"Setting snapshot path to {self.snapshot_path}")

    def save_snapshot(self):
        os.makedirs(self.snapshot_path, exist_ok=True)
        torch.save(self.state_dict(), self.snapshot_path / "snapshot.pth")

    def load_snapshot(self):
        self.load_state_dict(torch.load(self.snapshot_path / "snapshot.pth"))


def split_datasets(dataset, train_fraction=0.95, random_seed=42):
    dataset_length = len(dataset)
    lengths = [
        int(train_fraction * dataset_length),
        dataset_length - int(train_fraction * dataset_length),
    ]
    train_set, val_set = random_split(
        dataset, lengths, generator=torch.Generator().manual_seed(random_seed)
    )
    return train_set, val_set

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/libraries/loss_fn.py
```python
from typing import Optional, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

# Reference: https://github.com/pytorch/pytorch/issues/11959
def soft_cross_entropy(
    input: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        input: (batch_size, num_classes): tensor of raw logits
        target: (batch_size, num_classes): tensor of class probability; sum(target) == 1

    Returns:
        loss: (batch_size,)
    """
    log_probs = torch.log_softmax(input, dim=-1)
    # target is a distribution
    loss = F.kl_div(log_probs, target, reduction="batchmean")
    return loss


# Focal loss implementation
# Source: https://github.com/AdeelH/pytorch-multi-class-focal-loss/blob/master/focal_loss.py
# MIT License
#
# Copyright (c) 2020 Adeel Hassan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
class FocalLoss(nn.Module):
    """Focal Loss, as described in https://arxiv.org/abs/1708.02002.
    It is essentially an enhancement to cross entropy loss and is
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.
    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    """

    def __init__(
        self,
        alpha: Optional[Tensor] = None,
        gamma: float = 0.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        """Constructor.
        Args:
            alpha (Tensor, optional): Weights for each class. Defaults to None.
            gamma (float, optional): A constant, as described in the paper.
                Defaults to 0.
            reduction (str, optional): 'mean', 'sum' or 'none'.
                Defaults to 'mean'.
            ignore_index (int, optional): class label to ignore.
                Defaults to -100.
        """
        if reduction not in ("mean", "sum", "none"):
            raise ValueError('Reduction must be one of: "mean", "sum", "none".')

        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction

        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction="none", ignore_index=ignore_index
        )

    def __repr__(self):
        arg_keys = ["alpha", "gamma", "ignore_index", "reduction"]
        arg_vals = [self.__dict__[k] for k in arg_keys]
        arg_strs = [f"{k}={v}" for k, v in zip(arg_keys, arg_vals)]
        arg_str = ", ".join(arg_strs)
        return f"{type(self).__name__}({arg_str})"

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if x.ndim > 2:
            # (N, C, d1, d2, ..., dK) --> (N * d1 * ... * dK, C)
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            # (N, d1, d2, ..., dK) --> (N * d1 * ... * dK,)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return 0.0
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt) ** self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def focal_loss(
    alpha: Optional[Sequence] = None,
    gamma: float = 0.0,
    reduction: str = "mean",
    ignore_index: int = -100,
    device="cpu",
    dtype=torch.float32,
) -> FocalLoss:
    """Factory function for FocalLoss.
    Args:
        alpha (Sequence, optional): Weights for each class. Will be converted
            to a Tensor if not None. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 0.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
        device (str, optional): Device to move alpha to. Defaults to 'cpu'.
        dtype (torch.dtype, optional): dtype to cast alpha to.
            Defaults to torch.float32.
    Returns:
        A FocalLoss object
    """
    if alpha is not None:
        if not isinstance(alpha, Tensor):
            alpha = torch.tensor(alpha)
        alpha = alpha.to(device=device, dtype=dtype)

    fl = FocalLoss(
        alpha=alpha, gamma=gamma, reduction=reduction, ignore_index=ignore_index
    )
    return fl

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/libraries/mingpt/utils.py
```python
import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float("Inf")
    return out


@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = (
            x if x.size(1) <= block_size else x[:, -block_size:]
        )  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/libraries/mingpt/__init__.py
```python

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/libraries/mingpt/trainer.py
```python
"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 0  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        def run_epoch(loader, is_train):
            model.train(is_train)

            losses = []
            pbar = (
                tqdm(enumerate(loader), total=len(loader))
                if is_train
                else enumerate(loader)
            )
            for it, (x, y) in pbar:

                # place data on the correct device
                x = x.to(self.device)
                y = y.to(self.device)

                # forward the model
                with torch.set_grad_enabled(is_train):
                    logits, loss = model(x, y)
                    loss = (
                        loss.mean()
                    )  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                if is_train:

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), config.grad_norm_clip
                    )
                    optimizer.step()

                    # decay the learning rate based on our progress
                    if config.lr_decay:
                        self.tokens += (
                            y >= 0
                        ).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(
                                max(1, config.warmup_tokens)
                            )
                        else:
                            # cosine learning rate decay
                            progress = float(
                                self.tokens - config.warmup_tokens
                            ) / float(
                                max(1, config.final_tokens - config.warmup_tokens)
                            )
                            lr_mult = max(
                                0.1, 0.5 * (1.0 + math.cos(math.pi * progress))
                            )
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = lr
                    else:
                        lr = config.learning_rate

                    # report progress
                    pbar.set_description(  # type: ignore
                        f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}"
                    )

            if not is_train:
                test_loss = float(np.mean(losses))
                logger.info("test loss: %f", test_loss)
                return test_loss

        best_loss = float("inf")
        self.tokens = 0  # counter used for learning rate decay

        train_loader = DataLoader(
            self.train_dataset,
            shuffle=True,
            pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )
        if self.test_dataset is not None:
            test_loader = DataLoader(
                self.test_dataset,
                shuffle=True,
                pin_memory=True,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
            )

        for epoch in range(config.max_epochs):
            run_epoch(train_loader, is_train=True)
            if self.test_dataset is not None:
                test_loss = run_epoch(test_loader, is_train=False)

            # supports early stopping based on the test loss, or just save always if no test set is provided
            good_model = self.test_dataset is None or test_loss < best_loss
            if self.config.ckpt_path is not None and good_model:
                best_loss = test_loss
                self.save_checkpoint()

```

## reference_material/diffusion_policy_code/diffusion_policy/model/bet/libraries/mingpt/model.py
```python
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class GPTConfig:
    """base GPT config, params common to all GPT versions"""

    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    discrete_input = False
    input_size = 10
    n_embd = 768
    n_layer = 12

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """GPT-1 like network roughly 125M params"""

    n_layer = 12
    n_head = 12
    n_embd = 768


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )
        self.n_head = config.n_head

    def forward(self, x):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """the full GPT language model, with a context size of block_size"""

    def __init__(self, config: GPTConfig):
        super().__init__()

        # input embedding stem
        if config.discrete_input:
            self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        else:
            self.tok_emb = nn.Linear(config.input_size, config.n_embd)
        self.discrete_input = config.discrete_input
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.block_size = config.block_size
        self.apply(self._init_weights)

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GPT):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(
            optim_groups, lr=train_config.learning_rate, betas=train_config.betas
        )
        return optimizer

    def forward(self, idx, targets=None):
        if self.discrete_input:
            b, t = idx.size()
        else:
            b, t, dim = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        # forward the GPT model
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[
            :, :t, :
        ]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss

```

## reference_material/diffusion_policy_code/diffusion_policy/model/common/module_attr_mixin.py
```python
import torch.nn as nn

class ModuleAttrMixin(nn.Module):
    def __init__(self):
        super().__init__()
        self._dummy_variable = nn.Parameter()

    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

```

## reference_material/diffusion_policy_code/diffusion_policy/model/common/dict_of_tensor_mixin.py
```python
import torch
import torch.nn as nn

class DictOfTensorMixin(nn.Module):
    def __init__(self, params_dict=None):
        super().__init__()
        if params_dict is None:
            params_dict = nn.ParameterDict()
        self.params_dict = params_dict

    @property
    def device(self):
        return next(iter(self.parameters())).device

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        def dfs_add(dest, keys, value: torch.Tensor):
            if len(keys) == 1:
                dest[keys[0]] = value
                return

            if keys[0] not in dest:
                dest[keys[0]] = nn.ParameterDict()
            dfs_add(dest[keys[0]], keys[1:], value)

        def load_dict(state_dict, prefix):
            out_dict = nn.ParameterDict()
            for key, value in state_dict.items():
                value: torch.Tensor
                if key.startswith(prefix):
                    param_keys = key[len(prefix):].split('.')[1:]
                    # if len(param_keys) == 0:
                    #     import pdb; pdb.set_trace()
                    dfs_add(out_dict, param_keys, value.clone())
            return out_dict

        self.params_dict = load_dict(state_dict, prefix + 'params_dict')
        self.params_dict.requires_grad_(False)
        return 

```

## reference_material/diffusion_policy_code/diffusion_policy/model/common/shape_util.py
```python
from typing import Dict, List, Tuple, Callable
import torch
import torch.nn as nn

def get_module_device(m: nn.Module):
    device = torch.device('cpu')
    try:
        param = next(iter(m.parameters()))
        device = param.device
    except StopIteration:
        pass
    return device

@torch.no_grad()
def get_output_shape(
        input_shape: Tuple[int],
        net: Callable[[torch.Tensor], torch.Tensor]
    ):  
        device = get_module_device(net)
        test_input = torch.zeros((1,)+tuple(input_shape), device=device)
        test_output = net(test_input)
        output_shape = tuple(test_output.shape[1:])
        return output_shape

```

## reference_material/diffusion_policy_code/diffusion_policy/model/common/normalizer.py
```python
from typing import Union, Dict

import unittest
import zarr
import numpy as np
import torch
import torch.nn as nn
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.model.common.dict_of_tensor_mixin import DictOfTensorMixin


class LinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
        data: Union[Dict, torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
        if isinstance(data, dict):
            for key, value in data.items():
                self.params_dict[key] =  _fit(value, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
        else:
            self.params_dict['_default'] = _fit(data, 
                    last_n_dims=last_n_dims,
                    dtype=dtype,
                    mode=mode,
                    output_max=output_max,
                    output_min=output_min,
                    range_eps=range_eps,
                    fit_offset=fit_offset)
    
    def __call__(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)
    
    def __getitem__(self, key: str):
        return SingleFieldLinearNormalizer(self.params_dict[key])

    def __setitem__(self, key: str , value: 'SingleFieldLinearNormalizer'):
        self.params_dict[key] = value.params_dict

    def _normalize_impl(self, x, forward=True):
        if isinstance(x, dict):
            result = dict()
            for key, value in x.items():
                params = self.params_dict[key]
                result[key] = _normalize(value, params, forward=forward)
            return result
        else:
            if '_default' not in self.params_dict:
                raise RuntimeError("Not initialized")
            params = self.params_dict['_default']
            return _normalize(x, params, forward=forward)

    def normalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=True)

    def unnormalize(self, x: Union[Dict, torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self._normalize_impl(x, forward=False)

    def get_input_stats(self) -> Dict:
        if len(self.params_dict) == 0:
            raise RuntimeError("Not initialized")
        if len(self.params_dict) == 1 and '_default' in self.params_dict:
            return self.params_dict['_default']['input_stats']
        
        result = dict()
        for key, value in self.params_dict.items():
            if key != '_default':
                result[key] = value['input_stats']
        return result


    def get_output_stats(self, key='_default'):
        input_stats = self.get_input_stats()
        if 'min' in input_stats:
            # no dict
            return dict_apply(input_stats, self.normalize)
        
        result = dict()
        for key, group in input_stats.items():
            this_dict = dict()
            for name, value in group.items():
                this_dict[name] = self.normalize({key:value})[key]
            result[key] = this_dict
        return result


class SingleFieldLinearNormalizer(DictOfTensorMixin):
    avaliable_modes = ['limits', 'gaussian']
    
    @torch.no_grad()
    def fit(self,
            data: Union[torch.Tensor, np.ndarray, zarr.Array],
            last_n_dims=1,
            dtype=torch.float32,
            mode='limits',
            output_max=1.,
            output_min=-1.,
            range_eps=1e-4,
            fit_offset=True):
        self.params_dict = _fit(data, 
            last_n_dims=last_n_dims,
            dtype=dtype,
            mode=mode,
            output_max=output_max,
            output_min=output_min,
            range_eps=range_eps,
            fit_offset=fit_offset)
    
    @classmethod
    def create_fit(cls, data: Union[torch.Tensor, np.ndarray, zarr.Array], **kwargs):
        obj = cls()
        obj.fit(data, **kwargs)
        return obj
    
    @classmethod
    def create_manual(cls, 
            scale: Union[torch.Tensor, np.ndarray], 
            offset: Union[torch.Tensor, np.ndarray],
            input_stats_dict: Dict[str, Union[torch.Tensor, np.ndarray]]):
        def to_tensor(x):
            if not isinstance(x, torch.Tensor):
                x = torch.from_numpy(x)
            x = x.flatten()
            return x
        
        # check
        for x in [offset] + list(input_stats_dict.values()):
            assert x.shape == scale.shape
            assert x.dtype == scale.dtype
        
        params_dict = nn.ParameterDict({
            'scale': to_tensor(scale),
            'offset': to_tensor(offset),
            'input_stats': nn.ParameterDict(
                dict_apply(input_stats_dict, to_tensor))
        })
        return cls(params_dict)

    @classmethod
    def create_identity(cls, dtype=torch.float32):
        scale = torch.tensor([1], dtype=dtype)
        offset = torch.tensor([0], dtype=dtype)
        input_stats_dict = {
            'min': torch.tensor([-1], dtype=dtype),
            'max': torch.tensor([1], dtype=dtype),
            'mean': torch.tensor([0], dtype=dtype),
            'std': torch.tensor([1], dtype=dtype)
        }
        return cls.create_manual(scale, offset, input_stats_dict)

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=True)

    def unnormalize(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return _normalize(x, self.params_dict, forward=False)

    def get_input_stats(self):
        return self.params_dict['input_stats']

    def get_output_stats(self):
        return dict_apply(self.params_dict['input_stats'], self.normalize)

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.normalize(x)



def _fit(data: Union[torch.Tensor, np.ndarray, zarr.Array],
        last_n_dims=1,
        dtype=torch.float32,
        mode='limits',
        output_max=1.,
        output_min=-1.,
        range_eps=1e-4,
        fit_offset=True):
    assert mode in ['limits', 'gaussian']
    assert last_n_dims >= 0
    assert output_max > output_min

    # convert data to torch and type
    if isinstance(data, zarr.Array):
        data = data[:]
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if dtype is not None:
        data = data.type(dtype)

    # convert shape
    dim = 1
    if last_n_dims > 0:
        dim = np.prod(data.shape[-last_n_dims:])
    data = data.reshape(-1,dim)

    # compute input stats min max mean std
    input_min, _ = data.min(axis=0)
    input_max, _ = data.max(axis=0)
    input_mean = data.mean(axis=0)
    input_std = data.std(axis=0)

    # compute scale and offset
    if mode == 'limits':
        if fit_offset:
            # unit scale
            input_range = input_max - input_min
            ignore_dim = input_range < range_eps
            input_range[ignore_dim] = output_max - output_min
            scale = (output_max - output_min) / input_range
            offset = output_min - scale * input_min
            offset[ignore_dim] = (output_max + output_min) / 2 - input_min[ignore_dim]
            # ignore dims scaled to mean of output max and min
        else:
            # use this when data is pre-zero-centered.
            assert output_max > 0
            assert output_min < 0
            # unit abs
            output_abs = min(abs(output_min), abs(output_max))
            input_abs = torch.maximum(torch.abs(input_min), torch.abs(input_max))
            ignore_dim = input_abs < range_eps
            input_abs[ignore_dim] = output_abs
            # don't scale constant channels 
            scale = output_abs / input_abs
            offset = torch.zeros_like(input_mean)
    elif mode == 'gaussian':
        ignore_dim = input_std < range_eps
        scale = input_std.clone()
        scale[ignore_dim] = 1
        scale = 1 / scale

        if fit_offset:
            offset = - input_mean * scale
        else:
            offset = torch.zeros_like(input_mean)
    
    # save
    this_params = nn.ParameterDict({
        'scale': scale,
        'offset': offset,
        'input_stats': nn.ParameterDict({
            'min': input_min,
            'max': input_max,
            'mean': input_mean,
            'std': input_std
        })
    })
    for p in this_params.parameters():
        p.requires_grad_(False)
    return this_params


def _normalize(x, params, forward=True):
    assert 'scale' in params
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    scale = params['scale']
    offset = params['offset']
    x = x.to(device=scale.device, dtype=scale.dtype)
    src_shape = x.shape
    x = x.reshape(-1, scale.shape[0])
    if forward:
        x = x * scale + offset
    else:
        x = (x - offset) / scale
    x = x.reshape(src_shape)
    return x


def test():
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=1, fit_offset=False)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1., atol=1e-3)
    assert np.allclose(datan.min(), 0., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    data = torch.zeros((100,10,9,2)).uniform_()
    normalizer = SingleFieldLinearNormalizer()
    normalizer.fit(data, mode='gaussian', last_n_dims=0)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.mean(), 0., atol=1e-3)
    assert np.allclose(datan.std(), 1., atol=1e-3)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)


    # dict
    data = torch.zeros((100,10,9,2)).uniform_()
    data[...,0,0] = 0

    normalizer = LinearNormalizer()
    normalizer.fit(data, mode='limits', last_n_dims=2)
    datan = normalizer.normalize(data)
    assert datan.shape == data.shape
    assert np.allclose(datan.max(), 1.)
    assert np.allclose(datan.min(), -1.)
    dataun = normalizer.unnormalize(datan)
    assert torch.allclose(data, dataun, atol=1e-7)

    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    data = {
        'obs': torch.zeros((1000,128,9,2)).uniform_() * 512,
        'action': torch.zeros((1000,128,2)).uniform_() * 512
    }
    normalizer = LinearNormalizer()
    normalizer.fit(data)
    datan = normalizer.normalize(data)
    dataun = normalizer.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)
    
    input_stats = normalizer.get_input_stats()
    output_stats = normalizer.get_output_stats()

    state_dict = normalizer.state_dict()
    n = LinearNormalizer()
    n.load_state_dict(state_dict)
    datan = n.normalize(data)
    dataun = n.unnormalize(datan)
    for key in data:
        assert torch.allclose(data[key], dataun[key], atol=1e-4)

```

## reference_material/diffusion_policy_code/diffusion_policy/model/common/tensor_util.py
```python
"""
A collection of utilities for working with nested tensor structures consisting
of numpy arrays and torch tensors.
"""
import collections
import numpy as np
import torch


def recursive_dict_list_tuple_apply(x, type_func_dict):
    """
    Recursively apply functions to a nested dictionary or list or tuple, given a dictionary of 
    {data_type: function_to_apply}.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        type_func_dict (dict): a mapping from data types to the functions to be 
            applied for each data type.

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    assert(list not in type_func_dict)
    assert(tuple not in type_func_dict)
    assert(dict not in type_func_dict)

    if isinstance(x, (dict, collections.OrderedDict)):
        new_x = collections.OrderedDict() if isinstance(x, collections.OrderedDict) else dict()
        for k, v in x.items():
            new_x[k] = recursive_dict_list_tuple_apply(v, type_func_dict)
        return new_x
    elif isinstance(x, (list, tuple)):
        ret = [recursive_dict_list_tuple_apply(v, type_func_dict) for v in x]
        if isinstance(x, tuple):
            ret = tuple(ret)
        return ret
    else:
        for t, f in type_func_dict.items():
            if isinstance(x, t):
                return f(x)
        else:
            raise NotImplementedError(
                'Cannot handle data type %s' % str(type(x)))


def map_tensor(x, func):
    """
    Apply function @func to torch.Tensor objects in a nested dictionary or
    list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        func (function): function to apply to each tensor

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: func,
            type(None): lambda x: x,
        }
    )


def map_ndarray(x, func):
    """
    Apply function @func to np.ndarray objects in a nested dictionary or
    list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        func (function): function to apply to each array

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            np.ndarray: func,
            type(None): lambda x: x,
        }
    )


def map_tensor_ndarray(x, tensor_func, ndarray_func):
    """
    Apply function @tensor_func to torch.Tensor objects and @ndarray_func to 
    np.ndarray objects in a nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        tensor_func (function): function to apply to each tensor
        ndarray_Func (function): function to apply to each array

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: tensor_func,
            np.ndarray: ndarray_func,
            type(None): lambda x: x,
        }
    )


def clone(x):
    """
    Clones all torch tensors and numpy arrays in nested dictionary or list
    or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.clone(),
            np.ndarray: lambda x: x.copy(),
            type(None): lambda x: x,
        }
    )


def detach(x):
    """
    Detaches all torch tensors in nested dictionary or list
    or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.detach(),
        }
    )


def to_batch(x):
    """
    Introduces a leading batch dimension of 1 for all torch tensors and numpy 
    arrays in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x[None, ...],
            np.ndarray: lambda x: x[None, ...],
            type(None): lambda x: x,
        }
    )


def to_sequence(x):
    """
    Introduces a time dimension of 1 at dimension 1 for all torch tensors and numpy 
    arrays in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x[:, None, ...],
            np.ndarray: lambda x: x[:, None, ...],
            type(None): lambda x: x,
        }
    )


def index_at_time(x, ind):
    """
    Indexes all torch tensors and numpy arrays in dimension 1 with index @ind in
    nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        ind (int): index

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x[:, ind, ...],
            np.ndarray: lambda x: x[:, ind, ...],
            type(None): lambda x: x,
        }
    )


def unsqueeze(x, dim):
    """
    Adds dimension of size 1 at dimension @dim in all torch tensors and numpy arrays
    in nested dictionary or list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        dim (int): dimension

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.unsqueeze(dim=dim),
            np.ndarray: lambda x: np.expand_dims(x, axis=dim),
            type(None): lambda x: x,
        }
    )


def contiguous(x):
    """
    Makes all torch tensors and numpy arrays contiguous in nested dictionary or 
    list or tuple and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.contiguous(),
            np.ndarray: lambda x: np.ascontiguousarray(x),
            type(None): lambda x: x,
        }
    )


def to_device(x, device):
    """
    Sends all torch tensors in nested dictionary or list or tuple to device
    @device, and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        device (torch.Device): device to send tensors to

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, d=device: x.to(d),
            type(None): lambda x: x,
        }
    )


def to_tensor(x):
    """
    Converts all numpy arrays in nested dictionary or list or tuple to
    torch tensors (and leaves existing torch Tensors as-is), and returns 
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x,
            np.ndarray: lambda x: torch.from_numpy(x),
            type(None): lambda x: x,
        }
    )


def to_numpy(x):
    """
    Converts all torch tensors in nested dictionary or list or tuple to
    numpy (and leaves existing numpy arrays as-is), and returns 
    a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    def f(tensor):
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy()
        else:
            return tensor.detach().numpy()
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: f,
            np.ndarray: lambda x: x,
            type(None): lambda x: x,
        }
    )


def to_list(x):
    """
    Converts all torch tensors and numpy arrays in nested dictionary or list 
    or tuple to a list, and returns a new nested structure. Useful for
    json encoding.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    def f(tensor):
        if tensor.is_cuda:
            return tensor.detach().cpu().numpy().tolist()
        else:
            return tensor.detach().numpy().tolist()
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: f,
            np.ndarray: lambda x: x.tolist(),
            type(None): lambda x: x,
        }
    )


def to_float(x):
    """
    Converts all torch tensors and numpy arrays in nested dictionary or list 
    or tuple to float type entries, and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.float(),
            np.ndarray: lambda x: x.astype(np.float32),
            type(None): lambda x: x,
        }
    )


def to_uint8(x):
    """
    Converts all torch tensors and numpy arrays in nested dictionary or list 
    or tuple to uint8 type entries, and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.byte(),
            np.ndarray: lambda x: x.astype(np.uint8),
            type(None): lambda x: x,
        }
    )


def to_torch(x, device):
    """
    Converts all numpy arrays and torch tensors in nested dictionary or list or tuple to 
    torch tensors on device @device and returns a new nested structure.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        device (torch.Device): device to send tensors to

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return to_device(to_float(to_tensor(x)), device)


def to_one_hot_single(tensor, num_class):
    """
    Convert tensor to one-hot representation, assuming a certain number of total class labels.

    Args:
        tensor (torch.Tensor): tensor containing integer labels
        num_class (int): number of classes

    Returns:
        x (torch.Tensor): tensor containing one-hot representation of labels
    """
    x = torch.zeros(tensor.size() + (num_class,)).to(tensor.device)
    x.scatter_(-1, tensor.unsqueeze(-1), 1)
    return x


def to_one_hot(tensor, num_class):
    """
    Convert all tensors in nested dictionary or list or tuple to one-hot representation, 
    assuming a certain number of total class labels.

    Args:
        tensor (dict or list or tuple): a possibly nested dictionary or list or tuple
        num_class (int): number of classes

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(tensor, func=lambda x, nc=num_class: to_one_hot_single(x, nc))


def flatten_single(x, begin_axis=1):
    """
    Flatten a tensor in all dimensions from @begin_axis onwards.

    Args:
        x (torch.Tensor): tensor to flatten
        begin_axis (int): which axis to flatten from

    Returns:
        y (torch.Tensor): flattened tensor
    """
    fixed_size = x.size()[:begin_axis]
    _s = list(fixed_size) + [-1]
    return x.reshape(*_s)


def flatten(x, begin_axis=1):
    """
    Flatten all tensors in nested dictionary or list or tuple, from @begin_axis onwards.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): which axis to flatten from

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis: flatten_single(x, begin_axis=b),
        }
    )


def reshape_dimensions_single(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions in a tensor to a target dimension.

    Args:
        x (torch.Tensor): tensor to reshape
        begin_axis (int): begin dimension
        end_axis (int): end dimension
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (torch.Tensor): reshaped tensor
    """
    assert(begin_axis <= end_axis)
    assert(begin_axis >= 0)
    assert(end_axis < len(x.shape))
    assert(isinstance(target_dims, (tuple, list)))
    s = x.shape
    final_s = []
    for i in range(len(s)):
        if i == begin_axis:
            final_s.extend(target_dims)
        elif i < begin_axis or i > end_axis:
            final_s.append(s[i])
    return x.reshape(*final_s)


def reshape_dimensions(x, begin_axis, end_axis, target_dims):
    """
    Reshape selected dimensions for all tensors in nested dictionary or list or tuple 
    to a target dimension.
    
    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension
        target_dims (tuple or list): target shape for the range of dimensions
            (@begin_axis, @end_axis)

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=t),
            np.ndarray: lambda x, b=begin_axis, e=end_axis, t=target_dims: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=t),
            type(None): lambda x: x,
        }
    )


def join_dimensions(x, begin_axis, end_axis):
    """
    Joins all dimensions between dimensions (@begin_axis, @end_axis) into a flat dimension, for
    all tensors in nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        begin_axis (int): begin dimension
        end_axis (int): end dimension

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]),
            np.ndarray: lambda x, b=begin_axis, e=end_axis: reshape_dimensions_single(
                x, begin_axis=b, end_axis=e, target_dims=[-1]),
            type(None): lambda x: x,
        }
    )


def expand_at_single(x, size, dim):
    """
    Expand a tensor at a single dimension @dim by @size

    Args:
        x (torch.Tensor): input tensor
        size (int): size to expand
        dim (int): dimension to expand

    Returns:
        y (torch.Tensor): expanded tensor
    """
    assert dim < x.ndimension()
    assert x.shape[dim] == 1
    expand_dims = [-1] * x.ndimension()
    expand_dims[dim] = size
    return x.expand(*expand_dims)


def expand_at(x, size, dim):
    """
    Expand all tensors in nested dictionary or list or tuple at a single
    dimension @dim by @size.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        size (int): size to expand
        dim (int): dimension to expand

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(x, lambda t, s=size, d=dim: expand_at_single(t, s, d))


def unsqueeze_expand_at(x, size, dim):
    """
    Unsqueeze and expand a tensor at a dimension @dim by @size.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        size (int): size to expand
        dim (int): dimension to unsqueeze and expand

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    x = unsqueeze(x, dim)
    return expand_at(x, size, dim)


def repeat_by_expand_at(x, repeats, dim):
    """
    Repeat a dimension by combining expand and reshape operations.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        repeats (int): number of times to repeat the target dimension
        dim (int): dimension to repeat on

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    x = unsqueeze_expand_at(x, repeats, dim + 1)
    return join_dimensions(x, dim, dim + 1)


def named_reduce_single(x, reduction, dim):
    """
    Reduce tensor at a dimension by named reduction functions.

    Args:
        x (torch.Tensor): tensor to be reduced
        reduction (str): one of ["sum", "max", "mean", "flatten"]
        dim (int): dimension to be reduced (or begin axis for flatten)

    Returns:
        y (torch.Tensor): reduced tensor
    """
    assert x.ndimension() > dim
    assert reduction in ["sum", "max", "mean", "flatten"]
    if reduction == "flatten":
        x = flatten(x, begin_axis=dim)
    elif reduction == "max":
        x = torch.max(x, dim=dim)[0]  # [B, D]
    elif reduction == "sum":
        x = torch.sum(x, dim=dim)
    else:
        x = torch.mean(x, dim=dim)
    return x


def named_reduce(x, reduction, dim):
    """
    Reduces all tensors in nested dictionary or list or tuple at a dimension
    using a named reduction function.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        reduction (str): one of ["sum", "max", "mean", "flatten"]
        dim (int): dimension to be reduced (or begin axis for flatten)

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(x, func=lambda t, r=reduction, d=dim: named_reduce_single(t, r, d))


def gather_along_dim_with_dim_single(x, target_dim, source_dim, indices):
    """
    This function indexes out a target dimension of a tensor in a structured way,
    by allowing a different value to be selected for each member of a flat index 
    tensor (@indices) corresponding to a source dimension. This can be interpreted
    as moving along the source dimension, using the corresponding index value
    in @indices to select values for all other dimensions outside of the
    source and target dimensions. A common use case is to gather values
    in target dimension 1 for each batch member (target dimension 0).

    Args:
        x (torch.Tensor): tensor to gather values for
        target_dim (int): dimension to gather values along
        source_dim (int): dimension to hold constant and use for gathering values
            from the other dimensions
        indices (torch.Tensor): flat index tensor with same shape as tensor @x along
            @source_dim
    
    Returns:
        y (torch.Tensor): gathered tensor, with dimension @target_dim indexed out
    """
    assert len(indices.shape) == 1
    assert x.shape[source_dim] == indices.shape[0]

    # unsqueeze in all dimensions except the source dimension
    new_shape = [1] * x.ndimension()
    new_shape[source_dim] = -1
    indices = indices.reshape(*new_shape)

    # repeat in all dimensions - but preserve shape of source dimension,
    # and make sure target_dimension has singleton dimension
    expand_shape = list(x.shape)
    expand_shape[source_dim] = -1
    expand_shape[target_dim] = 1
    indices = indices.expand(*expand_shape)

    out = x.gather(dim=target_dim, index=indices)
    return out.squeeze(target_dim)


def gather_along_dim_with_dim(x, target_dim, source_dim, indices):
    """
    Apply @gather_along_dim_with_dim_single to all tensors in a nested 
    dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        target_dim (int): dimension to gather values along
        source_dim (int): dimension to hold constant and use for gathering values
            from the other dimensions
        indices (torch.Tensor): flat index tensor with same shape as tensor @x along
            @source_dim

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple
    """
    return map_tensor(x, 
        lambda y, t=target_dim, s=source_dim, i=indices: gather_along_dim_with_dim_single(y, t, s, i))
    

def gather_sequence_single(seq, indices):
    """
    Given a tensor with leading dimensions [B, T, ...], gather an element from each sequence in 
    the batch given an index for each sequence.

    Args:
        seq (torch.Tensor): tensor with leading dimensions [B, T, ...]
        indices (torch.Tensor): tensor indices of shape [B]

    Return:
        y (torch.Tensor): indexed tensor of shape [B, ....]
    """
    return gather_along_dim_with_dim_single(seq, target_dim=1, source_dim=0, indices=indices)


def gather_sequence(seq, indices):
    """
    Given a nested dictionary or list or tuple, gathers an element from each sequence of the batch
    for tensors with leading dimensions [B, T, ...].

    Args:
        seq (dict or list or tuple): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        indices (torch.Tensor): tensor indices of shape [B]

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple with tensors of shape [B, ...]
    """
    return gather_along_dim_with_dim(seq, target_dim=1, source_dim=0, indices=indices)


def pad_sequence_single(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad input tensor or array @seq in the time dimension (dimension 1).

    Args:
        seq (np.ndarray or torch.Tensor): sequence to be padded
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

    Returns:
        padded sequence (np.ndarray or torch.Tensor)
    """
    assert isinstance(seq, (np.ndarray, torch.Tensor))
    assert pad_same or pad_values is not None
    if pad_values is not None:
        assert isinstance(pad_values, float)
    repeat_func = np.repeat if isinstance(seq, np.ndarray) else torch.repeat_interleave
    concat_func = np.concatenate if isinstance(seq, np.ndarray) else torch.cat
    ones_like_func = np.ones_like if isinstance(seq, np.ndarray) else torch.ones_like
    seq_dim = 1 if batched else 0

    begin_pad = []
    end_pad = []

    if padding[0] > 0:
        pad = seq[[0]] if pad_same else ones_like_func(seq[[0]]) * pad_values
        begin_pad.append(repeat_func(pad, padding[0], seq_dim))
    if padding[1] > 0:
        pad = seq[[-1]] if pad_same else ones_like_func(seq[[-1]]) * pad_values
        end_pad.append(repeat_func(pad, padding[1], seq_dim))

    return concat_func(begin_pad + [seq] + end_pad, seq_dim)


def pad_sequence(seq, padding, batched=False, pad_same=True, pad_values=None):
    """
    Pad a nested dictionary or list or tuple of sequence tensors in the time dimension (dimension 1).

    Args:
        seq (dict or list or tuple): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        padding (tuple): begin and end padding, e.g. [1, 1] pads both begin and end of the sequence by 1
        batched (bool): if sequence has the batch dimension
        pad_same (bool): if pad by duplicating
        pad_values (scalar or (ndarray, Tensor)): values to be padded if not pad_same

    Returns:
        padded sequence (dict or list or tuple)
    """
    return recursive_dict_list_tuple_apply(
        seq,
        {
            torch.Tensor: lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values:
                pad_sequence_single(x, p, b, ps, pv),
            np.ndarray: lambda x, p=padding, b=batched, ps=pad_same, pv=pad_values:
                pad_sequence_single(x, p, b, ps, pv),
            type(None): lambda x: x,
        }
    )


def assert_size_at_dim_single(x, size, dim, msg):
    """
    Ensure that array or tensor @x has size @size in dim @dim.

    Args:
        x (np.ndarray or torch.Tensor): input array or tensor
        size (int): size that tensors should have at @dim
        dim (int): dimension to check
        msg (str): text to display if assertion fails
    """
    assert x.shape[dim] == size, msg


def assert_size_at_dim(x, size, dim, msg):
    """
    Ensure that arrays and tensors in nested dictionary or list or tuple have 
    size @size in dim @dim.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple
        size (int): size that tensors should have at @dim
        dim (int): dimension to check
    """
    map_tensor(x, lambda t, s=size, d=dim, m=msg: assert_size_at_dim_single(t, s, d, m))


def get_shape(x):
    """
    Get all shapes of arrays and tensors in nested dictionary or list or tuple.

    Args:
        x (dict or list or tuple): a possibly nested dictionary or list or tuple

    Returns:
        y (dict or list or tuple): new nested dict-list-tuple that contains each array or
            tensor's shape
    """
    return recursive_dict_list_tuple_apply(
        x,
        {
            torch.Tensor: lambda x: x.shape,
            np.ndarray: lambda x: x.shape,
            type(None): lambda x: x,
        }
    )


def list_of_flat_dict_to_dict_of_list(list_of_dict):
    """
    Helper function to go from a list of flat dictionaries to a dictionary of lists.
    By "flat" we mean that none of the values are dictionaries, but are numpy arrays,
    floats, etc.

    Args:
        list_of_dict (list): list of flat dictionaries

    Returns:
        dict_of_list (dict): dictionary of lists
    """
    assert isinstance(list_of_dict, list)
    dic = collections.OrderedDict()
    for i in range(len(list_of_dict)):
        for k in list_of_dict[i]:
            if k not in dic:
                dic[k] = []
            dic[k].append(list_of_dict[i][k])
    return dic


def flatten_nested_dict_list(d, parent_key='', sep='_', item_key=''):
    """
    Flatten a nested dict or list to a list.

    For example, given a dict
    {
        a: 1
        b: {
            c: 2
        }
        c: 3
    }

    the function would return [(a, 1), (b_c, 2), (c, 3)]

    Args:
        d (dict, list): a nested dict or list to be flattened
        parent_key (str): recursion helper
        sep (str): separator for nesting keys
        item_key (str): recursion helper
    Returns:
        list: a list of (key, value) tuples
    """
    items = []
    if isinstance(d, (tuple, list)):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for i, v in enumerate(d):
            items.extend(flatten_nested_dict_list(v, new_key, sep=sep, item_key=str(i)))
        return items
    elif isinstance(d, dict):
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        for k, v in d.items():
            assert isinstance(k, str)
            items.extend(flatten_nested_dict_list(v, new_key, sep=sep, item_key=k))
        return items
    else:
        new_key = parent_key + sep + item_key if len(parent_key) > 0 else item_key
        return [(new_key, d)]


def time_distributed(inputs, op, activation=None, inputs_as_kwargs=False, inputs_as_args=False, **kwargs):
    """
    Apply function @op to all tensors in nested dictionary or list or tuple @inputs in both the
    batch (B) and time (T) dimension, where the tensors are expected to have shape [B, T, ...].
    Will do this by reshaping tensors to [B * T, ...], passing through the op, and then reshaping
    outputs to [B, T, ...].

    Args:
        inputs (list or tuple or dict): a possibly nested dictionary or list or tuple with tensors
            of leading dimensions [B, T, ...]
        op: a layer op that accepts inputs
        activation: activation to apply at the output
        inputs_as_kwargs (bool): whether to feed input as a kwargs dict to the op
        inputs_as_args (bool) whether to feed input as a args list to the op
        kwargs (dict): other kwargs to supply to the op

    Returns:
        outputs (dict or list or tuple): new nested dict-list-tuple with tensors of leading dimension [B, T].
    """
    batch_size, seq_len = flatten_nested_dict_list(inputs)[0][1].shape[:2]
    inputs = join_dimensions(inputs, 0, 1)
    if inputs_as_kwargs:
        outputs = op(**inputs, **kwargs)
    elif inputs_as_args:
        outputs = op(*inputs, **kwargs)
    else:
        outputs = op(inputs, **kwargs)

    if activation is not None:
        outputs = map_tensor(outputs, activation)
    outputs = reshape_dimensions(outputs, begin_axis=0, end_axis=0, target_dims=(batch_size, seq_len))
    return outputs

```

## reference_material/diffusion_policy_code/diffusion_policy/model/common/lr_scheduler.py
```python
from diffusers.optimization import (
    Union, SchedulerType, Optional,
    Optimizer, TYPE_TO_SCHEDULER_FUNCTION
)

def get_scheduler(
    name: Union[str, SchedulerType],
    optimizer: Optimizer,
    num_warmup_steps: Optional[int] = None,
    num_training_steps: Optional[int] = None,
    **kwargs
):
    """
    Added kwargs vs diffuser's original implementation

    Unified API to get any scheduler from its name.

    Args:
        name (`str` or `SchedulerType`):
            The name of the scheduler to use.
        optimizer (`torch.optim.Optimizer`):
            The optimizer that will be used during training.
        num_warmup_steps (`int`, *optional*):
            The number of warmup steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
        num_training_steps (`int``, *optional*):
            The number of training steps to do. This is not required by all schedulers (hence the argument being
            optional), the function will raise an error if it's unset and the scheduler type requires it.
    """
    name = SchedulerType(name)
    schedule_func = TYPE_TO_SCHEDULER_FUNCTION[name]
    if name == SchedulerType.CONSTANT:
        return schedule_func(optimizer, **kwargs)

    # All other schedulers require `num_warmup_steps`
    if num_warmup_steps is None:
        raise ValueError(f"{name} requires `num_warmup_steps`, please provide that argument.")

    if name == SchedulerType.CONSTANT_WITH_WARMUP:
        return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, **kwargs)

    # All other schedulers require `num_training_steps`
    if num_training_steps is None:
        raise ValueError(f"{name} requires `num_training_steps`, please provide that argument.")

    return schedule_func(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps, **kwargs)

```

## reference_material/diffusion_policy_code/diffusion_policy/model/common/rotation_transformer.py
```python
from typing import Union
import pytorch3d.transforms as pt
import torch
import numpy as np
import functools

class RotationTransformer:
    valid_reps = [
        'axis_angle',
        'euler_angles',
        'quaternion',
        'rotation_6d',
        'matrix'
    ]

    def __init__(self, 
            from_rep='axis_angle', 
            to_rep='rotation_6d', 
            from_convention=None,
            to_convention=None):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == 'euler_angles':
            assert from_convention is not None
        if to_rep == 'euler_angles':
            assert to_convention is not None

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != 'matrix':
            funcs = [
                getattr(pt, f'{from_rep}_to_matrix'),
                getattr(pt, f'matrix_to_{from_rep}')
            ]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != 'matrix':
            funcs = [
                getattr(pt, f'matrix_to_{to_rep}'),
                getattr(pt, f'{to_rep}_to_matrix')
            ]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) 
                    for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])
        
        inverse_funcs = inverse_funcs[::-1]
        
        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: Union[np.ndarray, torch.Tensor], funcs: list) -> Union[np.ndarray, torch.Tensor]:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.from_numpy(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y
        
    def forward(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.forward_funcs)
    
    def inverse(self, x: Union[np.ndarray, torch.Tensor]
        ) -> Union[np.ndarray, torch.Tensor]:
        return self._apply_funcs(x, self.inverse_funcs)


def test():
    tf = RotationTransformer()

    rotvec = np.random.uniform(-2*np.pi,2*np.pi,size=(1000,3))
    rot6d = tf.forward(rotvec)
    new_rotvec = tf.inverse(rot6d)

    from scipy.spatial.transform import Rotation
    diff = Rotation.from_rotvec(rotvec) * Rotation.from_rotvec(new_rotvec).inv()
    dist = diff.magnitude()
    assert dist.max() < 1e-7

    tf = RotationTransformer('rotation_6d', 'matrix')
    rot6d_wrong = rot6d + np.random.normal(scale=0.1, size=rot6d.shape)
    mat = tf.forward(rot6d_wrong)
    mat_det = np.linalg.det(mat)
    assert np.allclose(mat_det, 1)
    # rotaiton_6d will be normalized to rotation matrix

```

## reference_material/diffusion_policy_code/diffusion_policy/model/diffusion/positional_embedding.py
```python
import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

```

## reference_material/diffusion_policy_code/diffusion_policy/model/diffusion/transformer_for_diffusion.py
```python
from typing import Union, Optional, Tuple
import logging
import torch
import torch.nn as nn
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin

logger = logging.getLogger(__name__)

class TransformerForDiffusion(ModuleAttrMixin):
    def __init__(self,
            input_dim: int,
            output_dim: int,
            horizon: int,
            n_obs_steps: int = None,
            cond_dim: int = 0,
            n_layer: int = 12,
            n_head: int = 12,
            n_emb: int = 768,
            p_drop_emb: float = 0.1,
            p_drop_attn: float = 0.1,
            causal_attn: bool=False,
            time_as_cond: bool=True,
            obs_as_cond: bool=False,
            n_cond_layers: int = 0
        ) -> None:
        super().__init__()

        # compute number of tokens for main trunk and condition encoder
        if n_obs_steps is None:
            n_obs_steps = horizon
        
        T = horizon
        T_cond = 1
        if not time_as_cond:
            T += 1
            T_cond -= 1
        obs_as_cond = cond_dim > 0
        if obs_as_cond:
            assert time_as_cond
            T_cond += n_obs_steps

        # input embedding stem
        self.input_emb = nn.Linear(input_dim, n_emb)
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        self.drop = nn.Dropout(p_drop_emb)

        # cond encoder
        self.time_emb = SinusoidalPosEmb(n_emb)
        self.cond_obs_emb = None
        
        if obs_as_cond:
            self.cond_obs_emb = nn.Linear(cond_dim, n_emb)

        self.cond_pos_emb = None
        self.encoder = None
        self.decoder = None
        encoder_only = False
        if T_cond > 0:
            self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
            if n_cond_layers > 0:
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=n_emb,
                    nhead=n_head,
                    dim_feedforward=4*n_emb,
                    dropout=p_drop_attn,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.encoder = nn.TransformerEncoder(
                    encoder_layer=encoder_layer,
                    num_layers=n_cond_layers
                )
            else:
                self.encoder = nn.Sequential(
                    nn.Linear(n_emb, 4 * n_emb),
                    nn.Mish(),
                    nn.Linear(4 * n_emb, n_emb)
                )
            # decoder
            decoder_layer = nn.TransformerDecoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True # important for stability
            )
            self.decoder = nn.TransformerDecoder(
                decoder_layer=decoder_layer,
                num_layers=n_layer
            )
        else:
            # encoder only BERT
            encoder_only = True

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4*n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer,
                num_layers=n_layer
            )

        # attention mask
        if causal_attn:
            # causal mask to ensure that attention is only applied to the left in the input sequence
            # torch.nn.Transformer uses additive mask as opposed to multiplicative mask in minGPT
            # therefore, the upper triangle should be -inf and others (including diag) should be 0.
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer("mask", mask)
            
            if time_as_cond and obs_as_cond:
                S = T_cond
                t, s = torch.meshgrid(
                    torch.arange(T),
                    torch.arange(S),
                    indexing='ij'
                )
                mask = t >= (s-1) # add one dimension since time is the first token in cond
                mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
                self.register_buffer('memory_mask', mask)
            else:
                self.memory_mask = None
        else:
            self.mask = None
            self.memory_mask = None

        # decoder head
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, output_dim)
            
        # constants
        self.T = T
        self.T_cond = T_cond
        self.horizon = horizon
        self.time_as_cond = time_as_cond
        self.obs_as_cond = obs_as_cond
        self.encoder_only = encoder_only

        # init
        self.apply(self._init_weights)
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _init_weights(self, module):
        ignore_types = (nn.Dropout, 
            SinusoidalPosEmb, 
            nn.TransformerEncoderLayer, 
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Mish,
            nn.Sequential)
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            weight_names = [
                'in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerForDiffusion):
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            raise RuntimeError("Unaccounted module {}".format(module))
    
    def get_optim_groups(self, weight_decay: float=1e-3):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.startswith("bias"):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("pos_emb")
        no_decay.add("_dummy_variable")
        if self.cond_pos_emb is not None:
            no_decay.add("cond_pos_emb")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups


    def configure_optimizers(self, 
            learning_rate: float=1e-4, 
            weight_decay: float=1e-3,
            betas: Tuple[float, float]=(0.9,0.95)):
        optim_groups = self.get_optim_groups(weight_decay=weight_decay)
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas
        )
        return optimizer

    def forward(self, 
        sample: torch.Tensor, 
        timestep: Union[torch.Tensor, float, int], 
        cond: Optional[torch.Tensor]=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        cond: (B,T',cond_dim)
        output: (B,T,input_dim)
        """
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)
        # (B,1,n_emb)

        # process input
        input_emb = self.input_emb(sample)

        if self.encoder_only:
            # BERT
            token_embeddings = torch.cat([time_emb, input_emb], dim=1)
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T+1,n_emb)
            x = self.encoder(src=x, mask=self.mask)
            # (B,T+1,n_emb)
            x = x[:,1:,:]
            # (B,T,n_emb)
        else:
            # encoder
            cond_embeddings = time_emb
            if self.obs_as_cond:
                cond_obs_emb = self.cond_obs_emb(cond)
                # (B,To,n_emb)
                cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)
            tc = cond_embeddings.shape[1]
            position_embeddings = self.cond_pos_emb[
                :, :tc, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(cond_embeddings + position_embeddings)
            x = self.encoder(x)
            memory = x
            # (B,T_cond,n_emb)
            
            # decoder
            token_embeddings = input_emb
            t = token_embeddings.shape[1]
            position_embeddings = self.pos_emb[
                :, :t, :
            ]  # each position maps to a (learnable) vector
            x = self.drop(token_embeddings + position_embeddings)
            # (B,T,n_emb)
            x = self.decoder(
                tgt=x,
                memory=memory,
                tgt_mask=self.mask,
                memory_mask=self.memory_mask
            )
            # (B,T,n_emb)
        
        # head
        x = self.ln_f(x)
        x = self.head(x)
        # (B,T,n_out)
        return x


def test():
    # GPT with time embedding
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)
    

    # GPT with time embedding and obs cond
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # GPT with time embedding and obs cond and encoder
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        cond_dim=10,
        causal_attn=True,
        # time_as_cond=False,
        n_cond_layers=4
    )
    opt = transformer.configure_optimizers()
    
    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    cond = torch.zeros((4,4,10))
    out = transformer(sample, timestep, cond)

    # BERT with time embedding token
    transformer = TransformerForDiffusion(
        input_dim=16,
        output_dim=16,
        horizon=8,
        n_obs_steps=4,
        # cond_dim=10,
        # causal_attn=True,
        time_as_cond=False,
        # n_cond_layers=4
    )
    opt = transformer.configure_optimizers()

    timestep = torch.tensor(0)
    sample = torch.zeros((4,8,16))
    out = transformer(sample, timestep)


```

## reference_material/diffusion_policy_code/diffusion_policy/model/diffusion/conditional_unet1d.py
```python
from typing import Union
import logging
import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self, 
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def forward(self, 
            sample: torch.Tensor, 
            timestep: Union[torch.Tensor, float, int], 
            local_cond=None, global_cond=None, **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x


```

## reference_material/diffusion_policy_code/diffusion_policy/model/diffusion/mask_generator.py
```python
from typing import Sequence, Optional
import torch
from torch import nn
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin


def get_intersection_slice_mask(
    shape: tuple, 
    dim_slices: Sequence[slice], 
    device: Optional[torch.device]=None
    ):
    assert(len(shape) == len(dim_slices))
    mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
    mask[dim_slices] = True
    return mask


def get_union_slice_mask(
    shape: tuple, 
    dim_slices: Sequence[slice], 
    device: Optional[torch.device]=None
    ):
    assert(len(shape) == len(dim_slices))
    mask = torch.zeros(size=shape, dtype=torch.bool, device=device)
    for i in range(len(dim_slices)):
        this_slices = [slice(None)] * len(shape)
        this_slices[i] = dim_slices[i]
        mask[this_slices] = True
    return mask


class DummyMaskGenerator(ModuleAttrMixin):
    def __init__(self):
        super().__init__()
    
    @torch.no_grad()
    def forward(self, shape):
        device = self.device
        mask = torch.ones(size=shape, dtype=torch.bool, device=device)
        return mask


class LowdimMaskGenerator(ModuleAttrMixin):
    def __init__(self,
        action_dim, obs_dim,
        # obs mask setup
        max_n_obs_steps=2, 
        fix_obs_steps=True, 
        # action mask
        action_visible=False
        ):
        super().__init__()
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.action_visible = action_visible

    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        assert D == (self.action_dim + self.obs_dim)

        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)

        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_obs_dim = ~is_action_dim

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        mask = obs_mask
        if self.action_visible:
            mask = mask | action_mask
        
        return mask


class KeypointMaskGenerator(ModuleAttrMixin):
    def __init__(self, 
            # dimensions
            action_dim, keypoint_dim,
            # obs mask setup
            max_n_obs_steps=2, fix_obs_steps=True, 
            # keypoint mask setup
            keypoint_visible_rate=0.7, time_independent=False,
            # action mask
            action_visible=False,
            context_dim=0, # dim for context
            n_context_steps=1
            ):
        super().__init__()
        self.action_dim = action_dim
        self.keypoint_dim = keypoint_dim
        self.context_dim = context_dim
        self.max_n_obs_steps = max_n_obs_steps
        self.fix_obs_steps = fix_obs_steps
        self.keypoint_visible_rate = keypoint_visible_rate
        self.time_independent = time_independent
        self.action_visible = action_visible
        self.n_context_steps = n_context_steps
    
    @torch.no_grad()
    def forward(self, shape, seed=None):
        device = self.device
        B, T, D = shape
        all_keypoint_dims = D - self.action_dim - self.context_dim
        n_keypoints = all_keypoint_dims // self.keypoint_dim
        
        # create all tensors on this device
        rng = torch.Generator(device=device)
        if seed is not None:
            rng = rng.manual_seed(seed)
        
        # generate dim mask
        dim_mask = torch.zeros(size=shape, 
            dtype=torch.bool, device=device)
        is_action_dim = dim_mask.clone()
        is_action_dim[...,:self.action_dim] = True
        is_context_dim = dim_mask.clone()
        if self.context_dim > 0:
            is_context_dim[...,-self.context_dim:] = True
        is_obs_dim = ~(is_action_dim | is_context_dim)
        # assumption trajectory=cat([action, keypoints, context], dim=-1)

        # generate obs mask
        if self.fix_obs_steps:
            obs_steps = torch.full((B,), 
            fill_value=self.max_n_obs_steps, device=device)
        else:
            obs_steps = torch.randint(
                low=1, high=self.max_n_obs_steps+1, 
                size=(B,), generator=rng, device=device)
            
        steps = torch.arange(0, T, device=device).reshape(1,T).expand(B,T)
        obs_mask = (steps.T < obs_steps).T.reshape(B,T,1).expand(B,T,D)
        obs_mask = obs_mask & is_obs_dim

        # generate action mask
        if self.action_visible:
            action_steps = torch.maximum(
                obs_steps - 1, 
                torch.tensor(0,
                    dtype=obs_steps.dtype, 
                    device=obs_steps.device))
            action_mask = (steps.T < action_steps).T.reshape(B,T,1).expand(B,T,D)
            action_mask = action_mask & is_action_dim

        # generate keypoint mask
        if self.time_independent:
            visible_kps = torch.rand(size=(B, T, n_keypoints), 
                generator=rng, device=device) < self.keypoint_visible_rate
            visible_dims = torch.repeat_interleave(visible_kps, repeats=self.keypoint_dim, dim=-1)
            visible_dims_mask = torch.cat([
                torch.ones((B, T, self.action_dim), 
                    dtype=torch.bool, device=device),
                visible_dims,
                torch.ones((B, T, self.context_dim), 
                    dtype=torch.bool, device=device),
            ], axis=-1)
            keypoint_mask = visible_dims_mask
        else:
            visible_kps = torch.rand(size=(B,n_keypoints), 
                generator=rng, device=device) < self.keypoint_visible_rate
            visible_dims = torch.repeat_interleave(visible_kps, repeats=self.keypoint_dim, dim=-1)
            visible_dims_mask = torch.cat([
                torch.ones((B, self.action_dim), 
                    dtype=torch.bool, device=device),
                visible_dims,
                torch.ones((B, self.context_dim), 
                    dtype=torch.bool, device=device),
            ], axis=-1)
            keypoint_mask = visible_dims_mask.reshape(B,1,D).expand(B,T,D)
        keypoint_mask = keypoint_mask & is_obs_dim

        # generate context mask
        context_mask = is_context_dim.clone()
        context_mask[:,self.n_context_steps:,:] = False

        mask = obs_mask & keypoint_mask 
        if self.action_visible:
            mask = mask | action_mask
        if self.context_dim > 0:
            mask = mask | context_mask

        return mask


def test():
    # kmg = KeypointMaskGenerator(2,2, random_obs_steps=True)
    # self = KeypointMaskGenerator(2,2,context_dim=2, action_visible=True)
    # self = KeypointMaskGenerator(2,2,context_dim=0, action_visible=True)
    self = LowdimMaskGenerator(2,20, max_n_obs_steps=3, action_visible=True)

```

## reference_material/diffusion_policy_code/diffusion_policy/model/diffusion/ema_model.py
```python
import copy
import torch
from torch.nn.modules.batchnorm import _BatchNorm

class EMAModel:
    """
    Exponential Moving Average of models weights
    """

    def __init__(
        self,
        model,
        update_after_step=0,
        inv_gamma=1.0,
        power=2 / 3,
        min_value=0.0,
        max_value=0.9999
    ):
        """
        @crowsonkb's notes on EMA Warmup:
            If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are good values for models you plan
            to train for a million or more steps (reaches decay factor 0.999 at 31.6K steps, 0.9999 at 1M steps),
            gamma=1, power=3/4 for models you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999
            at 215.4k steps).
        Args:
            inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
            power (float): Exponential factor of EMA warmup. Default: 2/3.
            min_value (float): The minimum EMA decay rate. Default: 0.
        """

        self.averaged_model = model
        self.averaged_model.eval()
        self.averaged_model.requires_grad_(False)

        self.update_after_step = update_after_step
        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value
        self.max_value = max_value

        self.decay = 0.0
        self.optimization_step = 0

    def get_decay(self, optimization_step):
        """
        Compute the decay factor for the exponential moving average.
        """
        step = max(0, optimization_step - self.update_after_step - 1)
        value = 1 - (1 + step / self.inv_gamma) ** -self.power

        if step <= 0:
            return 0.0

        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def step(self, new_model):
        self.decay = self.get_decay(self.optimization_step)

        # old_all_dataptrs = set()
        # for param in new_model.parameters():
        #     data_ptr = param.data_ptr()
        #     if data_ptr != 0:
        #         old_all_dataptrs.add(data_ptr)

        all_dataptrs = set()
        for module, ema_module in zip(new_model.modules(), self.averaged_model.modules()):            
            for param, ema_param in zip(module.parameters(recurse=False), ema_module.parameters(recurse=False)):
                # iterative over immediate parameters only.
                if isinstance(param, dict):
                    raise RuntimeError('Dict parameter not supported')
                
                # data_ptr = param.data_ptr()
                # if data_ptr != 0:
                #     all_dataptrs.add(data_ptr)

                if isinstance(module, _BatchNorm):
                    # skip batchnorms
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                elif not param.requires_grad:
                    ema_param.copy_(param.to(dtype=ema_param.dtype).data)
                else:
                    ema_param.mul_(self.decay)
                    ema_param.add_(param.data.to(dtype=ema_param.dtype), alpha=1 - self.decay)

        # verify that iterating over module and then parameters is identical to parameters recursively.
        # assert old_all_dataptrs == all_dataptrs
        self.optimization_step += 1

```

## reference_material/diffusion_policy_code/diffusion_policy/model/diffusion/conv1d_components.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops.layers.torch import Rearrange


class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            # Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            # Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


def test():
    cb = Conv1dBlock(256, 128, kernel_size=3)
    x = torch.zeros((1,256,16))
    o = cb(x)

```

## reference_material/diffusion_policy_code/diffusion_policy/model/vision/model_getter.py
```python
import torch
import torchvision

def get_resnet(name, weights=None, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", "r3m"
    """
    # load r3m weights
    if (weights == "r3m") or (weights == "R3M"):
        return get_r3m(name=name, **kwargs)

    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)
    resnet.fc = torch.nn.Identity()
    return resnet

def get_r3m(name, **kwargs):
    """
    name: resnet18, resnet34, resnet50
    """
    import r3m
    r3m.device = 'cpu'
    model = r3m.load_r3m(name)
    r3m_model = model.module
    resnet_model = r3m_model.convnet
    resnet_model = resnet_model.to('cpu')
    return resnet_model

```

## reference_material/diffusion_policy_code/diffusion_policy/model/vision/multi_image_obs_encoder.py
```python
from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from diffusion_policy.model.vision.crop_randomizer import CropRandomizer
from diffusion_policy.model.common.module_attr_mixin import ModuleAttrMixin
from diffusion_policy.common.pytorch_util import dict_apply, replace_submodules


class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)
                
                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16, 
                                num_channels=x.num_features)
                        )
                    key_model_map[key] = this_model
                
                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False
                        )
                    else:
                        this_normalizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                
                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                key_transform_map[key] = this_transform
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent models
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                img = self.key_transform_map[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)
        
        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)
        
        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result
    
    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape, 
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape

```

## reference_material/diffusion_policy_code/diffusion_policy/model/vision/crop_randomizer.py
```python
import torch
import torch.nn as nn
import torchvision.transforms.functional as ttf
import diffusion_policy.model.common.tensor_util as tu

class CropRandomizer(nn.Module):
    """
    Randomly sample crops at input, and then average across crop features at output.
    """
    def __init__(
        self,
        input_shape,
        crop_height, 
        crop_width, 
        num_crops=1,
        pos_enc=False,
    ):
        """
        Args:
            input_shape (tuple, list): shape of input (not including batch dimension)
            crop_height (int): crop height
            crop_width (int): crop width
            num_crops (int): number of random crops to take
            pos_enc (bool): if True, add 2 channels to the output to encode the spatial
                location of the cropped pixels in the source image
        """
        super().__init__()

        assert len(input_shape) == 3 # (C, H, W)
        assert crop_height < input_shape[1]
        assert crop_width < input_shape[2]

        self.input_shape = input_shape
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.num_crops = num_crops
        self.pos_enc = pos_enc

    def output_shape_in(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_in operation, where raw inputs (usually observation modalities)
        are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """

        # outputs are shape (C, CH, CW), or maybe C + 2 if using position encoding, because
        # the number of crops are reshaped into the batch dimension, increasing the batch
        # size from B to B * N
        out_c = self.input_shape[0] + 2 if self.pos_enc else self.input_shape[0]
        return [out_c, self.crop_height, self.crop_width]

    def output_shape_out(self, input_shape=None):
        """
        Function to compute output shape from inputs to this module. Corresponds to
        the @forward_out operation, where processed inputs (usually encoded observation
        modalities) are passed in.

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        
        # since the forward_out operation splits [B * N, ...] -> [B, N, ...]
        # and then pools to result in [B, ...], only the batch dimension changes,
        # and so the other dimensions retain their shape.
        return list(input_shape)

    def forward_in(self, inputs):
        """
        Samples N random crops for each input in the batch, and then reshapes
        inputs to [B * N, ...].
        """
        assert len(inputs.shape) >= 3 # must have at least (C, H, W) dimensions
        if self.training:
            # generate random crops
            out, _ = sample_random_image_crops(
                images=inputs,
                crop_height=self.crop_height, 
                crop_width=self.crop_width, 
                num_crops=self.num_crops,
                pos_enc=self.pos_enc,
            )
            # [B, N, ...] -> [B * N, ...]
            return tu.join_dimensions(out, 0, 1)
        else:
            # take center crop during eval
            out = ttf.center_crop(img=inputs, output_size=(
                self.crop_height, self.crop_width))
            if self.num_crops > 1:
                B,C,H,W = out.shape
                out = out.unsqueeze(1).expand(B,self.num_crops,C,H,W).reshape(-1,C,H,W)
                # [B * N, ...]
            return out

    def forward_out(self, inputs):
        """
        Splits the outputs from shape [B * N, ...] -> [B, N, ...] and then average across N
        to result in shape [B, ...] to make sure the network output is consistent with
        what would have happened if there were no randomization.
        """
        if self.num_crops <= 1:
            return inputs
        else:
            batch_size = (inputs.shape[0] // self.num_crops)
            out = tu.reshape_dimensions(inputs, begin_axis=0, end_axis=0, 
                target_dims=(batch_size, self.num_crops))
            return out.mean(dim=1)
    
    def forward(self, inputs):
        return self.forward_in(inputs)

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = header + "(input_shape={}, crop_size=[{}, {}], num_crops={})".format(
            self.input_shape, self.crop_height, self.crop_width, self.num_crops)
        return msg


def crop_image_from_indices(images, crop_indices, crop_height, crop_width):
    """
    Crops images at the locations specified by @crop_indices. Crops will be 
    taken across all channels.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_indices (torch.Tensor): batch of indices of shape [..., N, 2] where
            N is the number of crops to take per image and each entry corresponds
            to the pixel height and width of where to take the crop. Note that
            the indices can also be of shape [..., 2] if only 1 crop should
            be taken per image. Leading dimensions must be consistent with
            @images argument. Each index specifies the top left of the crop.
            Values must be in range [0, H - CH - 1] x [0, W - CW - 1] where
            H and W are the height and width of @images and CH and CW are
            @crop_height and @crop_width.

        crop_height (int): height of crop to take

        crop_width (int): width of crop to take

    Returns:
        crops (torch.Tesnor): cropped images of shape [..., C, @crop_height, @crop_width]
    """

    # make sure length of input shapes is consistent
    assert crop_indices.shape[-1] == 2
    ndim_im_shape = len(images.shape)
    ndim_indices_shape = len(crop_indices.shape)
    assert (ndim_im_shape == ndim_indices_shape + 1) or (ndim_im_shape == ndim_indices_shape + 2)

    # maybe pad so that @crop_indices is shape [..., N, 2]
    is_padded = False
    if ndim_im_shape == ndim_indices_shape + 2:
        crop_indices = crop_indices.unsqueeze(-2)
        is_padded = True

    # make sure leading dimensions between images and indices are consistent
    assert images.shape[:-3] == crop_indices.shape[:-2]

    device = images.device
    image_c, image_h, image_w = images.shape[-3:]
    num_crops = crop_indices.shape[-2]

    # make sure @crop_indices are in valid range
    assert (crop_indices[..., 0] >= 0).all().item()
    assert (crop_indices[..., 0] < (image_h - crop_height)).all().item()
    assert (crop_indices[..., 1] >= 0).all().item()
    assert (crop_indices[..., 1] < (image_w - crop_width)).all().item()

    # convert each crop index (ch, cw) into a list of pixel indices that correspond to the entire window.

    # 2D index array with columns [0, 1, ..., CH - 1] and shape [CH, CW]
    crop_ind_grid_h = torch.arange(crop_height).to(device)
    crop_ind_grid_h = tu.unsqueeze_expand_at(crop_ind_grid_h, size=crop_width, dim=-1)
    # 2D index array with rows [0, 1, ..., CW - 1] and shape [CH, CW]
    crop_ind_grid_w = torch.arange(crop_width).to(device)
    crop_ind_grid_w = tu.unsqueeze_expand_at(crop_ind_grid_w, size=crop_height, dim=0)
    # combine into shape [CH, CW, 2]
    crop_in_grid = torch.cat((crop_ind_grid_h.unsqueeze(-1), crop_ind_grid_w.unsqueeze(-1)), dim=-1)

    # Add above grid with the offset index of each sampled crop to get 2d indices for each crop.
    # After broadcasting, this will be shape [..., N, CH, CW, 2] and each crop has a [CH, CW, 2]
    # shape array that tells us which pixels from the corresponding source image to grab.
    grid_reshape = [1] * len(crop_indices.shape[:-1]) + [crop_height, crop_width, 2]
    all_crop_inds = crop_indices.unsqueeze(-2).unsqueeze(-2) + crop_in_grid.reshape(grid_reshape)

    # For using @torch.gather, convert to flat indices from 2D indices, and also
    # repeat across the channel dimension. To get flat index of each pixel to grab for 
    # each sampled crop, we just use the mapping: ind = h_ind * @image_w + w_ind
    all_crop_inds = all_crop_inds[..., 0] * image_w + all_crop_inds[..., 1] # shape [..., N, CH, CW]
    all_crop_inds = tu.unsqueeze_expand_at(all_crop_inds, size=image_c, dim=-3) # shape [..., N, C, CH, CW]
    all_crop_inds = tu.flatten(all_crop_inds, begin_axis=-2) # shape [..., N, C, CH * CW]

    # Repeat and flatten the source images -> [..., N, C, H * W] and then use gather to index with crop pixel inds
    images_to_crop = tu.unsqueeze_expand_at(images, size=num_crops, dim=-4)
    images_to_crop = tu.flatten(images_to_crop, begin_axis=-2)
    crops = torch.gather(images_to_crop, dim=-1, index=all_crop_inds)
    # [..., N, C, CH * CW] -> [..., N, C, CH, CW]
    reshape_axis = len(crops.shape) - 1
    crops = tu.reshape_dimensions(crops, begin_axis=reshape_axis, end_axis=reshape_axis, 
                    target_dims=(crop_height, crop_width))

    if is_padded:
        # undo padding -> [..., C, CH, CW]
        crops = crops.squeeze(-4)
    return crops

def sample_random_image_crops(images, crop_height, crop_width, num_crops, pos_enc=False):
    """
    For each image, randomly sample @num_crops crops of size (@crop_height, @crop_width), from
    @images.

    Args:
        images (torch.Tensor): batch of images of shape [..., C, H, W]

        crop_height (int): height of crop to take
        
        crop_width (int): width of crop to take

        num_crops (n): number of crops to sample

        pos_enc (bool): if True, also add 2 channels to the outputs that gives a spatial 
            encoding of the original source pixel locations. This means that the
            output crops will contain information about where in the source image 
            it was sampled from.

    Returns:
        crops (torch.Tensor): crops of shape (..., @num_crops, C, @crop_height, @crop_width) 
            if @pos_enc is False, otherwise (..., @num_crops, C + 2, @crop_height, @crop_width)

        crop_inds (torch.Tensor): sampled crop indices of shape (..., N, 2)
    """
    device = images.device

    # maybe add 2 channels of spatial encoding to the source image
    source_im = images
    if pos_enc:
        # spatial encoding [y, x] in [0, 1]
        h, w = source_im.shape[-2:]
        pos_y, pos_x = torch.meshgrid(torch.arange(h), torch.arange(w))
        pos_y = pos_y.float().to(device) / float(h)
        pos_x = pos_x.float().to(device) / float(w)
        position_enc = torch.stack((pos_y, pos_x)) # shape [C, H, W]

        # unsqueeze and expand to match leading dimensions -> shape [..., C, H, W]
        leading_shape = source_im.shape[:-3]
        position_enc = position_enc[(None,) * len(leading_shape)]
        position_enc = position_enc.expand(*leading_shape, -1, -1, -1)

        # concat across channel dimension with input
        source_im = torch.cat((source_im, position_enc), dim=-3)

    # make sure sample boundaries ensure crops are fully within the images
    image_c, image_h, image_w = source_im.shape[-3:]
    max_sample_h = image_h - crop_height
    max_sample_w = image_w - crop_width

    # Sample crop locations for all tensor dimensions up to the last 3, which are [C, H, W].
    # Each gets @num_crops samples - typically this will just be the batch dimension (B), so 
    # we will sample [B, N] indices, but this supports having more than one leading dimension,
    # or possibly no leading dimension.
    #
    # Trick: sample in [0, 1) with rand, then re-scale to [0, M) and convert to long to get sampled ints
    crop_inds_h = (max_sample_h * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds_w = (max_sample_w * torch.rand(*source_im.shape[:-3], num_crops).to(device)).long()
    crop_inds = torch.cat((crop_inds_h.unsqueeze(-1), crop_inds_w.unsqueeze(-1)), dim=-1) # shape [..., N, 2]

    crops = crop_image_from_indices(
        images=source_im, 
        crop_indices=crop_inds, 
        crop_height=crop_height, 
        crop_width=crop_width, 
    )

    return crops, crop_inds

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/kitchen_lowdim_wrapper.py
```python
from typing import List, Dict, Optional, Optional
import numpy as np
import gym
from gym.spaces import Box
from diffusion_policy.env.kitchen.base import KitchenBase

class KitchenLowdimWrapper(gym.Env):
    def __init__(self,
            env: KitchenBase,
            init_qpos: Optional[np.ndarray]=None,
            init_qvel: Optional[np.ndarray]=None,
            render_hw = (240,360)
        ):
        self.env = env
        self.init_qpos = init_qpos
        self.init_qvel = init_qvel
        self.render_hw = render_hw

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space

    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        if self.init_qpos is not None:
            # reset anyway to be safe, not very expensive
            _ = self.env.reset()
            # start from known state
            self.env.set_state(self.init_qpos, self.init_qvel)
            obs = self.env._get_obs()
            return obs
            # obs, _, _, _ = self.env.step(np.zeros_like(
            #     self.action_space.sample()))
            # return obs
        else:
            return self.env.reset()

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, width=w, height=h)
    
    def step(self, a):
        return self.env.step(a)

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_models/__init__.py
```python

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/franka/kitchen_multitask_v0.py
```python
""" Kitchen environment for long horizon manipulation """
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from adept_envs import robot_env
from adept_envs.utils.configurable import configurable
from gym import spaces
from dm_control.mujoco import engine

@configurable(pickleable=True)
class KitchenV0(robot_env.RobotEnv):

    CALIBRATION_PATHS = {
        'default':
        os.path.join(os.path.dirname(__file__), 'robot/franka_config.xml')
    }
    # Converted to velocity actuation
    ROBOTS = {'robot': 'adept_envs.franka.robot.franka_robot:Robot_VelAct'}
    MODEl = os.path.join(
        os.path.dirname(__file__),
        '../franka/assets/franka_kitchen_jntpos_act_ab.xml')
    N_DOF_ROBOT = 9
    N_DOF_OBJECT = 21

    def __init__(self, 
            robot_params={}, frame_skip=40, 
            use_abs_action=False):
        self.goal_concat = True
        self.obs_dict = {}
        self.robot_noise_ratio = 0.1  # 10% as per robot_config specs
        self.goal = np.zeros((30,))
        self.use_abs_action = use_abs_action
        if use_abs_action:
            self.ROBOTS = {'robot': 'adept_envs.franka.robot.franka_robot:Robot_PosAct'}

        super().__init__(
            self.MODEl,
            robot=self.make_robot(
                n_jnt=self.N_DOF_ROBOT,  #root+robot_jnts
                n_obj=self.N_DOF_OBJECT,
                **robot_params),
            frame_skip=frame_skip,
            camera_settings=dict(
                distance=4.5,
                azimuth=-66,
                elevation=-65,
            ),
        )
        self.init_qpos = self.sim.model.key_qpos[0].copy()

        # For the microwave kettle slide hinge
        self.init_qpos = np.array([ 1.48388023e-01, -1.76848573e+00,  1.84390296e+00, -2.47685760e+00,
                                    2.60252026e-01,  7.12533105e-01,  1.59515394e+00,  4.79267505e-02,
                                    3.71350919e-02, -2.66279850e-04, -5.18043486e-05,  3.12877220e-05,
                                   -4.51199853e-05, -3.90842156e-06, -4.22629655e-05,  6.28065475e-05,
                                    4.04984708e-05,  4.62730939e-04, -2.26906415e-04, -4.65501369e-04,
                                   -6.44129196e-03, -1.77048263e-03,  1.08009684e-03, -2.69397440e-01,
                                    3.50383255e-01,  1.61944683e+00,  1.00618764e+00,  4.06395120e-03,
                                   -6.62095997e-03, -2.68278933e-04])

        self.init_qvel = self.sim.model.key_qvel[0].copy()

        self.act_mid = np.zeros(self.N_DOF_ROBOT)
        self.act_amp = 2.0 * np.ones(self.N_DOF_ROBOT)

        act_lower = -1*np.ones((self.N_DOF_ROBOT,))
        act_upper =  1*np.ones((self.N_DOF_ROBOT,))
        if use_abs_action:
            act_lower = act_lower * 8.
            act_upper = act_upper * 8.
            self.act_amp = np.ones(self.N_DOF_ROBOT)

        self.action_space = spaces.Box(act_lower, act_upper)

        obs_upper = 8. * np.ones(self.obs_dim)
        obs_lower = -obs_upper
        self.observation_space = spaces.Box(obs_lower, obs_upper)

    def _get_reward_n_score(self, obs_dict):
        raise NotImplementedError()

    def step(self, a, b=None):
        if not self.use_abs_action:
            a = np.clip(a, -1.0, 1.0)

        if not self.initializing:
            a = self.act_mid + a * self.act_amp  # mean center and scale
        else:
            self.goal = self._get_task_goal()  # update goal if init

        self.robot.step(
            self, a, step_duration=self.skip * self.model.opt.timestep)

        # observations
        obs = self._get_obs()

        #rewards
        reward_dict, score = self._get_reward_n_score(self.obs_dict)

        # termination
        done = False

        # finalize step
        env_info = {
            'time': self.obs_dict['t'],
            'obs_dict': self.obs_dict,
            'rewards': reward_dict,
            'score': score,
            # don't render every frame
            # 'images': np.asarray(self.render(mode='rgb_array'))
        }
        # self.render()
        return obs, reward_dict['r_total'], done, env_info

    def _get_obs(self):
        t, qp, qv, obj_qp, obj_qv = self.robot.get_obs(
            self, robot_noise_ratio=self.robot_noise_ratio)

        self.obs_dict = {}
        self.obs_dict['t'] = t
        self.obs_dict['qp'] = qp
        self.obs_dict['qv'] = qv
        self.obs_dict['obj_qp'] = obj_qp
        self.obs_dict['obj_qv'] = obj_qv
        self.obs_dict['goal'] = self.goal
        if self.goal_concat:
            return np.concatenate([self.obs_dict['qp'], self.obs_dict['obj_qp'], self.obs_dict['goal']])

    def reset_model(self):
        reset_pos = self.init_qpos[:].copy()
        reset_vel = self.init_qvel[:].copy()
        self.robot.reset(self, reset_pos, reset_vel)
        self.sim.forward()
        self.goal = self._get_task_goal()  #sample a new goal on reset
        return self._get_obs()

    def evaluate_success(self, paths):
        # score
        mean_score_per_rollout = np.zeros(shape=len(paths))
        for idx, path in enumerate(paths):
            mean_score_per_rollout[idx] = np.mean(path['env_infos']['score'])
        mean_score = np.mean(mean_score_per_rollout)

        # success percentage
        num_success = 0
        num_paths = len(paths)
        for path in paths:
            num_success += bool(path['env_infos']['rewards']['bonus'][-1])
        success_percentage = num_success * 100.0 / num_paths

        # fuse results
        return np.sign(mean_score) * (
            1e6 * round(success_percentage, 2) + abs(mean_score))

    def close_env(self):
        self.robot.close()

    def set_goal(self, goal):
        self.goal = goal

    def _get_task_goal(self):
        return self.goal

    # Only include goal
    @property
    def goal_space(self):
        len_obs = self.observation_space.low.shape[0]
        env_lim = np.abs(self.observation_space.low[0])
        return spaces.Box(low=-env_lim, high=env_lim, shape=(len_obs//2,))

    def convert_to_active_observation(self, observation):
        return observation

class KitchenTaskRelaxV1(KitchenV0):
    """Kitchen environment with proper camera and goal setup"""

    def __init__(self, use_abs_action=False):
        super(KitchenTaskRelaxV1, self).__init__(
            use_abs_action=use_abs_action)

    def _get_reward_n_score(self, obs_dict):
        reward_dict = {}
        reward_dict['true_reward'] = 0.
        reward_dict['bonus'] = 0.
        reward_dict['r_total'] = 0.
        score = 0.
        return reward_dict, score

    def render(self, mode='human', width=1280, height=720, custom=True, **kwargs):
        if custom:
            camera = engine.MovableCamera(self.sim, height, width)
            if 'distance' not in kwargs:
                kwargs['distance'] = 2.2
            if 'lookat' not in kwargs:
                kwargs['lookat'] = [-0.2, .5, 2.]
            if 'azimuth' not in kwargs:
                kwargs['azimuth'] = 70
            if 'elevation' not in kwargs:
                kwargs['elevation'] = -35
            camera.set_pose(**kwargs)
            img = camera.render()
            return img
        else:
            return super(KitchenTaskRelaxV1, self).render(
                mode=mode, width=width, height=height, **kwargs)


```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/franka/__init__.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from gym.envs.registration import register

# Relax the robot
register(
    id='kitchen_relax-v1',
    entry_point='adept_envs.franka.kitchen_multitask_v0:KitchenTaskRelaxV1',
    max_episode_steps=280,
)
```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/franka/robot/franka_robot.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, getpass
import numpy as np
from termcolor import cprint
import time
import copy
import click

from adept_envs import base_robot
from adept_envs.utils.config import (get_config_root_node, read_config_from_node)

# observations structure
from collections import namedtuple
observation = namedtuple('observation', ['time', 'qpos_robot', 'qvel_robot', 'qpos_object', 'qvel_object'])



franka_interface = ''

class Robot(base_robot.BaseRobot):

    """
    Abstracts away the differences between the robot_simulation and robot_hardware

    """

    def __init__(self, *args, **kwargs):
        super(Robot, self).__init__(*args, **kwargs)
        global franka_interface

        # Read robot configurations
        self._read_specs_from_config(robot_configs=self.calibration_path)


        # Robot: Handware
        if self.is_hardware:
            if franka_interface == '':
                raise NotImplementedError()
                from handware.franka import franka

                # initialize franka
                self.franka_interface = franka()
                franka_interface = self.franka_interface
                cprint("Initializing %s Hardware (Status:%d)" % (self.robot_name, self.franka.okay(self.robot_hardware_dof)), 'white', 'on_grey')
            else:
                self.franka_interface = franka_interface
                cprint("Reusing previours Franka session", 'white', 'on_grey')

        # Robot: Simulation
        else:
            self.robot_name = "Franka"
            cprint("Initializing %s sim" % self.robot_name, 'white', 'on_grey')

        # Robot's time
        self.time_start = time.time()
        self.time = time.time()-self.time_start
        self.time_render = -1 # time of rendering


    # read specs from the calibration file
    def _read_specs_from_config(self, robot_configs):
        root, root_name = get_config_root_node(config_file_name=robot_configs)
        self.robot_name = root_name[0]
        self.robot_mode = np.zeros(self.n_dofs, dtype=int)
        self.robot_mj_dof = np.zeros(self.n_dofs, dtype=int)
        self.robot_hardware_dof = np.zeros(self.n_dofs, dtype=int)
        self.robot_scale = np.zeros(self.n_dofs, dtype=float)
        self.robot_offset = np.zeros(self.n_dofs, dtype=float)
        self.robot_pos_bound = np.zeros([self.n_dofs, 2], dtype=float)
        self.robot_vel_bound = np.zeros([self.n_dofs, 2], dtype=float)
        self.robot_pos_noise_amp = np.zeros(self.n_dofs, dtype=float)
        self.robot_vel_noise_amp = np.zeros(self.n_dofs, dtype=float)

        print("Reading configurations for %s" % self.robot_name)
        for i in range(self.n_dofs):
            self.robot_mode[i] = read_config_from_node(root, "qpos"+str(i), "mode", int)
            self.robot_mj_dof[i] = read_config_from_node(root, "qpos"+str(i), "mj_dof", int)
            self.robot_hardware_dof[i] = read_config_from_node(root, "qpos"+str(i), "hardware_dof", int)
            self.robot_scale[i] = read_config_from_node(root, "qpos"+str(i), "scale", float)
            self.robot_offset[i] = read_config_from_node(root, "qpos"+str(i), "offset", float)
            self.robot_pos_bound[i] = read_config_from_node(root, "qpos"+str(i), "pos_bound", float)
            self.robot_vel_bound[i] = read_config_from_node(root, "qpos"+str(i), "vel_bound", float)
            self.robot_pos_noise_amp[i] = read_config_from_node(root, "qpos"+str(i), "pos_noise_amp", float)
            self.robot_vel_noise_amp[i] = read_config_from_node(root, "qpos"+str(i), "vel_noise_amp", float)


    # convert to hardware space
    def _de_calib(self, qp_mj, qv_mj=None):
        qp_ad = (qp_mj-self.robot_offset)/self.robot_scale
        if qv_mj is not None:
            qv_ad = qv_mj/self.robot_scale
            return qp_ad, qv_ad
        else:
            return qp_ad

    # convert to mujoco space
    def _calib(self, qp_ad, qv_ad):
        qp_mj  =  qp_ad* self.robot_scale + self.robot_offset
        qv_mj  =  qv_ad* self.robot_scale
        return qp_mj, qv_mj


    # refresh the observation cache
    def _observation_cache_refresh(self, env):
        for _ in range(self.observation_cache_maxsize):
            self.get_obs(env, sim_mimic_hardware=False)

    # get past observation
    def get_obs_from_cache(self, env, index=-1):
        assert (index>=0 and index<self.observation_cache_maxsize) or \
                (index<0 and index>=-self.observation_cache_maxsize), \
                "cache index out of bound. (cache size is %2d)"%self.observation_cache_maxsize
        obs = self.observation_cache[index]
        if self.has_obj:
            return obs.time, obs.qpos_robot, obs.qvel_robot, obs.qpos_object, obs.qvel_object
        else:
            return obs.time, obs.qpos_robot, obs.qvel_robot


    # get observation
    def get_obs(self, env, robot_noise_ratio=1, object_noise_ratio=1, sim_mimic_hardware=True):
        if self.is_hardware:
            raise NotImplementedError()

        else:
            #Gather simulated observation
            qp = env.sim.data.qpos[:self.n_jnt].copy()
            qv = env.sim.data.qvel[:self.n_jnt].copy()
            if self.has_obj:
                qp_obj = env.sim.data.qpos[-self.n_obj:].copy()
                qv_obj = env.sim.data.qvel[-self.n_obj:].copy()
            else:
                qp_obj = None
                qv_obj = None
            self.time = env.sim.data.time

            # Simulate observation noise
            if not env.initializing:
                qp += robot_noise_ratio*self.robot_pos_noise_amp[:self.n_jnt]*env.np_random.uniform(low=-1., high=1., size=self.n_jnt)
                qv += robot_noise_ratio*self.robot_vel_noise_amp[:self.n_jnt]*env.np_random.uniform(low=-1., high=1., size=self.n_jnt)
                if self.has_obj:
                    qp_obj += robot_noise_ratio*self.robot_pos_noise_amp[-self.n_obj:]*env.np_random.uniform(low=-1., high=1., size=self.n_obj)
                    qv_obj += robot_noise_ratio*self.robot_vel_noise_amp[-self.n_obj:]*env.np_random.uniform(low=-1., high=1., size=self.n_obj)

        # cache observations
        obs = observation(time=self.time, qpos_robot=qp, qvel_robot=qv, qpos_object=qp_obj, qvel_object=qv_obj)
        self.observation_cache.append(obs)

        if self.has_obj:
            return obs.time, obs.qpos_robot, obs.qvel_robot, obs.qpos_object, obs.qvel_object
        else:
            return obs.time, obs.qpos_robot, obs.qvel_robot


    # enforce position specs.
    def ctrl_position_limits(self, ctrl_position):
        ctrl_feasible_position = np.clip(ctrl_position, self.robot_pos_bound[:self.n_jnt, 0], self.robot_pos_bound[:self.n_jnt, 1])
        return ctrl_feasible_position


    # step the robot env
    def step(self, env, ctrl_desired, step_duration, sim_override=False):

        # Populate observation cache during startup
        if env.initializing:
            self._observation_cache_refresh(env)

        # enforce velocity limits
        ctrl_feasible = self.ctrl_velocity_limits(ctrl_desired, step_duration)

        # enforce position limits
        ctrl_feasible = self.ctrl_position_limits(ctrl_feasible)

        # Send controls to the robot
        if self.is_hardware and (not sim_override):
            raise NotImplementedError()
        else:
            env.do_simulation(ctrl_feasible, int(step_duration/env.sim.model.opt.timestep)) # render is folded in here

        # Update current robot state on the overlay
        if self.overlay:
            env.sim.data.qpos[self.n_jnt:2*self.n_jnt] = env.desired_pose.copy()
            env.sim.forward()

        # synchronize time
        if self.is_hardware:
            time_now = (time.time()-self.time_start)
            time_left_in_step = step_duration - (time_now-self.time)
            if(time_left_in_step>0.0001):
                time.sleep(time_left_in_step)
        return 1


    def reset(self, env, reset_pose, reset_vel, overlay_mimic_reset_pose=True, sim_override=False):
        reset_pose = self.clip_positions(reset_pose)

        if self.is_hardware:
            raise NotImplementedError()
        else:
            env.sim.reset()
            env.sim.data.qpos[:self.n_jnt] = reset_pose[:self.n_jnt].copy()
            env.sim.data.qvel[:self.n_jnt] = reset_vel[:self.n_jnt].copy()
            if self.has_obj:
                env.sim.data.qpos[-self.n_obj:] = reset_pose[-self.n_obj:].copy()
                env.sim.data.qvel[-self.n_obj:] = reset_vel[-self.n_obj:].copy()
            env.sim.forward()

        if self.overlay:
            env.sim.data.qpos[self.n_jnt:2*self.n_jnt] = env.desired_pose[:self.n_jnt].copy()
            env.sim.forward()

        # refresh observation cache before exit
        self._observation_cache_refresh(env)


    def close(self):
        if self.is_hardware:
            cprint("Closing Franka hardware... ", 'white', 'on_grey', end='', flush=True)
            status = 0
            raise NotImplementedError()
            cprint("Closed (Status: {})".format(status), 'white', 'on_grey', flush=True)
        else:
            cprint("Closing Franka sim", 'white', 'on_grey', flush=True)


class Robot_PosAct(Robot):

    # enforce velocity sepcs.
    # ALERT: This depends on previous observation. This is not ideal as it breaks MDP addumptions. Be careful
    def ctrl_velocity_limits(self, ctrl_position, step_duration):
        last_obs = self.observation_cache[-1]
        ctrl_desired_vel = (ctrl_position-last_obs.qpos_robot[:self.n_jnt])/step_duration

        ctrl_feasible_vel = np.clip(ctrl_desired_vel, self.robot_vel_bound[:self.n_jnt, 0], self.robot_vel_bound[:self.n_jnt, 1])
        ctrl_feasible_position = last_obs.qpos_robot[:self.n_jnt] + ctrl_feasible_vel*step_duration
        return ctrl_feasible_position


class Robot_VelAct(Robot):

    # enforce velocity sepcs.
    # ALERT: This depends on previous observation. This is not ideal as it breaks MDP addumptions. Be careful
    def ctrl_velocity_limits(self, ctrl_velocity, step_duration):
        last_obs = self.observation_cache[-1]

        ctrl_feasible_vel = np.clip(ctrl_velocity, self.robot_vel_bound[:self.n_jnt, 0], self.robot_vel_bound[:self.n_jnt, 1])
        ctrl_feasible_position = last_obs.qpos_robot[:self.n_jnt] + ctrl_feasible_vel*step_duration
        return ctrl_feasible_position


```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/utils/config.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
try:
    import cElementTree as ET
except ImportError:
    try:
        # Python 2.5 need to import a different module
        import xml.etree.cElementTree as ET
    except ImportError:
        exit_err("Failed to import cElementTree from any known place")

CONFIG_XML_DATA = """
<config name='dClaw1 dClaw2'>
  <limits low="1 2" high="2 3"/>
  <scale joint="10 20"/>
  <data type="test1 test2"/>
</config>
"""


# Read config from root
def read_config_from_node(root_node, parent_name, child_name, dtype=int):
    # find parent
    parent_node = root_node.find(parent_name)
    if parent_node == None:
        quit("Parent %s not found" % parent_name)

    # get child data
    child_data = parent_node.get(child_name)
    if child_data == None:
        quit("Child %s not found" % child_name)

    config_val = np.array(child_data.split(), dtype=dtype)
    return config_val


# get config frlom file or string
def get_config_root_node(config_file_name=None, config_file_data=None):
    try:
        # get root
        if config_file_data is None:
            config_file_content = open(config_file_name, "r")
            config = ET.parse(config_file_content)
            root_node = config.getroot()
        else:
            root_node = ET.fromstring(config_file_data)

        # get root data
        root_data = root_node.get('name')
        root_name = np.array(root_data.split(), dtype=str)
    except:
        quit("ERROR: Unable to process config file %s" % config_file_name)

    return root_node, root_name


# Read config from config_file
def read_config_from_xml(config_file_name, parent_name, child_name, dtype=int):
    root_node, root_name = get_config_root_node(
        config_file_name=config_file_name)
    return read_config_from_node(root_node, parent_name, child_name, dtype)


# tests
if __name__ == '__main__':
    print("Read config and parse -------------------------")
    root, root_name = get_config_root_node(config_file_data=CONFIG_XML_DATA)
    print("Root:name \t", root_name)
    print("limit:low \t", read_config_from_node(root, "limits", "low", float))
    print("limit:high \t", read_config_from_node(root, "limits", "high", float))
    print("scale:joint \t", read_config_from_node(root, "scale", "joint",
                                                  float))
    print("data:type \t", read_config_from_node(root, "data", "type", str))

    # read straight from xml (dumb the XML data as duh.xml for this test)
    root, root_name = get_config_root_node(config_file_name="duh.xml")
    print("Read from xml --------------------------------")
    print("limit:low \t", read_config_from_xml("duh.xml", "limits", "low",
                                               float))
    print("limit:high \t",
          read_config_from_xml("duh.xml", "limits", "high", float))
    print("scale:joint \t",
          read_config_from_xml("duh.xml", "scale", "joint", float))
    print("data:type \t", read_config_from_xml("duh.xml", "data", "type", str))

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/utils/parse_demos.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import click
import glob
import pickle
import numpy as np
from parse_mjl import parse_mjl_logs, viz_parsed_mjl_logs
from mjrl.utils.gym_env import GymEnv
import adept_envs
import time as timer
import skvideo.io
import gym

# headless renderer
render_buffer = []  # rendering buffer


def viewer(env,
           mode='initialize',
           filename='video',
           frame_size=(640, 480),
           camera_id=0,
           render=None):
    if render == 'onscreen':
        env.mj_render()

    elif render == 'offscreen':

        global render_buffer
        if mode == 'initialize':
            render_buffer = []
            mode = 'render'

        if mode == 'render':
            curr_frame = env.render(mode='rgb_array')
            render_buffer.append(curr_frame)

        if mode == 'save':
            skvideo.io.vwrite(filename, np.asarray(render_buffer))
            print("\noffscreen buffer saved", filename)

    elif render == 'None':
        pass

    else:
        print("unknown render: ", render)


# view demos (physics ignored)
def render_demos(env, data, filename='demo_rendering.mp4', render=None):
    FPS = 30
    render_skip = max(1, round(1. / \
        (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

    viewer(env, mode='initialize', render=render)
    for i_frame in range(data['ctrl'].shape[0]):
        env.sim.data.qpos[:] = data['qpos'][i_frame].copy()
        env.sim.data.qvel[:] = data['qvel'][i_frame].copy()
        env.sim.forward()
        if i_frame % render_skip == 0:
            viewer(env, mode='render', render=render)
            print(i_frame, end=', ', flush=True)

    viewer(env, mode='save', filename=filename, render=render)
    print("time taken = %f" % (timer.time() - t0))


# playback demos and get data(physics respected)
def gather_training_data(env, data, filename='demo_playback.mp4', render=None):
    env = env.env
    FPS = 30
    render_skip = max(1, round(1. / \
        (FPS * env.sim.model.opt.timestep * env.frame_skip)))
    t0 = timer.time()

    # initialize
    env.reset()
    init_qpos = data['qpos'][0].copy()
    init_qvel = data['qvel'][0].copy()
    act_mid = env.act_mid
    act_rng = env.act_amp

    # prepare env
    env.sim.data.qpos[:] = init_qpos
    env.sim.data.qvel[:] = init_qvel
    env.sim.forward()
    viewer(env, mode='initialize', render=render)

    # step the env and gather data
    path_obs = None
    for i_frame in range(data['ctrl'].shape[0] - 1):
        # Reset every time step
        # if i_frame % 1 == 0:
        #     qp = data['qpos'][i_frame].copy()
        #     qv = data['qvel'][i_frame].copy()
        #     env.sim.data.qpos[:] = qp
        #     env.sim.data.qvel[:] = qv
        #     env.sim.forward()

        obs = env._get_obs()

        # Construct the action
        # ctrl = (data['qpos'][i_frame + 1][:9] - obs[:9]) / (env.skip * env.model.opt.timestep)
        ctrl = (data['ctrl'][i_frame] - obs[:9])/(env.skip*env.model.opt.timestep)
        act = (ctrl - act_mid) / act_rng
        act = np.clip(act, -0.999, 0.999)
        next_obs, reward, done, env_info = env.step(act)
        if path_obs is None:
            path_obs = obs
            path_act = act
        else:
            path_obs = np.vstack((path_obs, obs))
            path_act = np.vstack((path_act, act))

        # render when needed to maintain FPS
        if i_frame % render_skip == 0:
            viewer(env, mode='render', render=render)
            print(i_frame, end=', ', flush=True)

    # finalize
    if render:
        viewer(env, mode='save', filename=filename, render=render)

    t1 = timer.time()
    print("time taken = %f" % (t1 - t0))

    # note that <init_qpos, init_qvel> are one step away from <path_obs[0], path_act[0]>
    return path_obs, path_act, init_qpos, init_qvel


# MAIN =========================================================
@click.command(help="parse tele-op demos")
@click.option('--env', '-e', type=str, help='gym env name', required=True)
@click.option(
    '--demo_dir',
    '-d',
    type=str,
    help='directory with tele-op logs',
    required=True)
@click.option(
    '--skip',
    '-s',
    type=int,
    help='number of frames to skip (1:no skip)',
    default=1)
@click.option('--graph', '-g', type=bool, help='plot logs', default=False)
@click.option('--save_logs', '-l', type=bool, help='save logs', default=False)
@click.option(
    '--view', '-v', type=str, help='render/playback', default='render')
@click.option(
    '--render', '-r', type=str, help='onscreen/offscreen', default='onscreen')
def main(env, demo_dir, skip, graph, save_logs, view, render):

    gym_env = gym.make(env)
    paths = []
    print("Scanning demo_dir: " + demo_dir + "=========")
    for ind, file in enumerate(glob.glob(demo_dir + "*.mjl")):

        # process logs
        print("processing: " + file, end=': ')

        data = parse_mjl_logs(file, skip)

        print("log duration %0.2f" % (data['time'][-1] - data['time'][0]))

        # plot logs
        if (graph):
            print("plotting: " + file)
            viz_parsed_mjl_logs(data)

        # save logs
        if (save_logs):
            pickle.dump(data, open(file[:-4] + ".pkl", 'wb'))

        # render logs to video
        if view == 'render':
            render_demos(
                gym_env,
                data,
                filename=data['logName'][:-4] + '_demo_render.mp4',
                render=render)

        # playback logs and gather data
        elif view == 'playback':
            try:
                obs, act,init_qpos, init_qvel = gather_training_data(gym_env, data,\
                filename=data['logName'][:-4]+'_playback.mp4', render=render)
            except Exception as e:
                print(e)
                continue
            path = {
                'observations': obs,
                'actions': act,
                'goals': obs,
                'init_qpos': init_qpos,
                'init_qvel': init_qvel
            }
            paths.append(path)
            # accept = input('accept demo?')
            # if accept == 'n':
            #     continue
            pickle.dump(path, open(demo_dir + env + str(ind) + "_path.pkl", 'wb'))
            print(demo_dir + env + file + "_path.pkl")

if __name__ == '__main__':
    main()
```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/utils/configurable.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
import os

from gym.envs.registration import registry as gym_registry


def import_class_from_path(class_path):
    """Given 'path.to.module:object', imports and returns the object."""
    module_path, class_name = class_path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class ConfigCache(object):
    """Configuration class to store constructor arguments.

    This is used to store parameters to pass to Gym environments at init time.
    """

    def __init__(self):
        self._configs = {}
        self._default_config = {}

    def set_default_config(self, config):
        """Sets the default configuration used for all RobotEnv envs."""
        self._default_config = dict(config)

    def set_config(self, cls_or_env_id, config):
        """Sets the configuration for the given environment within a context.

        Args:
            cls_or_env_id (Class | str): A class type or Gym environment ID to
                configure.
            config (dict): The configuration parameters.
        """
        config_key = self._get_config_key(cls_or_env_id)
        self._configs[config_key] = dict(config)

    def get_config(self, cls_or_env_id):
        """Returns the configuration for the given env name.

        Args:
            cls_or_env_id (Class | str): A class type or Gym environment ID to
                get the configuration of.
        """
        config_key = self._get_config_key(cls_or_env_id)
        config = dict(self._default_config)
        config.update(self._configs.get(config_key, {}))
        return config

    def clear_config(self, cls_or_env_id):
        """Clears the configuration for the given ID."""
        config_key = self._get_config_key(cls_or_env_id)
        if config_key in self._configs:
            del self._configs[config_key]

    def _get_config_key(self, cls_or_env_id):
        if inspect.isclass(cls_or_env_id):
            return cls_or_env_id
        env_id = cls_or_env_id
        assert isinstance(env_id, str)
        if env_id not in gym_registry.env_specs:
            raise ValueError("Unregistered environment name {}.".format(env_id))
        entry_point = gym_registry.env_specs[env_id]._entry_point
        if callable(entry_point):
            return entry_point
        else:
            return import_class_from_path(entry_point)


# Global robot config.
global_config = ConfigCache()


def configurable(config_id=None, pickleable=False, config_cache=global_config):
    """Class decorator to allow injection of constructor arguments.

    This allows constructor arguments to be passed via ConfigCache.
    Example usage:

    @configurable()
    class A:
        def __init__(b=None, c=2, d='Wow'):
            ...

    global_config.set_config(A, {'b': 10, 'c': 20})
    a = A()      # b=10, c=20, d='Wow'
    a = A(b=30)  # b=30, c=20, d='Wow'

    Args:
        config_id: ID of the config to use. This defaults to the class type.
        pickleable: Whether this class is pickleable. If true, causes the pickle
            state to include the config and constructor arguments.
        config_cache: The ConfigCache to use to read config data from. Uses
            the global ConfigCache by default.
    """
    def cls_decorator(cls):
        assert inspect.isclass(cls)

        # Overwrite the class constructor to pass arguments from the config.
        base_init = cls.__init__
        def __init__(self, *args, **kwargs):

            config = config_cache.get_config(config_id or type(self))
            # Allow kwargs to override the config.
            kwargs = {**config, **kwargs}

            # print('Initializing {} with params: {}'.format(type(self).__name__,
                                                           # kwargs))

            if pickleable:
                self._pkl_env_args = args
                self._pkl_env_kwargs = kwargs

            base_init(self, *args, **kwargs)
        cls.__init__ = __init__

        # If the class is pickleable, overwrite the state methods to save
        # the constructor arguments and config.
        if pickleable:
            # Use same pickle keys as gym.utils.ezpickle for backwards compat.
            PKL_ARGS_KEY = '_ezpickle_args'
            PKL_KWARGS_KEY = '_ezpickle_kwargs'

            def __getstate__(self):
                return {
                    PKL_ARGS_KEY: self._pkl_env_args,
                    PKL_KWARGS_KEY: self._pkl_env_kwargs,
                }
            cls.__getstate__ = __getstate__

            def __setstate__(self, data):
                saved_args = data[PKL_ARGS_KEY]
                saved_kwargs = data[PKL_KWARGS_KEY]

                # Override the saved state with the current config.
                config = config_cache.get_config(config_id or type(self))
                # Allow kwargs to override the config.
                kwargs = {**saved_kwargs, **config}

                inst = type(self)(*saved_args, **kwargs)
                self.__dict__.update(inst.__dict__)
            cls.__setstate__ = __setstate__

        return cls
    return cls_decorator

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/utils/constants.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

ENVS_ROOT_PATH = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../../"))

MODELS_PATH = os.path.abspath(os.path.join(ENVS_ROOT_PATH, "../adept_models/"))

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/utils/quatmath.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
# For testing whether a number is close to zero
_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def mulQuat(qa, qb):
    res = np.zeros(4)
    res[0] = qa[0]*qb[0] - qa[1]*qb[1] - qa[2]*qb[2] - qa[3]*qb[3]
    res[1] = qa[0]*qb[1] + qa[1]*qb[0] + qa[2]*qb[3] - qa[3]*qb[2]
    res[2] = qa[0]*qb[2] - qa[1]*qb[3] + qa[2]*qb[0] + qa[3]*qb[1]
    res[3] = qa[0]*qb[3] + qa[1]*qb[2] - qa[2]*qb[1] + qa[3]*qb[0]
    return res

def negQuat(quat):
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])

def quat2Vel(quat, dt=1):
    axis = quat[1:].copy()
    sin_a_2 = np.sqrt(np.sum(axis**2))
    axis = axis/(sin_a_2+1e-8)
    speed = 2*np.arctan2(sin_a_2, quat[0])/dt
    return speed, axis

def quatDiff2Vel(quat1, quat2, dt):
    neg = negQuat(quat1)
    diff = mulQuat(quat2, neg)
    return quat2Vel(diff, dt)


def axis_angle2quat(axis, angle):
    c = np.cos(angle/2)
    s = np.sin(angle/2)
    return np.array([c, s*axis[0], s*axis[1], s*axis[2]])

def euler2mat(euler):
    """ Convert Euler Angles to Rotation Matrix.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


def euler2quat(euler):
    """ Convert Euler Angles to Quaternions.  See rotation.py for notes """
    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shape euler {}".format(euler)

    ai, aj, ak = euler[..., 2] / 2, -euler[..., 1] / 2, euler[..., 0] / 2
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    quat = np.empty(euler.shape[:-1] + (4,), dtype=np.float64)
    quat[..., 0] = cj * cc + sj * ss
    quat[..., 3] = cj * sc - sj * cs
    quat[..., 2] = -(cj * ss + sj * cc)
    quat[..., 1] = cj * cs - sj * sc
    return quat


def mat2euler(mat):
    """ Convert Rotation Matrix to Euler Angles.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(condition,
                             -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
                             -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]))
    euler[..., 1] = np.where(condition,
                             -np.arctan2(-mat[..., 0, 2], cy),
                             -np.arctan2(-mat[..., 0, 2], cy))
    euler[..., 0] = np.where(condition,
                             -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]),
                             0.0)
    return euler


def mat2quat(mat):
    """ Convert Rotation Matrix to Quaternion.  See rotation.py for notes """
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    Qxx, Qyx, Qzx = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    Qxy, Qyy, Qzy = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    Qxz, Qyz, Qzz = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]
    # Fill only lower half of symmetric matrix
    K = np.zeros(mat.shape[:-2] + (4, 4), dtype=np.float64)
    K[..., 0, 0] = Qxx - Qyy - Qzz
    K[..., 1, 0] = Qyx + Qxy
    K[..., 1, 1] = Qyy - Qxx - Qzz
    K[..., 2, 0] = Qzx + Qxz
    K[..., 2, 1] = Qzy + Qyz
    K[..., 2, 2] = Qzz - Qxx - Qyy
    K[..., 3, 0] = Qyz - Qzy
    K[..., 3, 1] = Qzx - Qxz
    K[..., 3, 2] = Qxy - Qyx
    K[..., 3, 3] = Qxx + Qyy + Qzz
    K /= 3.0
    # TODO: vectorize this -- probably could be made faster
    q = np.empty(K.shape[:-2] + (4,))
    it = np.nditer(q[..., 0], flags=['multi_index'])
    while not it.finished:
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = np.linalg.eigh(K[it.multi_index])
        # Select largest eigenvector, reorder to w,x,y,z quaternion
        q[it.multi_index] = vecs[[3, 0, 1, 2], np.argmax(vals)]
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[it.multi_index][0] < 0:
            q[it.multi_index] *= -1
        it.iternext()
    return q


def quat2euler(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    return mat2euler(quat2mat(quat))


def quat2mat(quat):
    """ Convert Quaternion to Euler Angles.  See rotation.py for notes """
    quat = np.asarray(quat, dtype=np.float64)
    assert quat.shape[-1] == 4, "Invalid shape quat {}".format(quat)

    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    Nq = np.sum(quat * quat, axis=-1)
    s = 2.0 / Nq
    X, Y, Z = x * s, y * s, z * s
    wX, wY, wZ = w * X, w * Y, w * Z
    xX, xY, xZ = x * X, x * Y, x * Z
    yY, yZ, zZ = y * Y, y * Z, z * Z

    mat = np.empty(quat.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 0, 0] = 1.0 - (yY + zZ)
    mat[..., 0, 1] = xY - wZ
    mat[..., 0, 2] = xZ + wY
    mat[..., 1, 0] = xY + wZ
    mat[..., 1, 1] = 1.0 - (xX + zZ)
    mat[..., 1, 2] = yZ - wX
    mat[..., 2, 0] = xZ - wY
    mat[..., 2, 1] = yZ + wX
    mat[..., 2, 2] = 1.0 - (xX + yY)
    return np.where((Nq > _FLOAT_EPS)[..., np.newaxis, np.newaxis], mat, np.eye(3))
```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/simulation/sim_robot.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for loading MuJoCo models."""

import os
from typing import Dict, Optional

from adept_envs.simulation import module
from adept_envs.simulation.renderer import DMRenderer, MjPyRenderer, RenderMode


class MujocoSimRobot:
    """Class that encapsulates a MuJoCo simulation.

    This class exposes methods that are agnostic to the simulation backend.
    Two backends are supported:
    1. mujoco_py - MuJoCo v1.50
    2. dm_control - MuJoCo v2.00
    """

    def __init__(self,
                 model_file: str,
                 use_dm_backend: bool = False,
                 camera_settings: Optional[Dict] = None):
        """Initializes a new simulation.

        Args:
            model_file: The MuJoCo XML model file to load.
            use_dm_backend: If True, uses DM Control's Physics (MuJoCo v2.0) as
              the backend for the simulation. Otherwise, uses mujoco_py (MuJoCo
              v1.5) as the backend.
            camera_settings: Settings to initialize the renderer's camera. This
              can contain the keys `distance`, `azimuth`, and `elevation`.
        """
        self._use_dm_backend = use_dm_backend

        if not os.path.isfile(model_file):
            raise ValueError(
                '[MujocoSimRobot] Invalid model file path: {}'.format(
                    model_file))

        if self._use_dm_backend:
            dm_mujoco = module.get_dm_mujoco()
            if model_file.endswith('.mjb'):
                self.sim = dm_mujoco.Physics.from_binary_path(model_file)
            else:
                self.sim = dm_mujoco.Physics.from_xml_path(model_file)
            self.model = self.sim.model
            self._patch_mjlib_accessors(self.model, self.sim.data)
            self.renderer = DMRenderer(
                self.sim, camera_settings=camera_settings)
        else:  # Use mujoco_py
            mujoco_py = module.get_mujoco_py()
            self.model = mujoco_py.load_model_from_path(model_file)
            self.sim = mujoco_py.MjSim(self.model)
            self.renderer = MjPyRenderer(
                self.sim, camera_settings=camera_settings)

        self.data = self.sim.data

    def close(self):
        """Cleans up any resources being used by the simulation."""
        self.renderer.close()

    def save_binary(self, path: str):
        """Saves the loaded model to a binary .mjb file."""
        if os.path.exists(path):
            raise ValueError(
                '[MujocoSimRobot] Path already exists: {}'.format(path))
        if not path.endswith('.mjb'):
            path = path + '.mjb'
        if self._use_dm_backend:
            self.model.save_binary(path)
        else:
            with open(path, 'wb') as f:
                f.write(self.model.get_mjb())

    def get_mjlib(self):
        """Returns an object that exposes the low-level MuJoCo API."""
        if self._use_dm_backend:
            return module.get_dm_mujoco().wrapper.mjbindings.mjlib
        else:
            return module.get_mujoco_py_mjlib()

    def _patch_mjlib_accessors(self, model, data):
        """Adds accessors to the DM Control objects to support mujoco_py API."""
        assert self._use_dm_backend
        mjlib = self.get_mjlib()

        def name2id(type_name, name):
            obj_id = mjlib.mj_name2id(model.ptr,
                                      mjlib.mju_str2Type(type_name.encode()),
                                      name.encode())
            if obj_id < 0:
                raise ValueError('No {} with name "{}" exists.'.format(
                    type_name, name))
            return obj_id

        if not hasattr(model, 'body_name2id'):
            model.body_name2id = lambda name: name2id('body', name)

        if not hasattr(model, 'geom_name2id'):
            model.geom_name2id = lambda name: name2id('geom', name)

        if not hasattr(model, 'site_name2id'):
            model.site_name2id = lambda name: name2id('site', name)

        if not hasattr(model, 'joint_name2id'):
            model.joint_name2id = lambda name: name2id('joint', name)

        if not hasattr(model, 'actuator_name2id'):
            model.actuator_name2id = lambda name: name2id('actuator', name)

        if not hasattr(model, 'camera_name2id'):
            model.camera_name2id = lambda name: name2id('camera', name)

        if not hasattr(data, 'body_xpos'):
            data.body_xpos = data.xpos

        if not hasattr(data, 'body_xquat'):
            data.body_xquat = data.xquat

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/simulation/module.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for caching Python modules related to simulation."""

import sys

_MUJOCO_PY_MODULE = None

_DM_MUJOCO_MODULE = None
_DM_VIEWER_MODULE = None
_DM_RENDER_MODULE = None

_GLFW_MODULE = None


def get_mujoco_py():
    """Returns the mujoco_py module."""
    global _MUJOCO_PY_MODULE
    if _MUJOCO_PY_MODULE:
        return _MUJOCO_PY_MODULE
    try:
        import mujoco_py
        # Override the warning function.
        from mujoco_py.builder import cymj
        cymj.set_warning_callback(_mj_warning_fn)
    except ImportError:
        print(
            'Failed to import mujoco_py. Ensure that mujoco_py (using MuJoCo '
            'v1.50) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _MUJOCO_PY_MODULE = mujoco_py
    return mujoco_py


def get_mujoco_py_mjlib():
    """Returns the mujoco_py mjlib module."""

    class MjlibDelegate:
        """Wrapper that forwards mjlib calls."""

        def __init__(self, lib):
            self._lib = lib

        def __getattr__(self, name: str):
            if name.startswith('mj'):
                return getattr(self._lib, '_' + name)
            raise AttributeError(name)

    return MjlibDelegate(get_mujoco_py().cymj)


def get_dm_mujoco():
    """Returns the DM Control mujoco module."""
    global _DM_MUJOCO_MODULE
    if _DM_MUJOCO_MODULE:
        return _DM_MUJOCO_MODULE
    try:
        from dm_control import mujoco
    except ImportError:
        print(
            'Failed to import dm_control.mujoco. Ensure that dm_control (using '
            'MuJoCo v2.00) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _DM_MUJOCO_MODULE = mujoco
    return mujoco


def get_dm_viewer():
    """Returns the DM Control viewer module."""
    global _DM_VIEWER_MODULE
    if _DM_VIEWER_MODULE:
        return _DM_VIEWER_MODULE
    try:
        from dm_control import viewer
    except ImportError:
        print(
            'Failed to import dm_control.viewer. Ensure that dm_control (using '
            'MuJoCo v2.00) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _DM_VIEWER_MODULE = viewer
    return viewer


def get_dm_render():
    """Returns the DM Control render module."""
    global _DM_RENDER_MODULE
    if _DM_RENDER_MODULE:
        return _DM_RENDER_MODULE
    try:
        try:
            from dm_control import _render
            render = _render
        except ImportError:
            print('Warning: DM Control is out of date.')
            from dm_control import render
    except ImportError:
        print(
            'Failed to import dm_control.render. Ensure that dm_control (using '
            'MuJoCo v2.00) is installed.',
            file=sys.stderr)
        sys.exit(1)
    _DM_RENDER_MODULE = render
    return render


def _mj_warning_fn(warn_data: bytes):
    """Warning function override for mujoco_py."""
    print('WARNING: Mujoco simulation is unstable (has NaNs): {}'.format(
        warn_data.decode()))

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/simulation/renderer.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module for viewing Physics objects in the DM Control viewer."""

import abc
import enum
import sys
from typing import Dict, Optional

import numpy as np

from adept_envs.simulation import module

# Default window dimensions.
DEFAULT_WINDOW_WIDTH = 1024
DEFAULT_WINDOW_HEIGHT = 768

DEFAULT_WINDOW_TITLE = 'MuJoCo Viewer'

_MAX_RENDERBUFFER_SIZE = 2048


class RenderMode(enum.Enum):
    """Rendering modes for offscreen rendering."""
    RGB = 0
    DEPTH = 1
    SEGMENTATION = 2


class Renderer(abc.ABC):
    """Base interface for rendering simulations."""

    def __init__(self, camera_settings: Optional[Dict] = None):
        self._camera_settings = camera_settings

    @abc.abstractmethod
    def close(self):
        """Cleans up any resources being used by the renderer."""

    @abc.abstractmethod
    def render_to_window(self):
        """Renders the simulation to a window."""

    @abc.abstractmethod
    def render_offscreen(self,
                         width: int,
                         height: int,
                         mode: RenderMode = RenderMode.RGB,
                         camera_id: int = -1) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """

    def _update_camera(self, camera):
        """Updates the given camera to move to the initial settings."""
        if not self._camera_settings:
            return
        distance = self._camera_settings.get('distance')
        azimuth = self._camera_settings.get('azimuth')
        elevation = self._camera_settings.get('elevation')
        lookat = self._camera_settings.get('lookat')

        if distance is not None:
            camera.distance = distance
        if azimuth is not None:
            camera.azimuth = azimuth
        if elevation is not None:
            camera.elevation = elevation
        if lookat is not None:
            camera.lookat[:] = lookat


class MjPyRenderer(Renderer):
    """Class for rendering mujoco_py simulations."""

    def __init__(self, sim, **kwargs):
        assert isinstance(sim, module.get_mujoco_py().MjSim), \
            'MjPyRenderer takes a mujoco_py MjSim object.'
        super().__init__(**kwargs)
        self._sim = sim
        self._onscreen_renderer = None
        self._offscreen_renderer = None

    def render_to_window(self):
        """Renders the simulation to a window."""
        if not self._onscreen_renderer:
            self._onscreen_renderer = module.get_mujoco_py().MjViewer(self._sim)
            self._update_camera(self._onscreen_renderer.cam)

        self._onscreen_renderer.render()

    def render_offscreen(self,
                         width: int,
                         height: int,
                         mode: RenderMode = RenderMode.RGB,
                         camera_id: int = -1) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """
        if not self._offscreen_renderer:
            self._offscreen_renderer = module.get_mujoco_py() \
                .MjRenderContextOffscreen(self._sim)

        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera(self._offscreen_renderer.cam)

        self._offscreen_renderer.render(width, height, camera_id)
        if mode == RenderMode.RGB:
            data = self._offscreen_renderer.read_pixels(
                width, height, depth=False)
            # Original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == RenderMode.DEPTH:
            data = self._offscreen_renderer.read_pixels(
                width, height, depth=True)[1]
            # Original image is upside-down, so flip it
            return data[::-1, :]
        else:
            raise NotImplementedError(mode)

    def close(self):
        """Cleans up any resources being used by the renderer."""


class DMRenderer(Renderer):
    """Class for rendering DM Control Physics objects."""

    def __init__(self, physics, **kwargs):
        assert isinstance(physics, module.get_dm_mujoco().Physics), \
            'DMRenderer takes a DM Control Physics object.'
        super().__init__(**kwargs)
        self._physics = physics
        self._window = None

        # Set the camera to lookat the center of the geoms. (mujoco_py does
        # this automatically.
        if 'lookat' not in self._camera_settings:
            self._camera_settings['lookat'] = [
                np.median(self._physics.data.geom_xpos[:, i]) for i in range(3)
            ]

    def render_to_window(self):
        """Renders the Physics object to a window.

        The window continuously renders the Physics in a separate thread.

        This function is a no-op if the window was already created.
        """
        if not self._window:
            self._window = DMRenderWindow()
            self._window.load_model(self._physics)
            self._update_camera(self._window.camera)
        self._window.run_frame()

    def render_offscreen(self,
                         width: int,
                         height: int,
                         mode: RenderMode = RenderMode.RGB,
                         camera_id: int = -1) -> np.ndarray:
        """Renders the camera view as a NumPy array of pixels.

        Args:
            width: The viewport width (pixels).
            height: The viewport height (pixels).
            mode: The rendering mode.
            camera_id: The ID of the camera to render from. By default, uses
                the free camera.

        Returns:
            A NumPy array of the pixels.
        """
        mujoco = module.get_dm_mujoco()
        # TODO(michaelahn): Consider caching the camera.
        camera = mujoco.Camera(
            physics=self._physics,
            height=height,
            width=width,
            camera_id=camera_id)

        # Update the camera configuration for the free-camera.
        if camera_id == -1:
            self._update_camera(
                camera._render_camera,  # pylint: disable=protected-access
            )

        image = camera.render(
            depth=(mode == RenderMode.DEPTH),
            segmentation=(mode == RenderMode.SEGMENTATION))
        camera._scene.free()  # pylint: disable=protected-access
        return image

    def close(self):
        """Cleans up any resources being used by the renderer."""
        if self._window:
            self._window.close()
            self._window = None


class DMRenderWindow:
    """Class that encapsulates a graphical window."""

    def __init__(self,
                 width: int = DEFAULT_WINDOW_WIDTH,
                 height: int = DEFAULT_WINDOW_HEIGHT,
                 title: str = DEFAULT_WINDOW_TITLE):
        """Creates a graphical render window.

        Args:
            width: The width of the window.
            height: The height of the window.
            title: The title of the window.
        """
        dmv = module.get_dm_viewer()
        self._viewport = dmv.renderer.Viewport(width, height)
        self._window = dmv.gui.RenderWindow(width, height, title)
        self._viewer = dmv.viewer.Viewer(self._viewport, self._window.mouse,
                                         self._window.keyboard)
        self._draw_surface = None
        self._renderer = dmv.renderer.NullRenderer()

    @property
    def camera(self):
        return self._viewer._camera._camera

    def close(self):
        self._viewer.deinitialize()
        self._renderer.release()
        self._draw_surface.free()
        self._window.close()

    def load_model(self, physics):
        """Loads the given Physics object to render."""
        self._viewer.deinitialize()

        self._draw_surface = module.get_dm_render().Renderer(
            max_width=_MAX_RENDERBUFFER_SIZE, max_height=_MAX_RENDERBUFFER_SIZE)
        self._renderer = module.get_dm_viewer().renderer.OffScreenRenderer(
            physics.model, self._draw_surface)

        self._viewer.initialize(physics, self._renderer, touchpad=False)

    def run_frame(self):
        """Renders one frame of the simulation.

        NOTE: This is extremely slow at the moment.
        """
        glfw = module.get_dm_viewer().gui.glfw_gui.glfw
        glfw_window = self._window._context.window
        if glfw.window_should_close(glfw_window):
            sys.exit(0)

        self._viewport.set_size(*self._window.shape)
        self._viewer.render()
        pixels = self._renderer.pixels

        with self._window._context.make_current() as ctx:
            ctx.call(self._window._update_gui_on_render_thread, glfw_window,
                     pixels)
        self._window._mouse.process_events()
        self._window._keyboard.process_events()

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/__init__.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import adept_envs.franka

from adept_envs.utils.configurable import global_config

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/mujoco_env.py
```python
"""Base environment for MuJoCo-based environments."""

#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections
import os
import time
from typing import Dict, Optional

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from adept_envs.simulation.sim_robot import MujocoSimRobot, RenderMode

DEFAULT_RENDER_SIZE = 480

USE_DM_CONTROL = True


class MujocoEnv(gym.Env):
    """Superclass for all MuJoCo environments."""

    def __init__(self,
                 model_path: str,
                 frame_skip: int,
                 camera_settings: Optional[Dict] = None,
                 use_dm_backend: Optional[bool] = None,
                 ):
        """Initializes a new MuJoCo environment.

        Args:
            model_path: The path to the MuJoCo XML file.
            frame_skip: The number of simulation steps per environment step. On
              hardware this influences the duration of each environment step.
            camera_settings: Settings to initialize the simulation camera. This
              can contain the keys `distance`, `azimuth`, and `elevation`.
            use_dm_backend: A boolean to switch between mujoco-py and dm_control.
        """
        self._seed()
        if not os.path.isfile(model_path):
            raise IOError(
                '[MujocoEnv]: Model path does not exist: {}'.format(model_path))
        self.frame_skip = frame_skip

        self.sim_robot = MujocoSimRobot(
            model_path,
            use_dm_backend=use_dm_backend or USE_DM_CONTROL,
            camera_settings=camera_settings)
        self.sim = self.sim_robot.sim
        self.model = self.sim_robot.model
        self.data = self.sim_robot.data

        self.metadata = {
            'render.modes': ['human', 'rgb_array', 'depth_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.mujoco_render_frames = False

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()
        observation, _reward, done, _info = self.step(np.zeros(self.model.nu))
        assert not done

        bounds = self.model.actuator_ctrlrange.copy()
        act_upper = bounds[:, 1]
        act_lower = bounds[:, 0]

        # Define the action and observation spaces.
        # HACK: MJRL is still using gym 0.9.x so we can't provide a dtype.
        try:
            self.action_space = spaces.Box(
                act_lower, act_upper, dtype=np.float32)
            if isinstance(observation, collections.Mapping):
                self.observation_space = spaces.Dict({
                k: spaces.Box(-np.inf, np.inf, shape=v.shape, dtype=np.float32) for k, v in observation.items()})
            else:
                self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size
                self.observation_space = spaces.Box(
                -np.inf, np.inf, observation.shape, dtype=np.float32)

        except TypeError:
            # Fallback case for gym 0.9.x
            self.action_space = spaces.Box(act_lower, act_upper)
            assert not isinstance(observation, collections.Mapping), 'gym 0.9.x does not support dictionary observation.'
            self.obs_dim = np.sum([o.size for o in observation]) if type(observation) is tuple else observation.size
            self.observation_space = spaces.Box(
                -np.inf, np.inf, observation.shape)

    def seed(self, seed=None):  # Compatibility with new gym
        return self._seed(seed)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """Reset the robot degrees of freedom (qpos and qvel).

        Implement this in each subclass.
        """
        raise NotImplementedError

    # -----------------------------

    def reset(self):  # compatibility with new gym
        return self._reset()

    def _reset(self):
        self.sim.reset()
        self.sim.forward()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        # we are directly manipulating mujoco state here
        data = self.sim.data # MjData
        for i in range(self.model.nq):
            data.qpos[i] = qpos[i]
        for i in range(self.model.nv):
            data.qvel[i] = qvel[i]
        # state = np.concatenate([self.data.qpos, self.data.qvel, self.data.act])
        # self.sim.set_state(state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        for i in range(self.model.nu):
            self.sim.data.ctrl[i] = ctrl[i]

        for _ in range(n_frames):
            self.sim.step()

            # TODO(michaelahn): Remove this; render should be called separately.
            if self.mujoco_render_frames is True:
                self.mj_render()

    def render(self,
               mode='human',
               width=DEFAULT_RENDER_SIZE,
               height=DEFAULT_RENDER_SIZE,
               camera_id=-1):
        """Renders the environment.

        Args:
            mode: The type of rendering to use.
                - 'human': Renders to a graphical window.
                - 'rgb_array': Returns the RGB image as an np.ndarray.
                - 'depth_array': Returns the depth image as an np.ndarray.
            width: The width of the rendered image. This only affects offscreen
                rendering.
            height: The height of the rendered image. This only affects
                offscreen rendering.
            camera_id: The ID of the camera to use. By default, this is the free
                camera. If specified, only affects offscreen rendering.
        """
        if mode == 'human':
            self.sim_robot.renderer.render_to_window()
        elif mode == 'rgb_array':
            assert width and height
            return self.sim_robot.renderer.render_offscreen(
                width, height, mode=RenderMode.RGB, camera_id=camera_id)
        elif mode == 'depth_array':
            assert width and height
            return self.sim_robot.renderer.render_offscreen(
                width, height, mode=RenderMode.DEPTH, camera_id=camera_id)
        else:
            raise NotImplementedError(mode)

    def close(self):
        self.sim_robot.close()

    def mj_render(self):
        """Backwards compatibility with MJRL."""
        self.render(mode='human')

    def state_vector(self):
        state = self.sim.get_state()
        return np.concatenate([state.qpos.flat, state.qvel.flat])
```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/base_robot.py
```python
#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from collections import deque

class BaseRobot(object):
    """Base class for all robot classes."""

    def __init__(self,
                 n_jnt,
                 n_obj,
                 pos_bounds=None,
                 vel_bounds=None,
                 calibration_path=None,
                 is_hardware=False,
                 device_name=None,
                 overlay=False,
                 calibration_mode=False,
                 observation_cache_maxsize=5):
        """Create a new robot.
        Args:
            n_jnt: The number of dofs in the robot.
            n_obj: The number of dofs in the object.
            pos_bounds: (n_jnt, 2)-shape matrix denoting the min and max joint
                position for each joint.
            vel_bounds: (n_jnt, 2)-shape matrix denoting the min and max joint
                velocity for each joint.
            calibration_path: File path to the calibration configuration file to
                use.
            is_hardware: Whether to run on hardware or not.
            device_name: The device path for the robot hardware. Only required
                in legacy mode.
            overlay: Whether to show a simulation overlay of the hardware.
            calibration_mode: Start with motors disengaged.
        """

        assert n_jnt > 0
        assert n_obj >= 0

        self._n_jnt = n_jnt
        self._n_obj = n_obj
        self._n_dofs = n_jnt + n_obj

        self._pos_bounds = None
        if pos_bounds is not None:
            pos_bounds = np.array(pos_bounds, dtype=np.float32)
            assert pos_bounds.shape == (self._n_dofs, 2)
            for low, high in pos_bounds:
                assert low < high
            self._pos_bounds = pos_bounds
        self._vel_bounds = None
        if vel_bounds is not None:
            vel_bounds = np.array(vel_bounds, dtype=np.float32)
            assert vel_bounds.shape == (self._n_dofs, 2)
            for low, high in vel_bounds:
                assert low < high
            self._vel_bounds = vel_bounds

        self._is_hardware = is_hardware
        self._device_name = device_name
        self._calibration_path = calibration_path
        self._overlay = overlay
        self._calibration_mode = calibration_mode
        self._observation_cache_maxsize = observation_cache_maxsize

        # Gets updated
        self._observation_cache = deque([], maxlen=self._observation_cache_maxsize)


    @property
    def n_jnt(self):
        return self._n_jnt

    @property
    def n_obj(self):
        return self._n_obj

    @property
    def n_dofs(self):
        return self._n_dofs

    @property
    def pos_bounds(self):
        return self._pos_bounds

    @property
    def vel_bounds(self):
        return self._vel_bounds

    @property
    def is_hardware(self):
        return self._is_hardware

    @property
    def device_name(self):
        return self._device_name

    @property
    def calibration_path(self):
        return self._calibration_path

    @property
    def overlay(self):
        return self._overlay

    @property
    def has_obj(self):
        return self._n_obj > 0

    @property
    def calibration_mode(self):
        return self._calibration_mode

    @property
    def observation_cache_maxsize(self):
        return self._observation_cache_maxsize

    @property
    def observation_cache(self):
        return self._observation_cache


    def clip_positions(self, positions):
        """Clips the given joint positions to the position bounds.

        Args:
            positions: The joint positions.

        Returns:
            The bounded joint positions.
        """
        if self.pos_bounds is None:
            return positions
        assert len(positions) == self.n_jnt or len(positions) == self.n_dofs
        pos_bounds = self.pos_bounds[:len(positions)]
        return np.clip(positions, pos_bounds[:, 0], pos_bounds[:, 1])


```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/relay_policy_learning/adept_envs/adept_envs/robot_env.py
```python
"""Base class for robotics environments."""

#!/usr/bin/python
#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import importlib
import inspect
import os
from typing import Dict, Optional

import numpy as np


from adept_envs import mujoco_env
from adept_envs.base_robot import BaseRobot
from adept_envs.utils.configurable import import_class_from_path
from adept_envs.utils.constants import MODELS_PATH


class RobotEnv(mujoco_env.MujocoEnv):
    """Base environment for all adept robots."""

    # Mapping of robot name to fully qualified class path.
    # e.g. 'robot': 'adept_envs.dclaw.robot.Robot'
    # Subclasses should override this to specify the Robot classes they support.
    ROBOTS = {}

    # Mapping of device path to the calibration file to use. If the device path
    # is not found, the 'default' key is used.
    # This can be overridden by subclasses.
    CALIBRATION_PATHS = {}

    def __init__(self,
                 model_path: str,
                 robot: BaseRobot,
                 frame_skip: int,
                 camera_settings: Optional[Dict] = None):
        """Initializes a robotics environment.

        Args:
            model_path: The path to the model to run. Relative paths will be
              interpreted as relative to the 'adept_models' folder.
            robot: The Robot object to use.
            frame_skip: The number of simulation steps per environment step. On
              hardware this influences the duration of each environment step.
            camera_settings: Settings to initialize the simulation camera. This
              can contain the keys `distance`, `azimuth`, and `elevation`.
        """
        self._robot = robot

        # Initial pose for first step.
        self.desired_pose = np.zeros(self.n_jnt)

        if not model_path.startswith('/'):
            model_path = os.path.abspath(os.path.join(MODELS_PATH, model_path))

        self.remote_viz = None

        try:
            from adept_envs.utils.remote_viz import RemoteViz
            self.remote_viz = RemoteViz(model_path)
        except ImportError:
            pass          


        self._initializing = True
        super(RobotEnv, self).__init__(
            model_path, frame_skip, camera_settings=camera_settings)
        self._initializing = False


    @property
    def robot(self):
        return self._robot

    @property
    def n_jnt(self):
        return self._robot.n_jnt

    @property
    def n_obj(self):
        return self._robot.n_obj

    @property
    def skip(self):
        """Alias for frame_skip. Needed for MJRL."""
        return self.frame_skip

    @property
    def initializing(self):
        return self._initializing

    def close_env(self):
        if self._robot is not None:
            self._robot.close()

    def make_robot(self,
                   n_jnt,
                   n_obj=0,
                   is_hardware=False,
                   device_name=None,
                   legacy=False,
                   **kwargs):
        """Creates a new robot for the environment.

        Args:
            n_jnt: The number of joints in the robot.
            n_obj: The number of object joints in the robot environment.
            is_hardware: Whether to run on hardware or not.
            device_name: The device path for the robot hardware.
            legacy: If true, runs using direct dynamixel communication rather
              than DDS.
            kwargs: See BaseRobot for other parameters.

        Returns:
            A Robot object.
        """
        if not self.ROBOTS:
            raise NotImplementedError('Subclasses must override ROBOTS.')

        if is_hardware and not device_name:
            raise ValueError('Must provide device name if running on hardware.')

        robot_name = 'dds_robot' if not legacy and is_hardware else 'robot'
        if robot_name not in self.ROBOTS:
            raise KeyError("Unsupported robot '{}', available: {}".format(
                robot_name, list(self.ROBOTS.keys())))

        cls = import_class_from_path(self.ROBOTS[robot_name])

        calibration_path = None
        if self.CALIBRATION_PATHS:
            if not device_name:
                calibration_name = 'default'
            elif device_name not in self.CALIBRATION_PATHS:
                print('Device "{}" not in CALIBRATION_PATHS; using default.'
                      .format(device_name))
                calibration_name = 'default'
            else:
                calibration_name = device_name

            calibration_path = self.CALIBRATION_PATHS[calibration_name]
            if not os.path.isfile(calibration_path):
                raise OSError('Could not find calibration file at: {}'.format(
                    calibration_path))

        return cls(
            n_jnt,
            n_obj,
            is_hardware=is_hardware,
            device_name=device_name,
            calibration_path=calibration_path,
            **kwargs)

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/__init__.py
```python
"""Environments using kitchen and Franka robot."""
from gym.envs.registration import register

register(
    id="kitchen-microwave-kettle-light-slider-v0",
    entry_point="diffusion_policy.env.kitchen.v0:KitchenMicrowaveKettleLightSliderV0",
    max_episode_steps=280,
    reward_threshold=1.0,
)

register(
    id="kitchen-microwave-kettle-burner-light-v0",
    entry_point="diffusion_policy.env.kitchen.v0:KitchenMicrowaveKettleBottomBurnerLightV0",
    max_episode_steps=280,
    reward_threshold=1.0,
)

register(
    id="kitchen-kettle-microwave-light-slider-v0",
    entry_point="diffusion_policy.env.kitchen.v0:KitchenKettleMicrowaveLightSliderV0",
    max_episode_steps=280,
    reward_threshold=1.0,
)

register(
    id="kitchen-all-v0",
    entry_point="diffusion_policy.env.kitchen.v0:KitchenAllV0",
    max_episode_steps=280,
    reward_threshold=1.0,
)

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/base.py
```python
import sys
import os
# hack to import adept envs
ADEPT_DIR = os.path.join(os.path.dirname(__file__), 'relay_policy_learning', 'adept_envs')
sys.path.append(ADEPT_DIR)

import logging
import numpy as np
import adept_envs
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3
logger = logging.getLogger()


class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    COMPLETE_IN_ANY_ORDER = (
        True  # This allows for the tasks to be completed in arbitrary order.
    )

    def __init__(
        self, dataset_url=None, ref_max_score=None, ref_min_score=None, 
        use_abs_action=False,
        **kwargs
    ):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.goal_masking = True
        super(KitchenBase, self).__init__(use_abs_action=use_abs_action, **kwargs)

    def set_goal_masking(self, goal_masking=True):
        """Sets goal masking for goal-conditioned approaches (like RPL)."""
        self.goal_masking = goal_masking

    def _get_task_goal(self, task=None, actually_return_goal=False):
        if task is None:
            task = ["microwave", "kettle", "bottom burner", "light switch"]
        new_goal = np.zeros_like(self.goal)
        if self.goal_masking and not actually_return_goal:
            return new_goal
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        reward = 0.0
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        next_goal = self._get_task_goal(
            task=self.TASK_ELEMENTS, actually_return_goal=True
        )  # obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(
                next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx]
            )
            complete = distance < BONUS_THRESH
            condition = (
                complete and all_completed_so_far
                if not self.COMPLETE_IN_ANY_ORDER
                else complete
            )
            if condition:  # element == self.tasks_to_complete[0]:
                print("Task {} completed!".format(element))
                completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        if self.TERMINATE_ON_WRONG_COMPLETE:
            all_goal = self._get_task_goal(task=self.ALL_TASKS)
            for wrong_task in list(set(self.ALL_TASKS) - set(self.TASK_ELEMENTS)):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(obs[..., element_idx] - all_goal[element_idx])
                complete = distance < BONUS_THRESH
                if complete:
                    done = True
                    break
        env_info["completed_tasks"] = set(self.TASK_ELEMENTS) - set(
            self.tasks_to_complete
        )
        return obs, reward, done, env_info

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError

    def _split_data_into_seqs(self, data):
        """Splits dataset object into list of sequence dicts."""
        seq_end_idxs = np.where(data["terminals"])[0]
        start = 0
        seqs = []
        for end_idx in seq_end_idxs:
            seqs.append(
                dict(
                    states=data["observations"][start : end_idx + 1],
                    actions=data["actions"][start : end_idx + 1],
                )
            )
            start = end_idx + 1
        return seqs

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/kitchen_util.py
```python
import struct
import numpy as np

def parse_mjl_logs(read_filename, skipamount):
    with open(read_filename, mode='rb') as file:
        fileContent = file.read()
    headers = struct.unpack('iiiiiii', fileContent[:28])
    nq = headers[0]
    nv = headers[1]
    nu = headers[2]
    nmocap = headers[3]
    nsensordata = headers[4]
    nuserdata = headers[5]
    name_len = headers[6]
    name = struct.unpack(str(name_len) + 's', fileContent[28:28+name_len])[0]
    rem_size = len(fileContent[28 + name_len:])
    num_floats = int(rem_size/4)
    dat = np.asarray(struct.unpack(str(num_floats) + 'f', fileContent[28+name_len:]))
    recsz = 1 + nq + nv + nu + 7*nmocap + nsensordata + nuserdata
    if rem_size % recsz != 0:
        print("ERROR")
    else:
        dat = np.reshape(dat, (int(len(dat)/recsz), recsz))
        dat = dat.T

    time = dat[0,:][::skipamount] - 0*dat[0, 0]
    qpos = dat[1:nq + 1, :].T[::skipamount, :]
    qvel = dat[nq+1:nq+nv+1,:].T[::skipamount, :]
    ctrl = dat[nq+nv+1:nq+nv+nu+1,:].T[::skipamount,:]
    mocap_pos = dat[nq+nv+nu+1:nq+nv+nu+3*nmocap+1,:].T[::skipamount, :]
    mocap_quat = dat[nq+nv+nu+3*nmocap+1:nq+nv+nu+7*nmocap+1,:].T[::skipamount, :]
    sensordata = dat[nq+nv+nu+7*nmocap+1:nq+nv+nu+7*nmocap+nsensordata+1,:].T[::skipamount,:]
    userdata = dat[nq+nv+nu+7*nmocap+nsensordata+1:,:].T[::skipamount,:]

    data = dict(nq=nq,
               nv=nv,
               nu=nu,
               nmocap=nmocap,
               nsensordata=nsensordata,
               name=name,
               time=time,
               qpos=qpos,
               qvel=qvel,
               ctrl=ctrl,
               mocap_pos=mocap_pos,
               mocap_quat=mocap_quat,
               sensordata=sensordata,
               userdata=userdata,
               logName = read_filename
               )
    return data

```

## reference_material/diffusion_policy_code/diffusion_policy/env/kitchen/v0.py
```python
from diffusion_policy.env.kitchen.base import KitchenBase


class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "bottom burner", "light switch"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "slide cabinet"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenKettleMicrowaveLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "microwave", "light switch", "slide cabinet"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenAllV0(KitchenBase):
    TASK_ELEMENTS = KitchenBase.ALL_TASKS

```

## reference_material/diffusion_policy_code/diffusion_policy/env/pusht/pusht_env.py
```python
import gym
from gym import spaces

import collections
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from diffusion_policy.env.pusht.pymunk_override import DrawOptions


def pymunk_to_shapely(body, shapes):
    geoms = list()
    for shape in shapes:
        if isinstance(shape, pymunk.shapes.Poly):
            verts = [body.local_to_world(v) for v in shape.get_vertices()]
            verts += [verts[0]]
            geoms.append(sg.Polygon(verts))
        else:
            raise RuntimeError(f'Unsupported shape type {type(shape)}')
    geom = sg.MultiPolygon(geoms)
    return geom

class PushTEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 10}
    reward_range = (0., 1.)

    def __init__(self,
            legacy=False, 
            block_cog=None, damping=None,
            render_action=True,
            render_size=96,
            reset_to_state=None
        ):
        self._seed = None
        self.seed()
        self.window_size = ws = 512  # The size of the PyGame window
        self.render_size = render_size
        self.sim_hz = 100
        # Local controller params.
        self.k_p, self.k_v = 100, 20    # PD control.z
        self.control_hz = self.metadata['video.frames_per_second']
        # legcay set_state for data compatibility
        self.legacy = legacy

        # agent_pos, block_pos, block_angle
        self.observation_space = spaces.Box(
            low=np.array([0,0,0,0,0], dtype=np.float64),
            high=np.array([ws,ws,ws,ws,np.pi*2], dtype=np.float64),
            shape=(5,),
            dtype=np.float64
        )

        # positional goal for agent
        self.action_space = spaces.Box(
            low=np.array([0,0], dtype=np.float64),
            high=np.array([ws,ws], dtype=np.float64),
            shape=(2,),
            dtype=np.float64
        )

        self.block_cog = block_cog
        self.damping = damping
        self.render_action = render_action

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        self.screen = None

        self.space = None
        self.teleop = None
        self.render_buffer = None
        self.latest_action = None
        self.reset_to_state = reset_to_state
    
    def reset(self):
        seed = self._seed
        self._setup()
        if self.block_cog is not None:
            self.block.center_of_gravity = self.block_cog
        if self.damping is not None:
            self.space.damping = self.damping
        
        # use legacy RandomState for compatibility
        state = self.reset_to_state
        if state is None:
            rs = np.random.RandomState(seed=seed)
            state = np.array([
                rs.randint(50, 450), rs.randint(50, 450),
                rs.randint(100, 400), rs.randint(100, 400),
                rs.randn() * 2 * np.pi - np.pi
                ])
        self._set_state(state)

        observation = self._get_obs()
        return observation

    def step(self, action):
        dt = 1.0 / self.sim_hz
        self.n_contact_points = 0
        n_steps = self.sim_hz // self.control_hz
        if action is not None:
            self.latest_action = action
            for i in range(n_steps):
                # Step PD control.
                # self.agent.velocity = self.k_p * (act - self.agent.position)    # P control works too.
                acceleration = self.k_p * (action - self.agent.position) + self.k_v * (Vec2d(0, 0) - self.agent.velocity)
                self.agent.velocity += acceleration * dt

                # Step physics.
                self.space.step(dt)

        # compute reward
        goal_body = self._get_goal_pose_body(self.goal_pose)
        goal_geom = pymunk_to_shapely(goal_body, self.block.shapes)
        block_geom = pymunk_to_shapely(self.block, self.block.shapes)

        intersection_area = goal_geom.intersection(block_geom).area
        goal_area = goal_geom.area
        coverage = intersection_area / goal_area
        reward = np.clip(coverage / self.success_threshold, 0, 1)
        done = coverage > self.success_threshold

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, done, info

    def render(self, mode):
        return self._render_frame(mode)

    def teleop_agent(self):
        TeleopAgent = collections.namedtuple('TeleopAgent', ['act'])
        def act(obs):
            act = None
            mouse_position = pymunk.pygame_util.from_pygame(Vec2d(*pygame.mouse.get_pos()), self.screen)
            if self.teleop or (mouse_position - self.agent.position).length < 30:
                self.teleop = True
                act = mouse_position
            return act
        return TeleopAgent(act)

    def _get_obs(self):
        obs = np.array(
            tuple(self.agent.position) \
            + tuple(self.block.position) \
            + (self.block.angle % (2 * np.pi),))
        return obs

    def _get_goal_pose_body(self, pose):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (50, 100))
        body = pymunk.Body(mass, inertia)
        # preserving the legacy assignment order for compatibility
        # the order here doesn't matter somehow, maybe because CoM is aligned with body origin
        body.position = pose[:2].tolist()
        body.angle = pose[2]
        return body
    
    def _get_info(self):
        n_steps = self.sim_hz // self.control_hz
        n_contact_points_per_step = int(np.ceil(self.n_contact_points / n_steps))
        info = {
            'pos_agent': np.array(self.agent.position),
            'vel_agent': np.array(self.agent.velocity),
            'block_pose': np.array(list(self.block.position) + [self.block.angle]),
            'goal_pose': self.goal_pose,
            'n_contacts': n_contact_points_per_step}
        return info

    def _render_frame(self, mode):

        if self.window is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        self.screen = canvas

        draw_options = DrawOptions(canvas)

        # Draw goal pose.
        goal_body = self._get_goal_pose_body(self.goal_pose)
        for shape in self.block.shapes:
            goal_points = [pymunk.pygame_util.to_pygame(goal_body.local_to_world(v), draw_options.surface) for v in shape.get_vertices()]
            goal_points += [goal_points[0]]
            pygame.draw.polygon(canvas, self.goal_color, goal_points)

        # Draw agent and block.
        self.space.debug_draw(draw_options)

        if mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # the clock is already ticked during in step for "human"


        img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        img = cv2.resize(img, (self.render_size, self.render_size))
        if self.render_action:
            if self.render_action and (self.latest_action is not None):
                action = np.array(self.latest_action)
                coord = (action / 512 * 96).astype(np.int32)
                marker_size = int(8/96*self.render_size)
                thickness = int(1/96*self.render_size)
                cv2.drawMarker(img, coord,
                    color=(255,0,0), markerType=cv2.MARKER_CROSS,
                    markerSize=marker_size, thickness=thickness)
        return img


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0,25536)
        self._seed = seed
        self.np_random = np.random.default_rng(seed)

    def _handle_collision(self, arbiter, space, data):
        self.n_contact_points += len(arbiter.contact_point_set.points)

    def _set_state(self, state):
        if isinstance(state, np.ndarray):
            state = state.tolist()
        pos_agent = state[:2]
        pos_block = state[2:4]
        rot_block = state[4]
        self.agent.position = pos_agent
        # setting angle rotates with respect to center of mass
        # therefore will modify the geometric position
        # if not the same as CoM
        # therefore should be modified first.
        if self.legacy:
            # for compatibility with legacy data
            self.block.position = pos_block
            self.block.angle = rot_block
        else:
            self.block.angle = rot_block
            self.block.position = pos_block

        # Run physics to take effect
        self.space.step(1.0 / self.sim_hz)
    
    def _set_state_local(self, state_local):
        agent_pos_local = state_local[:2]
        block_pose_local = state_local[2:]
        tf_img_obj = st.AffineTransform(
            translation=self.goal_pose[:2], 
            rotation=self.goal_pose[2])
        tf_obj_new = st.AffineTransform(
            translation=block_pose_local[:2],
            rotation=block_pose_local[2]
        )
        tf_img_new = st.AffineTransform(
            matrix=tf_img_obj.params @ tf_obj_new.params
        )
        agent_pos_new = tf_img_new(agent_pos_local)
        new_state = np.array(
            list(agent_pos_new[0]) + list(tf_img_new.translation) \
                + [tf_img_new.rotation])
        self._set_state(new_state)
        return new_state

    def _setup(self):
        self.space = pymunk.Space()
        self.space.gravity = 0, 0
        self.space.damping = 0
        self.teleop = False
        self.render_buffer = list()
        
        # Add walls.
        walls = [
            self._add_segment((5, 506), (5, 5), 2),
            self._add_segment((5, 5), (506, 5), 2),
            self._add_segment((506, 5), (506, 506), 2),
            self._add_segment((5, 506), (506, 506), 2)
        ]
        self.space.add(*walls)

        # Add agent, block, and goal zone.
        self.agent = self.add_circle((256, 400), 15)
        self.block = self.add_tee((256, 300), 0)
        self.goal_color = pygame.Color('LightGreen')
        self.goal_pose = np.array([256,256,np.pi/4])  # x, y, theta (in radians)

        # Add collision handling
        self.collision_handeler = self.space.add_collision_handler(0, 0)
        self.collision_handeler.post_solve = self._handle_collision
        self.n_contact_points = 0

        self.max_score = 50 * 100
        self.success_threshold = 0.95    # 95% coverage.

    def _add_segment(self, a, b, radius):
        shape = pymunk.Segment(self.space.static_body, a, b, radius)
        shape.color = pygame.Color('LightGray')    # https://htmlcolorcodes.com/color-names
        return shape

    def add_circle(self, position, radius):
        body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        body.position = position
        body.friction = 1
        shape = pymunk.Circle(body, radius)
        shape.color = pygame.Color('RoyalBlue')
        self.space.add(body, shape)
        return body

    def add_box(self, position, height, width):
        mass = 1
        inertia = pymunk.moment_for_box(mass, (height, width))
        body = pymunk.Body(mass, inertia)
        body.position = position
        shape = pymunk.Poly.create_box(body, (height, width))
        shape.color = pygame.Color('LightSlateGray')
        self.space.add(body, shape)
        return body

    def add_tee(self, position, angle, scale=30, color='LightSlateGray', mask=pymunk.ShapeFilter.ALL_MASKS()):
        mass = 1
        length = 4
        vertices1 = [(-length*scale/2, scale),
                                 ( length*scale/2, scale),
                                 ( length*scale/2, 0),
                                 (-length*scale/2, 0)]
        inertia1 = pymunk.moment_for_poly(mass, vertices=vertices1)
        vertices2 = [(-scale/2, scale),
                                 (-scale/2, length*scale),
                                 ( scale/2, length*scale),
                                 ( scale/2, scale)]
        inertia2 = pymunk.moment_for_poly(mass, vertices=vertices1)
        body = pymunk.Body(mass, inertia1 + inertia2)
        shape1 = pymunk.Poly(body, vertices1)
        shape2 = pymunk.Poly(body, vertices2)
        shape1.color = pygame.Color(color)
        shape2.color = pygame.Color(color)
        shape1.filter = pymunk.ShapeFilter(mask=mask)
        shape2.filter = pymunk.ShapeFilter(mask=mask)
        body.center_of_gravity = (shape1.center_of_gravity + shape2.center_of_gravity) / 2
        body.position = position
        body.angle = angle
        body.friction = 1
        self.space.add(body, shape1, shape2)
        return body

```

## reference_material/diffusion_policy_code/diffusion_policy/env/pusht/pusht_keypoints_env.py
```python
from typing import Dict, Sequence, Union, Optional
from gym import spaces
from diffusion_policy.env.pusht.pusht_env import PushTEnv
from diffusion_policy.env.pusht.pymunk_keypoint_manager import PymunkKeypointManager
import numpy as np

class PushTKeypointsEnv(PushTEnv):
    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96,
            keypoint_visible_rate=1.0, 
            agent_keypoints=False,
            draw_keypoints=False,
            reset_to_state=None,
            render_action=True,
            local_keypoint_map: Dict[str, np.ndarray]=None, 
            color_map: Optional[Dict[str, np.ndarray]]=None):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            reset_to_state=reset_to_state,
            render_action=render_action)
        ws = self.window_size

        if local_keypoint_map is None:
            # create default keypoint definition
            kp_kwargs = self.genenerate_keypoint_manager_params()
            local_keypoint_map = kp_kwargs['local_keypoint_map']
            color_map = kp_kwargs['color_map']

        # create observation spaces
        Dblockkps = np.prod(local_keypoint_map['block'].shape)
        Dagentkps = np.prod(local_keypoint_map['agent'].shape)
        Dagentpos = 2

        Do = Dblockkps
        if agent_keypoints:
            # blockkp + agnet_pos
            Do += Dagentkps
        else:
            # blockkp + agnet_kp
            Do += Dagentpos
        # obs + obs_mask
        Dobs = Do * 2

        low = np.zeros((Dobs,), dtype=np.float64)
        high = np.full_like(low, ws)
        # mask range 0-1
        high[Do:] = 1.

        # (block_kps+agent_kps, xy+confidence)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=np.float64
        )

        self.keypoint_visible_rate = keypoint_visible_rate
        self.agent_keypoints = agent_keypoints
        self.draw_keypoints = draw_keypoints
        self.kp_manager = PymunkKeypointManager(
            local_keypoint_map=local_keypoint_map,
            color_map=color_map)
        self.draw_kp_map = None

    @classmethod
    def genenerate_keypoint_manager_params(cls):
        env = PushTEnv()
        kp_manager = PymunkKeypointManager.create_from_pusht_env(env)
        kp_kwargs = kp_manager.kwargs
        return kp_kwargs

    def _get_obs(self):
        # get keypoints
        obj_map = {
            'block': self.block
        }
        if self.agent_keypoints:
            obj_map['agent'] = self.agent

        kp_map = self.kp_manager.get_keypoints_global(
            pose_map=obj_map, is_obj=True)
        # python dict guerentee order of keys and values
        kps = np.concatenate(list(kp_map.values()), axis=0)

        # select keypoints to drop
        n_kps = kps.shape[0]
        visible_kps = self.np_random.random(size=(n_kps,)) < self.keypoint_visible_rate
        kps_mask = np.repeat(visible_kps[:,None], 2, axis=1)

        # save keypoints for rendering
        vis_kps = kps.copy()
        vis_kps[~visible_kps] = 0
        draw_kp_map = {
            'block': vis_kps[:len(kp_map['block'])]
        }
        if self.agent_keypoints:
            draw_kp_map['agent'] = vis_kps[len(kp_map['block']):]
        self.draw_kp_map = draw_kp_map
        
        # construct obs
        obs = kps.flatten()
        obs_mask = kps_mask.flatten()
        if not self.agent_keypoints:
            # passing agent position when keypoints are not available
            agent_pos = np.array(self.agent.position)
            obs = np.concatenate([
                obs, agent_pos
            ])
            obs_mask = np.concatenate([
                obs_mask, np.ones((2,), dtype=bool)
            ])

        # obs, obs_mask
        obs = np.concatenate([
            obs, obs_mask.astype(obs.dtype)
        ], axis=0)
        return obs
    
    
    def _render_frame(self, mode):
        img = super()._render_frame(mode)
        if self.draw_keypoints:
            self.kp_manager.draw_keypoints(
                img, self.draw_kp_map, radius=int(img.shape[0]/96))
        return img

```

## reference_material/diffusion_policy_code/diffusion_policy/env/pusht/__init__.py
```python
from gym.envs.registration import register
import diffusion_policy.env.pusht

register(
    id='pusht-keypoints-v0',
    entry_point='envs.pusht.pusht_keypoints_env:PushTKeypointsEnv',
    max_episode_steps=200,
    reward_threshold=1.0
)
```

## reference_material/diffusion_policy_code/diffusion_policy/env/pusht/pusht_image_env.py
```python
from gym import spaces
from diffusion_policy.env.pusht.pusht_env import PushTEnv
import numpy as np
import cv2

class PushTImageEnv(PushTEnv):
    metadata = {"render.modes": ["rgb_array"], "video.frames_per_second": 10}

    def __init__(self,
            legacy=False,
            block_cog=None, 
            damping=None,
            render_size=96):
        super().__init__(
            legacy=legacy, 
            block_cog=block_cog,
            damping=damping,
            render_size=render_size,
            render_action=False)
        ws = self.window_size
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=1,
                shape=(3,render_size,render_size),
                dtype=np.float32
            ),
            'agent_pos': spaces.Box(
                low=0,
                high=ws,
                shape=(2,),
                dtype=np.float32
            )
        })
        self.render_cache = None
    
    def _get_obs(self):
        img = super()._render_frame(mode='rgb_array')

        agent_pos = np.array(self.agent.position)
        img_obs = np.moveaxis(img.astype(np.float32) / 255, -1, 0)
        obs = {
            'image': img_obs,
            'agent_pos': agent_pos
        }

        # draw action
        if self.latest_action is not None:
            action = np.array(self.latest_action)
            coord = (action / 512 * 96).astype(np.int32)
            marker_size = int(8/96*self.render_size)
            thickness = int(1/96*self.render_size)
            cv2.drawMarker(img, coord,
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=marker_size, thickness=thickness)
        self.render_cache = img

        return obs

    def render(self, mode):
        assert mode == 'rgb_array'

        if self.render_cache is None:
            self._get_obs()
        
        return self.render_cache

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/utils/pose3d.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple 6DOF pose container.
"""

import dataclasses
import numpy as np
from scipy.spatial import transform


class NoCopyAsDict(object):
    """Base class for dataclasses. Avoids a copy in the asdict() call."""

    def asdict(self):
        """Replacement for dataclasses.asdict.

        TF Dataset does not handle dataclasses.asdict, which uses copy.deepcopy when
        setting values in the output dict. This causes issues with tf.Dataset.
        Instead, shallow copy contents.

        Returns:
          dict containing contents of dataclass.
        """
        return {k.name: getattr(self, k.name) for k in dataclasses.fields(self)}


@dataclasses.dataclass
class Pose3d(NoCopyAsDict):
    """Simple container for translation and rotation."""

    rotation: transform.Rotation
    translation: np.ndarray

    @property
    def vec7(self):
        return np.concatenate([self.translation, self.rotation.as_quat()])

    def serialize(self):
        return {
            "rotation": self.rotation.as_quat().tolist(),
            "translation": self.translation.tolist(),
        }

    @staticmethod
    def deserialize(data):
        return Pose3d(
            rotation=transform.Rotation.from_quat(data["rotation"]),
            translation=np.array(data["translation"]),
        )

    def __eq__(self, other):
        return np.array_equal(
            self.rotation.as_quat(), other.rotation.as_quat()
        ) and np.array_equal(self.translation, other.translation)

    def __ne__(self, other):
        return not self.__eq__(other)

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/utils/utils_pybullet.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Assortment of utilities to interact with bullet within g3."""
import dataclasses
import datetime
import getpass
import gzip
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from absl import logging
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
import numpy as np
from scipy.spatial import transform
import six


import pybullet
import pybullet_data
import pybullet_utils.bullet_client as bullet_client

Vec3 = Tuple[float, float, float]
Vec4 = Tuple[float, float, float, float]
PYBULLET_STATE_VERSION = 2  # Basic versioning of serialized pybullet state.


# Note about rotation_to_matrix and matrix_to_rotation below:
# The abstractions below allow us to use older versions of scipy.
def rotation_to_matrix(rotation):
    if hasattr(rotation, "as_dcm"):
        return rotation.as_dcm()
    else:
        assert hasattr(rotation, "as_matrix")
        return rotation.as_matrix()


def matrix_to_rotation(matrix):
    if hasattr(transform.Rotation, "from_dcm"):
        return transform.Rotation.from_dcm(matrix)
    else:
        assert hasattr(transform.Rotation, "from_matrix")
        return transform.Rotation.from_matrix(matrix)


def load_urdf(pybullet_client, file_path, *args, **kwargs):
    """Loads the given URDF filepath."""

    # Handles most general file open case.
    try:
        if os.path.exists(file_path):
            return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet_client.error:
        pass

    try:
        import pathlib
        asset_path = str(pathlib.Path(__file__).parent.parent.joinpath('assets'))
        if file_path.startswith("third_party/py/envs/assets/"):
            pybullet_client.setAdditionalSearchPath(asset_path)
            file_path = file_path[len("third_party/py/envs/assets/") :]
        if file_path.startswith(
            "third_party/bullet/examples/pybullet/gym/pybullet_data/"
            ):
            pybullet_client.setAdditionalSearchPath(pybullet_data.getDataPath())
            file_path = file_path[55:]
        # logging.info("Loading URDF %s", file_path)
        return pybullet_client.loadURDF(file_path, *args, **kwargs)
    except pybullet.error:
        raise FileNotFoundError("Cannot load the URDF file {}".format(file_path))


def add_visual_sphere(client, center=(0, 0, 0), radius=0.1, rgba=(0.5, 0.5, 0.5, 0.5)):
    """Add a sphere to bullet scene (visual only, no physics).

    Args:
      client: pybullet client (or pybullet library handle).
      center: Center of sphere.
      radius: Sphere radius.
      rgba: rgba color of sphere.

    Returns:
      Unique integer bullet id of constructed object.
    """
    vis_obj_id = client.createVisualShape(
        client.GEOM_SPHERE, radius=radius, rgbaColor=rgba
    )
    obj_id = client.createMultiBody(
        baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis_obj_id, basePosition=center
    )
    return obj_id


def pybullet_mat_to_numpy_4x4(pybullet_matrix):
    assert len(pybullet_matrix) == 16, "pybullet matrix should be len 16"
    return np.transpose(np.reshape(np.array(pybullet_matrix, dtype=np.float64), (4, 4)))


def decompose_view_matrix(pybullet_view_matrix):
    """Decompose view matrix into pos + quat format (assumes mat is rigid!)."""
    # It would be MUCH better to use something from bullet, however pybullet does
    # not expose all of the linear algebra library.
    mat = pybullet_mat_to_numpy_4x4(pybullet_view_matrix)

    # View matrix is now:
    # | R_11 R_12 R_13 t_1 |
    # | R_21 R_22 R_23 t_2 |
    # | R_31 R_32 R_33 t_3 |
    # |    0    0    0   1 |

    # R is the inverse eye to target at orientation, and t is R * eye.
    mat_view_to_world = np.linalg.inv(mat)

    # mat_view_to_world is the view to world transform, therefore the translation
    # component of this matrix is simply the world space position (since mat *
    # (0, 0, 0, 1)) is just copying the right column.
    world_xyz_view = np.copy(mat_view_to_world[0:3, 3])

    mat_view_to_world[0:3, 3] = 0  # Zero out the position change.
    world_quat_view = matrix_to_rotation(mat_view_to_world).as_quat()

    return world_xyz_view, world_quat_view


def world_obj_to_view(world_xyz_obj, world_quat_obj, camera_view, client):
    """Transform object into view space."""
    world_xyz_view, world_quat_view = decompose_view_matrix(camera_view)
    view_xyz_world, view_quat_world = client.invertTransform(
        world_xyz_view, world_quat_view
    )
    view_xyz_obj, view_quat_obj = client.multiplyTransforms(
        view_xyz_world, view_quat_world, world_xyz_obj, world_quat_obj
    )

    return view_xyz_obj, view_quat_obj


def image_xy_to_view_ray(xy, cam_width, cam_height, proj_mat_inv):
    """Calculate view-space ray from pixel location."""
    # Recall (from http://www.songho.ca/opengl/gl_projectionmatrix.html):
    # xyzw_clip = M_proj * xyzw_eye, and
    # xyz_ndc = xyzw_clip[0:3] / xwzw_clip[3].
    xyz_ndc = np.array(
        [2.0 * xy[0] / cam_width - 1.0, -(2.0 * xy[1] / cam_height - 1.0), 0]
    )  # in [-1, 1]
    xyzw_clip = np.concatenate([xyz_ndc, [1]])
    xyzw_eye = proj_mat_inv @ xyzw_clip
    origin = np.zeros(3)
    vec = xyzw_eye[:3] / max(np.linalg.norm(xyzw_eye[:3]), 1e-6)
    return origin, vec


def view_ray_to_world_ray(origin, vec, view_mat_inv):
    """Transform view-space ray into world space."""
    origin = view_mat_inv @ np.concatenate([origin, [1]])
    vec = view_mat_inv @ np.concatenate([vec, [0]])

    return origin[:3], vec[:3]


def ray_to_plane_test(ray_origin, ray_vec, plane_origin, plane_normal):
    """Perform a ray-plane intersection test."""
    ln = np.dot(plane_normal, ray_vec)
    if abs(ln) < np.finfo(np.float32).eps:
        return None

    # Solve for the intersection fraction t.
    t = np.dot(plane_normal, plane_origin - ray_origin) / ln
    if t >= 0:
        return ray_origin + ray_vec * t
    else:
        return None


def get_workspace(env):
    (
        workspace_origin,
        workspace_quat,
    ) = env.pybullet_client.getBasePositionAndOrientation(env.workspace_uid)
    workspace_normal = rotation_to_matrix(transform.Rotation.from_quat(workspace_quat))[
        2, 0:3
    ]

    return workspace_origin, workspace_normal


def reset_camera_pose(env, view_type):
    """Reset camera pose to canonical frame."""
    p = env.pybullet_client

    if view_type == "POLICY":
        camera_info = p.getDebugVisualizerCamera()
        image_size = (camera_info[0], camera_info[1])

        viewm, _, front_position, lookat, _ = env.calc_camera_params(image_size)

        euler = matrix_to_rotation(pybullet_mat_to_numpy_4x4(viewm)[0:3, 0:3]).as_euler(
            "xyz", degrees=False
        )
        pitch = euler[1]
        yaw = -euler[2]
        # The distance is a bit far away (the GL view has higher FOV).
        distance = np.linalg.norm(front_position - lookat) * 0.6
    elif view_type == "TOP_DOWN":
        workspace_origin, _ = get_workspace(env)
        distance = 0.5
        lookat = workspace_origin
        yaw = np.pi / 2
        # Note: pi/2 pitch results in gimble lock and pybullet doesn't support it.
        pitch = -(np.pi / 2 - 1e-5)
    else:
        raise ValueError("unsupported view_type %s" % view_type)
    p.resetDebugVisualizerCamera(
        cameraDistance=distance,
        cameraYaw=360 * yaw / (2.0 * np.pi),
        cameraPitch=360 * pitch / (2.0 * np.pi),
        cameraTargetPosition=lookat,
    )


def _lists_to_tuple(obj):
    if isinstance(obj, list):
        return tuple([_lists_to_tuple(v) for v in obj])
    else:
        return obj


@dataclasses.dataclass
class ObjState:
    """A container for storing pybullet object state."""

    obj_id: int

    # base_pose: (xyz, quat).
    base_pose: Tuple[Vec3, Vec4]
    # base_vel: (vel, ang_vel).
    base_vel: Tuple[Vec3, Vec3]
    joint_info: Any
    joint_state: Any

    @staticmethod
    def get_bullet_state(client, obj_id):
        """Read Pybullet internal state."""
        base_pose = client.getBasePositionAndOrientation(obj_id)
        base_vel = client.getBaseVelocity(obj_id)

        joint_info = []
        joint_state = []
        for i in range(client.getNumJoints(obj_id)):
            joint_state.append(client.getJointState(obj_id, i))
            joint_info.append(ObjState._get_joint_info(client, obj_id, i))

        return ObjState(
            obj_id=obj_id,
            base_pose=base_pose,
            base_vel=base_vel,
            joint_info=tuple(joint_info),
            joint_state=tuple(joint_state),
        )

    @staticmethod
    def _get_joint_info(client, obj_id, joint_index):
        ji = client.getJointInfo(obj_id, joint_index)
        return tuple([v if not isinstance(v, bytes) else v.decode("utf-8") for v in ji])

    def set_bullet_state(self, client, obj_id):
        """Hard set the current bullet state."""
        xyz, quat = self.base_pose
        client.resetBasePositionAndOrientation(obj_id, xyz, quat)
        vel, ang_vel = self.base_vel
        client.resetBaseVelocity(obj_id, vel, ang_vel)

        njoints = client.getNumJoints(obj_id)
        if njoints != len(self.joint_info) or njoints != len(self.joint_state):
            raise ValueError("Incorrect number of joint info state pairs.")

        for i, (joint_info, joint_state) in enumerate(
            zip(self.joint_info, self.joint_state)
        ):
            joint_index = joint_info[0]
            if joint_index != i:
                raise ValueError("Joint index mismatch.")

            # Check that the current joint we're trying to restore state for has the
            # same info as the state joint.
            cur_joint_info = ObjState._get_joint_info(client, obj_id, joint_index)
            if cur_joint_info != joint_info:
                raise ValueError(
                    "joint_info mismatch %s vs %s (expected)"
                    % (str(cur_joint_info), str(joint_info))
                )
            joint_position = joint_state[0]
            joint_velocity = joint_state[1]
            client.resetJointState(
                obj_id, i, targetValue=joint_position, targetVelocity=joint_velocity
            )

    def serialize(self):
        return {
            "obj_id": self.obj_id,
            "base_pose": self.base_pose,
            "base_vel": self.base_vel,
            "joint_info": self.joint_info,
            "joint_state": self.joint_state,
        }

    @staticmethod
    def deserialize(data):
        return ObjState(
            obj_id=_lists_to_tuple(data["obj_id"]),
            base_pose=_lists_to_tuple(data["base_pose"]),
            base_vel=_lists_to_tuple(data["base_vel"]),
            joint_info=_lists_to_tuple(data["joint_info"]),
            joint_state=_lists_to_tuple(data["joint_state"]),
        )


@dataclasses.dataclass
class XarmState(ObjState):
    """A container for storing pybullet robot state."""

    # The set point of the robot's controller.
    target_effector_pose: Pose3d
    goal_translation: Optional[Vec3]

    @staticmethod
    def get_bullet_state(client, obj_id, target_effector_pose, goal_translation):
        if goal_translation is not None:
            goal_translation = tuple(goal_translation.tolist())
        return XarmState(
            **dataclasses.asdict(ObjState.get_bullet_state(client, obj_id)),
            target_effector_pose=target_effector_pose,
            goal_translation=goal_translation
        )

    def serialize(self):
        data = ObjState.serialize(self)
        data["target_effector_pose"] = self.target_effector_pose.serialize()
        if self.goal_translation is not None:
            data["goal_translation"] = self.goal_translation
        else:
            data["goal_translation"] = []
        return data

    @staticmethod
    def deserialize(data):
        goal_translation = (
            None
            if not data["goal_translation"]
            else _lists_to_tuple(data["goal_translation"])
        )
        return XarmState(
            obj_id=data["obj_id"],
            base_pose=_lists_to_tuple(data["base_pose"]),
            base_vel=_lists_to_tuple(data["base_vel"]),
            joint_info=_lists_to_tuple(data["joint_info"]),
            joint_state=_lists_to_tuple(data["joint_state"]),
            goal_translation=goal_translation,
            target_effector_pose=Pose3d.deserialize(data["target_effector_pose"]),
        )


def _serialize_pybullet_state(pybullet_state):
    """Convert data to POD types."""
    if isinstance(pybullet_state, list):
        return [_serialize_pybullet_state(entry) for entry in pybullet_state]
    elif isinstance(pybullet_state, dict):
        assert "_serialized_obj_name" not in pybullet_state
        return {
            key: _serialize_pybullet_state(value)
            for key, value in pybullet_state.items()
        }
    elif isinstance(pybullet_state, (XarmState, ObjState)):
        return {
            "_serialized_obj_name": type(pybullet_state).__name__,
            "_serialized_data": pybullet_state.serialize(),
        }
    elif isinstance(pybullet_state, int):
        return pybullet_state
    else:
        raise ValueError(
            "Unhandled type for object %s, type %s"
            % (str(pybullet_state), type(pybullet_state))
        )


def _deserialize_pybullet_state(state):
    """Parse data from POD types."""
    if isinstance(state, list):
        return [_deserialize_pybullet_state(item) for item in state]
    elif isinstance(state, dict):
        if "_serialized_obj_name" in state:
            if state["_serialized_obj_name"] == XarmState.__name__:
                return XarmState.deserialize(state["_serialized_data"])
            elif state["_serialized_obj_name"] == ObjState.__name__:
                return ObjState.deserialize(state["_serialized_data"])
            else:
                raise ValueError("Unsupported: %s" % state["_serialized_obj_name"])
        else:
            return {
                key: _deserialize_pybullet_state(value) for key, value in state.items()
            }
    elif isinstance(state, int):
        return state
    else:
        raise ValueError("Unhandled type for object %s" % str(state))


def write_pybullet_state(filename, pybullet_state, task, actions=None):
    """Serialize pybullet state to json file."""
    import torch
    data = {
        "pybullet_state": _serialize_pybullet_state(pybullet_state),
        "state_version": PYBULLET_STATE_VERSION,
        "ts_ms": int(time.mktime(datetime.datetime.now().timetuple())) * 1000,
        "user": getpass.getuser(),
        "task": task,
        "actions": actions if actions is not None else [],
    }
    torch.save(data, filename)


def read_pybullet_state(filename):
    """Deserialize pybullet state from json file."""
    import torch
    data = torch.load(filename)

    assert isinstance(data, dict)

    if data["state_version"] != PYBULLET_STATE_VERSION:
        raise ValueError(
            "incompatible state data (version %d, expected %d)"
            % (data["state_version"], PYBULLET_STATE_VERSION)
        )

    data["pybullet_state"] = _deserialize_pybullet_state(data["pybullet_state"])
    return data

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/utils/xarm_sim_robot.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XArm Robot Kinematics."""
from diffusion_policy.env.block_pushing.utils import utils_pybullet
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
import numpy as np
from scipy.spatial import transform
import pybullet

XARM_URDF_PATH = (
    "third_party/bullet/examples/pybullet/gym/pybullet_data/" "xarm/xarm6_robot.urdf"
)
SUCTION_URDF_PATH = "third_party/py/envs/assets/suction/" "suction-head-long.urdf"
CYLINDER_URDF_PATH = "third_party/py/envs/assets/suction/" "cylinder.urdf"
CYLINDER_REAL_URDF_PATH = "third_party/py/envs/assets/suction/" "cylinder_real.urdf"
HOME_JOINT_POSITIONS = np.deg2rad([0, -20, -80, 0, 100, -30])


class XArmSimRobot:
    """A simulated PyBullet XArm robot, mostly for forward/inverse kinematics."""

    def __init__(
        self,
        pybullet_client,
        initial_joint_positions=HOME_JOINT_POSITIONS,
        end_effector="none",
        color="default",
    ):
        self._pybullet_client = pybullet_client
        self.initial_joint_positions = initial_joint_positions

        if color == "default":
            self.xarm = utils_pybullet.load_urdf(
                pybullet_client, XARM_URDF_PATH, [0, 0, 0]
            )
        else:
            raise ValueError("Unrecognized xarm color %s" % color)

        # Get revolute joints of robot (skip fixed joints).
        joints = []
        joint_indices = []
        for i in range(self._pybullet_client.getNumJoints(self.xarm)):
            joint_info = self._pybullet_client.getJointInfo(self.xarm, i)
            if joint_info[2] == pybullet.JOINT_REVOLUTE:
                joints.append(joint_info[0])
                joint_indices.append(i)
                # Note examples in pybullet do this, but it is not clear what the
                # benefits are.
                self._pybullet_client.changeDynamics(
                    self.xarm, i, linearDamping=0, angularDamping=0
                )

        self._n_joints = len(joints)
        self._joints = tuple(joints)
        self._joint_indices = tuple(joint_indices)

        # Move robot to home joint configuration
        self.reset_joints(self.initial_joint_positions)
        self.effector_link = 6

        if (
            end_effector == "suction"
            or end_effector == "cylinder"
            or end_effector == "cylinder_real"
        ):
            self.end_effector = self._setup_end_effector(end_effector)
        else:
            if end_effector != "none":
                raise ValueError('end_effector "%s" is not supported.' % end_effector)
            self.end_effector = None

    def _setup_end_effector(self, end_effector):
        """Adds a suction or cylinder end effector."""
        pose = self.forward_kinematics()
        if end_effector == "suction":
            body = utils_pybullet.load_urdf(
                self._pybullet_client,
                SUCTION_URDF_PATH,
                pose.translation,
                pose.rotation.as_quat(),
            )
        elif end_effector == "cylinder":
            body = utils_pybullet.load_urdf(
                self._pybullet_client,
                CYLINDER_URDF_PATH,
                pose.translation,
                pose.rotation.as_quat(),
            )
        elif end_effector == "cylinder_real":
            body = utils_pybullet.load_urdf(
                self._pybullet_client,
                CYLINDER_REAL_URDF_PATH,
                pose.translation,
                pose.rotation.as_quat(),
            )
        else:
            raise ValueError('end_effector "%s" is not supported.' % end_effector)

        constraint_id = self._pybullet_client.createConstraint(
            parentBodyUniqueId=self.xarm,
            parentLinkIndex=6,
            childBodyUniqueId=body,
            childLinkIndex=-1,
            jointType=pybullet.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=(0, 0, 0),
        )
        self._pybullet_client.changeConstraint(constraint_id, maxForce=50)

        return body

    def reset_joints(self, joint_values):
        """Sets the position of the Robot's joints.

        *Note*: This should only be used at the start while not running the
                simulation resetJointState overrides all physics simulation.

        Args:
          joint_values: Iterable with desired joint positions.
        """
        for i in range(self._n_joints):
            self._pybullet_client.resetJointState(
                self.xarm, self._joints[i], joint_values[i]
            )

    def get_joints_measured(self):
        joint_states = self._pybullet_client.getJointStates(
            self.xarm, self._joint_indices
        )
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        joint_torques = np.array([state[3] for state in joint_states])
        return joint_positions, joint_velocities, joint_torques

    def get_joint_positions(self):
        joint_states = self._pybullet_client.getJointStates(
            self.xarm, self._joint_indices
        )
        joint_positions = np.array([state[0] for state in joint_states])
        return joint_positions

    def forward_kinematics(self):
        """Forward kinematics."""
        effector_state = self._pybullet_client.getLinkState(
            self.xarm, self.effector_link
        )
        return Pose3d(
            translation=np.array(effector_state[0]),
            rotation=transform.Rotation.from_quat(effector_state[1]),
        )

    def inverse_kinematics(
        self, world_effector_pose, max_iterations=100, residual_threshold=1e-10
    ):
        """Inverse kinematics.

        Args:
          world_effector_pose: Target Pose3d for the robot's end effector.
          max_iterations: Refine the IK solution until the distance between target
            and actual end effector position is below this threshold, or the
            maxNumIterations is reached. Default is 20 iterations.
          residual_threshold: Refine the IK solution until the distance between
            target and actual end effector position is below this threshold, or the
            maxNumIterations is reached.

        Returns:
          Numpy array with required joint angles to reach the requested pose.
        """
        return np.array(
            self._pybullet_client.calculateInverseKinematics(
                self.xarm,
                self.effector_link,
                world_effector_pose.translation,
                world_effector_pose.rotation.as_quat(),  # as_quat returns xyzw.
                lowerLimits=[-17] * 6,
                upperLimits=[17] * 6,
                jointRanges=[17] * 6,
                restPoses=[0, 0] + self.get_joint_positions()[2:].tolist(),
                maxNumIterations=max_iterations,
                residualThreshold=residual_threshold,
            )
        )

    def set_target_effector_pose(self, world_effector_pose):
        target_joint_positions = self.inverse_kinematics(world_effector_pose)
        self.set_target_joint_positions(target_joint_positions)

    def set_target_joint_velocities(self, target_joint_velocities):
        self._pybullet_client.setJointMotorControlArray(
            self.xarm,
            self._joint_indices,
            pybullet.VELOCITY_CONTROL,
            targetVelocities=target_joint_velocities,
            forces=[5 * 240.0] * 6,
        )

    def set_target_joint_positions(self, target_joint_positions):
        self._pybullet_client.setJointMotorControlArray(
            self.xarm,
            self._joint_indices,
            pybullet.POSITION_CONTROL,
            targetPositions=target_joint_positions,
            forces=[5 * 240.0] * 6,
        )

    def set_alpha_transparency(self, alpha):
        visual_shape_data = self._pybullet_client.getVisualShapeData(self.xarm)

        for i in range(self._pybullet_client.getNumJoints(self.xarm)):
            object_id, link_index, _, _, _, _, _, rgba_color = visual_shape_data[i]
            assert object_id == self.xarm, "xarm id mismatch."
            assert link_index == i, "Link visual data was returned out of order."
            rgba_color = list(rgba_color[0:3]) + [alpha]
            self._pybullet_client.changeVisualShape(
                self.xarm, linkIndex=i, rgbaColor=rgba_color
            )

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/oracles/pushing_info.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataclass holding info needed for pushing oracles."""
import dataclasses
from typing import Any


@dataclasses.dataclass
class PushingInfo:
    """Holds onto info necessary for pushing state machine."""

    xy_block: Any = None
    xy_ee: Any = None
    xy_pre_block: Any = None
    xy_delta_to_nexttoblock: Any = None
    xy_delta_to_touchingblock: Any = None
    xy_dir_block_to_ee: Any = None
    theta_threshold_to_orient: Any = None
    theta_threshold_flat_enough: Any = None
    theta_error: Any = None
    obstacle_poses: Any = None
    distance_to_target: Any = None

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/oracles/discontinuous_push_oracle.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pushes to first target, waits, then pushes to second target."""

import diffusion_policy.env.block_pushing.oracles.oriented_push_oracle as oriented_push_oracle_module
import numpy as np
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import


class DiscontinuousOrientedPushOracle(oriented_push_oracle_module.OrientedPushOracle):
    """Pushes to first target, waits, then pushes to second target."""

    def __init__(self, env, goal_tolerance=0.04, wait=0):
        super(DiscontinuousOrientedPushOracle, self).__init__(env)
        self._countdown = 0
        self._wait = wait
        self._goal_dist_tolerance = goal_tolerance

    def reset(self):
        self.phase = "move_to_pre_block"
        self._countdown = 0

    def _action(self, time_step, policy_state):
        if time_step.is_first():
            self.reset()
            # Move to first target first.
            self._current_target = "target"
            self._has_switched = False

        def _block_target_dist(block, target):
            dist = np.linalg.norm(
                time_step.observation["%s_translation" % block]
                - time_step.observation["%s_translation" % target]
            )
            return dist

        d1 = _block_target_dist("block", "target")
        if d1 < self._goal_dist_tolerance and not self._has_switched:
            self._countdown = self._wait
            # If first block has been pushed to first target, switch to second block.
            self._has_switched = True
            self._current_target = "target2"

        xy_delta = self._get_action_for_block_target(
            time_step, block="block", target=self._current_target
        )

        if self._countdown > 0:
            xy_delta = np.zeros_like(xy_delta)
            self._countdown -= 1

        return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/oracles/oriented_push_oracle.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Oracle for pushing task which orients the block then pushes it."""

import diffusion_policy.env.block_pushing.oracles.pushing_info as pushing_info_module
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import


class OrientedPushOracle(py_policy.PyPolicy):
    """Oracle for pushing task which orients the block then pushes it."""

    def __init__(self, env, action_noise_std=0.0):
        super(OrientedPushOracle, self).__init__(
            env.time_step_spec(), env.action_spec()
        )
        self._env = env
        self._np_random_state = np.random.RandomState(0)
        self.phase = "move_to_pre_block"
        self._action_noise_std = action_noise_std

    def reset(self):
        self.phase = "move_to_pre_block"

    def get_theta_from_vector(self, vector):
        return np.arctan2(vector[1], vector[0])

    def theta_to_rotation2d(self, theta):
        r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        return r

    def rotate(self, theta, xy_dir_block_to_ee):
        rot_2d = self.theta_to_rotation2d(theta)
        return rot_2d @ xy_dir_block_to_ee

    def _get_action_info(self, time_step, block, target):
        xy_block = time_step.observation["%s_translation" % block][:2]
        theta_block = time_step.observation["%s_orientation" % block]
        xy_target = time_step.observation["%s_translation" % target][:2]
        xy_ee = time_step.observation["effector_target_translation"][:2]

        xy_block_to_target = xy_target - xy_block
        xy_dir_block_to_target = (xy_block_to_target) / np.linalg.norm(
            xy_block_to_target
        )
        theta_to_target = self.get_theta_from_vector(xy_dir_block_to_target)

        theta_error = theta_to_target - theta_block
        # Block has 4-way symmetry.
        while theta_error > np.pi / 4:
            theta_error -= np.pi / 2.0
        while theta_error < -np.pi / 4:
            theta_error += np.pi / 2.0

        xy_pre_block = xy_block + -xy_dir_block_to_target * 0.05
        xy_nexttoblock = xy_block + -xy_dir_block_to_target * 0.03
        xy_touchingblock = xy_block + -xy_dir_block_to_target * 0.01
        xy_delta_to_nexttoblock = xy_nexttoblock - xy_ee
        xy_delta_to_touchingblock = xy_touchingblock - xy_ee

        xy_block_to_ee = xy_ee - xy_block
        xy_dir_block_to_ee = xy_block_to_ee / np.linalg.norm(xy_block_to_ee)

        theta_threshold_to_orient = 0.2
        theta_threshold_flat_enough = 0.03
        return pushing_info_module.PushingInfo(
            xy_block=xy_block,
            xy_ee=xy_ee,
            xy_pre_block=xy_pre_block,
            xy_delta_to_nexttoblock=xy_delta_to_nexttoblock,
            xy_delta_to_touchingblock=xy_delta_to_touchingblock,
            xy_dir_block_to_ee=xy_dir_block_to_ee,
            theta_threshold_to_orient=theta_threshold_to_orient,
            theta_threshold_flat_enough=theta_threshold_flat_enough,
            theta_error=theta_error,
        )

    def _get_move_to_preblock(self, xy_pre_block, xy_ee):
        max_step_velocity = 0.3
        # Go 5 cm away from the block, on the line between the block and target.
        xy_delta_to_preblock = xy_pre_block - xy_ee
        diff = np.linalg.norm(xy_delta_to_preblock)
        if diff < 0.001:
            self.phase = "move_to_block"
        xy_delta = xy_delta_to_preblock
        return xy_delta, max_step_velocity

    def _get_move_to_block(
        self, xy_delta_to_nexttoblock, theta_threshold_to_orient, theta_error
    ):
        diff = np.linalg.norm(xy_delta_to_nexttoblock)
        if diff < 0.001:
            self.phase = "push_block"
        # If need to re-oorient, then re-orient.
        if theta_error > theta_threshold_to_orient:
            self.phase = "orient_block_left"
        if theta_error < -theta_threshold_to_orient:
            self.phase = "orient_block_right"
        # Otherwise, push into the block.
        xy_delta = xy_delta_to_nexttoblock
        return xy_delta

    def _get_push_block(
        self, theta_error, theta_threshold_to_orient, xy_delta_to_touchingblock
    ):
        # If need to reorient, go back to move_to_pre_block, move_to_block first.
        if theta_error > theta_threshold_to_orient:
            self.phase = "move_to_pre_block"
        if theta_error < -theta_threshold_to_orient:
            self.phase = "move_to_pre_block"
        xy_delta = xy_delta_to_touchingblock
        return xy_delta

    def _get_orient_block_left(
        self,
        xy_dir_block_to_ee,
        orient_circle_diameter,
        xy_block,
        xy_ee,
        theta_error,
        theta_threshold_flat_enough,
    ):
        xy_dir_block_to_ee = self.rotate(0.2, xy_dir_block_to_ee)
        xy_block_to_ee = xy_dir_block_to_ee * orient_circle_diameter
        xy_push_left_spot = xy_block + xy_block_to_ee
        xy_delta = xy_push_left_spot - xy_ee
        if theta_error < theta_threshold_flat_enough:
            self.phase = "move_to_pre_block"
        return xy_delta

    def _get_orient_block_right(
        self,
        xy_dir_block_to_ee,
        orient_circle_diameter,
        xy_block,
        xy_ee,
        theta_error,
        theta_threshold_flat_enough,
    ):
        xy_dir_block_to_ee = self.rotate(-0.2, xy_dir_block_to_ee)
        xy_block_to_ee = xy_dir_block_to_ee * orient_circle_diameter
        xy_push_left_spot = xy_block + xy_block_to_ee
        xy_delta = xy_push_left_spot - xy_ee
        if theta_error > -theta_threshold_flat_enough:
            self.phase = "move_to_pre_block"
        return xy_delta

    def _get_action_for_block_target(self, time_step, block="block", target="target"):
        # Specifying this as velocity makes it independent of control frequency.
        max_step_velocity = 0.35
        info = self._get_action_info(time_step, block, target)

        if self.phase == "move_to_pre_block":
            xy_delta, max_step_velocity = self._get_move_to_preblock(
                info.xy_pre_block, info.xy_ee
            )

        if self.phase == "move_to_block":
            xy_delta = self._get_move_to_block(
                info.xy_delta_to_nexttoblock,
                info.theta_threshold_to_orient,
                info.theta_error,
            )

        if self.phase == "push_block":
            xy_delta = self._get_push_block(
                info.theta_error,
                info.theta_threshold_to_orient,
                info.xy_delta_to_touchingblock,
            )

        orient_circle_diameter = 0.025

        if self.phase == "orient_block_left" or self.phase == "orient_block_right":
            max_step_velocity = 0.15

        if self.phase == "orient_block_left":
            xy_delta = self._get_orient_block_left(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough,
            )

        if self.phase == "orient_block_right":
            xy_delta = self._get_orient_block_right(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough,
            )

        if self._action_noise_std != 0.0:
            xy_delta += self._np_random_state.randn(2) * self._action_noise_std

        max_step_distance = max_step_velocity * (1 / self._env.get_control_frequency())
        length = np.linalg.norm(xy_delta)
        if length > max_step_distance:
            xy_direction = xy_delta / length
            xy_delta = xy_direction * max_step_distance
        return xy_delta

    def _action(self, time_step, policy_state):
        if time_step.is_first():
            self.reset()
        xy_delta = self._get_action_for_block_target(
            time_step, block="block", target="target"
        )
        return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))


class OrientedPushNormalizedOracle(py_policy.PyPolicy):
    """Oracle for pushing task which orients the block then pushes it."""

    def __init__(self, env):
        super(OrientedPushNormalizedOracle, self).__init__(
            env.time_step_spec(), env.action_spec()
        )
        self._oracle = OrientedPushOracle(env)
        self._env = env

    def reset(self):
        self._oracle.reset()

    def _action(self, time_step, policy_state):
        time_step = time_step._asdict()
        time_step["observation"] = self._env.calc_unnormalized_state(
            time_step["observation"]
        )
        step = self._oracle._action(
            ts.TimeStep(**time_step), policy_state
        )  # pylint: disable=protected-access
        return policy_step.PolicyStep(
            action=self._env.calc_normalized_action(step.action)
        )

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/oracles/multimodal_push_oracle.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Oracle for multimodal pushing task."""
import diffusion_policy.env.block_pushing.oracles.oriented_push_oracle as oriented_push_oracle_module
import numpy as np
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import


class MultimodalOrientedPushOracle(oriented_push_oracle_module.OrientedPushOracle):
    """Oracle for multimodal pushing task."""

    def __init__(self, env, goal_dist_tolerance=0.04, action_noise_std=0.0):
        super(MultimodalOrientedPushOracle, self).__init__(env)
        self._goal_dist_tolerance = goal_dist_tolerance
        self._action_noise_std = action_noise_std

    def reset(self):
        self.origin = None
        self.first_preblock = None
        self.phase = "move_to_pre_block"

    def _get_move_to_preblock(self, xy_pre_block, xy_ee):
        max_step_velocity = 0.3
        # Go 5 cm away from the block, on the line between the block and target.
        xy_delta_to_preblock = xy_pre_block - xy_ee
        diff = np.linalg.norm(xy_delta_to_preblock)
        if diff < 0.001:
            self.phase = "move_to_block"
            if self.first_preblock is None:
                self.first_preblock = np.copy(xy_pre_block)
        xy_delta = xy_delta_to_preblock
        return xy_delta, max_step_velocity

    def _get_action_for_block_target(self, time_step, block="block", target="target"):
        # Specifying this as velocity makes it independent of control frequency.
        max_step_velocity = 0.35

        info = self._get_action_info(time_step, block, target)

        if self.origin is None:
            self.origin = np.copy(info.xy_ee)

        if self.phase == "move_to_pre_block":
            xy_delta, max_step_velocity = self._get_move_to_preblock(
                info.xy_pre_block, info.xy_ee
            )

        if self.phase == "return_to_first_preblock":
            max_step_velocity = 0.3
            if self.first_preblock is None:
                self.first_preblock = self.origin
            # Return to the first preblock.
            xy_delta_to_origin = self.first_preblock - info.xy_ee
            diff = np.linalg.norm(xy_delta_to_origin)
            if diff < 0.001:
                self.phase = "return_to_origin"
            xy_delta = xy_delta_to_origin

        if self.phase == "return_to_origin":
            max_step_velocity = 0.3
            # Go 5 cm away from the block, on the line between the block and target.
            xy_delta_to_origin = self.origin - info.xy_ee
            diff = np.linalg.norm(xy_delta_to_origin)
            if diff < 0.001:
                self.phase = "move_to_pre_block"
            xy_delta = xy_delta_to_origin

        if self.phase == "move_to_block":
            xy_delta = self._get_move_to_block(
                info.xy_delta_to_nexttoblock,
                info.theta_threshold_to_orient,
                info.theta_error,
            )

        if self.phase == "push_block":
            xy_delta = self._get_push_block(
                info.theta_error,
                info.theta_threshold_to_orient,
                info.xy_delta_to_touchingblock,
            )

        orient_circle_diameter = 0.025

        if self.phase == "orient_block_left" or self.phase == "orient_block_right":
            max_step_velocity = 0.15

        if self.phase == "orient_block_left":
            xy_delta = self._get_orient_block_left(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough,
            )

        if self.phase == "orient_block_right":
            xy_delta = self._get_orient_block_right(
                info.xy_dir_block_to_ee,
                orient_circle_diameter,
                info.xy_block,
                info.xy_ee,
                info.theta_error,
                info.theta_threshold_flat_enough,
            )

        if self._action_noise_std != 0.0:
            xy_delta += self._np_random_state.randn(2) * self._action_noise_std

        max_step_distance = max_step_velocity * (1 / self._env.get_control_frequency())
        length = np.linalg.norm(xy_delta)
        if length > max_step_distance:
            xy_direction = xy_delta / length
            xy_delta = xy_direction * max_step_distance
        return xy_delta

    def _choose_goal_order(self):
        """Chooses block->target order for multimodal pushing."""
        # Define all possible ((first_block, first_target),
        # (second_block, second_target)).
        possible_orders = [
            (("block", "target"), ("block2", "target2")),
            (("block", "target2"), ("block2", "target")),
            (("block2", "target"), ("block", "target2")),
            (("block2", "target2"), ("block", "target")),
        ]
        # import pdb; pdb.set_trace()
        # result = random.choice(possible_orders)
        result = possible_orders[self._env._rng.choice(len(possible_orders))]
        return result

    def _action(self, time_step, policy_state):
        if time_step.is_first():
            self.reset()
            (
                (self._first_block, self._first_target),
                (self._second_block, self._second_target),
            ) = self._choose_goal_order()
            self._current_block, self._current_target = (
                self._first_block,
                self._first_target,
            )
            self._has_switched = False

        def _block_target_dist(block, target):
            dist = np.linalg.norm(
                time_step.observation["%s_translation" % block]
                - time_step.observation["%s_translation" % target]
            )
            return dist

        if (
            _block_target_dist(self._first_block, self._first_target)
            < self._goal_dist_tolerance
            and not self._has_switched
        ):
            # If first block has been pushed to first target, switch to second block.
            self._current_block, self._current_target = (
                self._second_block,
                self._second_target,
            )
            self._has_switched = True
            self.phase = "return_to_first_preblock"

        xy_delta = self._get_action_for_block_target(
            time_step, block=self._current_block, target=self._current_target
        )

        return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/oracles/reach_oracle.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reach oracle."""
import numpy as np
from tf_agents.policies import py_policy
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types

# Only used for debug visualization.
import pybullet  # pylint: disable=unused-import


class ReachOracle(py_policy.PyPolicy):
    """Oracle for moving to a specific spot relative to the block and target."""

    def __init__(self, env, block_pushing_oracles_action_std=0.0):
        super(ReachOracle, self).__init__(env.time_step_spec(), env.action_spec())
        self._env = env
        self._np_random_state = np.random.RandomState(0)
        self._block_pushing_oracles_action_std = block_pushing_oracles_action_std

    def _action(self, time_step, policy_state):

        # Specifying this as velocity makes it independent of control frequency.
        max_step_velocity = 0.2

        xy_ee = time_step.observation["effector_target_translation"]

        # This should be observable from block and target translation,
        # but re-using the computation from the env so that it's only done once, and
        # used for reward / completion computation.
        xy_pre_block = self._env.reach_target_translation

        xy_delta = xy_pre_block - xy_ee

        if self._block_pushing_oracles_action_std != 0.0:
            xy_delta += (
                self._np_random_state.randn(2) * self._block_pushing_oracles_action_std
            )

        max_step_distance = max_step_velocity * (1 / self._env.get_control_frequency())
        length = np.linalg.norm(xy_delta)
        if length > max_step_distance:
            xy_direction = xy_delta / length
            xy_delta = xy_direction * max_step_distance

        return policy_step.PolicyStep(action=np.asarray(xy_delta, dtype=np.float32))

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/block_pushing_multimodal.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multimodal block environments for the XArm."""

import collections
import logging
import math
from typing import Dict, List, Optional, Union
import copy
import time

from gym import spaces
from gym.envs import registration
from diffusion_policy.env.block_pushing import block_pushing
from diffusion_policy.env.block_pushing.utils import utils_pybullet
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
from diffusion_policy.env.block_pushing.utils.utils_pybullet import ObjState
from diffusion_policy.env.block_pushing.utils.utils_pybullet import XarmState
import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client

# pytype: skip-file
BLOCK2_URDF_PATH = "third_party/py/envs/assets/block2.urdf"
ZONE2_URDF_PATH = "third_party/py/envs/assets/zone2.urdf"

# When resetting multiple targets, they should all be this far apart.
MIN_BLOCK_DIST = 0.1
MIN_TARGET_DIST = 0.12
# pylint: enable=line-too-long
NUM_RESET_ATTEMPTS = 1000

# Random movement of blocks
RANDOM_X_SHIFT = 0.1
RANDOM_Y_SHIFT = 0.15

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode="w",
)
logger = logging.getLogger()


def build_env_name(task, shared_memory, use_image_obs):
    """Construct the env name from parameters."""
    del task
    env_name = "BlockPushMultimodal"

    if use_image_obs:
        env_name = env_name + "Rgb"

    if shared_memory:
        env_name = "Shared" + env_name

    env_name = env_name + "-v0"

    return env_name


class BlockPushEventManager:
    def __init__(self):
        self.event_steps = {
            'REACH_0': -1,
            'REACH_1': -1,
            'TARGET_0_0': -1,
            'TARGET_0_1': -1,
            'TARGET_1_0': -1,
            'TARGET_1_1': -1
        }
    
    def reach(self, step, block_id):
        key = f'REACH_{block_id}'
        if self.event_steps[key] < 0:
            self.event_steps[key] = step
    
    def target(self, step, block_id, target_id):
        key = f'TARGET_{block_id}_{target_id}'
        if self.event_steps[key] < 0:
            self.event_steps[key] = step

    def reset(self):
        for key in list(self.event_steps):
            self.event_steps[key] = -1
    
    def get_info(self):
        return copy.deepcopy(self.event_steps)

class BlockPushMultimodal(block_pushing.BlockPush):
    """2 blocks, 2 targets."""

    def __init__(
        self,
        control_frequency=10.0,
        task=block_pushing.BlockTaskVariant.PUSH,
        image_size=None,
        shared_memory=False,
        seed=None,
        goal_dist_tolerance=0.05,
        abs_action=False
    ):
        """Creates an env instance.

        Args:
          control_frequency: Control frequency for the arm. Each env step will
            advance the simulation by 1/control_frequency seconds.
          task: enum for which task, see BlockTaskVariant enum.
          image_size: Optional image size (height, width). If None, no image
            observations will be used.
          shared_memory: If True `pybullet.SHARED_MEMORY` is used to connect to
            pybullet. Useful to debug.
          seed: Optional seed for the environment.
          goal_dist_tolerance: float, how far away from the goal to terminate.
        """
        self._target_ids = None
        self._target_poses = None
        self._event_manager = BlockPushEventManager()
        super(BlockPushMultimodal, self).__init__(
            control_frequency=control_frequency,
            task=task,
            image_size=image_size,
            shared_memory=shared_memory,
            seed=seed,
            goal_dist_tolerance=goal_dist_tolerance,
        )
        self._init_distance = [-1.0, -1.0]
        self._in_target = [[-1.0, -1.0], [-1.0, -1.0]]
        self._first_move = [-1, -1]
        self._step_num = 0
        self._abs_action = abs_action

    @property
    def target_poses(self):
        return self._target_poses

    def get_goal_translation(self):
        """Return the translation component of the goal (2D)."""
        if self._target_poses:
            return [i.translation for i in self._target_poses]
        else:
            return None

    def _setup_pybullet_scene(self):
        self._pybullet_client = bullet_client.BulletClient(self._connection_mode)

        # Temporarily disable rendering to speed up loading URDFs.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self._setup_workspace_and_robot()

        self._target_ids = [
            utils_pybullet.load_urdf(self._pybullet_client, i, useFixedBase=True)
            for i in [block_pushing.ZONE_URDF_PATH, ZONE2_URDF_PATH]
        ]
        self._block_ids = []
        for i in [block_pushing.BLOCK_URDF_PATH, BLOCK2_URDF_PATH]:
            self._block_ids.append(
                utils_pybullet.load_urdf(self._pybullet_client, i, useFixedBase=False)
            )

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.step_simulation_to_stabilize()

    def _reset_block_poses(self, workspace_center_x):
        """Resets block poses."""

        # Helper for choosing random block position.
        def _reset_block_pose(idx, add=0.0, avoid=None):
            def _get_random_translation():
                block_x = (
                    workspace_center_x
                    + add
                    + self._rng.uniform(low=-RANDOM_X_SHIFT, high=RANDOM_X_SHIFT)
                )
                block_y = -0.2 + self._rng.uniform(
                    low=-RANDOM_Y_SHIFT, high=RANDOM_Y_SHIFT
                )
                block_translation = np.array([block_x, block_y, 0])
                return block_translation

            if avoid is None:
                block_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    block_translation = _get_random_translation()
                    dist = np.linalg.norm(block_translation[0] - avoid[0])
                    # print('block inner try_idx %d, dist %.3f' % (try_idx, dist))
                    if dist > MIN_BLOCK_DIST:
                        break
            block_sampled_angle = self._rng.uniform(math.pi)
            block_rotation = transform.Rotation.from_rotvec([0, 0, block_sampled_angle])
            self._pybullet_client.resetBasePositionAndOrientation(
                self._block_ids[idx],
                block_translation.tolist(),
                block_rotation.as_quat().tolist(),
            )
            return block_translation

        # Reject targets too close to `avoid`.
        for _ in range(NUM_RESET_ATTEMPTS):
            # Reset first block.
            b0_translation = _reset_block_pose(0)
            # Reset second block away from first block.
            b1_translation = _reset_block_pose(1, avoid=b0_translation)
            dist = np.linalg.norm(b0_translation[0] - b1_translation[0])
            if dist > MIN_BLOCK_DIST:
                break
        else:
            raise ValueError("could not find matching block")
        assert dist > MIN_BLOCK_DIST

    def _reset_target_poses(self, workspace_center_x):
        """Resets target poses."""

        def _reset_target_pose(idx, add=0.0, avoid=None):
            def _get_random_translation():
                # Choose x,y randomly.
                target_x = (
                    workspace_center_x
                    + add
                    + self._rng.uniform(
                        low=-0.05 * RANDOM_X_SHIFT, high=0.05 * RANDOM_X_SHIFT
                    )
                )
                target_y = 0.2 + self._rng.uniform(
                    low=-0.05 * RANDOM_Y_SHIFT, high=0.05 * RANDOM_Y_SHIFT
                )
                target_translation = np.array([target_x, target_y, 0.020])
                return target_translation

            if avoid is None:
                target_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    target_translation = _get_random_translation()
                    dist = np.linalg.norm(target_translation[0] - avoid[0])
                    # print('target inner try_idx %d, dist %.3f' % (try_idx, dist))
                    if dist > MIN_TARGET_DIST:
                        break
            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 30, high=math.pi / 30
            )
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle]
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_ids[idx],
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),
            )
            self._target_poses[idx] = Pose3d(
                rotation=target_rotation, translation=target_translation
            )

        if self._target_poses is None:
            self._target_poses = [None for _ in range(len(self._target_ids))]

        for _ in range(NUM_RESET_ATTEMPTS):
            # Choose the first target.
            add = 0.12 * self._rng.choice([-1, 1])
            # Randomly flip the location of the targets.
            _reset_target_pose(0, add=add)
            _reset_target_pose(1, add=-add, avoid=self._target_poses[0].translation)
            dist = np.linalg.norm(
                self._target_poses[0].translation[0]
                - self._target_poses[1].translation[0]
            )
            if dist > MIN_TARGET_DIST:
                break
        else:
            raise ValueError("could not find matching target")
        assert dist > MIN_TARGET_DIST

    def _reset_object_poses(self, workspace_center_x, workspace_center_y):
        # Reset block poses.
        self._reset_block_poses(workspace_center_x)

        # Reset target poses.
        self._reset_target_poses(workspace_center_x)

        self._init_distance = [-1.0, -1.0]
        self._in_target = [[-1.0, -1.0], [-1.0, -1.0]]
        self._step_num = 0

    def reset(self, reset_poses=True):
        workspace_center_x = 0.4
        workspace_center_y = 0.0

        if reset_poses:
            self._pybullet_client.restoreState(self._saved_state)

            rotation = transform.Rotation.from_rotvec([0, math.pi, 0])
            translation = np.array([0.3, -0.4, block_pushing.EFFECTOR_HEIGHT])
            starting_pose = Pose3d(rotation=rotation, translation=translation)
            self._set_robot_target_effector_pose(starting_pose)
            self._reset_object_poses(workspace_center_x, workspace_center_y)

        # else:
        self._target_poses = [
            self._get_target_pose(idx) for idx in self._target_ids
        ]

        if reset_poses:
            self.step_simulation_to_stabilize()

        state = self._compute_state()
        self._previous_state = state
        self._event_manager.reset()
        return state

    def _get_target_pose(self, idx):
        (
            target_translation,
            target_orientation_quat,
        ) = self._pybullet_client.getBasePositionAndOrientation(idx)
        target_rotation = transform.Rotation.from_quat(target_orientation_quat)
        target_translation = np.array(target_translation)
        return Pose3d(rotation=target_rotation, translation=target_translation)

    def _compute_reach_target(self, state):
        xy_block = state["block_translation"]
        xy_target = state["target_translation"]

        xy_block_to_target = xy_target - xy_block
        xy_dir_block_to_target = (xy_block_to_target) / np.linalg.norm(
            xy_block_to_target
        )
        self.reach_target_translation = xy_block + -1 * xy_dir_block_to_target * 0.05

    def _compute_state(self):
        effector_pose = self._robot.forward_kinematics()

        def _get_block_pose(idx):
            block_position_and_orientation = (
                self._pybullet_client.getBasePositionAndOrientation(
                    self._block_ids[idx]
                )
            )
            block_pose = Pose3d(
                rotation=transform.Rotation.from_quat(
                    block_position_and_orientation[1]
                ),
                translation=block_position_and_orientation[0],
            )
            return block_pose

        block_poses = [_get_block_pose(i) for i in range(len(self._block_ids))]

        def _yaw_from_pose(pose):
            return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1] % np.pi])

        obs = collections.OrderedDict(
            block_translation=block_poses[0].translation[0:2],
            block_orientation=_yaw_from_pose(block_poses[0]),
            block2_translation=block_poses[1].translation[0:2],
            block2_orientation=_yaw_from_pose(block_poses[1]),
            effector_translation=effector_pose.translation[0:2],
            effector_target_translation=self._target_effector_pose.translation[0:2],
            target_translation=self._target_poses[0].translation[0:2],
            target_orientation=_yaw_from_pose(self._target_poses[0]),
            target2_translation=self._target_poses[1].translation[0:2],
            target2_orientation=_yaw_from_pose(self._target_poses[1]),
        )

        for i in range(2):
            new_distance = np.linalg.norm(
                block_poses[i].translation[0:2]
            )  # + np.linalg.norm(_yaw_from_pose(block_poses[i]))
            if self._init_distance[i] == -1:
                self._init_distance[i] = new_distance
            else:
                if self._init_distance[i] != 100:
                    if np.abs(new_distance - self._init_distance[i]) > 1e-3:
                        logger.info(f"Block {i} moved on step {self._step_num}")
                        self._event_manager.reach(step=self._step_num, block_id=i)
                        self._init_distance[i] = 100

        self._step_num += 1
        if self._image_size is not None:
            obs["rgb"] = self._render_camera(self._image_size)
        return obs

    def step(self, action):
        self._step_robot_and_sim(action)

        state = self._compute_state()
        done = False
        reward = self._get_reward(state)
        if reward >= 0.5:
            # Terminate the episode if both blocks are close enough to the targets.
            done = True

        info = self._event_manager.get_info()
        return state, reward, done, info
    
    def _step_robot_and_sim(self, action):
        """Steps the robot and pybullet sim."""
        # Compute target_effector_pose by shifting the effector's pose by the
        # action.
        if self._abs_action:
            target_effector_translation = np.array([action[0], action[1], 0])
        else:
            target_effector_translation = np.array(
                self._target_effector_pose.translation
            ) + np.array([action[0], action[1], 0])

        target_effector_translation[0:2] = np.clip(
            target_effector_translation[0:2],
            self.workspace_bounds[0],
            self.workspace_bounds[1],
        )
        target_effector_translation[-1] = self.effector_height
        target_effector_pose = Pose3d(
            rotation=block_pushing.EFFECTOR_DOWN_ROTATION, translation=target_effector_translation
        )

        self._set_robot_target_effector_pose(target_effector_pose)

        # Update sleep time dynamically to stay near real-time.
        frame_sleep_time = 0
        if self._connection_mode == pybullet.SHARED_MEMORY:
            cur_time = time.time()
            if self._last_loop_time is not None:
                # Calculate the total, non-sleeping time from the previous frame, this
                # includes the actual step as well as any compute that happens in the
                # caller thread (model inference, etc).
                compute_time = (
                    cur_time
                    - self._last_loop_time
                    - self._last_loop_frame_sleep_time * self._sim_steps_per_step
                )
                # Use this to calculate the current frame's total sleep time to ensure
                # that env.step runs at policy rate. This is an estimate since the
                # previous frame's compute time may not match the current frame.
                total_sleep_time = max((1 / self._control_frequency) - compute_time, 0)
                # Now spread this out over the inner sim steps. This doesn't change
                # control in any way, but makes the animation appear smooth.
                frame_sleep_time = total_sleep_time / self._sim_steps_per_step
            else:
                # No estimate of the previous frame's compute, assume it is zero.
                frame_sleep_time = 1 / self._step_frequency

            # Cache end of this loop time, to compute sleep time on next iteration.
            self._last_loop_time = cur_time
            self._last_loop_frame_sleep_time = frame_sleep_time

        for _ in range(self._sim_steps_per_step):
            if self._connection_mode == pybullet.SHARED_MEMORY:
                block_pushing.sleep_spin(frame_sleep_time)
            self._pybullet_client.stepSimulation()

    def _get_reward(self, state):
        # Reward is 1. if both blocks are inside targets, but not the same target.
        targets = ["target", "target2"]

        def _block_target_dist(block, target):
            return np.linalg.norm(
                state["%s_translation" % block] - state["%s_translation" % target]
            )

        def _closest_target(block):
            # Distances to all targets.
            dists = [_block_target_dist(block, t) for t in targets]
            # Which is closest.
            closest_target = targets[np.argmin(dists)]
            closest_dist = np.min(dists)
            # Is it in the closest target?
            in_target = closest_dist < self.goal_dist_tolerance
            return closest_target, in_target

        blocks = ["block", "block2"]

        reward = 0.0

        for t_i, t in enumerate(targets):
            for b_i, b in enumerate(blocks):
                if self._in_target[t_i][b_i] == -1:
                    dist = _block_target_dist(b, t)
                    if dist < self.goal_dist_tolerance:
                        self._in_target[t_i][b_i] = 0
                        logger.info(
                            f"Block {b_i} entered target {t_i} on step {self._step_num}"
                        )
                        self._event_manager.target(step=self._step_num, block_id=b_i, target_id=t_i)
                        reward += 0.49

        b0_closest_target, b0_in_target = _closest_target("block")
        b1_closest_target, b1_in_target = _closest_target("block2")
        # reward = 0.0
        if b0_in_target and b1_in_target and (b0_closest_target != b1_closest_target):
            reward = 0.51
        return reward

    def _compute_goal_distance(self, state):
        blocks = ["block", "block2"]

        def _target_block_dist(target, block):
            return np.linalg.norm(
                state["%s_translation" % block] - state["%s_translation" % target]
            )

        def _closest_block_dist(target):
            dists = [_target_block_dist(target, b) for b in blocks]
            closest_dist = np.min(dists)
            return closest_dist

        t0_closest_dist = _closest_block_dist("target")
        t1_closest_dist = _closest_block_dist("target2")
        return np.mean([t0_closest_dist, t1_closest_dist])

    @property
    def succeeded(self):
        state = self._compute_state()
        reward = self._get_reward(state)
        if reward >= 0.5:
            return True
        return False

    def _create_observation_space(self, image_size):
        pi2 = math.pi * 2

        obs_dict = collections.OrderedDict(
            block_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            block2_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block2_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            effector_translation=spaces.Box(
                low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
                high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            effector_target_translation=spaces.Box(
                low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
                high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            target_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
            target2_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target2_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
        )
        if image_size is not None:
            obs_dict["rgb"] = spaces.Box(
                low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)

    def get_pybullet_state(self):
        """Save pybullet state of the scene.

        Returns:
          dict containing 'robots', 'robot_end_effectors', 'targets', 'objects',
            each containing a list of ObjState.
        """
        state: Dict[str, List[ObjState]] = {}

        state["robots"] = [
            XarmState.get_bullet_state(
                self._pybullet_client,
                self.robot.xarm,
                target_effector_pose=self._target_effector_pose,
                goal_translation=None,
            )
        ]

        state["robot_end_effectors"] = []
        if self.robot.end_effector:
            state["robot_end_effectors"].append(
                ObjState.get_bullet_state(
                    self._pybullet_client, self.robot.end_effector
                )
            )

        state["targets"] = []
        if self._target_ids:
            for target_id in self._target_ids:
                state["targets"].append(
                    ObjState.get_bullet_state(self._pybullet_client, target_id)
                )

        state["objects"] = []
        for obj_id in self.get_obj_ids():
            state["objects"].append(
                ObjState.get_bullet_state(self._pybullet_client, obj_id)
            )

        return state

    def set_pybullet_state(self, state):
        """Restore pyullet state.

        WARNING: py_environment wrapper assumes environments aren't reset in their
        constructor and will often reset the environment unintentionally. It is
        always recommended that you call env.reset on the tfagents wrapper before
        playback (replaying pybullet_state).

        Args:
          state: dict containing 'robots', 'robot_end_effectors', 'targets',
            'objects', each containing a list of ObjState.
        """

        assert isinstance(state["robots"][0], XarmState)
        xarm_state: XarmState = state["robots"][0]
        xarm_state.set_bullet_state(self._pybullet_client, self.robot.xarm)
        self._set_robot_target_effector_pose(xarm_state.target_effector_pose)

        def _set_state_safe(obj_state, obj_id):
            if obj_state is not None:
                assert obj_id is not None, "Cannot set state for missing object."
                obj_state.set_bullet_state(self._pybullet_client, obj_id)
            else:
                assert obj_id is None, f"No state found for obj_id {obj_id}"

        robot_end_effectors = state["robot_end_effectors"]
        _set_state_safe(
            None if not robot_end_effectors else robot_end_effectors[0],
            self.robot.end_effector,
        )

        for target_state, target_id in zip(state["targets"], self._target_ids):
            _set_state_safe(target_state, target_id)

        obj_ids = self.get_obj_ids()
        assert len(state["objects"]) == len(obj_ids), "State length mismatch"
        for obj_state, obj_id in zip(state["objects"], obj_ids):
            _set_state_safe(obj_state, obj_id)

        self.reset(reset_poses=False)


class BlockPushHorizontalMultimodal(BlockPushMultimodal):
    def _reset_object_poses(self, workspace_center_x, workspace_center_y):
        # Reset block poses.
        self._reset_block_poses(workspace_center_y)

        # Reset target poses.
        self._reset_target_poses(workspace_center_y)

    def _reset_block_poses(self, workspace_center_y):
        """Resets block poses."""

        # Helper for choosing random block position.
        def _reset_block_pose(idx, add=0.0, avoid=None):
            def _get_random_translation():
                block_x = 0.35 + 0.5 * self._rng.uniform(
                    low=-RANDOM_X_SHIFT, high=RANDOM_X_SHIFT
                )
                block_y = (
                    workspace_center_y
                    + add
                    + 0.5 * self._rng.uniform(low=-RANDOM_Y_SHIFT, high=RANDOM_Y_SHIFT)
                )
                block_translation = np.array([block_x, block_y, 0])
                return block_translation

            if avoid is None:
                block_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    block_translation = _get_random_translation()
                    dist = np.linalg.norm(block_translation[0] - avoid[0])
                    # print('block inner try_idx %d, dist %.3f' % (try_idx, dist))
                    if dist > MIN_BLOCK_DIST:
                        break
            block_sampled_angle = self._rng.uniform(math.pi)
            block_rotation = transform.Rotation.from_rotvec([0, 0, block_sampled_angle])
            self._pybullet_client.resetBasePositionAndOrientation(
                self._block_ids[idx],
                block_translation.tolist(),
                block_rotation.as_quat().tolist(),
            )
            return block_translation

        # Reject targets too close to `avoid`.
        for _ in range(NUM_RESET_ATTEMPTS):
            # Reset first block.
            add = 0.2 * self._rng.choice([-1, 1])
            b0_translation = _reset_block_pose(0, add=add)
            # Reset second block away from first block.
            b1_translation = _reset_block_pose(1, add=-add, avoid=b0_translation)
            dist = np.linalg.norm(b0_translation[0] - b1_translation[0])
            if dist > MIN_BLOCK_DIST:
                break
        else:
            raise ValueError("could not find matching block")
        assert dist > MIN_BLOCK_DIST

    def _reset_target_poses(self, workspace_center_y):
        """Resets target poses."""

        def _reset_target_pose(idx, add=0.0, avoid=None):
            def _get_random_translation():
                # Choose x,y randomly.
                target_x = 0.5 + self._rng.uniform(
                    low=-0.05 * RANDOM_X_SHIFT, high=0.05 * RANDOM_X_SHIFT
                )
                target_y = (
                    workspace_center_y
                    + add
                    + self._rng.uniform(
                        low=-0.05 * RANDOM_Y_SHIFT, high=0.05 * RANDOM_Y_SHIFT
                    )
                )
                target_translation = np.array([target_x, target_y, 0.020])
                return target_translation

            if avoid is None:
                target_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    target_translation = _get_random_translation()
                    dist = np.linalg.norm(target_translation[0] - avoid[0])
                    # print('target inner try_idx %d, dist %.3f' % (try_idx, dist))
                    if dist > MIN_TARGET_DIST:
                        break
            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 30, high=math.pi / 30
            )
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle]
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_ids[idx],
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),
            )
            self._target_poses[idx] = Pose3d(
                rotation=target_rotation, translation=target_translation
            )

        if self._target_poses is None:
            self._target_poses = [None for _ in range(len(self._target_ids))]

        for _ in range(NUM_RESET_ATTEMPTS):
            # Choose the first target.
            add = 0.2 * self._rng.choice([-1, 1])
            # Randomly flip the location of the targets.
            _reset_target_pose(0, add=add)
            _reset_target_pose(1, add=-add, avoid=self._target_poses[0].translation)
            dist = np.linalg.norm(
                self._target_poses[0].translation[0]
                - self._target_poses[1].translation[0]
            )
            break
            # if dist > MIN_TARGET_DIST:
            #     break
        else:
            raise ValueError("could not find matching target")
        # assert dist > MIN_TARGET_DIST


if "BlockPushMultimodal-v0" in registration.registry.env_specs:
    del registration.registry.env_specs["BlockPushMultimodal-v0"]

registration.register(
    id="BlockPushMultimodal-v0", entry_point=BlockPushMultimodal, max_episode_steps=350
)

registration.register(
    id="BlockPushMultimodalFlipped-v0",
    entry_point=BlockPushHorizontalMultimodal,
    max_episode_steps=25,
)

registration.register(
    id="SharedBlockPushMultimodal-v0",
    entry_point=BlockPushMultimodal,
    kwargs=dict(shared_memory=True),
    max_episode_steps=350,
)
registration.register(
    id="BlockPushMultimodalRgb-v0",
    entry_point=BlockPushMultimodal,
    max_episode_steps=350,
    kwargs=dict(image_size=(block_pushing.IMAGE_HEIGHT, block_pushing.IMAGE_WIDTH)),
)

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/block_pushing_discontinuous.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Discontinuous block pushing."""
import collections
import enum
import math
from typing import List, Optional

from gym import spaces
from gym.envs import registration
from diffusion_policy.env.block_pushing import block_pushing
from diffusion_policy.env.block_pushing.utils import utils_pybullet
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client

ZONE2_URDF_PATH = "third_party/py/envs/assets/zone2.urdf"

MIN_TARGET_DIST = 0.15
NUM_RESET_ATTEMPTS = 1000


def build_env_name(task, shared_memory, use_image_obs):
    """Construct the env name from parameters."""
    del task
    env_name = "BlockPushDiscontinuous"

    if use_image_obs:
        env_name = env_name + "Rgb"

    if shared_memory:
        env_name = "Shared" + env_name

    env_name = env_name + "-v0"

    return env_name


class BlockTaskVariant(enum.Enum):
    REACH = "Reach"
    REACH_NORMALIZED = "ReachNormalized"
    PUSH = "Push"
    PUSH_NORMALIZED = "PushNormalized"
    INSERT = "Insert"


# pytype: skip-file
class BlockPushDiscontinuous(block_pushing.BlockPush):
    """Discontinuous block pushing."""

    def __init__(
        self,
        control_frequency=10.0,
        task=BlockTaskVariant.PUSH,
        image_size=None,
        shared_memory=False,
        seed=None,
        goal_dist_tolerance=0.04,
    ):
        super(BlockPushDiscontinuous, self).__init__(
            control_frequency=control_frequency,
            task=task,
            image_size=image_size,
            shared_memory=shared_memory,
            seed=seed,
            goal_dist_tolerance=goal_dist_tolerance,
        )

    @property
    def target_poses(self):
        return self._target_poses

    def get_goal_translation(self):
        """Return the translation component of the goal (2D)."""
        if self._target_poses:
            return [i.translation for i in self._target_poses]
        else:
            return None

    def _setup_pybullet_scene(self):
        self._pybullet_client = bullet_client.BulletClient(self._connection_mode)

        # Temporarily disable rendering to speed up loading URDFs.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self._setup_workspace_and_robot()
        target_urdf_path = block_pushing.ZONE_URDF_PATH

        self._target_ids = []
        for _ in [block_pushing.ZONE_URDF_PATH, ZONE2_URDF_PATH]:
            self._target_ids.append(
                utils_pybullet.load_urdf(
                    self._pybullet_client, target_urdf_path, useFixedBase=True
                )
            )
        self._block_ids = [
            utils_pybullet.load_urdf(
                self._pybullet_client, block_pushing.BLOCK_URDF_PATH, useFixedBase=False
            )
        ]

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.step_simulation_to_stabilize()

    def _reset_target_poses(self, workspace_center_x):
        """Resets target poses."""
        self._target_poses = [None for _ in range(len(self._target_ids))]

        def _reset_target_pose(idx, avoid=None):
            def _get_random_translation():
                # Choose x,y randomly.
                target_x = workspace_center_x + self._rng.uniform(low=-0.10, high=0.10)
                # Fix ys for this environment.
                if idx == 0:
                    target_y = 0
                else:
                    target_y = 0.4
                target_translation = np.array([target_x, target_y, 0.020])
                return target_translation

            if avoid is None:
                target_translation = _get_random_translation()
            else:
                # Reject targets too close to `avoid`.
                for _ in range(NUM_RESET_ATTEMPTS):
                    target_translation = _get_random_translation()
                    dist = np.linalg.norm(target_translation[0] - avoid[0])
                    if dist > MIN_TARGET_DIST:
                        break
            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 6, high=math.pi / 6
            )
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle]
            )
            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_ids[idx],
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),
            )
            self._target_poses[idx] = Pose3d(
                rotation=target_rotation, translation=target_translation
            )

        try_idx = 0
        while True:
            # Choose the first target.
            _reset_target_pose(0)
            # Choose the second target, avoiding the first.
            _reset_target_pose(1, avoid=self._target_poses[0].translation)
            dist = np.linalg.norm(
                self._target_poses[0].translation[0]
                - self._target_poses[1].translation[0]
            )
            if dist > MIN_TARGET_DIST:
                break
            try_idx += 1
            if try_idx >= NUM_RESET_ATTEMPTS:
                raise ValueError("could not find matching target")
        assert dist > MIN_TARGET_DIST

    def reset(self):
        self._pybullet_client.restoreState(self._saved_state)

        rotation = transform.Rotation.from_rotvec([0, math.pi, 0])
        translation = np.array([0.3, -0.4, block_pushing.EFFECTOR_HEIGHT])
        starting_pose = Pose3d(rotation=rotation, translation=translation)
        self._set_robot_target_effector_pose(starting_pose)

        workspace_center_x = 0.4

        # Reset block pose.
        block_x = workspace_center_x + self._rng.uniform(low=-0.1, high=0.1)
        block_y = -0.2 + self._rng.uniform(low=-0.15, high=0.15)
        block_translation = np.array([block_x, block_y, 0])
        block_sampled_angle = self._rng.uniform(math.pi)
        block_rotation = transform.Rotation.from_rotvec([0, 0, block_sampled_angle])

        self._pybullet_client.resetBasePositionAndOrientation(
            self._block_ids[0],
            block_translation.tolist(),
            block_rotation.as_quat().tolist(),
        )

        # Reset target pose.
        self._reset_target_poses(workspace_center_x)

        self.step_simulation_to_stabilize()
        state = self._compute_state()
        self._previous_state = state
        self.min_dist_to_first_goal = np.inf
        self.min_dist_to_second_goal = np.inf
        self.steps = 0
        return state

    def _compute_goal_distance(self, state):
        # Reward is 1. blocks is inside any target.
        return np.mean([self.min_dist_to_first_goal, self.min_dist_to_second_goal])

    def _compute_state(self):
        effector_pose = self._robot.forward_kinematics()
        block_position_and_orientation = (
            self._pybullet_client.getBasePositionAndOrientation(self._block_ids[0])
        )
        block_pose = Pose3d(
            rotation=transform.Rotation.from_quat(block_position_and_orientation[1]),
            translation=block_position_and_orientation[0],
        )

        def _yaw_from_pose(pose):
            return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1]])

        obs = collections.OrderedDict(
            block_translation=block_pose.translation[0:2],
            block_orientation=_yaw_from_pose(block_pose),
            effector_translation=effector_pose.translation[0:2],
            effector_target_translation=self._target_effector_pose.translation[0:2],
            target_translation=self._target_poses[0].translation[0:2],
            target_orientation=_yaw_from_pose(self._target_poses[0]),
            target2_translation=self._target_poses[1].translation[0:2],
            target2_orientation=_yaw_from_pose(self._target_poses[1]),
        )
        if self._image_size is not None:
            obs["rgb"] = self._render_camera(self._image_size)
        return obs

    def step(self, action):
        self._step_robot_and_sim(action)
        state = self._compute_state()
        reward = self._get_reward(state)
        done = False
        if reward > 0.0:
            done = True
        # Cache so we can compute success.
        self.state = state
        return state, reward, done, {}

    def dist(self, state, target):
        # Reward is 1. blocks is inside any target.
        return np.linalg.norm(
            state["block_translation"] - state["%s_translation" % target]
        )

    def _get_reward(self, state):
        """Reward is 1.0 if agent hits both goals and stays at second."""
        # This also statefully updates these values.
        self.min_dist_to_first_goal = min(
            self.dist(state, "target"), self.min_dist_to_first_goal
        )
        self.min_dist_to_second_goal = min(
            self.dist(state, "target2"), self.min_dist_to_second_goal
        )

        def _reward(thresh):
            reward_first = True if self.min_dist_to_first_goal < thresh else False
            reward_second = True if self.min_dist_to_second_goal < thresh else False
            return 1.0 if (reward_first and reward_second) else 0.0

        reward = _reward(self.goal_dist_tolerance)
        return reward

    @property
    def succeeded(self):
        thresh = self.goal_dist_tolerance
        hit_first = True if self.min_dist_to_first_goal < thresh else False
        hit_second = True if self.min_dist_to_first_goal < thresh else False
        current_distance_to_second = self.dist(self.state, "target2")
        still_at_second = True if current_distance_to_second < thresh else False
        return hit_first and hit_second and still_at_second

    def _create_observation_space(self, image_size):
        pi2 = math.pi * 2

        obs_dict = collections.OrderedDict(
            block_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            effector_translation=spaces.Box(
                # Small buffer for to IK noise.
                low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
                high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            effector_target_translation=spaces.Box(
                # Small buffer for to IK noise.
                low=block_pushing.WORKSPACE_BOUNDS[0] - 0.1,
                high=block_pushing.WORKSPACE_BOUNDS[1] + 0.1,
            ),  # x,y
            target_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
            target2_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target2_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
        )
        if image_size is not None:
            obs_dict["rgb"] = spaces.Box(
                low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)


if "BlockPushDiscontinuous-v0" in registration.registry.env_specs:
    del registration.registry.env_specs["BlockPushDiscontinuous-v0"]

registration.register(
    id="BlockPushDiscontinuous-v0",
    entry_point=BlockPushDiscontinuous,
    max_episode_steps=200,
)

registration.register(
    id="BlockPushDiscontinuousRgb-v0",
    entry_point=BlockPushDiscontinuous,
    max_episode_steps=200,
    kwargs=dict(image_size=(block_pushing.IMAGE_HEIGHT, block_pushing.IMAGE_WIDTH)),
)

```

## reference_material/diffusion_policy_code/diffusion_policy/env/block_pushing/block_pushing.py
```python
# coding=utf-8
# Copyright 2022 The Reach ML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple block environments for the XArm."""

import collections
import enum
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import gym
from gym import spaces
from gym.envs import registration
from diffusion_policy.env.block_pushing.utils import utils_pybullet
from diffusion_policy.env.block_pushing.utils import xarm_sim_robot
from diffusion_policy.env.block_pushing.utils.pose3d import Pose3d
from diffusion_policy.env.block_pushing.utils.utils_pybullet import ObjState
from diffusion_policy.env.block_pushing.utils.utils_pybullet import XarmState
import numpy as np
from scipy.spatial import transform
import pybullet
import pybullet_utils.bullet_client as bullet_client

import matplotlib.pyplot as plt

BLOCK_URDF_PATH = "third_party/py/envs/assets/block.urdf"
PLANE_URDF_PATH = "third_party/bullet/examples/pybullet/gym/pybullet_data/" "plane.urdf"
WORKSPACE_URDF_PATH = "third_party/py/envs/assets/workspace.urdf"
ZONE_URDF_PATH = "third_party/py/envs/assets/zone.urdf"
INSERT_URDF_PATH = "third_party/py/envs/assets/insert.urdf"

EFFECTOR_HEIGHT = 0.06
EFFECTOR_DOWN_ROTATION = transform.Rotation.from_rotvec([0, math.pi, 0])

WORKSPACE_BOUNDS = np.array(((0.15, -0.5), (0.7, 0.5)))

# Min/max bounds calculated from oracle data using:
# ibc/environments/board2d_dataset_statistics.ipynb
# to calculate [mean - 3 * std, mean + 3 * std] using the oracle data.
# pylint: disable=line-too-long
ACTION_MIN = np.array([-0.02547718, -0.02090043], np.float32)
ACTION_MAX = np.array([0.02869084, 0.04272365], np.float32)
EFFECTOR_TARGET_TRANSLATION_MIN = np.array(
    [0.1774151772260666, -0.6287994794547558], np.float32
)
EFFECTOR_TARGET_TRANSLATION_MAX = np.array(
    [0.5654461532831192, 0.5441607423126698], np.float32
)
EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MIN = np.array(
    [-0.07369826920330524, -0.11395704373717308], np.float32
)
EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MAX = np.array(
    [0.10131562314927578, 0.19391131028532982], np.float32
)
EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MIN = np.array(
    [-0.17813862301409245, -0.3309651017189026], np.float32
)
EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MAX = np.array(
    [0.23726161383092403, 0.8404090404510498], np.float32
)
BLOCK_ORIENTATION_COS_SIN_MIN = np.array(
    [-2.0649861991405487, -0.6154364347457886], np.float32
)
BLOCK_ORIENTATION_COS_SIN_MAX = np.array(
    [1.6590178310871124, 1.8811014890670776], np.float32
)
TARGET_ORIENTATION_COS_SIN_MIN = np.array(
    [-1.0761439241468906, -0.8846937336493284], np.float32
)
TARGET_ORIENTATION_COS_SIN_MAX = np.array(
    [-0.8344330154359341, 0.8786859593819827], np.float32
)

# Hardcoded Pose joints to make sure we don't have surprises from using the
# IK solver on reset. The joint poses correspond to the Pose with:
#   rotation = rotation3.Rotation3.from_axis_angle([0, 1, 0], math.pi)
#   translation = np.array([0.3, -0.4, 0.07])
INITIAL_JOINT_POSITIONS = np.array(
    [
        -0.9254632489674508,
        0.6990770671568564,
        -1.106629064060494,
        0.0006653351931553931,
        0.3987969742311386,
        -4.063402065624296,
    ]
)

DEFAULT_CAMERA_POSE = (1.0, 0, 0.75)
DEFAULT_CAMERA_ORIENTATION = (np.pi / 4, np.pi, -np.pi / 2)
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
CAMERA_INTRINSICS = (
    0.803 * IMAGE_WIDTH,  # fx
    0,
    IMAGE_WIDTH / 2.0,  # cx
    0,
    0.803 * IMAGE_WIDTH,  # fy
    IMAGE_HEIGHT / 2.0,  # cy
    0,
    0,
    1,
)

# "Realistic" visuals.
X_MIN_REAL = 0.15
X_MAX_REAL = 0.6
Y_MIN_REAL = -0.3048
Y_MAX_REAL = 0.3048
WORKSPACE_BOUNDS_REAL = np.array(((X_MIN_REAL, Y_MIN_REAL), (X_MAX_REAL, Y_MAX_REAL)))
WORKSPACE_URDF_PATH_REAL = "third_party/py/ibc/environments/assets/workspace_real.urdf"
CAMERA_POSE_REAL = (0.75, 0, 0.5)
CAMERA_ORIENTATION_REAL = (np.pi / 5, np.pi, -np.pi / 2)

IMAGE_WIDTH_REAL = 320
IMAGE_HEIGHT_REAL = 180
CAMERA_INTRINSICS_REAL = (
    0.803 * IMAGE_WIDTH_REAL,  # fx
    0,
    IMAGE_WIDTH_REAL / 2.0,  # cx
    0,
    0.803 * IMAGE_WIDTH_REAL,  # fy
    IMAGE_HEIGHT_REAL / 2.0,  # cy
    0,
    0,
    1,
)
# pylint: enable=line-too-long


def build_env_name(task, shared_memory, use_image_obs, use_normalized_env=False):
    """Construct the env name from parameters."""
    if isinstance(task, str):
        task = BlockTaskVariant[task]
    env_name = "Block" + task.value

    if use_image_obs:
        env_name = env_name + "Rgb"
    if use_normalized_env:
        env_name = env_name + "Normalized"
    if shared_memory:
        env_name = "Shared" + env_name

    env_name = env_name + "-v0"

    return env_name


class BlockTaskVariant(enum.Enum):
    REACH = "Reach"
    REACH_NORMALIZED = "ReachNormalized"
    PUSH = "Push"
    PUSH_NORMALIZED = "PushNormalized"
    INSERT = "Insert"


def sleep_spin(sleep_time_sec):
    """Spin wait sleep. Avoids time.sleep accuracy issues on Windows."""
    if sleep_time_sec <= 0:
        return
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < sleep_time_sec:
        pass


class BlockPush(gym.Env):
    """Simple XArm environment for block pushing."""

    def __init__(
        self,
        control_frequency=10.0,
        task=BlockTaskVariant.PUSH,
        image_size=None,
        shared_memory=False,
        seed=None,
        goal_dist_tolerance=0.01,
        effector_height=None,
        visuals_mode="default",
        abs_action=False
    ):
        """Creates an env instance.

        Args:
          control_frequency: Control frequency for the arm. Each env step will
            advance the simulation by 1/control_frequency seconds.
          task: enum for which task, see BlockTaskVariant enum.
          image_size: Optional image size (height, width). If None, no image
            observations will be used.
          shared_memory: If True `pybullet.SHARED_MEMORY` is used to connect to
            pybullet. Useful to debug.
          seed: Optional seed for the environment.
          goal_dist_tolerance: float, how far away from the goal to terminate.
          effector_height: float, custom height for end effector.
          visuals_mode: 'default' or 'real'.
        """
        # pybullet.connect(pybullet.GUI)
        # pybullet.resetDebugVisualizerCamera(
        #     cameraDistance=1.5,
        #     cameraYaw=0,
        #     cameraPitch=-40,
        #     cameraTargetPosition=[0.55, -0.35, 0.2],
        # )
        if visuals_mode != "default" and visuals_mode != "real":
            raise ValueError("visuals_mode must be `real` or `default`.")
        self._task = task
        self._connection_mode = pybullet.DIRECT
        if shared_memory:
            self._connection_mode = pybullet.SHARED_MEMORY

        self.goal_dist_tolerance = goal_dist_tolerance

        self.effector_height = effector_height or EFFECTOR_HEIGHT

        self._visuals_mode = visuals_mode
        if visuals_mode == "default":
            self._camera_pose = DEFAULT_CAMERA_POSE
            self._camera_orientation = DEFAULT_CAMERA_ORIENTATION
            self.workspace_bounds = WORKSPACE_BOUNDS
            self._image_size = image_size
            self._camera_instrinsics = CAMERA_INTRINSICS
            self._workspace_urdf_path = WORKSPACE_URDF_PATH
        else:
            self._camera_pose = CAMERA_POSE_REAL
            self._camera_orientation = CAMERA_ORIENTATION_REAL
            self.workspace_bounds = WORKSPACE_BOUNDS_REAL
            self._image_size = image_size
            self._camera_instrinsics = CAMERA_INTRINSICS_REAL
            self._workspace_urdf_path = WORKSPACE_URDF_PATH_REAL

        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))  # x, y
        self.observation_space = self._create_observation_space(image_size)

        self._rng = np.random.RandomState(seed=seed)
        self._block_ids = None
        self._previous_state = None
        self._robot = None
        self._workspace_uid = None
        self._target_id = None
        self._target_pose = None
        self._target_effector_pose = None
        self._pybullet_client = None
        self.reach_target_translation = None
        self._setup_pybullet_scene()
        self._saved_state = None

        assert isinstance(self._pybullet_client, bullet_client.BulletClient)
        self._control_frequency = control_frequency
        self._step_frequency = (
            1 / self._pybullet_client.getPhysicsEngineParameters()["fixedTimeStep"]
        )

        self._last_loop_time = None
        self._last_loop_frame_sleep_time = None
        if self._step_frequency % self._control_frequency != 0:
            raise ValueError(
                "Control frequency should be a multiple of the "
                "configured Bullet TimeStep."
            )
        self._sim_steps_per_step = int(self._step_frequency / self._control_frequency)

        self.rendered_img = None
        self._abs_action = abs_action

        # Use saved_state and restore to make reset safe as no simulation state has
        # been updated at this state, but the assets are now loaded.
        self.save_state()
        self.reset()

    @property
    def pybullet_client(self):
        return self._pybullet_client

    @property
    def robot(self):
        return self._robot

    @property
    def workspace_uid(self):
        return self._workspace_uid

    @property
    def target_effector_pose(self):
        return self._target_effector_pose

    @property
    def target_pose(self):
        return self._target_pose

    @property
    def control_frequency(self):
        return self._control_frequency

    @property
    def connection_mode(self):
        return self._connection_mode

    def save_state(self):
        self._saved_state = self._pybullet_client.saveState()

    def set_goal_dist_tolerance(self, val):
        self.goal_dist_tolerance = val

    def get_control_frequency(self):
        return self._control_frequency

    def compute_state(self):
        return self._compute_state()

    def get_goal_translation(self):
        """Return the translation component of the goal (2D)."""
        if self._task == BlockTaskVariant.REACH:
            return np.concatenate([self.reach_target_translation, [0]])
        else:
            return self._target_pose.translation if self._target_pose else None

    def get_obj_ids(self):
        return self._block_ids

    def _setup_workspace_and_robot(self, end_effector="suction"):
        self._pybullet_client.resetSimulation()
        self._pybullet_client.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        self._pybullet_client.setPhysicsEngineParameter(enableFileCaching=0)
        self._pybullet_client.setGravity(0, 0, -9.8)

        utils_pybullet.load_urdf(
            self._pybullet_client, PLANE_URDF_PATH, basePosition=[0, 0, -0.001]
        )
        self._workspace_uid = utils_pybullet.load_urdf(
            self._pybullet_client,
            self._workspace_urdf_path,
            basePosition=[0.35, 0, 0.0],
        )

        self._robot = xarm_sim_robot.XArmSimRobot(
            self._pybullet_client,
            initial_joint_positions=INITIAL_JOINT_POSITIONS,
            end_effector=end_effector,
            color="white" if self._visuals_mode == "real" else "default",
        )

    def _setup_pybullet_scene(self):
        self._pybullet_client = bullet_client.BulletClient(self._connection_mode)

        # Temporarily disable rendering to speed up loading URDFs.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 0)

        self._setup_workspace_and_robot()

        if self._task == BlockTaskVariant.INSERT:
            target_urdf_path = INSERT_URDF_PATH
        else:
            target_urdf_path = ZONE_URDF_PATH

        self._target_id = utils_pybullet.load_urdf(
            self._pybullet_client, target_urdf_path, useFixedBase=True
        )
        self._block_ids = [
            utils_pybullet.load_urdf(
                self._pybullet_client, BLOCK_URDF_PATH, useFixedBase=False
            )
        ]

        # Re-enable rendering.
        pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_RENDERING, 1)

        self.step_simulation_to_stabilize()

    def step_simulation_to_stabilize(self, nsteps=100):
        for _ in range(nsteps):
            self._pybullet_client.stepSimulation()

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed=seed)

    def _set_robot_target_effector_pose(self, pose):
        self._target_effector_pose = pose
        self._robot.set_target_effector_pose(pose)

    def reset(self, reset_poses=True):
        workspace_center_x = 0.4

        if reset_poses:
            self._pybullet_client.restoreState(self._saved_state)

            rotation = transform.Rotation.from_rotvec([0, math.pi, 0])
            translation = np.array([0.3, -0.4, self.effector_height])
            starting_pose = Pose3d(rotation=rotation, translation=translation)
            self._set_robot_target_effector_pose(starting_pose)

            # Reset block pose.
            block_x = workspace_center_x + self._rng.uniform(low=-0.1, high=0.1)
            block_y = -0.2 + self._rng.uniform(low=-0.15, high=0.15)
            block_translation = np.array([block_x, block_y, 0])
            block_sampled_angle = self._rng.uniform(math.pi)
            block_rotation = transform.Rotation.from_rotvec([0, 0, block_sampled_angle])

            self._pybullet_client.resetBasePositionAndOrientation(
                self._block_ids[0],
                block_translation.tolist(),
                block_rotation.as_quat().tolist(),
            )

            # Reset target pose.
            target_x = workspace_center_x + self._rng.uniform(low=-0.10, high=0.10)
            target_y = 0.2 + self._rng.uniform(low=-0.15, high=0.15)
            target_translation = np.array([target_x, target_y, 0.020])

            target_sampled_angle = math.pi + self._rng.uniform(
                low=-math.pi / 6, high=math.pi / 6
            )
            target_rotation = transform.Rotation.from_rotvec(
                [0, 0, target_sampled_angle]
            )

            self._pybullet_client.resetBasePositionAndOrientation(
                self._target_id,
                target_translation.tolist(),
                target_rotation.as_quat().tolist(),
            )
        else:
            (
                target_translation,
                target_orientation_quat,
            ) = self._pybullet_client.getBasePositionAndOrientation(self._target_id)
            target_rotation = transform.Rotation.from_quat(target_orientation_quat)
            target_translation = np.array(target_translation)

        self._target_pose = Pose3d(
            rotation=target_rotation, translation=target_translation
        )

        if reset_poses:
            self.step_simulation_to_stabilize()

        state = self._compute_state()
        self._previous_state = state

        if self._task == BlockTaskVariant.REACH:
            self._compute_reach_target(state)

        self._init_goal_distance = self._compute_goal_distance(state)
        init_goal_eps = 1e-7
        assert self._init_goal_distance > init_goal_eps
        self.best_fraction_reduced_goal_dist = 0.0

        return state

    def _compute_goal_distance(self, state):
        goal_translation = self.get_goal_translation()
        if self._task != BlockTaskVariant.REACH:
            goal_distance = np.linalg.norm(
                state["block_translation"] - goal_translation[0:2]
            )
        else:
            goal_distance = np.linalg.norm(
                state["effector_translation"] - goal_translation[0:2]
            )
        return goal_distance

    def _compute_reach_target(self, state):
        xy_block = state["block_translation"]
        xy_target = state["target_translation"]

        xy_block_to_target = xy_target - xy_block
        xy_dir_block_to_target = (xy_block_to_target) / np.linalg.norm(
            xy_block_to_target
        )
        self.reach_target_translation = xy_block + -1 * xy_dir_block_to_target * 0.05

    def _compute_state(self):
        effector_pose = self._robot.forward_kinematics()
        block_position_and_orientation = (
            self._pybullet_client.getBasePositionAndOrientation(self._block_ids[0])
        )
        block_pose = Pose3d(
            rotation=transform.Rotation.from_quat(block_position_and_orientation[1]),
            translation=block_position_and_orientation[0],
        )

        def _yaw_from_pose(pose):
            return np.array([pose.rotation.as_euler("xyz", degrees=False)[-1]])

        obs = collections.OrderedDict(
            block_translation=block_pose.translation[0:2],
            block_orientation=_yaw_from_pose(block_pose),
            effector_translation=effector_pose.translation[0:2],
            effector_target_translation=self._target_effector_pose.translation[0:2],
            target_translation=self._target_pose.translation[0:2],
            target_orientation=_yaw_from_pose(self._target_pose),
        )
        if self._image_size is not None:
            obs["rgb"] = self._render_camera(self._image_size)
        return obs

    def _step_robot_and_sim(self, action):
        """Steps the robot and pybullet sim."""
        # Compute target_effector_pose by shifting the effector's pose by the
        # action.
        if self._abs_action:
            target_effector_translation = np.array([action[0], action[1], 0])
        else:
            target_effector_translation = np.array(
                self._target_effector_pose.translation
            ) + np.array([action[0], action[1], 0])

        target_effector_translation[0:2] = np.clip(
            target_effector_translation[0:2],
            self.workspace_bounds[0],
            self.workspace_bounds[1],
        )
        target_effector_translation[-1] = self.effector_height
        target_effector_pose = Pose3d(
            rotation=EFFECTOR_DOWN_ROTATION, translation=target_effector_translation
        )

        self._set_robot_target_effector_pose(target_effector_pose)

        # Update sleep time dynamically to stay near real-time.
        frame_sleep_time = 0
        if self._connection_mode == pybullet.SHARED_MEMORY:
            cur_time = time.time()
            if self._last_loop_time is not None:
                # Calculate the total, non-sleeping time from the previous frame, this
                # includes the actual step as well as any compute that happens in the
                # caller thread (model inference, etc).
                compute_time = (
                    cur_time
                    - self._last_loop_time
                    - self._last_loop_frame_sleep_time * self._sim_steps_per_step
                )
                # Use this to calculate the current frame's total sleep time to ensure
                # that env.step runs at policy rate. This is an estimate since the
                # previous frame's compute time may not match the current frame.
                total_sleep_time = max((1 / self._control_frequency) - compute_time, 0)
                # Now spread this out over the inner sim steps. This doesn't change
                # control in any way, but makes the animation appear smooth.
                frame_sleep_time = total_sleep_time / self._sim_steps_per_step
            else:
                # No estimate of the previous frame's compute, assume it is zero.
                frame_sleep_time = 1 / self._step_frequency

            # Cache end of this loop time, to compute sleep time on next iteration.
            self._last_loop_time = cur_time
            self._last_loop_frame_sleep_time = frame_sleep_time

        for _ in range(self._sim_steps_per_step):
            if self._connection_mode == pybullet.SHARED_MEMORY:
                sleep_spin(frame_sleep_time)
            self._pybullet_client.stepSimulation()

    def step(self, action):
        self._step_robot_and_sim(action)

        state = self._compute_state()

        goal_distance = self._compute_goal_distance(state)
        fraction_reduced_goal_distance = 1.0 - (
            goal_distance / self._init_goal_distance
        )
        if fraction_reduced_goal_distance > self.best_fraction_reduced_goal_dist:
            self.best_fraction_reduced_goal_dist = fraction_reduced_goal_distance

        done = False
        reward = self.best_fraction_reduced_goal_dist

        # Terminate the episode if the block is close enough to the target.
        if goal_distance < self.goal_dist_tolerance:
            reward = 1.0
            done = True

        return state, reward, done, {}

    @property
    def succeeded(self):
        state = self._compute_state()
        goal_distance = self._compute_goal_distance(state)
        if goal_distance < self.goal_dist_tolerance:
            return True
        return False

    @property
    def goal_distance(self):
        state = self._compute_state()
        return self._compute_goal_distance(state)

    def render(self, mode="rgb_array"):
        if self._image_size is not None:
            image_size = self._image_size
        else:
            # This allows rendering even for state-only obs,
            # for visualization.
            image_size = (IMAGE_HEIGHT, IMAGE_WIDTH)

        data = self._render_camera(image_size=(image_size[0], image_size[1]))
        if mode == "human":
            if self.rendered_img is None:
                self.rendered_img = plt.imshow(
                    np.zeros((image_size[0], image_size[1], 4))
                )
            else:
                self.rendered_img.set_data(data)
            plt.draw()
            plt.pause(0.00001)
        return data

    def close(self):
        self._pybullet_client.disconnect()

    def calc_camera_params(self, image_size):
        # Mimic RealSense D415 camera parameters.
        intrinsics = self._camera_instrinsics

        # Set default camera poses.
        front_position = self._camera_pose
        front_rotation = self._camera_orientation
        front_rotation = self._pybullet_client.getQuaternionFromEuler(front_rotation)
        # Default camera configs.
        zrange = (0.01, 10.0)

        # OpenGL camera settings.
        lookdir = np.float32([0, 0, 1]).reshape(3, 1)
        updir = np.float32([0, -1, 0]).reshape(3, 1)
        rotation = self._pybullet_client.getMatrixFromQuaternion(front_rotation)
        rotm = np.float32(rotation).reshape(3, 3)
        lookdir = (rotm @ lookdir).reshape(-1)
        updir = (rotm @ updir).reshape(-1)
        lookat = front_position + lookdir
        focal_len = intrinsics[0]
        znear, zfar = zrange
        viewm = self._pybullet_client.computeViewMatrix(front_position, lookat, updir)
        fovh = (image_size[0] / 2) / focal_len
        fovh = 180 * np.arctan(fovh) * 2 / np.pi

        # Notes: 1) FOV is vertical FOV 2) aspect must be float
        aspect_ratio = image_size[1] / image_size[0]
        projm = self._pybullet_client.computeProjectionMatrixFOV(
            fovh, aspect_ratio, znear, zfar
        )

        return viewm, projm, front_position, lookat, updir

    def _render_camera(self, image_size):
        """Render RGB image with RealSense configuration."""
        viewm, projm, _, _, _ = self.calc_camera_params(image_size)

        # Render with OpenGL camera settings.
        _, _, color, _, _ = self._pybullet_client.getCameraImage(
            width=image_size[1],
            height=image_size[0],
            viewMatrix=viewm,
            projectionMatrix=projm,
            flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            renderer=pybullet.ER_BULLET_HARDWARE_OPENGL,
        )

        # Get color image.
        color_image_size = (image_size[0], image_size[1], 4)
        color = np.array(color, dtype=np.uint8).reshape(color_image_size)
        color = color[:, :, :3]  # remove alpha channel

        return color.astype(np.uint8)

    def _create_observation_space(self, image_size):
        pi2 = math.pi * 2

        obs_dict = collections.OrderedDict(
            block_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            block_orientation=spaces.Box(low=-pi2, high=pi2, shape=(1,)),  # phi
            effector_translation=spaces.Box(
                low=self.workspace_bounds[0] - 0.1,  # Small buffer for to IK noise.
                high=self.workspace_bounds[1] + 0.1,
            ),  # x,y
            effector_target_translation=spaces.Box(
                low=self.workspace_bounds[0] - 0.1,  # Small buffer for to IK noise.
                high=self.workspace_bounds[1] + 0.1,
            ),  # x,y
            target_translation=spaces.Box(low=-5, high=5, shape=(2,)),  # x,y
            target_orientation=spaces.Box(
                low=-pi2,
                high=pi2,
                shape=(1,),
            ),  # theta
        )
        if image_size is not None:
            obs_dict["rgb"] = spaces.Box(
                low=0, high=255, shape=(image_size[0], image_size[1], 3), dtype=np.uint8
            )
        return spaces.Dict(obs_dict)

    def get_pybullet_state(self):
        """Save pybullet state of the scene.

        Returns:
          dict containing 'robots', 'robot_end_effectors', 'targets', 'objects',
            each containing a list of ObjState.
        """
        state: Dict[str, List[ObjState]] = {}

        state["robots"] = [
            XarmState.get_bullet_state(
                self._pybullet_client,
                self.robot.xarm,
                target_effector_pose=self._target_effector_pose,
                goal_translation=self.get_goal_translation(),
            )
        ]

        state["robot_end_effectors"] = []
        if self.robot.end_effector:
            state["robot_end_effectors"].append(
                ObjState.get_bullet_state(
                    self._pybullet_client, self.robot.end_effector
                )
            )

        state["targets"] = []
        if self._target_id:
            state["targets"].append(
                ObjState.get_bullet_state(self._pybullet_client, self._target_id)
            )

        state["objects"] = []
        for obj_id in self.get_obj_ids():
            state["objects"].append(
                ObjState.get_bullet_state(self._pybullet_client, obj_id)
            )

        return state

    def set_pybullet_state(self, state):
        """Restore pyullet state.

        WARNING: py_environment wrapper assumes environments aren't reset in their
        constructor and will often reset the environment unintentionally. It is
        always recommended that you call env.reset on the tfagents wrapper before
        playback (replaying pybullet_state).

        Args:
          state: dict containing 'robots', 'robot_end_effectors', 'targets',
            'objects', each containing a list of ObjState.
        """

        assert isinstance(state["robots"][0], XarmState)
        xarm_state: XarmState = state["robots"][0]
        xarm_state.set_bullet_state(self._pybullet_client, self.robot.xarm)
        self._set_robot_target_effector_pose(xarm_state.target_effector_pose)

        def _set_state_safe(obj_state, obj_id):
            if obj_state is not None:
                assert obj_id is not None, "Cannot set state for missing object."
                obj_state.set_bullet_state(self._pybullet_client, obj_id)
            else:
                assert obj_id is None, f"No state found for obj_id {obj_id}"

        robot_end_effectors = state["robot_end_effectors"]
        _set_state_safe(
            None if not robot_end_effectors else robot_end_effectors[0],
            self.robot.end_effector,
        )

        targets = state["targets"]
        _set_state_safe(None if not targets else targets[0], self._target_id)

        obj_ids = self.get_obj_ids()
        assert len(state["objects"]) == len(obj_ids), "State length mismatch"
        for obj_state, obj_id in zip(state["objects"], obj_ids):
            _set_state_safe(obj_state, obj_id)

        self.reset(reset_poses=False)


class BlockPushNormalized(gym.Env):
    """Simple XArm environment for block pushing, normalized state and actions."""

    def __init__(
        self,
        control_frequency=10.0,
        task=BlockTaskVariant.PUSH_NORMALIZED,
        image_size=None,
        shared_memory=False,
        seed=None,
    ):
        """Creates an env instance.

        Args:
          control_frequency: Control frequency for the arm. Each env step will
            advance the simulation by 1/control_frequency seconds.
          task: enum for which task, see BlockTaskVariant enum.
          image_size: Optional image size (height, width). If None, no image
            observations will be used.
          shared_memory: If True `pybullet.SHARED_MEMORY` is used to connect to
            pybullet. Useful to debug.
          seed: Optional seed for the environment.
        """
        # Map normalized task to unnormalized task.
        if task == BlockTaskVariant.PUSH_NORMALIZED:
            env_task = BlockTaskVariant.PUSH
        elif task == BlockTaskVariant.REACH_NORMALIZED:
            env_task = BlockTaskVariant.REACH
        else:
            raise ValueError("Unsupported task %s" % str(task))
        self._env = BlockPush(
            control_frequency, env_task, image_size, shared_memory, seed
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,))
        self.observation_space = spaces.Dict(
            collections.OrderedDict(
                effector_target_translation=spaces.Box(low=-1, high=1, shape=(2,)),
                effector_target_to_block_translation=spaces.Box(
                    low=-1, high=1, shape=(2,)
                ),
                block_orientation_cos_sin=spaces.Box(low=-1, high=1, shape=(2,)),
                effector_target_to_target_translation=spaces.Box(
                    low=-1, high=1, shape=(2,)
                ),
                target_orientation_cos_sin=spaces.Box(low=-1, high=1, shape=(2,)),
            )
        )
        self.reset()

    def get_control_frequency(self):
        return self._env.get_control_frequency()

    @property
    def reach_target_translation(self):
        return self._env.reach_target_translation

    def seed(self, seed=None):
        self._env.seed(seed)

    def reset(self):
        state = self._env.reset()
        return self.calc_normalized_state(state)

    def step(self, action):
        # The environment is normalized [mean-3*std, mean+3*std] -> [-1, 1].
        action = np.clip(action, a_min=-1.0, a_max=1.0)
        state, reward, done, info = self._env.step(
            self.calc_unnormalized_action(action)
        )
        state = self.calc_normalized_state(state)
        reward = reward * 100  # Keep returns in [0, 100]
        return state, reward, done, info

    def render(self, mode="rgb_array"):
        return self._env.render(mode)

    def close(self):
        self._env.close()

    @staticmethod
    def _normalize(values, values_min, values_max):
        offset = (values_max + values_min) * 0.5
        scale = (values_max - values_min) * 0.5
        return (values - offset) / scale  # [min, max] -> [-1, 1]

    @staticmethod
    def _unnormalize(values, values_min, values_max):
        offset = (values_max + values_min) * 0.5
        scale = (values_max - values_min) * 0.5
        return values * scale + offset  # [-1, 1] -> [min, max]

    @classmethod
    def calc_normalized_action(cls, action):
        return cls._normalize(action, ACTION_MIN, ACTION_MAX)

    @classmethod
    def calc_unnormalized_action(cls, norm_action):
        return cls._unnormalize(norm_action, ACTION_MIN, ACTION_MAX)

    @classmethod
    def calc_normalized_state(cls, state):

        effector_target_translation = cls._normalize(
            state["effector_target_translation"],
            EFFECTOR_TARGET_TRANSLATION_MIN,
            EFFECTOR_TARGET_TRANSLATION_MAX,
        )

        effector_target_to_block_translation = cls._normalize(
            state["block_translation"] - state["effector_target_translation"],
            EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MIN,
            EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MAX,
        )
        ori = state["block_orientation"][0]
        block_orientation_cos_sin = cls._normalize(
            np.array([math.cos(ori), math.sin(ori)], np.float32),
            BLOCK_ORIENTATION_COS_SIN_MIN,
            BLOCK_ORIENTATION_COS_SIN_MAX,
        )

        effector_target_to_target_translation = cls._normalize(
            state["target_translation"] - state["effector_target_translation"],
            EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MIN,
            EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MAX,
        )
        ori = state["target_orientation"][0]
        target_orientation_cos_sin = cls._normalize(
            np.array([math.cos(ori), math.sin(ori)], np.float32),
            TARGET_ORIENTATION_COS_SIN_MIN,
            TARGET_ORIENTATION_COS_SIN_MAX,
        )

        # Note: We do not include effector_translation in the normalized state.
        # This means the unnormalized -> normalized mapping is not invertable.
        return collections.OrderedDict(
            effector_target_translation=effector_target_translation,
            effector_target_to_block_translation=effector_target_to_block_translation,
            block_orientation_cos_sin=block_orientation_cos_sin,
            effector_target_to_target_translation=effector_target_to_target_translation,
            target_orientation_cos_sin=target_orientation_cos_sin,
        )

    @classmethod
    def calc_unnormalized_state(cls, norm_state):

        effector_target_translation = cls._unnormalize(
            norm_state["effector_target_translation"],
            EFFECTOR_TARGET_TRANSLATION_MIN,
            EFFECTOR_TARGET_TRANSLATION_MAX,
        )
        # Note: normalized state does not include effector_translation state, this
        # means this component will be missing (and is marked nan).
        effector_translation = np.array([np.nan, np.nan], np.float32)

        effector_target_to_block_translation = cls._unnormalize(
            norm_state["effector_target_to_block_translation"],
            EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MIN,
            EFFECTOR_TARGET_TO_BLOCK_TRANSLATION_MAX,
        )
        block_translation = (
            effector_target_to_block_translation + effector_target_translation
        )
        ori_cos_sin = cls._unnormalize(
            norm_state["block_orientation_cos_sin"],
            BLOCK_ORIENTATION_COS_SIN_MIN,
            BLOCK_ORIENTATION_COS_SIN_MAX,
        )
        block_orientation = np.array(
            [math.atan2(ori_cos_sin[1], ori_cos_sin[0])], np.float32
        )

        effector_target_to_target_translation = cls._unnormalize(
            norm_state["effector_target_to_target_translation"],
            EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MIN,
            EFFECTOR_TARGET_TO_TARGET_TRANSLATION_MAX,
        )
        target_translation = (
            effector_target_to_target_translation + effector_target_translation
        )
        ori_cos_sin = cls._unnormalize(
            norm_state["target_orientation_cos_sin"],
            TARGET_ORIENTATION_COS_SIN_MIN,
            TARGET_ORIENTATION_COS_SIN_MAX,
        )
        target_orientation = np.array(
            [math.atan2(ori_cos_sin[1], ori_cos_sin[0])], np.float32
        )

        return collections.OrderedDict(
            block_translation=block_translation,
            block_orientation=block_orientation,
            effector_translation=effector_translation,
            effector_target_translation=effector_target_translation,
            target_translation=target_translation,
            target_orientation=target_orientation,
        )

    def get_pybullet_state(self):
        return self._env.get_pybullet_state()

    def set_pybullet_state(self, state):
        return self._env.set_pybullet_state(state)

    @property
    def pybullet_client(self):
        return self._env.pybullet_client

    def calc_camera_params(self, image_size):
        return self._env.calc_camera_params(image_size)

    def _compute_state(self):
        return self.calc_normalized_state(
            self._env._compute_state()
        )  # pylint: disable=protected-access


# Make sure we only register once to allow us to reload the module in colab for
# debugging.
if "BlockPush-v0" in registration.registry.env_specs:
    del registration.registry.env_specs["BlockInsert-v0"]
    del registration.registry.env_specs["BlockPush-v0"]
    del registration.registry.env_specs["BlockPushNormalized-v0"]
    del registration.registry.env_specs["BlockPushRgbNormalized-v0"]
    del registration.registry.env_specs["BlockReach-v0"]
    del registration.registry.env_specs["BlockReachNormalized-v0"]
    del registration.registry.env_specs["BlockReachRgbNormalized-v0"]
    del registration.registry.env_specs["SharedBlockInsert-v0"]
    del registration.registry.env_specs["SharedBlockPush-v0"]
    del registration.registry.env_specs["SharedBlockReach-v0"]

registration.register(
    id="BlockInsert-v0",
    entry_point=BlockPush,
    kwargs=dict(task=BlockTaskVariant.INSERT),
    max_episode_steps=50,
)
registration.register(id="BlockPush-v0", entry_point=BlockPush, max_episode_steps=100)
registration.register(
    id="BlockPushNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(task=BlockTaskVariant.PUSH_NORMALIZED),
    max_episode_steps=100,
)
registration.register(
    id="BlockPushRgb-v0",
    entry_point=BlockPush,
    max_episode_steps=100,
    kwargs=dict(image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
)
registration.register(
    id="BlockPushRgbNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(
        task=BlockTaskVariant.PUSH_NORMALIZED, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    ),
    max_episode_steps=100,
)
registration.register(
    id="BlockReach-v0",
    entry_point=BlockPush,
    kwargs=dict(task=BlockTaskVariant.REACH),
    max_episode_steps=50,
)
registration.register(
    id="BlockReachRgb-v0",
    entry_point=BlockPush,
    max_episode_steps=100,
    kwargs=dict(task=BlockTaskVariant.REACH, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)),
)
registration.register(
    id="BlockReachNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(task=BlockTaskVariant.REACH_NORMALIZED),
    max_episode_steps=50,
)
registration.register(
    id="BlockReachRgbNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(
        task=BlockTaskVariant.REACH_NORMALIZED, image_size=(IMAGE_HEIGHT, IMAGE_WIDTH)
    ),
    max_episode_steps=50,
)

registration.register(
    id="SharedBlockInsert-v0",
    entry_point=BlockPush,
    kwargs=dict(task=BlockTaskVariant.INSERT, shared_memory=True),
    max_episode_steps=50,
)
registration.register(
    id="SharedBlockPush-v0",
    entry_point=BlockPush,
    kwargs=dict(shared_memory=True),
    max_episode_steps=100,
)
registration.register(
    id="SharedBlockPushNormalized-v0",
    entry_point=BlockPushNormalized,
    kwargs=dict(task=BlockTaskVariant.PUSH_NORMALIZED, shared_memory=True),
    max_episode_steps=100,
)
registration.register(
    id="SharedBlockReach-v0",
    entry_point=BlockPush,
    kwargs=dict(task=BlockTaskVariant.REACH, shared_memory=True),
    max_episode_steps=50,
)

```

## reference_material/diffusion_policy_code/diffusion_policy/env/robomimic/robomimic_lowdim_wrapper.py
```python
from typing import List, Dict, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite

class RobomimicLowdimWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview'
        ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.seed_state_map = dict()
        self._seed = None
        
        # setup spaces
        low = np.full(env.action_dimension, fill_value=-1)
        high = np.full(env.action_dimension, fill_value=1)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )

    def get_observation(self):
        raw_obs = self.env.get_observation()
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)


def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = '/home/cchi/dev/diffusion_policy/data/robomimic/datasets/square/ph/low_dim.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=False, 
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'
        ]
    )

    states = list()
    for _ in range(2):
        wrapper.seed(0)
        wrapper.reset()
        states.append(wrapper.env.get_state()['states'])
    assert np.allclose(states[0], states[1])

    img = wrapper.render()
    plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])

```

## reference_material/diffusion_policy_code/diffusion_policy/env/robomimic/robomimic_image_wrapper.py
```python
from typing import List, Optional
from matplotlib.pyplot import fill
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite

class RobomimicImageWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        
        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")
            
            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()
        
        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs
    
    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img


def test():
    import os
    from omegaconf import OmegaConf
    cfg_path = os.path.expanduser('~/dev/diffusion_policy/diffusion_policy/config/task/lift_image.yaml')
    cfg = OmegaConf.load(cfg_path)
    shape_meta = cfg['shape_meta']


    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = os.path.expanduser('~/dev/diffusion_policy/data/robomimic/datasets/square/ph/image.hdf5')
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=True, 
    )

    wrapper = RobomimicImageWrapper(
        env=env,
        shape_meta=shape_meta
    )
    wrapper.seed(0)
    obs = wrapper.reset()
    img = wrapper.render()
    plt.imshow(img)


    # states = list()
    # for _ in range(2):
    #     wrapper.seed(0)
    #     wrapper.reset()
    #     states.append(wrapper.env.get_state()['states'])
    # assert np.allclose(states[0], states[1])

    # img = wrapper.render()
    # plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])

```
