# All Code from gridworld

## gridworld/alg/__init__.py
```python
from .ppo import PPOWrapper
from .utils import HistoryLoggerCallback

ALGORITHM = {
    'PPO': PPOWrapper,
}
```

## gridworld/alg/ppo.py
```python
import torch
from stable_baselines3 import PPO


class PPOWrapper(PPO):
    def __init__(self, config, env, seed, log_dir):
        policy = config['policy']
        n_steps = config['n_steps']
        batch_size = config['batch_size']
        n_epochs = config['n_epochs']
        lr = config['source_lr']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = env

        super(PPOWrapper, self).__init__(policy=policy,
                                         env=env,
                                         learning_rate=lr,
                                         n_steps=n_steps,
                                         batch_size=batch_size,
                                         n_epochs=n_epochs,
                                         verbose=0,
                                         seed=seed,
                                         device=device,
                                         tensorboard_log=log_dir)
```

## gridworld/alg/utils.py
```python
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class HistoryLoggerCallback(BaseCallback):
    def __init__(self, env_name, env_idx, history=None):
        super(HistoryLoggerCallback, self).__init__()
        self.env_name = env_name
        self.env_idx = env_idx

        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

        self.history = history

        self.episode_rewards = []
        self.episode_success = []

    def _on_step(self) -> bool:
        # Capture state, action, and reward at each step
        self.states.append(self.locals["obs_tensor"].cpu().numpy())
        self.next_states.append(self.locals["new_obs"])
        self.actions.append(self.locals["actions"])

        self.rewards.append(self.locals["rewards"].copy())
        self.dones.append(self.locals["dones"])

        self.episode_rewards.append(self.locals['rewards'])
        
        if self.locals['dones'][0]:
            mean_reward = np.mean(np.mean(self.episode_rewards, axis=0))
            self.logger.record('rollout/mean_reward', mean_reward)
            self.episode_rewards = []
                        
        return True

    def _on_training_end(self):
        self.history[self.env_idx] = {
            'states': np.array(self.states, dtype=np.int32),
            'actions': np.array(self.actions, dtype=np.int32),
            'rewards': np.array(self.rewards, dtype=np.int32),
            'next_states': np.array(self.next_states, dtype=np.int32),
            'dones': np.array(self.dones, dtype=np.bool_)
        }

```

## gridworld/collect_data.py
```python
import os
from datetime import datetime

import yaml

import sys
sys.path.append(os.path.dirname(sys.path[0]))

from env import SAMPLE_ENVIRONMENT, make_env, Darkroom, DarkroomPermuted, DarkKeyToDoor
from alg import ALGORITHM, HistoryLoggerCallback
import argparse
import multiprocessing
from utils import get_config, get_traj_file_name
import h5py
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=True, help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=True, help="Environment config")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Directory for history saving")
    parser.add_argument('--override', '-o', default='')
    args = parser.parse_args()
    return args


def worker(arg, config, traj_dir, env_idx, history, file_name):
    
    if config['env'] == 'darkroom':
        env = DummyVecEnv([make_env(config, goal=arg)] * config['n_stream'])
    elif config['env'] == 'darkroompermuted':
        env = DummyVecEnv([make_env(config, perm_idx=arg)] * config['n_stream'])
    elif config['env'] == 'darkkeytodoor':
        env = DummyVecEnv([make_env(config, key=arg[:2], goal=arg[2:])] * config['n_stream'])
    else:
        raise ValueError('Invalid environment')
    
    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx
    
    # Initialize algorithm
    alg = ALGORITHM[alg_name](config, env, seed, traj_dir)

    callback = HistoryLoggerCallback(config['env'], env_idx, history)

    log_name = f'{file_name}_{env_idx}'

    # Execute learning algorithm
    alg.learn(total_timesteps=config['total_source_timesteps'],
              callback=callback,
              log_interval=1,
              tb_log_name=log_name,
              reset_num_timesteps=True,
              progress_bar=False)
    
    env.close()


if __name__ == '__main__':
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))

    # Ensure the log directory exists
    if not os.path.exists(args.traj_dir):
        os.makedirs(args.traj_dir, exist_ok=True)

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)
    
    train_args, test_args = SAMPLE_ENVIRONMENT[config['env']](config, shuffle=False)

    total_args = train_args + test_args
    n_envs = len(total_args)

    file_name = get_traj_file_name(config)
    path = f'{args.traj_dir}/{file_name}.hdf5'
    
    start_time = datetime.now()
    print(f'Training started at {start_time}')

    with multiprocessing.Manager() as manager:
        history = manager.dict()

        # Create a pool with a maximum of n_workers
        with multiprocessing.Pool(processes=config['n_process']) as pool:
            # Map the worker function to the environments with the other arguments
            pool.starmap(worker, [(total_args[i], config, args.traj_dir, i, history, file_name) for i in range(n_envs)])

        # Save the history dictionary
        with h5py.File(path, 'w-') as f:
            for i in range(n_envs):
                env_group = f.create_group(f'{i}')
                for key, value in history[i].items():
                    env_group.create_dataset(key, data=value)
    
    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    
    start_time = datetime.now()
    print(f'Annotating optimal actions for DPT started at {start_time}')
    
    with h5py.File(path, 'a') as f:
        for i in range(n_envs):            
            if config['env'] == 'darkroom':
                states = f[f'{i}']['states'][()].transpose(1, 0, 2)
                actions = f[f'{i}']['actions'][()].transpose(1, 0)
                rewards = f[f'{i}']['rewards'][()].transpose(1, 0)
                env = Darkroom(config, goal=total_args[i])
                optimal_actions = np.zeros_like(actions)
                
                for stream_idx in range(states.shape[0]):
                    for step_idx in range(states.shape[1]):
                        optimal_actions[stream_idx, step_idx] = env.get_optimal_action(states[stream_idx, step_idx])
            
                group = f[f'{i}']
                group.create_dataset('optimal_actions', data=optimal_actions)
                
            elif config['env'] == 'darkroompermuted':
                states = f[f'{i}']['states'][()].transpose(1, 0, 2)
                actions = f[f'{i}']['actions'][()].transpose(1, 0)
                rewards = f[f'{i}']['rewards'][()].transpose(1, 0)
                env = DarkroomPermuted(config, perm_idx=i)
                optimal_actions = np.zeros_like(actions)
                
                for stream_idx in range(states.shape[0]):
                    for step_idx in range(states.shape[1]):
                        optimal_actions[stream_idx, step_idx] = env.get_optimal_action(states[stream_idx, step_idx])
                
                group = f[f'{i}']
                group.create_dataset('optimal_actions', data=optimal_actions)
                
            elif config['env'] == 'darkkeytodoor':
                states = f[f'{i}']['states'][()].transpose(1, 0, 2).reshape(100, -1, config['horizon'], 2)
                actions = f[f'{i}']['actions'][()].transpose(1, 0).reshape(100, -1, config['horizon'])
                rewards = f[f'{i}']['rewards'][()].transpose(1, 0).reshape(100, -1, config['horizon'])
                env = DarkKeyToDoor(config, key=total_args[i][:2], goal=total_args[i][2:])
                optimal_actions = np.zeros_like(actions)
                
                for stream_idx in range(states.shape[0]):
                    for episode_idx in range(states.shape[1]):
                        have_key=False
                        for step_idx in range(states.shape[2]):
                            optimal_actions[stream_idx, episode_idx, step_idx] = env.get_optimal_action(states[stream_idx, episode_idx, step_idx], have_key)
                            if not have_key and rewards[stream_idx, episode_idx, step_idx] > 0:
                                have_key = True
                        
                group = f[f'{i}']
                group.create_dataset('optimal_actions', data=optimal_actions.reshape(100, -1))
                
            else:
                raise ValueError('Invalid environment')
                
    end_time = datetime.now()
    print()
    print(f'Annotating ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
```

## gridworld/dataset.py
```python
from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
import random
from einops import rearrange, repeat


class ADDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        elif self.env == 'darkroompermuted':
            n_total_envs = 120
        elif self.env == 'darkkeytodoor':
            n_total_envs = config['grid_size'] ** 4
        else:
            raise ValueError('Invalid env')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
    
    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
            
        traj = {
            'query_states': self.states[history_idx, transition_idx + self.n_transit - 1],
            'target_actions': self.actions[history_idx, transition_idx + self.n_transit - 1],
            'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
        }
        
        if self.dynamics:
            traj.update({
                'target_next_states': self.next_states[history_idx, transition_idx + self.n_transit - 1],
                'target_rewards': self.rewards[history_idx, transition_idx + self.n_transit - 1],
            })
        
        return traj
    
    
class DPTDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        elif self.env == 'darkroompermuted':
            n_total_envs = 120
        elif self.env == 'darkkeytodoor':
            n_total_envs = config['grid_size'] ** 4
        else:
            raise ValueError('Invalid env')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []
        optimal_actions = []
        
        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                optimal_actions.append(f[f'{i}']['optimal_actions'][()][:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        self.optimal_actions = np.concatenate(optimal_actions, axis=0)

    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
            
        traj = {
            'query_states': self.states[history_idx, transition_idx + self.n_transit - 1],
            'target_actions': self.optimal_actions[history_idx, transition_idx + self.n_transit - 1],
            'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit - 1],
            'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit - 1],
        }
        
        if self.dynamics:
            traj.update({
                'query_actions': self.actions[history_idx, transition_idx + self.n_transit - 1],
                'target_next_states': self.next_states[history_idx, transition_idx + self.n_transit - 1],
                'target_rewards': self.rewards[history_idx, transition_idx + self.n_transit - 1]
            })
        
        return traj
        

class IDTDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.dynamics = config['dynamics']
        
        if self.env == 'darkroom':
            n_total_envs = config['grid_size'] ** 2
        elif self.env == 'darkroompermuted':
            n_total_envs = 120
        elif self.env == 'darkkeytodoor':
            n_total_envs = config['grid_size'] ** 4
        else:
            raise ValueError('Invalid env')

        total_env_idx = list(range(n_total_envs))
        random.seed(config['env_split_seed'])
        random.shuffle(total_env_idx)
        
        n_train_envs = round(n_total_envs * config['train_env_ratio'])
        
        if mode == 'train':
            env_idx = total_env_idx[:n_train_envs]
        elif mode == 'test':
            env_idx = total_env_idx[n_train_envs:]
        elif mode == 'all':
            env_idx = total_env_idx
        else:
            raise ValueError('Invalid mode')

        states = []
        actions = []
        rewards = []
        next_states = []

        with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
            for i in env_idx:
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
                    
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.return_to_go = self.get_return_to_go(self.rewards)
        self.next_states = np.concatenate(next_states, axis=0)
        
        self.sort_episodes()
        
        self.return_to_go = self.relabel_return_to_go(self.return_to_go)
        
    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
    
    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
            
        traj = {
            'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit],
            'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit],
            'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit],
            'return_to_go': self.return_to_go[history_idx, transition_idx:transition_idx + self.n_transit],
            'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit],
        }
        
        return traj
    
    def get_return_to_go(self, rewards):
        episode_rewards = rewards.reshape(-1, rewards.shape[1] // self.config['horizon'], self.config['horizon'])
        return np.flip(np.flip(episode_rewards, axis=-1).cumsum(axis=-1), axis=-1).reshape(-1, rewards.shape[1])
    
    def sort_episodes(self):
        return_to_go = rearrange(self.return_to_go, 'traj (epi time) -> traj epi time', time=self.config['horizon'])        
        sorted_episode_idx = np.argsort(return_to_go[:, :, 0])
        sorted_episode_idx = repeat(sorted_episode_idx, 'traj epi -> traj epi time', time=self.config['horizon'])
        
        return_to_go = np.take_along_axis(return_to_go, sorted_episode_idx, axis=1)
        self.return_to_go = rearrange(return_to_go, 'traj epi time -> traj (epi time)')
        
        actions = rearrange(self.actions, 'traj (epi time) -> traj epi time', time=self.config['horizon'])
        actions = np.take_along_axis(actions, sorted_episode_idx, axis=1)
        self.actions = rearrange(actions, 'traj epi time -> traj (epi time)')
        
        rewards = rearrange(self.rewards, 'traj (epi time) -> traj epi time', time=self.config['horizon'])
        rewards = np.take_along_axis(rewards, sorted_episode_idx, axis=1)
        self.rewards = rearrange(rewards, 'traj epi time -> traj (epi time)')
        
        sorted_episode_idx = repeat(sorted_episode_idx, 'traj epi time -> traj epi time dim', dim=self.states.shape[-1])
        
        states = rearrange(self.states, 'traj (epi time) dim -> traj epi time dim', time=self.config['horizon'])
        states = np.take_along_axis(states, sorted_episode_idx, axis=1)
        self.states = rearrange(states, 'traj epi time dim -> traj (epi time) dim')
        
        next_states = rearrange(self.next_states, 'traj (epi time) dim -> traj epi time dim', time=self.config['horizon'])
        next_states = np.take_along_axis(next_states, sorted_episode_idx, axis=1)
        self.next_states = rearrange(next_states, 'traj epi time dim -> traj (epi time) dim')
    
    def relabel_return_to_go(self, rtg):
        max_episode_rtg = rtg.max(axis=-1) # (num_traj, )
        max_episode_rtg = repeat(max_episode_rtg, 'traj -> traj epi', epi=rtg.shape[1] // self.config['horizon'])
        
        episode_rtg = rtg.reshape(-1, rtg.shape[1] // self.config['horizon'], self.config['horizon'])
        
        episode_offset = max_episode_rtg - episode_rtg[:, :, 0]
        offset = repeat(episode_offset, 'traj epi -> traj epi time', time=self.config['horizon'])
        
        return (episode_rtg + offset).reshape(-1, rtg.shape[1])
```

## gridworld/env/__init__.py
```python
from .darkroom import sample_darkroom, sample_darkroom_permuted, Darkroom, DarkroomPermuted, map_dark_states, map_dark_states_inverse
from .dark_key_to_door import DarkKeyToDoor, sample_dark_key_to_door


ENVIRONMENT = {
    'darkroom': Darkroom,
    'darkroompermuted': DarkroomPermuted,
    'darkkeytodoor': DarkKeyToDoor,
}


SAMPLE_ENVIRONMENT = {
    'darkroom': sample_darkroom,
    'darkroompermuted': sample_darkroom_permuted,
    'darkkeytodoor': sample_dark_key_to_door,
}


def make_env(config, **kwargs):
    def _init():
            return ENVIRONMENT[config['env']](config, **kwargs)
    return _init
```

## gridworld/env/dark_key_to_door.py
```python
from __future__ import annotations

import random
from typing import Any, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType


def sample_dark_key_to_door(config, shuffle=True):
    keys_goals = [np.array([i, j, k, l])
             for i in range(config['grid_size']) for j in range(config['grid_size'])
             for k in range(config['grid_size']) for l in range(config['grid_size'])]
    
    if shuffle:
        random.seed(config['env_split_seed'])
        random.shuffle(keys_goals)

    n_train_envs = round(config['grid_size'] ** 4 * config['train_env_ratio'])

    train_keys_goals = keys_goals[:n_train_envs]
    test_keys_goals = keys_goals[n_train_envs:]

    return train_keys_goals, test_keys_goals


class DarkKeyToDoor(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, config, **kwargs):
        super(DarkKeyToDoor, self).__init__()
        self.grid_size = config['grid_size']
        if 'key' in kwargs:
            self.key = kwargs['key']
        else:
            self.key = np.random.randint(0, self.grid_size, 2)
        if 'goal' in kwargs:
            self.goal = kwargs['goal']
        else:
            self.goal = np.random.randint(0, self.grid_size, 2)
        self.horizon = config['horizon']
        self.dim_obs = 2
        self.dim_action = 5
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(self.dim_obs,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.dim_action)

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> Tuple[ObsType, dict[str, Any]]:
        self.current_step = 0

        center = self.grid_size // 2
        self.state = np.array([center, center])
        self.have_key = False
        self.reach_goal = False

        return self.state, {}

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        s = np.array(self.state)
        a = action

        # Action handling
        if a == 0:
            s[0] += 1
        elif a == 1:
            s[0] -= 1
        elif a == 2:
            s[1] += 1
        elif a == 3:
            s[1] -= 1

        s = np.clip(s, 0, self.grid_size - 1)
        self.state = s

        info = {}
        info['already_success'] = self.reach_goal

        if not self.have_key and np.array_equal(s, self.key):
            self.have_key = True
            reward = 1
        elif self.have_key and not self.reach_goal and np.array_equal(s, self.goal):
            self.reach_goal = True
            reward = 1
        else:
            reward = 0

        self.current_step += 1

        done = self.current_step >= self.horizon
        
        info['success'] = self.reach_goal

        return s.copy(), reward, done, done, info
    
    def get_optimal_action(self, state, have_key=False):
        if have_key:
            if state[0] < self.goal[0]:
                a = 0
            elif state[0] > self.goal[0]:
                a = 1
            elif state[1] < self.goal[1]:
                a = 2
            elif state[1] > self.goal[1]:
                a = 3
            else:
                a = 4
        else:
            if state[0] < self.key[0]:
                a = 0
            elif state[0] > self.key[0]:
                a = 1
            elif state[1] < self.key[1]:
                a = 2
            elif state[1] > self.key[1]:
                a = 3
            else:
                a = 4
            
        return a
    
    def get_max_return(self):
        return 2
```

## gridworld/env/darkroom.py
```python
from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObsType
import torch
from typing import Any, Tuple
import random
import itertools


######################
# 说明：
# - 下面这些固定种子会让环境内部的随机过程（包括默认的噪声 RNG）可复现。
# - 如果你更希望每次生成的数据集噪声模式都不一样，可按需修改/移除这些全局 seed。
np.random.seed(0)
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
######################


def map_dark_states(states, grid_size):
    return torch.sum(
        states
        * torch.tensor((grid_size, 1), device=states.device, requires_grad=False),
        dim=-1,
    )


def map_dark_states_inverse(index, grid_size):
    return torch.stack((index // grid_size, index % grid_size), dim=-1)


def sample_darkroom(config, shuffle=True):
    goals = [
        np.array([i, j])
        for i in range(config["grid_size"])
        for j in range(config["grid_size"])
    ]

    if shuffle:
        random.seed(config["env_split_seed"])
        random.shuffle(goals)

    n_train_envs = round(config["grid_size"] ** 2 * config["train_env_ratio"])

    train_goals = goals[:n_train_envs]
    test_goals = goals[n_train_envs:]

    return train_goals, test_goals


def sample_darkroom_permuted(config, shuffle=True):
    perms = list(range(120))

    if shuffle:
        random.seed(config["env_split_seed"])
        random.shuffle(perms)

    n_train_envs = round(120 * config["train_env_ratio"])

    train_perms = perms[:n_train_envs]
    test_perms = perms[n_train_envs:]

    return train_perms, test_perms


class Darkroom(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config, **kwargs):
        super(Darkroom, self).__init__()
        self.grid_size = config["grid_size"]
        if "goal" in kwargs:
            self.goal = kwargs["goal"]  # 训练/评估时由外部传入
        self.horizon = config["horizon"]
        self.dim_obs = 2
        self.dim_action = 1
        self.num_action = 5
        self.observation_space = spaces.Box(
            low=0, high=self.grid_size - 1, shape=(self.dim_obs,), dtype=np.int32
        )
        self.action_space = spaces.Discrete(self.num_action)

        # ===== 噪声配置（可通过 YAML 或 --override 设置）=====
        # 噪声概率（0/0.5/0.7 等），默认 0 表示无噪声
        self.noise_prob: float = float(config.get("noise_prob", 0.0))
        # 是否打印每一步是否使用了随机动作（默认 False，避免刷屏）
        self.noise_verbose: bool = bool(config.get("noise_verbose", False))
        # 为噪声单独设置随机种子（独立于全局 random），便于精确复现；None 则使用全局 random
        noise_seed = config.get("noise_seed", None)
        self._noise_rng = random.Random(noise_seed) if noise_seed is not None else None
        # =====================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[ObsType, dict[str, Any]]:
        self.current_step = 0

        center = self.grid_size // 2
        self.state = np.array([center, center])

        return self.state, {}

    def _rng_random(self) -> float:
        """在有独立噪声 RNG 时使用它，否则用全局 random。"""
        if self._noise_rng is not None:
            return self._noise_rng.random()
        return random.random()

    def _rng_randrange(self, n: int) -> int:
        """在有独立噪声 RNG 时使用它，否则用全局 random。"""
        if self._noise_rng is not None:
            return self._noise_rng.randrange(n)
        return random.randrange(n)

    def step(self, action):
        if self.current_step >= self.horizon:
            raise ValueError("Episode has already ended")

        s = np.array(self.state)

        # ---------- 带噪声的执行策略 ----------
        # 以 noise_prob 的概率覆盖为随机动作（均匀从 0..num_action-1 采样）
        if self.noise_prob > 0.0 and self._rng_random() < self.noise_prob:
            a = self._rng_randrange(self.num_action)
            if self.noise_verbose:
                print("\033[31m Using [random policy] now! \033[0m")
        else:
            a = action
            if self.noise_verbose:
                print("\033[34m Using [PPO policy] now! \033[0m")
        # ------------------------------------

        # Action handling
        if a == 0:
            s[0] += 1
        elif a == 1:
            s[0] -= 1
        elif a == 2:
            s[1] += 1
        elif a == 3:
            s[1] -= 1
        # a == 4 表示原地不动

        s = np.clip(s, 0, self.grid_size - 1)
        self.state = s

        reward = 1 if np.array_equal(s, self.goal) else 0
        self.current_step += 1
        done = self.current_step >= self.horizon
        info = {}
        return s.copy(), reward, done, done, info

    def get_optimal_action(self, state):
        if state[0] < self.goal[0]:
            a = 0
        elif state[0] > self.goal[0]:
            a = 1
        elif state[1] < self.goal[1]:
            a = 2
        elif state[1] > self.goal[1]:
            a = 3
        else:
            a = 4
        return a

    def transit(self, s, a):
        if a == 0:
            s[0] += 1
        elif a == 1:
            s[0] -= 1
        elif a == 2:
            s[1] += 1
        elif a == 3:
            s[1] -= 1
        elif a == 4:
            pass
        else:
            raise ValueError("Invalid action")

        s = np.clip(s, 0, self.grid_size - 1)

        if np.all(s == self.goal):
            r = 1
        else:
            r = 0

        return s, r

    def get_max_return(self):
        center = self.grid_size // 2
        return (
            self.horizon
            + 1
            - np.sum(np.absolute(self.goal - np.array([center, center])))
        ).clip(0, self.horizon)


class DarkroomPermuted(Darkroom):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)

        self.perm_idx = kwargs["perm_idx"]
        self.goal = np.array([self.grid_size - 1, self.grid_size - 1])

        assert self.perm_idx < 120  # 5! permutations in darkroom

        actions = np.arange(self.action_space.n)
        permutations = list(itertools.permutations(actions))
        self.perm = permutations[self.perm_idx]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[ObsType, dict[str, Any]]:
        self.current_step = 0
        self.state = np.array([0, 0])
        return self.state, {}

    def step(self, action):
        # 这里仍然会走父类的带噪声逻辑（在 super().step 内）
        return super().step(self.perm[action])

    def transit(self, s, a):
        return super().transit(s, self.perm[a])

    def get_optimal_action(self, state):
        action = super().get_optimal_action(state)
        return self.perm.index(action)

    def get_max_return(self):
        return self.horizon + 1 - np.sum(np.absolute(self.goal - np.array([0, 0])))

```

## gridworld/evaluate.py
```python
from datetime import datetime

import argparse
from glob import glob

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', '-c', required=True, help="Checkpoint directory")
    parser.add_argument('--beam-k', '-k', required=False, type=int, default=10, help="Beam_k for planning")
    parser.add_argument('--seed', '-s', required=False, type=int, default=0, help="Torch seed")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    ckpt_paths = sorted(glob(path.join(args.ckpt_dir, 'ckpt-*.pt')))
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt = torch.load(ckpt_path)
        print(f'Checkpoint loaded from {ckpt_path}')
        config = ckpt['config']
    else:
        raise ValueError('No checkpoint found')

    model_name = config['model']
    
    model = MODEL[model_name](config).to(device)
    
    model.load_state_dict(ckpt['model'])
    model.eval()

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    
    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in test_env_args])
    elif env_name == "darkkeytodoor":
        envs = SubprocVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in test_env_args])
    elif env_name == 'darkroompermuted':
        envs = SubprocVecEnv([make_env(config, perm_idx=arg) for arg in test_env_args])
    else:
        raise NotImplementedError(f'Environment {env_name} is not supported')
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    start_time = datetime.now()
    print(f'Generation started at {start_time}')
    
    with torch.no_grad():
        if config['dynamics']:
            test_rewards = model.evaluate_in_context(vec_env=envs,
                                                     eval_timesteps=config['horizon'] * 100,
                                                     beam_k=args.beam_k)['reward_episode']
            path = path.join(args.ckpt_dir, f'eval_result_k{args.beam_k}.npy')

        else:
            test_rewards = model.evaluate_in_context(vec_env=envs,
                                                     eval_timesteps=config['horizon'] * 50)['reward_episode']
            path = path.join(args.ckpt_dir, 'eval_result.npy')

    end_time = datetime.now()
    print()
    print(f'Generation ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    
    envs.close()
    
    with open(path, 'wb') as f:
        np.save(f, test_rewards)
```

## gridworld/model/__init__.py
```python
from .ad import AD
from .dpt import DPT
from .idt import IDT

MODEL = {
    "AD": AD,
    "DPT": DPT,
    "IDT": IDT,
}
```

## gridworld/model/ad.py
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, rearrange, repeat

from env import map_dark_states, map_dark_states_inverse

from .tiny_llama.model import Transformer


class AD(torch.nn.Module):
    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']
        self.dynamics = config['dynamics']

        self.transformer = Transformer(config)
        
        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, config['tf_n_embd'])
        self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], config['tf_n_embd'])
        self.pred_action = nn.Linear(config['tf_n_embd'], config['num_actions'])

        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])
        
        if self.dynamics:
            self.embed_query_action = nn.Embedding(config['num_actions'], config['tf_n_embd'])
            self.pred_reward = nn.Linear(config['tf_n_embd'], 2)
            self.pred_next_state = nn.Linear(config['tf_n_embd'], self.grid_size * self.grid_size)
        
    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_state)
        target_actions = x['target_actions'].to(self.device)  # (batch_size,)
        states = x['states'].to(self.device)  # (batch_size, num_transit, dim_state)
        actions = x['actions'].to(self.device)  # (batch_size, num_transit, num_actions)
        next_states = x['next_states'].to(self.device)  # (batch_size, num_transit, dim_state)
        rewards = x['rewards'].to(self.device)  # (batch_size, num_transit)
        rewards = rearrange(rewards, 'b n -> b n 1')

        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size).to(torch.long))
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')
        
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        context_embed = self.embed_context(context)
        context_embed, _ = pack([context_embed, query_states_embed], 'b * d')
        
        if self.dynamics:
            query_actions = x['target_actions'].to(self.device)  # (batch_size, )
            query_actions_embed = self.embed_query_action(query_actions)
            context_embed, _ = pack([context_embed, query_actions_embed], 'b * d')
        
        transformer_output = self.transformer(context_embed,
                                              max_seq_length=self.max_seq_length,
                                              dtype=self.mixed_precision)

        result = {}

        logits_actions = self.pred_action(transformer_output[:, self.n_transit-1])  # (batch_size, dim_action)
        
        loss_full_action = self.loss_fn(logits_actions, target_actions)
        acc_full_action = (logits_actions.argmax(dim=-1) == target_actions).float().mean()
        
        result['loss_action'] = loss_full_action
        result['acc_action'] = acc_full_action
        
        if self.dynamics:
            logit_rewards = self.pred_reward(transformer_output[:, -1])
            target_rewards = x['target_rewards'].to(self.device)  # (batch_size, )
            result['loss_reward'] = self.loss_fn(logit_rewards, target_rewards)
            result['acc_reward'] = (logit_rewards.argmax(dim=-1) == target_rewards).float().mean()
            
            logits_states = self.pred_next_state(transformer_output[:, -1])
            target_states = x['target_next_states'].to(self.device)  # (batch_size, )
            result['loss_next_state'] = self.loss_fn(logits_states, target_states)
            result['acc_next_state'] = (logits_states.argmax(dim=-1) == target_states).float().mean()
            
        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {}
        outputs['reward_episode'] = []

        reward_episode = np.zeros(vec_env.num_envs)
        
        # Get inital states embeddings
        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))
        transformer_input = query_states_embed
        
        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach().to(torch.float)

            # position = step % self.config['horizon']
            if self.dynamics and beam_k > 0 and step >= self.n_transit:
                actions = self.beam_search(x=transformer_input.clone().detach(),
                                           query_states=query_states_prev.clone().detach(),
                                           position=step % self.config['horizon'],
                                           beam_k=beam_k,
                                           sample=sample)
            else:
                output = self.transformer(transformer_input,
                                          max_seq_length=self.max_seq_length,
                                          dtype='fp32')
                
                logits = self.pred_action(output[:, -1])
                
                if sample:
                    log_probs = F.log_softmax(logits, dim=-1)
                    actions = torch.multinomial(log_probs.exp(), num_samples=1)
                    actions = rearrange(actions, 'e 1 -> e')
                else:
                    actions = logits.argmax(dim=-1)
                                
            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            actions = rearrange(actions, 'e -> e 1 1')
            actions = F.one_hot(actions, num_classes=self.config['num_actions'])

            reward_episode += rewards
            rewards = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards = rearrange(rewards, 'e -> e 1 1')

            query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.long)
            query_states = rearrange(query_states, 'e d -> e 1 d')
            
            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                
                states_next = torch.tensor(np.stack([info['terminal_observation'] for info in infos]), device=self.device, dtype=torch.float)
                
                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach().to(torch.float)
            
            query_states_embed = self.embed_query_state(map_dark_states(query_states, self.grid_size))

            context, _ = pack([query_states_prev, actions, rewards, states_next], 'e i *')
            context_embed = self.embed_context(context)
            
            if transformer_input.size(1) > 1:
                context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
                context_embed = context_embed[:, -(self.n_transit-1):]
                
            transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')
            
        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)

        return outputs
    
    def beam_search(self, x, query_states, position, beam_k=5, sample=True):
        batch_size = x.size(0)
        
        output = self.transformer(x,
                                  max_seq_length=self.max_seq_length,
                                  dtype="fp32")
                
        logit_actions = self.pred_action(output[:, -1])
        
        if sample:
            log_probs = F.log_softmax(logit_actions, dim=-1)            
            all_actions = torch.multinomial(log_probs.exp(), num_samples=self.config['num_actions'])
        else:
            all_actions = logit_actions.argsort(dim=-1, descending=True) # (batch_size, num_actions)
        
        # Query all actions
        all_actions_embed = self.embed_query_action(all_actions)
        all_actions_embed = rearrange(all_actions_embed, 'b a h -> b a 1 h')
        
        x = repeat(x, 'b n h -> b a n h', a=self.config['num_actions'])
        x, _ = pack([x, all_actions_embed], 'b a * h')
        
        output = self.transformer(rearrange(x, 'b a n h -> (b a) n h'),
                                  max_seq_length=self.max_seq_length,
                                  dtype="fp32")
        
        output = rearrange(output, '(b a) n h -> b a n h', a=self.config['num_actions'])
        
        # Get rewards
        logits_rewards = self.pred_reward(output[:, :, -1])
        rewards = logits_rewards.argmax(dim=-1)  # (batch_size, num_actions)
        
        # Get next states
        logit_next_states = self.pred_next_state(output[:, :, -1])
        next_states = logit_next_states.argmax(dim=-1)  # (batch_size, num_actions)
        
        # Initialize cumulative rewards
        cum_rewards = rewards.clone().detach()
        
        # Sort actions according to rewards
        rewards_sort = cum_rewards.sort(dim=-1, descending=True, stable=True)
        cum_rewards = rewards_sort.values[:, :beam_k]
        indices_k = rewards_sort.indices[:, :beam_k]
        
        # Update cumulative rewards        
        beam = torch.gather(all_actions, 1, indices_k)
        beam = rearrange(beam, 'b k -> b k 1')
        
        if self.config['env'] == 'darkroom':
            max_beam_steps = self.grid_size - 1
        elif self.config['env'] == 'darkkeytodoor' or self.config['env'] == 'darkroompermuted':
            max_beam_steps = (self.grid_size - 1) * 2
        else:
            raise ValueError('Invalid environment')
        
        position += 1
        beam_step = 1
        
        while position < self.config['horizon'] and beam_step < max_beam_steps:
            # Sort and cutoff variables
            x = torch.gather(x, 1, repeat(indices_k, 'b k -> b k n h', n=x.size(2), h=x.size(3)))
            actions_onehot = F.one_hot(beam[:, :, -1], num_classes=self.config['num_actions'])
            rewards = torch.gather(rewards, 1, indices_k)
            rewards = rearrange(rewards, 'b k -> b k 1')
            next_states = torch.gather(next_states, 1, indices_k)
            next_states_coord = map_dark_states_inverse(next_states, self.config['grid_size'])
            query_states = repeat(query_states, 'b k d -> b (k a) d', a=self.config['num_actions'])
            query_states = torch.gather(query_states, 1, repeat(indices_k, 'b k -> b k d', d=query_states.size(2)))
            
            # Make new context transition
            new_context, _ = pack([query_states, actions_onehot, rewards, next_states_coord], 'b k *')
            new_context_embed = self.embed_context(new_context.float())
            new_context_embed = repeat(new_context_embed, 'b k h -> b (k a) 1 h', a=self.config['num_actions'])

            # Make new query states            
            query_states_embed = self.embed_query_state(next_states)
            query_states_embed = repeat(query_states_embed, 'b k h -> b (k a) 1 h', a=self.config['num_actions'])
            
            query_states = next_states_coord  # (batch_size, beam_k, dim_state)

            # Make transformer input
            x = repeat(x, 'b k n h -> b (k a) n h', a=self.config['num_actions'])
            
            all_actions = torch.arange(self.config['num_actions'], device=self.device)
            all_actions_embed = self.embed_query_action(all_actions)
            all_actions_embed = repeat(all_actions_embed, 'a h -> b (k a) 1 h', b=batch_size, k=rewards.size(1))
            
            x, _ = pack([x[:, :, 1:self.config['n_transit']-1], new_context_embed, query_states_embed, all_actions_embed], 'b ka * h')
            
            assert x.size(2) == self.config['n_transit'] + 1
            
            # query states & actions
            output = self.transformer(rearrange(x, 'b ka n h -> (b ka) n h'),
                                      max_seq_length=self.max_seq_length,
                                      dtype="fp32")

            output = rearrange(output, '(b ka) n h -> b ka n h', b=batch_size)
            
            # Get rewards
            logit_rewards = self.pred_reward(output[:, :, -1])
            rewards = logit_rewards.argmax(dim=-1)  # (batch_size, beam_k * num_actions)
            
            # Get next states
            logit_next_states = self.pred_next_state(output[:, :, -1])
            next_states = logit_next_states.argmax(dim=-1)  # (batch_size, beam_k * num_actions)
            
            # Update cumulative rewards
            cum_rewards = repeat(cum_rewards, 'b k -> b (k a)', a=self.config['num_actions'])
            cum_rewards = cum_rewards + rewards
            rewards_sort = cum_rewards.sort(dim=-1, descending=True, stable=True)
            cum_rewards = rewards_sort.values[:, :beam_k]
            indices_k = rewards_sort.indices[:, :beam_k]
            
            new_actions = repeat(all_actions, 'a -> b (k a) 1', b=batch_size, k=beam.size(1))
            beam = repeat(beam, 'b k s -> b (k a) s', a=self.config['num_actions'])
            beam, _ = pack([beam, new_actions], 'b ka *')
            beam = torch.gather(beam, 1, repeat(indices_k, 'b k -> b k s', s=beam.size(2)))
            
            position += 1
            beam_step += 1
            
        return beam[:, 0, 0]
```

## gridworld/model/dpt.py
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tiny_llama.model import Transformer, LLaMAMLP
from einops import pack, rearrange, repeat
import numpy as np
from env import map_dark_states, map_dark_states_inverse


class DPT(nn.Module):
    def __init__(self, config):
        super(DPT, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']
        self.grid_size = config['grid_size']
        self.dynamics = config['dynamics']

        self.transformer = Transformer(config)
        
        self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, config['tf_n_embd'])
        self.embed_query_state = nn.Linear(config['dim_states'], config['tf_n_embd'])
        self.pred_actions = nn.Linear(config['tf_n_embd'], config['num_actions'])

        self.embed_query_action = nn.Embedding(config['num_actions'], config['tf_n_embd'])
        
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])
        
        if self.dynamics:
            self.pred_rewards = nn.Linear(config['tf_n_embd'], 2)
            self.pred_next_states = nn.Linear(config['tf_n_embd'], self.grid_size * self.grid_size)
        
    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_state)
        target_actions = x['target_actions'].to(self.device)  # (batch_size,)
        states = x['states'].to(self.device)  # (batch_size, num_transit, dim_state)
        actions = x['actions'].to(self.device)  # (batch_size, num_transit, num_actions)
        next_states = x['next_states'].to(self.device)  # (batch_size, num_transit, dim_state)
        rewards = x['rewards'].to(self.device)  # (batch_size, num_transit)
        rewards = rearrange(rewards, 'b n -> b n 1')

        query_states = F.pad(query_states, (0, actions.size(2) + rewards.size(2) + next_states.size(2)))
        query_states = rearrange(query_states, 'b d -> b 1 d')
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
                
        context, _ = pack([query_states, context], 'b * d')
        context_embed = self.embed_context(context)
    
        if self.dynamics:
            query_actions = x['query_actions'].to(self.device)  # (batch_size, num_actions)
            query_actions_embed = self.embed_query_action(query_actions)
            context_embed, _ = pack([context_embed, query_actions_embed], 'b * d')
        
        transformer_output = self.transformer(context_embed,
                                              max_seq_length=self.max_seq_length,
                                              dtype=self.mixed_precision)

        result = {}

        logits_actions = self.pred_actions(transformer_output[:, 1:self.n_transit])  # (batch_size, num_transit-1 , dim_action)
        target_actions_repeated = repeat(target_actions, 'b -> b n', n=logits_actions.size(1))
        
        result['loss_action'] = self.loss_fn(rearrange(logits_actions, 'b n a -> (b n) a'),
                                        rearrange(target_actions_repeated, 'b n -> (b n)'))
        result['acc_action'] = (logits_actions.argmax(dim=-1) == target_actions_repeated).float().mean()
        
        if self.dynamics:
            logits_rewards = self.pred_rewards(transformer_output[:, -1])
            target_rewards = x['target_rewards'].to(self.device)  # (batch_size, )
                        
            result['loss_reward'] = self.loss_fn(logits_rewards, target_rewards)
            result['acc_reward'] = (logits_rewards.argmax(dim=-1) == target_rewards).float().mean()
            
            logits_states = self.pred_next_states(transformer_output[:, -1])
            target_states = x['target_next_states'].to(self.device)  # (batch_size, )
            
            result['loss_next_state'] = self.loss_fn(logits_states, target_states)
            result['acc_next_state'] = (logits_states.argmax(dim=-1) == target_states).float().mean()
            
        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):

        outputs = {}
        outputs['reward_episode'] = []

        reward_episode = np.zeros(vec_env.num_envs)
        
        # Get inital states embeddings
        query_states = vec_env.reset()
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.float)
        query_states_padded = F.pad(query_states, (0, self.config['dim_states'] + self.config['num_actions'] + 1))
        query_states_padded = rearrange(query_states_padded, 'e d -> e 1 d')
        # query_states = rearrange(query_states, 'e d -> e 1 d')
        
        transformer_input = self.embed_context(query_states_padded)
        # transformer_input = self.embed_query_state(query_states_padded)
        
        
        for step in range(eval_timesteps):
            query_states_prev = query_states_padded[:,:,:self.config['dim_states']].clone().detach()
            # query_states_prev = query_states.clone().detach()

            # position = step % self.config['horizon']
            if self.dynamics and beam_k > 0 and step >= self.n_transit:
                actions = self.beam_search(x=transformer_input.clone().detach(),
                                           query_states=query_states.clone().detach(),
                                           position=step % self.config['horizon'],
                                           beam_k=beam_k,
                                           sample=sample)
            else:
                output = self.transformer(transformer_input,
                                          max_seq_length=self.max_seq_length,
                                          dtype='fp32')
                
                logits = self.pred_actions(output[:, -1])
                
                if sample:
                    log_probs = F.log_softmax(logits, dim=-1)
                    actions = torch.multinomial(log_probs.exp(), num_samples=1)
                    actions = rearrange(actions, 'e 1 -> e')
                else:
                    actions = logits.argmax(dim=-1)
                                
            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            actions = rearrange(actions, 'e -> e 1 1')
            actions = F.one_hot(actions, num_classes=self.config['num_actions'])

            reward_episode += rewards
            rewards = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards = rearrange(rewards, 'e -> e 1 1')

            query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.float)
            query_states = rearrange(query_states, 'e d -> e 1 d')
            
            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                
                states_next = torch.tensor(np.stack([info['terminal_observation'] for info in infos]), device=self.device, dtype=torch.float)
                
                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach()
            
            query_states_padded = F.pad(query_states, (0, self.config['dim_states'] + self.config['num_actions'] + 1))
            query_states_embed = self.embed_context(query_states_padded)
            # query_states_embed = self.embed_query_state(query_states)

            context, _ = pack([query_states_prev, actions, rewards, states_next], 'e n *')
            context_embed = self.embed_context(context)
            
            if transformer_input.size(1) > 1:
                context_embed, _ = pack([transformer_input[:, 1:], context_embed], 'e * h')
                context_embed = context_embed[:, -(self.n_transit-1):]
                
            transformer_input, _ = pack([query_states_embed, context_embed], 'e * h')
            
        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)

        return outputs
    
    def beam_search(self, x, query_states, position, beam_k=5, sample=True):
        batch_size = x.size(0)
        
        output = self.transformer(x,
                                  max_seq_length=self.max_seq_length,
                                  dtype="fp32")
                
        logit_actions = self.pred_actions(output[:, -1])
        
        if sample:
            log_probs = F.log_softmax(logit_actions, dim=-1)            
            all_actions = torch.multinomial(log_probs.exp(), num_samples=self.config['num_actions'])
        else:
            all_actions = logit_actions.argsort(dim=-1, descending=True) # (batch_size, num_actions)
        
        # Query all actions
        # all_actions_onehot = F.one_hot(all_actions, num_classes=self.config['num_actions']) # (batch_size, num_actions, num_actions)
        # all_actions_onehot = rearrange(all_actions_onehot, 'b a d -> b a 1 d')
        all_actions_embed = self.embed_query_action(all_actions)
        all_actions_embed = rearrange(all_actions_embed, 'b a h -> b a 1 h')
        
        # query_states = repeat(query_states, 'b 1 d -> b a 1 d', a=self.config['num_actions'])
        # query_states_actions, _ = pack([query_states, all_actions_onehot], 'b a i *')
        # query_states_actions = F.pad(query_states_actions, (0, 1+self.config['dim_states']))
        # query_states_actions_embed = self.embed_context(query_states_actions)
        
        x = repeat(x, 'b n h -> b a n h', a=self.config['num_actions'])
        x, _ = pack([x, all_actions_embed], 'b a * h')
        
        output = self.transformer(rearrange(x, 'b a n h -> (b a) n h'),
                                  max_seq_length=self.max_seq_length,
                                  dtype="fp32")
        
        output = rearrange(output, '(b a) n h -> b a n h', a=self.config['num_actions'])
        
        # Get rewards
        logit_rewards = self.pred_rewards(output[:, :, -1])
        rewards = logit_rewards.argmax(dim=-1)  # (batch_size, num_actions)
        
        # # Get next states
        logit_next_states = self.pred_next_states(output[:, :, -1])
        next_states = logit_next_states.argmax(dim=-1)  # (batch_size, num_actions)
        
        # Initialize cumulative rewards
        cum_rewards = rewards.clone().detach()
        
        # Sort actions according to rewards
        rewards_sort = cum_rewards.sort(dim=-1, descending=True, stable=True)
        cum_rewards = rewards_sort.values[:, :beam_k]
        indices_k = rewards_sort.indices[:, :beam_k]
        
        beam = torch.gather(all_actions, 1, indices_k)
        beam = rearrange(beam, 'b k -> b k 1')
                
        position += 1
        if self.config['env'] == 'darkroom':
            max_beam_steps = self.grid_size - 1
        elif self.config['env'] == 'darkkeytodoor' or self.config['env'] == 'darkroompermuted':
            max_beam_steps = (self.grid_size - 1) * 2
        else:
            raise ValueError('Invalid environment')
        
        beam_step = 1
                     
        while position < self.config['horizon'] and beam_step < max_beam_steps:
            # Sort and cutoff variables
            x = torch.gather(x, 1, repeat(indices_k, 'b k -> b k n h', n=x.size(2), h=x.size(3)))
            actions_onehot = F.one_hot(beam[:, :, -1], num_classes=self.config['num_actions'])
            rewards = torch.gather(rewards, 1, indices_k)
            rewards = rearrange(rewards, 'b k -> b k 1')
            next_states = torch.gather(next_states, 1, indices_k)
            next_states_coord = map_dark_states_inverse(next_states, self.config['grid_size'])
            query_states = repeat(query_states, 'b k d -> b (k a) d', a=self.config['num_actions'])
            query_states = torch.gather(query_states, 1, repeat(indices_k, 'b k -> b k d', d=query_states.size(2)))
            
            # Make new context transition
            new_context, _ = pack([query_states, actions_onehot, rewards, next_states_coord], 'b k *')
            new_context_embed = self.embed_context(new_context.float())
            new_context_embed = repeat(new_context_embed, 'b k h -> b (k a) 1 h', a=self.config['num_actions'])
            
            # Make new query states  
            next_states_padded = F.pad(next_states_coord, (0, self.config['dim_states'] + self.config['num_actions'] + 1))
            query_states_embed = self.embed_context(next_states_padded.to(torch.float))
            query_states_embed = repeat(query_states_embed, 'b k h -> b (k a) 1 h', a=self.config['num_actions'])
            
            query_states = next_states_coord  # (batch_size, beam_k, dim_state)
            
            # Make transformer input
            x = repeat(x, 'b k n h -> b (k a) n h', a=self.config['num_actions'])
            
            all_actions = torch.arange(self.config['num_actions'], device=self.device)
            all_actions_embed = self.embed_query_action(all_actions)
            all_actions_embed = repeat(all_actions_embed, 'a h -> b (k a) 1 h', b=batch_size, k=rewards.size(1))
            
            x, _ = pack([query_states_embed, x[:, :, 2:self.config['n_transit']], new_context_embed, all_actions_embed], 'b ka * h')
            
            assert x.size(2) == self.config['n_transit'] + 1
            
            # query (states, actions)
            output = self.transformer(rearrange(x, 'b ka n h -> (b ka) n h'),
                                      max_seq_length=self.max_seq_length,
                                      dtype="fp32")
                        
            output = rearrange(output, '(b ka) n h -> b ka n h', b=batch_size)

            # Get rewards
            logit_rewards = self.pred_rewards(output[:, :, -1])
            rewards = logit_rewards.argmax(dim=-1)  # (batch_size, beam_k * num_actions)
            
            # Get next states
            logit_next_states = self.pred_next_states(output[:, :, -1])
            next_states = logit_next_states.argmax(dim=-1)  # (batch_size, beam_k * num_actions)
            
            # Update cumulative rewards
            cum_rewards = repeat(cum_rewards, 'b k -> b (k a)', a=self.config['num_actions'])
            cum_rewards = cum_rewards + rewards
            rewards_sort = cum_rewards.sort(dim=-1, descending=True, stable=True)
            cum_rewards = rewards_sort.values[:, :beam_k]
            indices_k = rewards_sort.indices[:, :beam_k]

            new_actions = repeat(all_actions, 'a -> b (k a) 1', b=batch_size, k=beam.size(1))
            beam = repeat(beam, 'b k s -> b (k a) s', a=self.config['num_actions'])
            beam, _ = pack([beam, new_actions], 'b ka *')
            beam = torch.gather(beam, 1, repeat(indices_k, 'b k -> b k s', s=beam.size(2)))
            
            position += 1
            beam_step += 1
            
        return beam[:, 0, 0]
```

## gridworld/model/idt.py
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tiny_llama.model import Transformer
from einops import pack, rearrange, repeat
from env import map_dark_states


class IDT(nn.Module):
    def __init__(self, config):
        super(IDT, self).__init__()
        self.config = config
        self.device = config['device']
        self.low_per_high = config['low_per_high']
        self.n_transit = config['n_transit']
        
        assert self.config['horizon'] % self.low_per_high == 0
        
        self.reviewing_decisions = ReviewingDecisions(config)
        self.h_decision_transformer = HDecisionTransformer(config)
        self.decisions_to_go = DecisionsToGo(config)
        
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=config['label_smoothing'])

        
    def forward(self, x):
        states = map_dark_states(x['states'].to(self.device), self.config['grid_size'])
        actions = x['actions'].to(self.device)
        rewards = x['rewards'].to(self.device)
        next_states = map_dark_states(x['next_states'].to(self.device), self.config['grid_size'])
        return_to_go = x['return_to_go'].to(self.device)
        
        states = rearrange(states, 'b (high low) -> b high low', low=self.low_per_high)
        actions = rearrange(actions, 'b (high low) -> b high low', low=self.low_per_high)
        rewards = rearrange(rewards, 'b (high low) -> b high low', low=self.low_per_high)
        next_states = rearrange(next_states, 'b (high low) -> b high low', low=self.low_per_high)
        return_to_go = rearrange(return_to_go, 'b (high low) -> b high low', low=self.low_per_high)
        
        z_dists = self.reviewing_decisions(states, actions, rewards, next_states) # (batch_size, n_high, 2 * dim_z)
        
        h_states = states[:, :, 0]
        h_return_to_go = return_to_go[:, :, 0]
        
        z_pred = self.h_decision_transformer(h_return_to_go, h_states, z_dists)
        
        z = self.get_gaussian_sample(z_pred[:, :, :self.config['dim_z']], z_pred[:, :, self.config['dim_z']:])
                
        logit_states, logit_actions, logit_rewards = self.decisions_to_go(z, states, actions, rewards)
        
        output = {}
        
        output['loss_action'] = self.loss_fn(rearrange(logit_actions, 'b high low a -> (b high low) a'),
                                                  rearrange(actions, 'b high low -> (b high low)'))
        output['acc_action'] = (logit_actions.argmax(dim=-1) == actions).float().mean()
        
        if self.config['dynamics']:
            output['loss_reward'] = self.loss_fn(rearrange(logit_rewards, 'b high low r -> (b high low) r'),
                                                 rearrange(rewards, 'b high low -> (b high low)'))
            output['acc_reward'] = (logit_rewards.argmax(dim=-1) == rewards).float().mean()
            
            output['loss_next_state'] = self.loss_fn(rearrange(logit_states, 'b high low s -> (b high low) s'),
                                                     rearrange(next_states, 'b high low -> (b high low)'))
            output['acc_next_state']  = (logit_states.argmax(dim=-1) == next_states).float().mean()
                                                 
        return output
    
    def evaluate_in_context(self, vec_env, eval_timesteps, beam_k=0, sample=True):
        outputs = {}
        outputs['reward_episode'] = []

        reward_episode = np.zeros(vec_env.num_envs)
        
        # Get inital states embeddings
        states = vec_env.reset()
        states = torch.tensor(states, device=self.device, requires_grad=False, dtype=torch.long)
        states = rearrange(states, 'e d -> e 1 d')
        states = map_dark_states(states, self.config['grid_size'])
                
        return_to_go = np.array(vec_env.env_method('get_max_return', indices=list(range(vec_env.num_envs))))
        return_to_go = torch.tensor(return_to_go, device=self.device, requires_grad=False, dtype=torch.float)
        return_to_go = rearrange(return_to_go, 'e -> e 1')
        
        z_dists = None
        
        step = 0
        while step < eval_timesteps:
                        
            z_pred = self.h_decision_transformer(return_to_go, states, z_dists)[:, -1:] # (batch_size, 1, 2 * dim_z)
            z = self.get_gaussian_sample(z_pred[:, :, :self.config['dim_z']], z_pred[:, :, self.config['dim_z']:])[:, 0]
            
            reward_low, next_states, dones, history_states, history_actions, history_rewards, history_next_states = self.decisions_to_go.predict_actions(vec_env, z, states[:, -1:], 
                                                                                                                                                         beam_k=beam_k if step >= self.n_transit else 0, 
                                                                                                                                                         sample=sample)
            reward_episode += reward_low
            
            # update z
            history_states = rearrange(history_states, 'e low -> e 1 low')
            history_actions = rearrange(history_actions, 'e low -> e 1 low')
            history_rewards = rearrange(history_rewards, 'e low -> e 1 low')
            history_next_states = rearrange(history_next_states, 'e low -> e 1 low')
            z_reviewed = self.reviewing_decisions(history_states, history_actions, history_rewards, history_next_states)
            if z_dists is None:
                z_dists = z_reviewed
            else:
                z_dists, _ = pack([z_dists, z_reviewed], 'b * h')
            
            # update state
            states, _ = pack([states, next_states], 'e *')
            
            if states.size(1) > self.n_transit // self.low_per_high:
                states = states[:, 1:]
                return_to_go = return_to_go[:, 1:]
                z_dists = z_dists[:, 1:]
            
            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                
                next_return_to_go = np.array(vec_env.env_method('get_max_return', indices=list(range(vec_env.num_envs))))
                next_return_to_go = torch.tensor(next_return_to_go, device=self.device, requires_grad=False, dtype=torch.float)
                next_return_to_go = rearrange(next_return_to_go, 'e -> e 1')
            else:
                next_return_to_go = return_to_go[:, -1] - torch.tensor(reward_low, device=self.device, dtype=torch.float)
                next_return_to_go = rearrange(next_return_to_go, 'e -> e 1')
            
            # update rtg
            return_to_go, _ = pack([return_to_go, next_return_to_go], 'e *')
            
            step += self.low_per_high
        
        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        
        return outputs
    
    def get_gaussian_sample(self, mean, logvar):
        std = logvar.div(2).exp()
        std = repeat(std, 'b high d -> b high low d', low=self.low_per_high)
        mean = repeat(mean, 'b high d -> b high low d', low=self.low_per_high)
        eps = torch.randn_like(mean)
        return eps * std + mean
    

class ReviewingDecisions(nn.Module):
    def __init__(self, config):
        super(ReviewingDecisions, self).__init__()
        self.config = config
        self.low_per_high = config['low_per_high']
        
        self.transformer = Transformer(config)

        self.embed_state = nn.Embedding(config['grid_size'] * config['grid_size'], config['tf_n_embd'])
        self.embed_action = nn.Embedding(config['num_actions'], config['tf_n_embd'])
        self.embed_reward = nn.Embedding(2, config['tf_n_embd'])
        self.embed_next_state = nn.Embedding(config['grid_size'] * config['grid_size'], config['tf_n_embd'])

        self.predict_z = nn.Linear(config['tf_n_embd'], 2 * config['dim_z'])

    def forward(self, states, actions, rewards, next_states):
        batch_size = states.size(0)
        
        states_embed = self.embed_state(states)
        actions_embed = self.embed_action(actions)
        rewards_embed = self.embed_reward(rewards)
        next_states_embed = self.embed_next_state(next_states)
        
        transformer_input = states_embed + actions_embed + rewards_embed + next_states_embed
        
        transformer_output = self.transformer(rearrange(transformer_input, 'b high low h -> (b high) low h'),
                                                max_seq_length=self.low_per_high,
                                                dtype=self.config['mixed_precision'])
        
        transformer_output = rearrange(transformer_output, '(b high) low h -> b high low h', b=batch_size)
        z_dists = self.predict_z(transformer_output[:, :, -1])

        return z_dists


class HDecisionTransformer(nn.Module):
    def __init__(self, config):
        super(HDecisionTransformer, self).__init__()
        self.config = config
        self.transformer = Transformer(config)

        self.embed_return = nn.Linear(1, config['tf_n_embd'])
        self.embed_state = nn.Embedding(config['grid_size'] * config['grid_size'], config['tf_n_embd'])
        self.embed_z = nn.Linear(2 * config['dim_z'], config['tf_n_embd'])

        self.pred_z_dist = nn.Linear(config['tf_n_embd'], 2 * config['dim_z'])

    def forward(self, return_to_go, states, z_dists=None):
        if z_dists is None:
            z_dists = torch.zeros(states.size(0), 1, 2 * self.config['dim_z'], device=states.device)
        elif z_dists.size(1) < return_to_go.size(1):
            z_dists = F.pad(z_dists, (0, 0, 0, 1))
            
        return_to_go = rearrange(return_to_go, 'b n -> b n 1')
        return_embed = self.embed_return(return_to_go)
        states_embed = self.embed_state(states)
        z_embed = self.embed_z(z_dists)
        
        return_embed = rearrange(return_embed, 'b n h -> b n 1 h')
        states_embed = rearrange(states_embed, 'b n h -> b n 1 h')
        z_embed = rearrange(z_embed, 'b n h -> b n 1 h')
        
        transformer_input, _ = pack([return_embed, states_embed, z_embed], 'b n * h')

        transformer_output = self.transformer(rearrange(transformer_input, 'b n rsz h -> b (n rsz) h'),
                                              max_seq_length=self.config['n_transit'],
                                              dtype=self.config['mixed_precision'])
        
        transformer_output = rearrange(transformer_output, 'b (n rsz) h -> b n rsz h', rsz=3)

        z_pred = self.pred_z_dist(transformer_output[:, :, 1])

        return z_pred


class DecisionsToGo(nn.Module):
    def __init__(self, config):
        super(DecisionsToGo, self).__init__()
        self.config = config
        self.low_per_high = config['low_per_high']
        self.dim_z = config['dim_z']
        
        self.transformer = Transformer(config)

        self.embed_z = nn.Linear(config['dim_z'], config['tf_n_embd'])

        self.embed_state = nn.Embedding(config['grid_size'] * config['grid_size'], config['tf_n_embd'])
        self.embed_action = nn.Embedding(config['num_actions'], config['tf_n_embd'])
        self.embed_reward = nn.Embedding(2, config['tf_n_embd'])

        self.pred_action = nn.Linear(config['tf_n_embd'], config['num_actions'])
        
        if config['dynamics']:
            self.pred_next_state = nn.Linear(config['tf_n_embd'], config['grid_size'] * config['grid_size'])
            self.pred_reward = nn.Linear(config['tf_n_embd'], 2)

    def forward(self, z, states, actions, rewards):
        batch_size = states.size(0)
        
        z_embed = self.embed_z(z) # (batch_size, high, low_per_high, hidden)
        states_embed = self.embed_state(states) # (batch_size, high, low_per_high, hidden)
        actions_embed = self.embed_action(actions) # (batch_size, high, low_per_high, hidden)
        rewards_embed = self.embed_reward(rewards) # (batch_size, high, low_per_high, hidden)
        
        z_embed = rearrange(z_embed, 'b high low h -> b high low 1 h')
        states_embed = rearrange(states_embed, 'b high low h -> b high low 1 h')
        actions_embed = rearrange(actions_embed, 'b high low h -> b high low 1 h')
        rewards_embed = rearrange(rewards_embed, 'b high low h -> b high low 1 h')
        
        transformer_input, _ = pack([z_embed, states_embed, actions_embed, rewards_embed], 'b high low * h')
        
        transformer_output = self.transformer(rearrange(transformer_input, 'b high low zsar h -> (b high) (low zsar) h'),
                                              max_seq_length=4*self.config['low_per_high'],
                                              dtype=self.config['mixed_precision'])
        
        transformer_output = rearrange(transformer_output, '(b high) (low zsar) h -> b high low zsar h', b=batch_size, zsar=4)

        logit_actions = self.pred_action(transformer_output[:, :, :, 1])
        
        if self.config['dynamics']:
            logit_states = self.pred_next_state(transformer_output[:, :, :, 2])
            logit_rewards = self.pred_reward(transformer_output[:, :, :, 2])
        else:
            logit_states = None
            logit_rewards = None

        return logit_states, logit_actions, logit_rewards
    
    def predict_actions(self, vec_env, z, states, beam_k=0, sample=True):
        reward_low = np.zeros(vec_env.num_envs)

        z_embed = self.embed_z(z)
        states_embed = self.embed_state(states)
        
        assert z_embed.size(1) == self.low_per_high
        assert states_embed.size(1) == 1
        
        history_states = states
        
        transformer_input, _ = pack([z_embed[:, :1], states_embed], 'e * h')
        
        for step in range(self.low_per_high):
            
            if self.config['dynamics'] and beam_k > 0:
                actions = self.beam_search(x=transformer_input.clone().detach(),
                                           z_embed=z_embed.clone().detach(),
                                           beam_k=beam_k,
                                           sample=sample)
            else:
                transformer_output = self.transformer(transformer_input,
                                                      max_seq_length=4*self.config['low_per_high'],
                                                      dtype=self.config['mixed_precision'])
                
                logit_actions = self.pred_action(transformer_output[:, -1]) # (batch_size, num_actions)
                
                if sample:
                    log_probs = F.log_softmax(logit_actions, dim=-1)
                    actions = torch.multinomial(log_probs.exp(), num_samples=1)
                    actions = rearrange(actions, 'e 1 -> e')
                else:
                    actions = logit_actions.argmax(dim=-1) # (batch_size,)
            
            next_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            
            actions = rearrange(actions, 'e -> e 1')
            actions_embed = self.embed_action(actions)
            
            reward_low += rewards
            rewards = torch.tensor(rewards, device=self.config['device'], requires_grad=False, dtype=torch.long)
            rewards = rearrange(rewards, 'e -> e 1')
            rewards_embed = self.embed_reward(rewards)
            
            next_states = torch.tensor(next_states, device=self.config['device'], requires_grad=False, dtype=torch.long)
            next_states = rearrange(next_states, 'e d -> e 1 d')
            next_states = map_dark_states(next_states, self.config['grid_size'])
            next_states_embed = self.embed_state(next_states)
            
            if step < self.low_per_high - 1:
                history_states, _ = pack([history_states, next_states], 'e *')
                
            if step == 0:
                history_actions = actions
                history_rewards = rewards
                history_next_states = next_states
            else:
                history_actions, _ = pack([history_actions, actions], 'e *')
                history_rewards, _ = pack([history_rewards, rewards], 'e *')
                history_next_states, _ = pack([history_next_states, next_states], 'e *')
            
            if step < self.low_per_high - 1:
                transformer_input, _ = pack([transformer_input, actions_embed, rewards_embed, z_embed[:, step+1:step+2], next_states_embed], 'e * h')
            
        return reward_low, next_states, dones, history_states, history_actions, history_rewards, history_next_states
    
    def beam_search(self, x, z_embed, beam_k, sample=True):
        output = self.transformer(x,
                                  max_seq_length=4*self.config['low_per_high'],
                                  dtype=self.config['mixed_precision']) 
        
        logit_actions = self.pred_action(output[:, -1])
        
        if sample:
            log_probs = F.log_softmax(logit_actions, dim=-1)            
            all_actions = torch.multinomial(log_probs.exp(), num_samples=self.config['num_actions'])
        else:
            all_actions = logit_actions.argsort(dim=-1, descending=True) # (batch_size, num_actions)
            
        # Query all actions
        all_actions_embed = self.embed_action(all_actions)
        all_actions_embed = rearrange(all_actions_embed, 'b a h -> b a 1 h')
        
        x = repeat(x, 'b n h -> b a n h', a=self.config['num_actions'])
        x, _ = pack([x, all_actions_embed], 'b a * h')
        
        output = self.transformer(rearrange(x, 'b a n h -> (b a) n h'),
                                  max_seq_length=4*self.config['low_per_high'],
                                  dtype="fp32")
        
        output = rearrange(output, '(b a) n h -> b a n h', a=self.config['num_actions'])
        
        # Get rewards
        logit_rewards = self.pred_reward(output[:, :, -1])
        rewards = logit_rewards.argmax(dim=-1)  # (batch_size, num_actions)
        
        # Get next states
        logit_next_states = self.pred_next_state(output[:, :, -1])
        next_states = logit_next_states.argmax(dim=-1)  # (batch_size, num_actions)

        # Initialize cumulative rewards
        cum_rewards = rewards.clone().detach()
        
        # Sort actions according to rewards
        rewards_sort = cum_rewards.sort(dim=-1, descending=True, stable=True)
        cum_rewards = rewards_sort.values[:, :beam_k]
        indices_k = rewards_sort.indices[:, :beam_k]
        
        beam = torch.gather(all_actions, 1, indices_k)
        beam = rearrange(beam, 'b k -> b k 1')
        
        beam_step = 1
                
        while beam_step < self.low_per_high:
            # Sort and cutoff variables
            x = torch.gather(x, 1, repeat(indices_k, 'b k -> b k n h', n=x.size(2), h=x.size(3)))
            
            rewards = torch.gather(rewards, 1, indices_k)
            rewards_embed = self.embed_reward(rewards)  # (batch_size, beam_k, hidden)
            rewards_embed = rearrange(rewards_embed, 'b k h -> b k 1 h')
            
            next_states = torch.gather(next_states, 1, indices_k)
            next_states_embed = self.embed_state(next_states)  # (batch_size, beam_k, hidden)
            next_states_embed = rearrange(next_states_embed, 'b k h -> b k 1 h')
            
            z_embed_sliced = repeat(z_embed[:, beam_step:beam_step+1], 'b i h -> b k i h', k=x.size(1))
            x, _ = pack([x, rewards_embed, z_embed_sliced, next_states_embed], 'b k * h')
            
            # Query all actions
            all_actions = torch.arange(self.config['num_actions'], device=x.device)
            actions_embed = self.embed_action(all_actions)
            actions_embed = repeat(actions_embed, 'a h -> b k a 1 h', b=x.size(0), k=x.size(1))
            x = repeat(x, 'b k n h -> b k a n h', a=self.config['num_actions'])
            x, _ = pack([x, actions_embed], 'b k a * h')
            x = rearrange(x, 'b k a n h -> b (k a) n h')
            
            output = self.transformer(rearrange(x, 'b ka n h -> (b ka) n h'),
                                      max_seq_length=4*self.config['low_per_high'],
                                      dtype="fp32")
            
            output = rearrange(output, '(b ka) n h -> b ka n h', b=x.size(0))
            
            # Get rewards
            logit_rewards = self.pred_reward(output[:, :, -1])
            rewards = logit_rewards.argmax(dim=-1)  # (batch_size, beam_k * num_actions)

            # Get next states
            logit_next_states = self.pred_next_state(output[:, :, -1])
            next_states = logit_next_states.argmax(dim=-1)  # (batch_size, beam_k * num_actions)
            
            # Sort actions according to rewards
            cum_rewards = repeat(cum_rewards, 'b k -> b (k a)', a=self.config['num_actions'])
            cum_rewards = cum_rewards + rewards
            rewards_sort = cum_rewards.sort(dim=-1, descending=True, stable=True)
            cum_rewards = rewards_sort.values[:, :beam_k]
            indices_k = rewards_sort.indices[:, :beam_k]
            
            new_actions = repeat(all_actions, 'a -> b (k a) 1', b=x.size(0), k=beam.size(1))
            beam = repeat(beam, 'b k s -> b (k a) s', a=self.config['num_actions'])
            beam, _ = pack([beam, new_actions], 'b ka *')
            beam = torch.gather(beam, 1, repeat(indices_k, 'b k -> b k s', s=beam.size(2)))
            
            beam_step += 1
            
        return beam[:, 0, 0]
```

## gridworld/model/tiny_llama/__init__.py
```python

```

## gridworld/model/tiny_llama/config.py
```python
from dataclasses import dataclass
from typing import Literal, Optional, Type

import torch


@dataclass
class Config:
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    rotary_percentage: float
    parallel_residual: bool
    bias: bool
    dropout: float
    attention_dropout: float

    shared_attention_norm: bool
    _norm_class: Literal["LayerNorm", "RMSNorm", "FusedRMSNorm"]
    _mlp_class: Literal["GptNeoxMLP", "LLaMAMLP"]
    n_query_groups: Optional[int] = None
    norm_eps: float = 1e-5
    intermediate_size: Optional[int] = None
    condense_ratio: int = 1
    flash_attn: bool = True

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        if self.n_query_groups is not None:
            assert self.n_head % self.n_query_groups == 0
        else:
            self.n_query_groups = self.n_head
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.n_embd

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def norm_class(self) -> Type:
        if self._norm_class == "RMSNorm":
            from .rmsnorm import RMSNorm

            return RMSNorm
        elif self._norm_class == "FusedRMSNorm":
            from .rmsnorm import FusedRMSNorm

            return FusedRMSNorm
        return getattr(torch.nn, self._norm_class)

```

## gridworld/model/tiny_llama/fused_rotary_embedding.py
```python
import rotary_emb
import torch
from einops import rearrange


class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin, interleaved=False, inplace=False):
        batch, seqlen, nheads, headdim = x.shape
        rotary_seqlen, rotary_dim = cos.shape
        rotary_dim *= 2
        assert rotary_dim <= headdim
        assert seqlen <= rotary_seqlen
        assert sin.shape == (rotary_seqlen, rotary_dim // 2)
        x_ro = x[..., :rotary_dim]
        x1, x2 = (
            x_ro.chunk(2, dim=-1)
            if not interleaved
            else (x_ro[..., ::2], x_ro[..., 1::2])
        )
        out = torch.empty_like(x) if not inplace else x
        out_ro = out[..., :rotary_dim]
        if inplace:
            o1, o2 = x1, x2
        else:
            o1, o2 = (
                out_ro.chunk(2, dim=-1)
                if not interleaved
                else (out_ro[..., ::2], out_ro[..., 1::2])
            )  
            
        rotary_emb.apply_rotary(
            x1,
            x2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            o1,
            o2,
            False,
        )
        if not inplace and rotary_dim < headdim:
            out[..., rotary_dim:].copy_(x[..., rotary_dim:])
        ctx.save_for_backward(cos, sin)
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do):
        cos, sin = ctx.saved_tensors
        _, seqlen, _, headdim = do.shape
        rotary_dim = cos.shape[-1]
        rotary_dim *= 2
        inplace = ctx.inplace
        do_ro = do[..., :rotary_dim]
        do1, do2 = (
            do_ro.chunk(2, dim=-1)
            if not ctx.interleaved
            else (do_ro[..., ::2], do_ro[..., 1::2])
        )
        dx = torch.empty_like(do) if not inplace else do
        if inplace:
            dx1, dx2 = do1, do2
        else:
            dx_ro = dx[..., :rotary_dim]
            dx1, dx2 = (
                dx_ro.chunk(2, dim=-1)
                if not ctx.interleaved
                else (dx_ro[..., ::2], dx_ro[..., 1::2])
            )
        rotary_emb.apply_rotary(
            do1,
            do2,
            rearrange(cos[:seqlen], "s d -> s 1 d"),
            rearrange(sin[:seqlen], "s d -> s 1 d"),
            dx1,
            dx2,
            True,
        )
        if not inplace and rotary_dim < headdim:
            dx[..., rotary_dim:].copy_(do[..., rotary_dim:])
        return dx, None, None, None, None


apply_rotary_emb_func = ApplyRotaryEmb.apply

```

## gridworld/model/tiny_llama/lora.py
```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """LoRA injected linear module."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r if r > 0 else 1.0
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        if r > 0:
            self.lora_A = nn.Parameter(torch.zeros(r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, r))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
            self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()
            # freeze base weights
            self.weight.requires_grad = False
            if bias:
                self.bias.requires_grad = False
        else:
            self.lora_A = None
            self.lora_B = None
            self.lora_dropout = nn.Identity()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.weight.size(1))
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            lora_out = F.linear(self.lora_dropout(x), self.lora_A)
            lora_out = F.linear(lora_out, self.lora_B) * self.scaling
            result = result + lora_out
        return result

```

## gridworld/model/tiny_llama/model.py
```python
import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
# from flash_attn import flash_attn_func
from lightning_utilities.core.imports import RequirementCache
from xformers.ops import SwiGLU
import peft  # noqa: F401

from .lora import LoRALinear

from .config import Config
from .fused_rotary_embedding import apply_rotary_emb_func

from torch.nn.attention import sdpa_kernel, SDPBackend

RoPECache = Tuple[torch.Tensor, torch.Tensor]
KVCache = Tuple[torch.Tensor, torch.Tensor]
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")


class Transformer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = Config(
            block_size=3*config['n_transit'],
            n_layer=config['tf_n_layer'],
            n_head=config['tf_n_head'],
            n_embd=config['tf_n_embd'],
            bias=True,
            rotary_percentage=1.0,
            parallel_residual=False,
            shared_attention_norm=False,
            _norm_class="FusedRMSNorm",
            _mlp_class="LLaMAMLP",
            dropout=config['tf_dropout'],
            attention_dropout=config['tf_attn_dropout'],
            intermediate_size=config['tf_n_inner'],
            flash_attn=config['flash_attn'],
        )
        # LoRA parameters
        self.config.use_lora = config.get('use_lora', False)
        self.config.lora_r = config.get('lora_r', 0)
        self.config.lora_alpha = config.get('lora_alpha', 1)
        self.config.lora_dropout = config.get('lora_dropout', 0.0)
        self.device = config['device']
        self.blocks = nn.ModuleList(Block(self.config) for _ in range(config['tf_n_layer']))
        self.rope_cache_fp16 = self.build_rope_cache(device=self.device, dtype=torch.float16)
        self.rope_cache_bf16 = self.build_rope_cache(device=self.device, dtype=torch.bfloat16)
        self.rope_cache_fp32 = self.build_rope_cache(device=self.device, dtype=torch.float32)

    def forward(self,
                x: torch.Tensor, 
                max_seq_length: int,
                mask: Optional[torch.Tensor] = None, 
                dtype="bf16") -> Tuple[torch.Tensor, Optional[KVCache]]:

        if dtype == "bf16":
            cos, sin = self.rope_cache_bf16
        elif dtype == "fp16":
            cos, sin = self.rope_cache_fp16
        elif dtype == "fp32":
            cos, sin = self.rope_cache_fp32
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
                
        for block in self.blocks:
            x, *_ = block(x,
                          (cos[:x.size(1)], sin[:x.size(1)]),
                          max_seq_length,
                          mask)
        return x

    def build_rope_cache(self, device, dtype) -> RoPECache:
        return build_rope_cache(
            seq_len=self.config.block_size,
            n_elem=int(self.config.rotary_percentage * self.config.head_size),
            dtype=dtype,
            device=device,
            condense_ratio=self.config.condense_ratio,
            )



class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.norm_1 = config.norm_class(
            config.n_embd, eps=config.norm_eps, dropout=config.dropout
        )
        self.attn = CausalSelfAttention(config)
        if not config.shared_attention_norm:
            self.norm_2 = config.norm_class(
                config.n_embd, eps=config.norm_eps, dropout=config.dropout
            )
        self.mlp = getattr(sys.modules[__name__], config._mlp_class)(config)
        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        n_1 = self.norm_1(x)
        h, new_kv_cache = self.attn(
            n_1, rope, max_seq_length, mask, input_pos, kv_cache
        )
        if self.config.parallel_residual:
            n_2 = n_1 if self.config.shared_attention_norm else self.norm_2(x)
            x = x + h + self.mlp(n_2)
        else:
            if self.config.shared_attention_norm:
                raise NotImplementedError(
                    "No checkpoint amongst the ones we support uses this configuration"
                    " (non-parallel residual and shared attention norm)."
                )

            x = x + h
            x = x + self.mlp(self.norm_2(x))
        return x, new_kv_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        shape = (config.n_head + 2 * config.n_query_groups) * config.head_size
        if config.use_lora and config.lora_r > 0:
            self.attn = LoRALinear(
                config.n_embd,
                shape,
                bias=config.bias,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
            self.proj = LoRALinear(
                config.n_embd,
                config.n_embd,
                bias=config.bias,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )
        else:
            self.attn = nn.Linear(config.n_embd, shape, bias=config.bias)
            self.proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.config = config

    def forward(
        self,
        x: torch.Tensor,
        rope: RoPECache,
        max_seq_length: int,
        mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[KVCache]]:
        (
            B,
            T,
            C,
        ) = x.size()

        qkv = self.attn(x)

        q_per_kv = self.config.n_head // self.config.n_query_groups
        total_qkv = q_per_kv + 2
        qkv = qkv.view(
            B, T, self.config.n_query_groups, total_qkv, self.config.head_size
        )

        q, k, v = qkv.split((q_per_kv, 1, 1), dim=-2)

        q = q.reshape(B, T, -1, self.config.head_size)  # (B, T, nh_q, hs)
        k = k.reshape(B, T, -1, self.config.head_size)
        v = v.reshape(B, T, -1, self.config.head_size)

        cos, sin = rope
        q = apply_rotary_emb_func(q, cos, sin, False, True)
        k = apply_rotary_emb_func(k, cos, sin, False, True)

        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            cache_k, cache_v = cache_k.to(dtype=k.dtype), cache_v.to(dtype=v.dtype)
            if input_pos[-1] >= max_seq_length:
                input_pos = torch.tensor(max_seq_length - 1, device=input_pos.device)
                cache_k = torch.roll(cache_k, -1, dims=2)
                cache_v = torch.roll(cache_v, -1, dims=2)
            k = cache_k.index_copy_(2, input_pos, k)
            v = cache_v.index_copy_(2, input_pos, v)
            kv_cache = k, v

        y = self.scaled_dot_product_attention(q, k, v, mask=mask)

        y = y.reshape(B, T, C)

        y = self.proj(y)

        return y, kv_cache

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        scale = 1.0 / math.sqrt(self.config.head_size)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.size() != k.size():
            k = k.repeat_interleave(q.shape[1]//k.shape[1], dim=1)
            v = v.repeat_interleave(q.shape[1]//v.shape[1], dim=1)
        
        if (
            FlashAttention2Available
            and mask is None
            and q.device.type == "cuda"
            and q.dtype in (torch.float16, torch.bfloat16)
            and self.config.flash_attn
        ):
            attn_type = SDPBackend.FLASH_ATTENTION
        else:
            attn_type = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]
            
        with sdpa_kernel(attn_type):
                y = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, 
                    attn_mask=mask, 
                    dropout_p=self.config.attention_dropout if self.training else 0.0, 
                    scale=scale,
                    is_causal=mask is None
                )
                
        
        return y.transpose(1, 2)


class GptNeoxMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc = nn.Linear(config.n_embd, config.intermediate_size, bias=config.bias)
        self.proj = nn.Linear(config.intermediate_size, config.n_embd, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = torch.nn.functional.gelu(x)
        return self.proj(x)


class LLaMAMLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.swiglu = SwiGLU(
            config.n_embd, config.intermediate_size, bias=False, _pack_weights=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.swiglu(x)


def build_rope_cache(
    seq_len: int,
    n_elem: int,
    dtype: torch.dtype,
    device: torch.device,
    base: int = 10000,
    condense_ratio: int = 1,
) -> RoPECache:
    
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device) / n_elem))

    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    idx_theta = torch.outer(seq_idx, theta)

    cos, sin = torch.cos(idx_theta), torch.sin(idx_theta)

    if dtype == torch.bfloat16:
        return cos.bfloat16(), sin.bfloat16()
    if dtype in (torch.float16, torch.bfloat16, torch.int8):
        return cos.half(), sin.half()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    head_size = x.size(-1)
    x1 = x[..., : head_size // 2]  # (B, nh, T, hs/2)
    x2 = x[..., head_size // 2 :]  # (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # (B, nh, T, hs)
    roped = (x * cos) + (rotated * sin)
    return roped.type_as(x)

```

## gridworld/model/tiny_llama/rmsnorm.py
```python
import dropout_layer_norm
import torch
from torch.nn import init


def maybe_align(x, alignment_in_bytes=16):
    return x if x.data_ptr() % alignment_in_bytes == 0 else x.clone()


def _dropout_add_layer_norm_forward(
    x0,
    residual,
    gamma,
    beta,
    rowscale,
    colscale,
    dropout_p,
    epsilon,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat,
        residualmat,
        gamma,
        beta,
        rowscale,
        colscale,
        None,
        None,
        dropout_p,
        epsilon,
        1.0,
        0,
        None,
        residual_in_fp32,
        is_rms_norm,
    )
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def _dropout_add_layer_norm_backward(
    dz,
    dx,
    x,
    x0,
    dmask,
    mu,
    rsigma,
    gamma,
    rowscale,
    colscale,
    dropout_p,
    has_residual,
    is_rms_norm=False,
):
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(xmat.shape)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    rowscale = rowscale.view(-1) if rowscale is not None else None
    if colscale is not None:
        assert x0 is not None, "x0 is required to compute the gradient of colscale"
    (
        dx0mat,
        dresidualmat,
        dgamma,
        dbeta,
        _,
        _,
        *rest,
    ) = dropout_layer_norm.dropout_add_ln_bwd(
        dzmat,
        dxmat,
        xmat,
        x0mat,
        dmask,
        mu,
        rsigma,
        gamma,
        rowscale,
        colscale,
        None,
        None,
        dropout_p,
        1.0,
        0,
        has_residual,
        is_rms_norm,
    )
    if colscale is None:
        return dx0mat, dresidualmat, dgamma, dbeta
    else:
        dcolscale = rest[0]
        return dx0mat, dresidualmat, dgamma, dbeta, dcolscale


def _dropout_add_layer_norm_subset_forward(
    x0,
    residual,
    gamma,
    beta,
    colscale,
    x0_subset,
    out_subset,
    dropout_p,
    epsilon,
    rowscale_const,
    out_numrows,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    hidden_size = gamma.numel()
    x0mat = x0.view((-1, hidden_size))
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    x0_subset = x0_subset.view(-1) if x0_subset is not None else None
    out_subset = out_subset.view(-1) if out_subset is not None else None
    zmat, xmat, dmask, mu, rsigma = dropout_layer_norm.dropout_add_ln_fwd(
        x0mat,
        residualmat,
        gamma,
        beta,
        None,
        colscale,
        x0_subset,
        out_subset,
        dropout_p,
        epsilon,
        rowscale_const,
        out_numrows,
        None,
        residual_in_fp32,
        is_rms_norm,
    )
    return zmat, xmat if xmat is not None else x0mat, dmask, mu, rsigma


def _dropout_add_layer_norm_subset_backward(
    dz,
    dx,
    x,
    x0,
    dmask,
    mu,
    rsigma,
    gamma,
    colscale,
    x0_subset,
    out_subset,
    dropout_p,
    rowscale_const,
    x0_numrows,
    has_residual,
    is_rms_norm=False,
):
    hidden_size = gamma.numel()
    xmat = x.view((-1, hidden_size))
    dzmat = dz.view(-1, hidden_size)
    dxmat = dx.view(xmat.shape) if dx is not None else None
    x0mat = x0.view((-1, hidden_size)) if x0 is not None else None
    x0_subset = x0_subset.view(-1) if x0_subset is not None else None
    out_subset = out_subset.view(-1) if out_subset is not None else None
    if colscale is not None:
        assert x0 is not None, "x0 is required to compute the gradient of colscale"
    (
        dx0mat,
        dresidualmat,
        dgamma,
        dbeta,
        _,
        _,
        *rest,
    ) = dropout_layer_norm.dropout_add_ln_bwd(
        dzmat,
        dxmat,
        xmat,
        x0mat,
        dmask,
        mu,
        rsigma,
        gamma,
        None,
        colscale,
        x0_subset,
        out_subset,
        dropout_p,
        rowscale_const,
        x0_numrows,
        has_residual,
        is_rms_norm,
    )
    if colscale is None:
        return dx0mat, dresidualmat, dgamma, dbeta
    else:
        dcolscale = rest[0]
        return dx0mat, dresidualmat, dgamma, dbeta, dcolscale


def _dropout_add_layer_norm_parallel_residual_forward(
    x0,
    x1,
    residual,
    gamma0,
    beta0,
    gamma1,
    beta1,
    dropout_p,
    epsilon,
    residual_in_fp32=False,
    is_rms_norm=False,
):
    hidden_size = gamma0.numel()
    x0mat = x0.view((-1, hidden_size))
    x1mat = x1.view((-1, hidden_size)) if x1 is not None else None
    residualmat = residual.view((-1, hidden_size)) if residual is not None else None
    (
        z0mat,
        z1mat,
        xmat,
        dmask0,
        dmask1,
        mu,
        rsigma,
    ) = dropout_layer_norm.dropout_add_ln_parallel_residual_fwd(
        x0mat,
        x1mat,
        residualmat,
        gamma0,
        beta0,
        gamma1,
        beta1,
        dropout_p,
        epsilon,
        None,
        residual_in_fp32,
        is_rms_norm,
    )
    return z0mat, z1mat, xmat if xmat is not None else x0mat, dmask0, dmask1, mu, rsigma


def _dropout_add_layer_norm_parallel_residual_backward(
    dz0,
    dz1,
    dx,
    x,
    dmask0,
    dmask1,
    mu,
    rsigma,
    gamma0,
    gamma1,
    dropout_p,
    has_x1,
    has_residual,
    is_rms_norm=False,
):
    hidden_size = gamma0.numel()
    xmat = x.view((-1, hidden_size))
    dz0mat = dz0.view(xmat.shape)
    dz1mat = dz1.view(xmat.shape) if dz1 is not None else None
    dxmat = dx.view(xmat.shape) if dx is not None else None
    (
        dx0mat,
        dx1mat,
        dresidualmat,
        dgamma0,
        dbeta0,
        dgamma1,
        dbeta1,
        *rest,
    ) = dropout_layer_norm.dropout_add_ln_parallel_residual_bwd(
        dz0mat,
        dz1mat,
        dxmat,
        xmat,
        dmask0,
        dmask1,
        mu,
        rsigma,
        gamma0,
        gamma1,
        dropout_p,
        has_x1,
        has_residual,
        is_rms_norm,
    )
    return dx0mat, dx1mat, dresidualmat, dgamma0, dbeta0, dgamma1, dbeta1


class DropoutAddLayerNormFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0,
        residual,
        gamma,
        beta,
        rowscale,
        colscale,
        dropout_p,
        epsilon,
        residual_in_fp32=False,
        prenorm=False,
        is_rms_norm=False,
        return_dmask=False,
    ):
        x0 = maybe_align(x0.contiguous(), 16)
        residual = (
            maybe_align(residual.contiguous(), 16) if residual is not None else None
        )
        gamma = maybe_align(gamma.contiguous(), 16)
        beta = maybe_align(beta.contiguous(), 16) if beta is not None else None
        rowscale = (
            maybe_align(rowscale.contiguous(), 16) if rowscale is not None else None
        )
        colscale = (
            maybe_align(colscale.contiguous(), 16) if colscale is not None else None
        )
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_forward(
            x0,
            residual,
            gamma,
            beta,
            rowscale,
            colscale,
            dropout_p,
            epsilon,
            residual_in_fp32,
            is_rms_norm,
        )
        x0_saved = x0 if colscale is not None else None
        ctx.save_for_backward(
            xmat.view(x0.shape), x0_saved, dmask, gamma, mu, rsigma, rowscale, colscale
        )
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta is not None
        if not return_dmask:
            return (
                zmat.view(x0.shape)
                if not prenorm
                else (zmat.view(x0.shape), xmat.view(x0.shape))
            )
        else:
            dmask = (
                dmask.view(x0.shape)
                if dropout_p > 0.0
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            ctx.mark_non_differentiable(dmask)
            return (
                (zmat.view(x0.shape), dmask)
                if not prenorm
                else (zmat.view(x0.shape), xmat.view(x0.shape), dmask)
            )

    @staticmethod
    def backward(ctx, dz, *args):
        dz = maybe_align(dz.contiguous(), 16)  # this happens!
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        x, x0, dmask, gamma, mu, rsigma, rowscale, colscale = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        dx0mat, dresidualmat, dgamma, dbeta, *rest = _dropout_add_layer_norm_backward(
            dz,
            dx,
            x,
            x0,
            dmask,
            mu,
            rsigma,
            gamma,
            rowscale,
            colscale,
            dropout_p,
            has_residual,
            ctx.is_rms_norm,
        )
        dx0 = dx0mat.view(x.shape)
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return (
            dx0,
            dresidual,
            dgamma,
            dbeta if ctx.has_beta else None,
            None,
            dcolscale,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DropoutAddLayerNormSubsetFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0,
        residual,
        gamma,
        beta,
        colscale,
        x0_subset,
        out_subset,
        dropout_p,
        epsilon,
        rowscale_const,
        out_numrows,
        residual_in_fp32=False,
        prenorm=False,
        is_rms_norm=False,
        return_dmask=False,
    ):
        x0 = maybe_align(x0.contiguous(), 16)
        residual = (
            maybe_align(residual.contiguous(), 16) if residual is not None else None
        )
        gamma = maybe_align(gamma.contiguous(), 16)
        beta = maybe_align(beta.contiguous(), 16) if beta is not None else None
        colscale = (
            maybe_align(colscale.contiguous(), 16) if colscale is not None else None
        )
        zmat, xmat, dmask, mu, rsigma = _dropout_add_layer_norm_subset_forward(
            x0,
            residual,
            gamma,
            beta,
            colscale,
            x0_subset,
            out_subset,
            dropout_p,
            epsilon,
            rowscale_const,
            out_numrows,
            residual_in_fp32,
            is_rms_norm,
        )
        x0_saved = x0 if colscale is not None else None
        x_shape = (-1, *x0.shape[1:])
        ctx.save_for_backward(
            xmat.view(x_shape),
            x0_saved,
            dmask,
            gamma,
            mu,
            rsigma,
            colscale,
            x0_subset,
            out_subset,
        )
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.rowscale_const = rowscale_const
        ctx.x0_numrows = x0.shape[:-1].numel()
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta is not None
        z_shape = (-1, *x0.shape[1:])
        if not return_dmask:
            return (
                zmat.view(z_shape)
                if not prenorm
                else (zmat.view(z_shape), xmat.view(x0.shape))
            )
        else:
            z = zmat.view(z_shape)
            dmask = (
                dmask.view(x0.shape)
                if dropout_p > 0.0
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            ctx.mark_non_differentiable(dmask)
            return (z, dmask) if not prenorm else (z, xmat.view(x_shape), dmask)

    @staticmethod
    def backward(ctx, dz, *args):
        dz = maybe_align(dz.contiguous(), 16)
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        (
            x,
            x0,
            dmask,
            gamma,
            mu,
            rsigma,
            colscale,
            x0_subset,
            out_subset,
        ) = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_residual = ctx.has_residual
        (
            dx0mat,
            dresidualmat,
            dgamma,
            dbeta,
            *rest,
        ) = _dropout_add_layer_norm_subset_backward(
            dz,
            dx,
            x,
            x0,
            dmask,
            mu,
            rsigma,
            gamma,
            colscale,
            x0_subset,
            out_subset,
            dropout_p,
            ctx.rowscale_const,
            ctx.x0_numrows,
            has_residual,
            ctx.is_rms_norm,
        )
        dx0 = dx0mat.view(-1, *x.shape[1:])
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        dcolscale = rest[0] if colscale is not None else None
        return (
            dx0,
            dresidual,
            dgamma,
            dbeta if ctx.has_beta else None,
            dcolscale,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


class DropoutAddLayerNormParallelResidualFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x0,
        x1,
        residual,
        gamma0,
        beta0,
        gamma1,
        beta1,
        dropout_p,
        epsilon,
        residual_in_fp32=False,
        prenorm=False,
        is_rms_norm=False,
        return_dmask=False,
    ):
        x0 = maybe_align(x0.contiguous(), 16)
        x1 = maybe_align(x1.contiguous(), 16) if x1 is not None else None
        residual = (
            maybe_align(residual.contiguous(), 16) if residual is not None else None
        )
        gamma0 = maybe_align(gamma0.contiguous(), 16)
        beta0 = maybe_align(beta0.contiguous(), 16) if beta0 is not None else None
        gamma1 = maybe_align(gamma1.contiguous(), 16) if gamma1 is not None else None
        beta1 = maybe_align(beta1.contiguous(), 16) if beta1 is not None else None
        (
            z0mat,
            z1mat,
            xmat,
            dmask0,
            dmask1,
            mu,
            rsigma,
        ) = _dropout_add_layer_norm_parallel_residual_forward(
            x0,
            x1,
            residual,
            gamma0,
            beta0,
            gamma1,
            beta1,
            dropout_p,
            epsilon,
            residual_in_fp32,
            is_rms_norm,
        )
        ctx.save_for_backward(
            xmat.view(x0.shape), dmask0, dmask1, gamma0, gamma1, mu, rsigma
        )
        ctx.prenorm = prenorm
        ctx.dropout_p = dropout_p
        ctx.has_x1 = x1 is not None
        ctx.has_residual = residual is not None
        ctx.is_rms_norm = is_rms_norm
        ctx.has_beta = beta0 is not None
        z = (z0mat.view(x0.shape), z1mat.view(x0.shape) if z1mat is not None else None)
        if not return_dmask:
            return z if not prenorm else (*z, xmat.view(x0.shape))
        else:
            dmask0 = (
                dmask0.view(x0.shape)
                if dropout_p > 0.0
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            dmask1 = (
                dmask1.view(x0.shape)
                if dropout_p > 0.0 and x1 is not None
                else torch.ones(x0.shape, dtype=torch.uint8, device=x0.device)
            )
            ctx.mark_non_differentiable(dmask0)
            ctx.mark_non_differentiable(dmask1)
            return (
                (*z, dmask0, dmask1)
                if not prenorm
                else (*z, xmat.view(x0.shape), dmask0, dmask1)
            )

    @staticmethod
    def backward(ctx, dz0, dz1, *args):
        dz0 = maybe_align(dz0.contiguous(), 16)
        dz1 = maybe_align(dz1.contiguous(), 16) if dz1 is not None else None
        dx = maybe_align(args[0].contiguous(), 16) if ctx.prenorm else None
        x, dmask0, dmask1, gamma0, gamma1, mu, rsigma = ctx.saved_tensors
        dropout_p = ctx.dropout_p
        has_x1 = ctx.has_x1
        has_residual = ctx.has_residual
        (
            dx0mat,
            dx1mat,
            dresidualmat,
            dgamma0,
            dbeta0,
            dgamma1,
            dbeta1,
        ) = _dropout_add_layer_norm_parallel_residual_backward(
            dz0,
            dz1,
            dx,
            x,
            dmask0,
            dmask1,
            mu,
            rsigma,
            gamma0,
            gamma1,
            dropout_p,
            has_x1,
            has_residual,
            ctx.is_rms_norm,
        )
        dx0 = dx0mat.view(x.shape)
        dx1 = dx1mat.view(x.shape) if dx1mat is not None else None
        dresidual = dresidualmat.view(x.shape) if dresidualmat is not None else None
        return (
            dx0,
            dx1,
            dresidual,
            dgamma0,
            dbeta0 if ctx.has_beta else None,
            dgamma1,
            dbeta1 if ctx.has_beta else None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def layer_norm(x, weight, bias, epsilon):
    return DropoutAddLayerNormFn.apply(
        x, None, weight, bias, None, None, 0.0, epsilon, False
    )


def dropout_add_layer_norm(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    rowscale=None,
    layerscale=None,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    return DropoutAddLayerNormFn.apply(
        x0,
        residual,
        weight,
        bias,
        rowscale,
        layerscale,
        dropout_p,
        epsilon,
        residual_in_fp32,
        prenorm,
        False,
        return_dropout_mask,
    )


def dropout_add_layer_norm_subset(
    x0,
    residual,
    weight,
    bias,
    dropout_p,
    epsilon,
    layerscale=None,
    x0_subset=None,
    out_subset=None,
    rowscale_const=1.0,
    out_numrows=0,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    return DropoutAddLayerNormSubsetFn.apply(
        x0,
        residual,
        weight,
        bias,
        layerscale,
        x0_subset,
        out_subset,
        dropout_p,
        epsilon,
        rowscale_const,
        out_numrows,
        residual_in_fp32,
        prenorm,
        False,
        return_dropout_mask,
    )


def dropout_add_layer_norm_parallel_residual(
    x0,
    x1,
    residual,
    weight0,
    bias0,
    weight1,
    bias1,
    dropout_p,
    epsilon,
    prenorm=False,
    residual_in_fp32=False,
    return_dropout_mask=False,
):
    return DropoutAddLayerNormParallelResidualFn.apply(
        x0,
        x1,
        residual,
        weight0,
        bias0,
        weight1,
        bias1,
        dropout_p,
        epsilon,
        residual_in_fp32,
        prenorm,
        False,
        return_dropout_mask,
    )


class DropoutAddLayerNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        prenorm=False,
        p=0.0,
        eps=1e-5,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.prenorm = prenorm
        self.p = p
        self.eps = eps
        self.residual_in_fp32 = residual_in_fp32
        self.weight = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.bias = torch.nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)
        init.zeros_(self.bias)

    def forward(self, x0, residual=None):
        return dropout_add_layer_norm(
            x0,
            residual,
            self.weight,
            self.bias,
            self.p if self.training else 0.0,
            self.eps,
            prenorm=self.prenorm,
            residual_in_fp32=self.residual_in_fp32,
        )


def rms_norm(x, weight, epsilon, dropout):
    return DropoutAddLayerNormFn.apply(
        x, None, weight, None, None, None, dropout, epsilon, False, False, True
    )


class FusedRMSNorm(torch.nn.Module):
    def __init__(
        self, size: int, dim: int = -1, eps: float = 1e-5, dropout: float = 0.0
    ):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.dim = dim
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.weight)

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps, self.dropout if self.training else 0.0)


class RMSNorm(torch.nn.Module):
    def __init__(self, size: int, dim: int = -1, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(size))
        self.eps = eps
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x * x, dim=self.dim, keepdim=True)
        x_normed = x * torch.rsqrt(norm_x + self.eps)
        return self.weight * x_normed

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

```

## gridworld/model/tiny_llama/utils.py
```python
"""Utility functions for training and inference."""

import pickle
import sys
import warnings
from contextlib import contextmanager
from functools import partial
from io import BytesIO
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Mapping, Optional, Type, TypeVar, Union

import torch
import torch.nn as nn
import torch.utils._device
from lightning.fabric.loggers import CSVLogger
from torch.serialization import normalize_storage_type


def find_multiple(n: int, k: int) -> int:
    assert k > 0
    if n % k == 0:
        return n
    return n + k - (n % k)


def num_parameters(module: nn.Module, requires_grad: Optional[bool] = None) -> int:
    return sum(p.numel() for p in module.parameters() if requires_grad is None or p.requires_grad == requires_grad)


@contextmanager
def quantization(mode: Optional[str] = None):
    if mode is None:
        yield
        return

    if mode == "bnb.int8":
        from quantize.bnb import InferenceLinear8bitLt

        quantized_linear_cls = InferenceLinear8bitLt
    elif mode == "bnb.fp4":
        from quantize.bnb import Linear4bit

        class QuantizedLinear(Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="fp4", compress_statistics=False, **kwargs)

        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.fp4-dq":
        from quantize.bnb import Linear4bit

        class QuantizedLinear(Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="fp4", compress_statistics=True, **kwargs)

        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.nf4":
        from quantize.bnb import Linear4bit

        class QuantizedLinear(Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="nf4", compress_statistics=False, **kwargs)

        quantized_linear_cls = QuantizedLinear
    elif mode == "bnb.nf4-dq":
        from quantize.bnb import Linear4bit

        class QuantizedLinear(Linear4bit):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, quant_type="nf4", compress_statistics=True, **kwargs)

        quantized_linear_cls = QuantizedLinear
    elif mode == "gptq.int4":
        from quantize.gptq import ColBlockQuantizedLinear

        class QuantizedLinear(ColBlockQuantizedLinear):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, bits=4, tile_cols=-1, **kwargs)

        quantized_linear_cls = QuantizedLinear
    else:
        raise ValueError(f"Unknown quantization mode: {mode}")

    torch_linear_cls = torch.nn.Linear
    torch.nn.Linear = quantized_linear_cls
    yield
    torch.nn.Linear = torch_linear_cls


class NotYetLoadedTensor:
    def __init__(self, metatensor, archiveinfo, storageinfo, rebuild_args):
        self.metatensor = metatensor
        self.archiveinfo = archiveinfo
        self.storageinfo = storageinfo
        self.rebuild_args = rebuild_args

    @classmethod
    def rebuild_from_type_v2(cls, func, new_type, args, state, *, archiveinfo=None):
        ret = func(*args)
        if isinstance(ret, NotYetLoadedTensor):
            old_lt = ret._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._tensor._rebuild_from_type_v2(lambda: t, new_type, (), state)

            ret._load_tensor = _load_tensor
            return ret
        return torch._tensor._rebuild_from_type_v2(func, new_type, args, state)

    @classmethod
    def rebuild_parameter(cls, data, requires_grad, backward_hooks, *, archiveinfo=None):
        if isinstance(data, NotYetLoadedTensor):
            old_lt = data._load_tensor

            def _load_tensor():
                t = old_lt()
                return torch._utils._rebuild_parameter(t, requires_grad, backward_hooks)

            data._load_tensor = _load_tensor
            return data
        return torch._utils._rebuild_parameter(data, requires_grad, backward_hooks)

    @classmethod
    def rebuild_tensor_v2(
        cls, storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata=None, *, archiveinfo=None
    ):
        rebuild_args = (storage_offset, size, stride, requires_grad, backward_hooks, metadata)
        metatensor = torch._utils._rebuild_tensor_v2(
            storage, storage_offset, size, stride, requires_grad, backward_hooks, metadata
        )
        storageinfo = storage.archiveinfo
        return NotYetLoadedTensor(metatensor, archiveinfo, storageinfo, rebuild_args)

    def _load_tensor(self):
        name, storage_cls, fn, device, size = self.storageinfo
        dtype = self.metatensor.dtype

        uts = (
            self.archiveinfo.zipfile_context.zf.get_storage_from_record(
                f"data/{fn}", size * torch._utils._element_size(dtype), torch.UntypedStorage
            )
            ._typed_storage()
            ._untyped_storage
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            storage = torch.storage.TypedStorage(wrap_storage=uts, dtype=self.metatensor.dtype, _internal=True)
        return torch._utils._rebuild_tensor_v2(storage, *self.rebuild_args)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        loaded_args = [(a._load_tensor() if isinstance(a, NotYetLoadedTensor) else a) for a in args]
        return func(*loaded_args, **kwargs)

    def __getattr__(self, name):
        if name in {
            "dtype",
            "grad",
            "grad_fn",
            "layout",
            "names",
            "ndim",
            "output_nr",
            "requires_grad",
            "retains_grad",
            "shape",
            "volatile",
        }:
            return getattr(self.metatensor, name)
        if name in {"size"}:
            return getattr(self.metatensor, name)
        if name in {"contiguous"}:
            return getattr(self._load_tensor(), name)

        raise AttributeError(f"{type(self)} does not have {name}")

    def __repr__(self):
        return f"NotYetLoadedTensor({repr(self.metatensor)})"


class LazyLoadingUnpickler(pickle.Unpickler):
    def __init__(self, file, zipfile_context):
        super().__init__(file)
        self.zipfile_context = zipfile_context

    def find_class(self, module, name):
        res = super().find_class(module, name)
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return partial(NotYetLoadedTensor.rebuild_tensor_v2, archiveinfo=self)
        if module == "torch._tensor" and name == "_rebuild_from_type_v2":
            return partial(NotYetLoadedTensor.rebuild_from_type_v2, archiveinfo=self)
        if module == "torch._utils" and name == "_rebuild_parameter":
            return partial(NotYetLoadedTensor.rebuild_parameter, archiveinfo=self)
        return res

    def persistent_load(self, pid):
        name, cls, fn, device, size = pid
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = torch.storage.TypedStorage(dtype=cls().dtype, device="meta")
        s.archiveinfo = pid
        return s


class lazy_load:
    def __init__(self, fn):
        self.zf = torch._C.PyTorchFileReader(str(fn))
        with BytesIO(self.zf.get_record("data.pkl")) as pkl:
            mup = LazyLoadingUnpickler(pkl, self)
            self.sd = mup.load()

    def __enter__(self):
        return self.sd

    def __exit__(self, exc_type, exc_val, exc_tb):
        del self.zf
        self.zf = None


def check_valid_checkpoint_dir(checkpoint_dir: Path) -> None:
    files = {
        "lit_model.pth": (checkpoint_dir / "lit_model.pth").is_file(),
        "lit_config.json": (checkpoint_dir / "lit_config.json").is_file(),
        "tokenizer.json OR tokenizer.model": (checkpoint_dir / "tokenizer.json").is_file() or (
            checkpoint_dir / "tokenizer.model"
        ).is_file(),
        "tokenizer_config.json": (checkpoint_dir / "tokenizer_config.json").is_file(),
    }
    if checkpoint_dir.is_dir():
        if all(files.values()):
            return
        problem = f" is missing the files: {[f for f, exists in files.items() if not exists]!r}"
    else:
        problem = " is not a checkpoint directory"

    available = list(Path("checkpoints").glob("*/*"))
    if available:
        options = "\n --checkpoint_dir ".join([""] + [repr(str(p.resolve())) for p in available])
        extra = f"\nYou have downloaded locally:{options}\n"
    else:
        extra = ""

    error_message = (
        f"--checkpoint_dir {str(checkpoint_dir.absolute())!r}{problem}."
        "\nFind download instructions at ."
        f"{extra}\nSee all download options by running:\n python scripts/download.py"
    )
    print(error_message, file=sys.stderr)
    raise SystemExit(1)


class SavingProxyForStorage:
    def __init__(self, obj, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.saver = saver
        if not (isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj)):
            raise TypeError(f"expected storage, not {type(obj)}")

        if isinstance(obj, torch.storage.TypedStorage):
            storage = obj._untyped_storage
            storage_type_str = obj._pickle_storage_type()
            storage_type = getattr(torch, storage_type_str)
            storage_numel = obj._size()
        else:
            storage = obj
            storage_type = normalize_storage_type(type(obj))
            storage_numel = storage.nbytes()

        storage_key = saver._write_storage_and_return_key(storage)
        location = torch.serialization.location_tag(storage)

        self.storage_info = ("storage", storage_type, storage_key, location, storage_numel)

    def __reduce_ex__(self, protocol_version):
        assert False, "this should be handled with out of band"


class SavingProxyForTensor:
    def __init__(self, tensor, saver, protocol_version=5):
        self.protocol_version = protocol_version
        self.reduce_ret_fn, (storage, *other_reduce_args) = tensor.__reduce_ex__(protocol_version)
        assert isinstance(storage, torch.storage.TypedStorage), "Please check for updates"
        storage_proxy = SavingProxyForStorage(storage, saver, protocol_version=protocol_version)
        self.reduce_args = (storage_proxy, *other_reduce_args)

    def __reduce_ex__(self, protocol_version):
        if protocol_version != self.protocol_version:
            raise RuntimeError(f"Unexpected protocol version: expected {self.protocol_version}, got {protocol_version}")
        return self.reduce_ret_fn, self.reduce_args


class IncrementalPyTorchPickler(pickle.Pickler):
    def __init__(self, saver, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage_dtypes = {}
        self.saver = saver
        self.id_map = {}

    def persistent_id(self, obj):
        if isinstance(obj, SavingProxyForStorage):
            return obj.storage_info

        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            if storage.data_ptr() != 0:
                if storage.data_ptr() in self.storage_dtypes:
                    if storage_dtype != self.storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that view the same data as different types"
                        )
                else:
                    self.storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = self.id_map.get(storage._cdata)
            if storage_key is None:
                storage_key = self.saver._write_storage_and_return_key(storage)
                self.id_map[storage._cdata] = storage_key
            location = torch.serialization.location_tag(storage)

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None


class incremental_save:
    def __init__(self, name):
        self.name = name
        self.zipfile = torch._C.PyTorchFileWriter(str(name))
        self.has_saved = False
        self.next_key = 0

    def __enter__(self):
        return self

    def store_early(self, tensor):
        if isinstance(tensor, torch.Tensor):
            return SavingProxyForTensor(tensor, self)
        raise TypeError(f"can only store tensors early, not {type(tensor)}")

    def save(self, obj):
        if self.has_saved:
            raise RuntimeError("have already saved")
        data_buf = BytesIO()
        pickler = IncrementalPyTorchPickler(self, data_buf, protocol=5)
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        self.zipfile.write_record("data.pkl", data_value, len(data_value))
        self.has_saved = True

    def _write_storage_and_return_key(self, storage):
        if self.has_saved:
            raise RuntimeError("have already saved")
        key = self.next_key
        self.next_key += 1
        name = f"data/{key}"
        if storage.device.type != "cpu":
            storage = storage.cpu()
        num_bytes = storage.nbytes()
        self.zipfile.write_record(name, storage.data_ptr(), num_bytes)
        return key

    def __exit__(self, type, value, traceback):
        self.zipfile.write_end_of_file()


T = TypeVar("T")


def step_csv_logger(*args: Any, cls: Type[T] = CSVLogger, **kwargs: Any) -> T:
    logger = cls(*args, **kwargs)

    def merge_by(dicts, key):
        from collections import defaultdict

        out = defaultdict(dict)
        for d in dicts:
            if key in d:
                out[d[key]].update(d)
        return [v for _, v in sorted(out.items())]

    def save(self) -> None:
        import csv

        if not self.metrics:
            return
        metrics = merge_by(self.metrics, "step")
        keys = sorted({k for m in metrics for k in m})
        with self._fs.open(self.metrics_file_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)

    logger.experiment.save = MethodType(save, logger.experiment)

    return logger


def chunked_cross_entropy(
    logits: Union[torch.Tensor, List[torch.Tensor]], targets: torch.Tensor, chunk_size: int = 128
) -> torch.Tensor:

    if isinstance(logits, list):
        if chunk_size == 0:
            logits = torch.cat(logits, dim=1)
            logits = logits.reshape(-1, logits.size(-1))
            targets = targets.reshape(-1)
            return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

        logit_chunks = [logit_chunk.reshape(-1, logit_chunk.size(-1)) for logit_chunk in logits]
        target_chunks = [target_chunk.reshape(-1) for target_chunk in targets.split(logits[0].size(1), dim=1)]
        loss_chunks = [
            torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=-1, reduction="none")
            for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
        ]
        return torch.cat(loss_chunks).mean()

    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    if chunk_size == 0:
        return torch.nn.functional.cross_entropy(logits, targets, ignore_index=-1)

    logit_chunks = logits.split(chunk_size)
    target_chunks = targets.split(chunk_size)
    loss_chunks = [
        torch.nn.functional.cross_entropy(logit_chunk, target_chunk, ignore_index=-1, reduction="none")
        for logit_chunk, target_chunk in zip(logit_chunks, target_chunks)
    ]
    return torch.cat(loss_chunks).mean()


def map_old_state_dict_weights(state_dict: Dict, mapping: Mapping, prefix: str) -> Dict:
    for checkpoint_name, attribute_name in mapping.items():
        full_checkpoint_name = prefix + checkpoint_name
        if full_checkpoint_name in state_dict:
            full_attribute_name = prefix + attribute_name
            state_dict[full_attribute_name] = state_dict.pop(full_checkpoint_name)
    return state_dict


def get_default_supported_precision(training: bool, tpu: bool = False) -> str:
    if tpu:
        return "32-true"
    if not torch.cuda.is_available() or torch.cuda.is_bf16_supported():
        return "bf16-mixed" if training else "bf16-true"
    return "16-mixed" if training else "16-true"
```

## gridworld/plot_best_streams.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gridworld/plot_best_streams.py
"""
Compare mean return of ALL streams vs. TOP-p% best streams (by a ranking criterion).
Saves a figure for each selected env instance.

Usage (example, from gridworld/):
  python gridworld/plot_best_streams.py \
    -ec cfg/env/darkroom.yaml \
    -ac cfg/alg/ppo_dr.yaml \
    -t datasets_dr_noise50 \
    -o plots \
    --env-index train \
    --max-envs 1 \
    --n-stream 100 \
    --source-timesteps -1 \
    --top-percent 50 \
    --rank-by mean \
    --seed 0
"""

import os, argparse, random
import numpy as np
import h5py
import matplotlib.pyplot as plt

from utils import get_config, get_traj_file_name  # provided in repo


# ---------- helpers: align T to horizon, returns computation ----------


def compute_use_T_like_dataset(T, horizon, source_timesteps):
    """
    Dataset-style behavior:
      - If source_timesteps < 0: fallback to the largest multiple of horizon not exceeding T.
      - Else: use min(source_timesteps, T); must be divisible by horizon.
    """
    if source_timesteps is None:
        raise ValueError("source_timesteps 不能为空；请显式设置。")
    if source_timesteps < 0:
        use_T = (T // horizon) * horizon
        if use_T <= 0:
            raise ValueError(
                f"T={T} 与 horizon={horizon} 不匹配，无法形成一个完整 episode。"
            )
        print(f"[INFO] source_timesteps<0: 回退 use_T={use_T}（与 horizon 对齐）")
        return use_T
    use_T = min(int(source_timesteps), T)
    if use_T % horizon != 0:
        raise ValueError(
            f"dataset 风格要求 source_timesteps 可被 horizon 整除；"
            f"当前 source_timesteps={source_timesteps}, horizon={horizon}."
        )
    return use_T


def returns_first_n_streams_dataset_style(
    rewards_TS, horizon, source_timesteps, n_stream
):
    """
    Parameters
    ----------
    rewards_TS : np.ndarray, shape (T, S)
    horizon : int
    source_timesteps : int
    n_stream : int

    Returns
    -------
    returns_SE : np.ndarray, shape (S0, E)
      per-stream per-episode returns
    E : int, number of episodes that fit
    use_T : int, number of timesteps used
    S0 : int, number of streams used
    """
    T, S = rewards_TS.shape
    S0 = min(int(n_stream), S)
    use_T = compute_use_T_like_dataset(T, horizon, source_timesteps)
    E = use_T // horizon
    # reshape to (E, horizon, S0) then sum over horizon → (E, S0) → transpose to (S0, E)
    ret_SE = rewards_TS[:use_T, :S0].reshape(E, horizon, S0).sum(axis=1).T
    return ret_SE, E, use_T, S0


# ---------- env 选择：与数据集一致（先打乱再切分） ----------


def resolve_env_indices_dataset_style(cfg, which):
    if which in ("train", "test", "all"):
        if cfg["env"] == "darkroom":
            n_total = cfg["grid_size"] ** 2
        elif cfg["env"] == "darkroompermuted":
            n_total = 120
        elif cfg["env"] == "darkkeytodoor":
            # 按你仓库 dataset 绘图脚本的用法（避免超大），可适当缩减；这里保持与之前一致
            n_total = round((cfg["grid_size"] ** 4) / 4)
        else:
            raise ValueError("Unsupported env")

        idx = list(range(n_total))
        rng = random.Random(int(cfg.get("env_split_seed", 0)))
        rng.shuffle(idx)

        n_train = round(n_total * cfg["train_env_ratio"])
        if which == "train":
            return idx[:n_train]
        elif which == "test":
            return idx[n_train:]
        else:
            return idx
    # Comma list of indices
    return [int(x) for x in which.split(",") if x != ""]


# ---------- 工作目录自适应：为 include 相对路径兜底 ----------


def _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs):
    candidates = []
    for p in (env_cfg_abs, alg_cfg_abs):
        if p:
            d = os.path.abspath(os.path.join(os.path.dirname(p), "..", ".."))
            candidates.append(d)
    for root in candidates:
        if os.path.isdir(os.path.join(root, "cfg")) and os.path.isfile(
            os.path.join(root, "utils.py")
        ):
            try:
                os.chdir(root)
                print(f"[INFO] 工作目录已切换到 {root} 以适配 include 相对路径。")
            except Exception as e:
                print(f"[WARN] 切换工作目录到 {root} 失败：{e}")
            break


# ---------- 根据数据集目录名生成子目录标签 ----------


def compute_dataset_tag(traj_dir_abs, cfg):
    tag = os.path.basename(os.path.normpath(traj_dir_abs))
    if tag.lower() in {"", ".", "datasets", "dataset", "data", "trajectories", "traj"}:
        tag = get_traj_file_name(cfg)
    return tag


def sanitize_tag(tag: str) -> str:
    safe_chars = []
    for c in tag:
        if c.isalnum() or c in "._-":
            safe_chars.append(c)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


# ---------- 排名与绘图 ----------


def rank_streams(returns_SE: np.ndarray, criterion: str):
    """
    returns_SE: (S, E)
    criterion: 'mean' | 'last' | 'max' | 'auc'
    returns: ndarray of shape (S,), higher is better
    """
    if criterion == "mean":
        score = returns_SE.mean(axis=1)
    elif criterion == "last":
        score = returns_SE[:, -1]
    elif criterion == "max":
        score = returns_SE.max(axis=1)
    elif criterion == "auc":
        # sum across episodes (equiv. to mean*E)
        score = returns_SE.sum(axis=1)
    else:
        raise ValueError(f"Unknown rank criterion: {criterion}")
    return score


def plot_best_vs_all(returns_SE, top_idx, out_path, title):
    """
    Plot mean curves of ALL vs TOP
    """
    all_mean = returns_SE.mean(axis=0)
    top_mean = (
        returns_SE[top_idx].mean(axis=0)
        if len(top_idx) > 0
        else np.zeros_like(all_mean)
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(all_mean, label=f"ALL (n={returns_SE.shape[0]})", linewidth=2)
    ax.plot(top_mean, label=f"TOP (n={len(top_idx)})", linewidth=2)

    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(title)
    ax.grid(True, linestyle=":", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


# ---------- 主流程 ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--env-config",
        "-ec",
        required=True,
        help="eg. cfg/env/darkroom.yaml 或 gridworld/cfg/env/darkroom.yaml",
    )
    ap.add_argument(
        "--alg-config",
        "-ac",
        required=True,
        help="eg. cfg/alg/ppo_dr.yaml 或 gridworld/cfg/alg/ppo_dr.yaml",
    )
    ap.add_argument("--traj-dir", "-t", default="./datasets", help="目录，内含 *.hdf5")
    ap.add_argument("--out", "-o", default="./plots", help="输出图的根目录")
    ap.add_argument(
        "--env-index",
        "-g",
        default="train",
        help='绘制的 env 组索引："0" 或 "0,1,2" 或 "train"/"test"/"all"（与数据集一致）',
    )
    ap.add_argument(
        "--max-envs",
        type=int,
        default=1,
        help="最多绘制多少个 env（从 --env-index 里取前K；0 表示不限制）",
    )
    ap.add_argument(
        "--n-stream",
        type=int,
        default=100,
        help="每个 env 参与统计的 stream 数（只看前 n_stream 条）",
    )
    ap.add_argument(
        "--source-timesteps",
        type=int,
        default=-1,
        help="用于统计的时间步数；<0 将回退到不超过 T 的最大整倍数（与 horizon 对齐）",
    )
    ap.add_argument(
        "--top-percent", type=float, default=50.0, help="选择前 p%% 的最佳流，默认 50%%"
    )
    ap.add_argument(
        "--rank-by",
        choices=["mean", "last", "max", "auc"],
        default="mean",
        help="最优流的排序指标，默认 mean",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="用于 env 索引随机性的种子；默认使用 config['env_split_seed']",
    )
    args = ap.parse_args()

    # 绝对路径 & 工作目录自适应
    env_cfg_abs = os.path.abspath(args.env_config)
    alg_cfg_abs = os.path.abspath(args.alg_config)
    traj_dir_abs = os.path.abspath(args.traj_dir)
    out_root_abs = os.path.abspath(args.out)

    _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs)

    # 读取配置
    cfg = get_config(env_cfg_abs)
    cfg.update(get_config(alg_cfg_abs))

    # 随机种子（仅用于 env 切分的一致性）
    seed_to_use = (
        args.seed if args.seed is not None else int(cfg.get("env_split_seed", 0))
    )
    random.seed(seed_to_use)
    np.random.seed(seed_to_use % (2**32 - 1))

    # 路径与输出目录
    h5_path = os.path.join(traj_dir_abs, get_traj_file_name(cfg) + ".hdf5")
    dataset_tag = compute_dataset_tag(traj_dir_abs, cfg)
    dataset_tag_safe = sanitize_tag(dataset_tag)
    out_dir_by_dataset = os.path.join(out_root_abs, dataset_tag)
    os.makedirs(out_dir_by_dataset, exist_ok=True)
    print(
        f"[INFO] 输出目录：{out_dir_by_dataset}  （dataset_tag='{dataset_tag}', filename_tag='{dataset_tag_safe}'）"
    )

    # env indices
    env_indices = resolve_env_indices_dataset_style(cfg, args.env_index)
    if args.max_envs and args.max_envs > 0:
        env_indices = env_indices[: args.max_envs]
    print(f"[INFO] 将绘制的 env 实例索引：{env_indices}")

    # 安全检查
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"未找到 HDF5 文件：{h5_path}")

    top_percent = float(args.top_percent)
    if not (0 < top_percent <= 100):
        raise ValueError(f"--top-percent 必须在 (0, 100]，收到 {top_percent}")

    with h5py.File(h5_path, "r") as f:
        for i in env_indices:
            # 读取 rewards(T,S) 并计算 returns(S,E)
            rewards_TS = f[str(i)]["rewards"][()]  # (T, S)
            T, S = rewards_TS.shape
            horizon = cfg["horizon"]

            returns_SE, E, use_T, S0 = returns_first_n_streams_dataset_style(
                rewards_TS, horizon, args.source_timesteps, args.n_stream
            )
            # 排名与选择 top-k
            score = rank_streams(returns_SE, args.rank_by)  # shape (S0,)
            order_desc = np.argsort(-score)  # 高分在前
            k = max(1, int(np.ceil(S0 * top_percent / 100.0)))
            top_idx = order_desc[:k]

            print(
                f"[env {i}] S0={S0}, E={E}, use_T={use_T}, "
                f"rank_by={args.rank_by}, top_percent={top_percent:.1f}%, top_k={k}"
            )
            print(f"Top-{k} indices (by {args.rank_by}): {top_idx.tolist()}")

            # 画图
            title = (
                f"Env {i} — {cfg['env']} — Mean return: "
                f"ALL (n={S0}) vs TOP-{k} ({top_percent:.1f}%, by {args.rank_by})"
            )
            out_fig = os.path.join(
                out_dir_by_dataset,
                f"best_vs_all_{dataset_tag_safe}_env{i}_top{int(round(top_percent))}_{args.rank_by}.png",
            )
            plot_best_vs_all(returns_SE, top_idx, out_fig, title)
            print(f"[OK] 保存：{out_fig}")


if __name__ == "__main__":
    main()

```

## gridworld/plot_returns.py
```python
import argparse, h5py, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
import math

def smooth_same(y, win=1):
    if win <= 1 or len(y) < 2: return y
    k = np.ones(win, dtype=float) / win
    return np.convolve(y, k, mode="same")

def accumulate_returns_one_group(rewards, dones, cap=None):
    """
    rewards/dones: 形状 (T, N_stream)
    返回 list[stream] -> list[episode return]
    """
    T, N = rewards.shape
    acc = np.zeros(N, dtype=float)
    returns = [[] for _ in range(N)]
    for t in range(T):
        acc += rewards[t]
        done_idx = np.nonzero(dones[t])[0]
        for s in done_idx:
            returns[s].append(acc[s])
            acc[s] = 0.0
    if cap is not None:
        returns = [r[:cap] for r in returns]
    return returns

def grid_dims(n):
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    return rows, cols

def plot_grid(returns, env_id, out_path, smooth=1, title_prefix=""):
    n_stream = len(returns)
    rows, cols = grid_dims(n_stream)
    fig_w, fig_h = cols * 2.2, rows * 1.9
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h), sharex=True, sharey=True)
    axes = np.atleast_2d(axes).reshape(rows, cols)

    for s in range(n_stream):
        ax = axes[s // cols, s % cols]
        y = np.asarray(returns[s], dtype=float)
        x = np.arange(1, len(y) + 1)
        ax.plot(x, smooth_same(y, smooth), linewidth=1.0)
        ax.set_title(f"S{s}", fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.4)

    # 去掉多余子图
    for k in range(n_stream, rows * cols):
        axes[k // cols, k % cols].axis("off")

    fig.supxlabel("episode")
    fig.supylabel("return")
    fig.suptitle(f"{title_prefix}Returns per Stream (env {env_id})", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    print(f"✅ 网格总览已保存: {out_path}")

def save_each_stream(returns, out_dir, smooth=1):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s, r in enumerate(returns):
        y = np.asarray(r, dtype=float)
        x = np.arange(1, len(y) + 1)
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(x, smooth_same(y, smooth), linewidth=1.2)
        ax.set_xlabel("episode"); ax.set_ylabel("return"); ax.set_title(f"stream {s}")
        ax.grid(True, linestyle=":", linewidth=0.6)
        fig.tight_layout()
        fp = out_dir / f"stream_{s:03d}.png"
        fig.savefig(fp, dpi=200)
        plt.close(fig)
    print(f"✅ 单流图已保存到目录: {out_dir}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--h5", required=True, help="HDF5 数据文件")
    p.add_argument("--env-id", type=int, default=0, help="只画该分组（group）的流")
    p.add_argument("--num", type=int, default=None, help="每条 stream 取前多少个 episode（默认取完）")
    p.add_argument("--smooth", type=int, default=1, help="滑动平均窗口，>1 可平滑曲线")
    p.add_argument("--outgrid", default=None, help="网格总览图输出路径（缺省自动生成）")
    p.add_argument("--outdir", default="", help="如提供，则逐流输出到该目录（每个 stream 一张图）")
    args = p.parse_args()

    h5_path = Path(args.h5)
    assert h5_path.exists(), f"找不到文件：{h5_path}"

    with h5py.File(h5_path, "r") as f:
        assert str(args.env_id) in f.keys(), f"env-id {args.env_id} 不在 HDF5 分组中"
        g = f[str(args.env_id)]
        rewards = g["rewards"][()]  # (T, N_stream)
        dones = g["dones"][()]      # (T, N_stream) bool
        returns = accumulate_returns_one_group(rewards, dones, cap=args.num)
        per_stream_eps = dones.sum(axis=0)
        print(f"[env {args.env_id}] n_stream={rewards.shape[1]}, episodes/stream 平均={per_stream_eps.mean():.0f} (min={per_stream_eps.min()}, max={per_stream_eps.max()})")

    outgrid = args.outgrid or (h5_path.parent / f"returns_env{args.env_id}_grid.png")
    plot_grid(returns, args.env_id, outgrid, smooth=args.smooth)

    if args.outdir:
        save_each_stream(returns, args.outdir, smooth=args.smooth)

if __name__ == "__main__":
    main()

```

## gridworld/plot_sampled_streams.py
```python
#!/usr/bin/env python3
# gridworld/plot_sampled_streams.py
# -*- coding: utf-8 -*-
import os, argparse, random
import numpy as np
import h5py
import matplotlib.pyplot as plt

from utils import get_config, get_traj_file_name  # 同目录 utils.py


# ---------- 与 dataset_linear_all.py 一致的 returns/metric/采样 ----------


def compute_use_T_like_dataset(T, horizon, source_timesteps):
    if source_timesteps is None:
        raise ValueError("source_timesteps 不能为空；请显式设置。")
    if source_timesteps < 0:
        use_T = (T // horizon) * horizon
        print(
            f"[WARN] dataset 风格通常要求 source_timesteps>0 且可被 horizon 整除；"
            f"收到 {source_timesteps}，这里退回到 use_T={use_T}（与 horizon 对齐）。"
        )
        return use_T
    use_T = min(int(source_timesteps), T)
    if use_T % horizon != 0:
        raise ValueError(
            f"dataset 风格要求 source_timesteps 可被 horizon 整除；"
            f"当前 source_timesteps={source_timesteps}, horizon={horizon}."
        )
    return use_T


def returns_first_n_streams_dataset_style(
    rewards_TS, horizon, source_timesteps, n_stream
):
    T, S = rewards_TS.shape
    S0 = min(int(n_stream), S)
    use_T = compute_use_T_like_dataset(T, horizon, source_timesteps)
    E = use_T // horizon
    ret_SE = rewards_TS[:use_T, :S0].reshape(E, horizon, S0).sum(axis=1).T
    return ret_SE, E, use_T, S0


def select_streams_like_dataset(
    returns_SE, max_reward, n_select, stability_coeff=1.0, dataset_type="ad"
):
    S0, E = returns_SE.shape
    metric_lst = []

    for j in range(S0):
        all_return_per_stream = returns_SE[j]  # (E,)
        return_diff = np.diff(all_return_per_stream)
        negative_diff = return_diff[return_diff <= 0]
        stability = 1 + np.mean(negative_diff) / max_reward  # 可能为 NaN

        mean_all = float(np.mean(all_return_per_stream))
        gap_all = float(np.max(all_return_per_stream) - np.min(all_return_per_stream))
        improvement = (mean_all + gap_all) / 2.0 / max_reward

        metric = stability_coeff * stability + 1.0 * improvement
        metric_lst.append(float(metric))

    sorted_metric_lst = sorted(metric_lst)
    metric_array = np.array(sorted_metric_lst, dtype=float)

    if dataset_type.lower() == "dpt":
        scale_a, scale_b = 0.0, 1.0
        if (np.max(metric_array) - np.min(metric_array)) == 0:
            probabilities_lst = np.zeros(S0, dtype=float)
        else:
            new_linear_vals = (metric_array - np.min(metric_array)) / (
                np.max(metric_array) - np.min(metric_array)
            )
            probabilities_lst = new_linear_vals * (scale_b - scale_a) + scale_a
    else:
        new_linear_vals = metric_array - np.min(metric_array)
        if np.max(new_linear_vals) == 0:
            probabilities_lst = np.zeros(S0, dtype=float)
        else:
            probabilities_lst = new_linear_vals / np.max(new_linear_vals)

    print(
        f"Scaled softmax probability (sorted from small to large) = {probabilities_lst}"
    )

    sorted_index_lst = sorted(range(len(metric_lst)), key=lambda i: metric_lst[i])

    temp_slice = []
    if np.max(probabilities_lst) > 0:
        while len(temp_slice) < n_select:
            for potato in reversed(range(S0)):  # 从高到低扫描
                selected_stream = sorted_index_lst[potato]
                selected_stream_prob = probabilities_lst[potato]
                if random.uniform(0, 1) <= selected_stream_prob:
                    temp_slice.append(selected_stream)
                if len(temp_slice) >= n_select:
                    break

    sel_idx = np.array(temp_slice[:n_select], dtype=np.int32)
    return (
        sel_idx,
        np.array(metric_lst, dtype=float),
        probabilities_lst,
        sorted_index_lst,
    )


# ---------- 绘图 ----------


def plot_grid(returns_SE, selected_idx, out_path, title, cols=10):
    if len(selected_idx) == 0:
        print("[WARN] 没有可绘制的 stream，输出占位图：", title)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No streams in this group", ha="center", va="center")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=240)
        plt.close(fig)
        return

    selR = returns_SE[selected_idx]
    rows = int(np.ceil(len(selected_idx) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.1, rows * 1.7), sharex=True, sharey=True
    )
    axes = np.ravel(axes)
    for k, ax in enumerate(axes):
        if k < len(selected_idx):
            ax.plot(selR[k])
            ax.set_title(f"s{selected_idx[k]}", fontsize=8)
        else:
            ax.axis("off")
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.02, "episode", ha="center")
    fig.text(0.02, 0.5, "return", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_mean_three_groups(returns_SE, selected_idx, out_path, title):
    """
    修复点：Selected 的均值用 selected_idx（含重复）直接索引求均值，
    这样重复被选中的“好流”在均值里拥有更高权重，反映 LHF 的放大效应。
    Not-selected 仍然以 unique 的补集来计算（不受重复影响）。
    """
    all_mean = returns_SE.mean(axis=0)
    S0 = returns_SE.shape[0]

    # 保留重复：按频次加权
    if len(selected_idx) > 0:
        selected_mean = returns_SE[selected_idx].mean(axis=0)  # 关键修复：含重复
        sel_unique = np.unique(selected_idx)
    else:
        selected_mean = np.zeros_like(all_mean)
        sel_unique = np.array([], dtype=int)

    # Not-selected 用 unique 的补集
    all_idx = np.arange(S0, dtype=int)
    not_sel = np.setdiff1d(all_idx, sel_unique, assume_unique=True)
    if not_sel.size > 0:
        not_sel_mean = returns_SE[not_sel].mean(axis=0)
    else:
        not_sel_mean = np.zeros_like(all_mean)

    # 画图
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(all_mean, label=f"all candidates (n={S0})", linewidth=2)

    # 在图例中标注 selections 与 unique 数量，明确这是“按重复加权”的均值
    ax.plot(
        selected_mean,
        label=f"selected (dup-weighted, selections={len(selected_idx)}, unique={sel_unique.size})",
        linewidth=2,
    )

    ax.plot(
        not_sel_mean,
        label=f"not selected (n={not_sel.size})",
        linewidth=2,
        linestyle="--",
    )

    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


# ---------- env 选择：与数据集一致（先打乱再切分） ----------


def resolve_env_indices_dataset_style(cfg, which):
    if which in ("train", "test", "all"):
        if cfg["env"] == "darkroom":
            n_total = cfg["grid_size"] ** 2
        elif cfg["env"] == "darkroompermuted":
            n_total = 120
        elif cfg["env"] == "darkkeytodoor":
            n_total = round((cfg["grid_size"] ** 4) / 4)
        else:
            raise ValueError("Unsupported env")

        idx = list(range(n_total))
        rng = random.Random(int(cfg.get("env_split_seed", 0)))
        rng.shuffle(idx)

        n_train = round(n_total * cfg["train_env_ratio"])
        if which == "train":
            return idx[:n_train]
        elif which == "test":
            return idx[n_train:]
        else:
            return idx
    return [int(x) for x in which.split(",") if x != ""]


# ---------- 工作目录自适应：为 include 相对路径兜底 ----------


def _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs):
    candidates = []
    for p in (env_cfg_abs, alg_cfg_abs):
        if p:
            d = os.path.abspath(os.path.join(os.path.dirname(p), "..", ".."))
            candidates.append(d)
    for root in candidates:
        if os.path.isdir(os.path.join(root, "cfg")) and os.path.isfile(
            os.path.join(root, "utils.py")
        ):
            try:
                os.chdir(root)
                print(f"[INFO] 工作目录已切换到 {root} 以适配 include 相对路径。")
            except Exception as e:
                print(f"[WARN] 切换工作目录到 {root} 失败：{e}")
            break


# ---------- 根据数据集目录名生成子目录标签 ----------


def compute_dataset_tag(traj_dir_abs, cfg):
    """
    取 traj_dir 的最后一层目录名作为数据集标签；若过于通用或为空，则回退为 HDF5 基名。
    """
    tag = os.path.basename(os.path.normpath(traj_dir_abs))
    if tag.lower() in {"", ".", "datasets", "dataset", "data", "trajectories", "traj"}:
        tag = get_traj_file_name(cfg)
    return tag


def sanitize_tag(tag: str) -> str:
    """
    将不适合文件名的字符替换为下划线，仅保留字母/数字/._-。
    """
    safe_chars = []
    for c in tag:
        if c.isalnum() or c in "._-":
            safe_chars.append(c)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


# ---------- 主流程 ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--env-config",
        "-ec",
        required=True,
        help="eg. cfg/env/darkroom.yaml 或 gridworld/cfg/env/darkroom.yaml",
    )
    ap.add_argument(
        "--alg-config",
        "-ac",
        required=True,
        help="eg. cfg/alg/ppo_dr.yaml 或 gridworld/cfg/alg/ppo_dr.yaml",
    )
    ap.add_argument("--traj-dir", "-t", default="./datasets")
    ap.add_argument("--out", "-o", default="./plots")
    ap.add_argument(
        "--env-index",
        "-g",
        default="train",
        help='绘制的 env 组索引："0" 或 "0,1,2" 或 "train"/"test"/"all"（与数据集一致，train/test 会先打乱再切分）',
    )
    ap.add_argument(
        "--max-envs",
        type=int,
        default=1,
        help="最多绘制多少个 env（在 --env-index 选出的列表里取前K个；0 表示不限制）。默认 1。",
    )
    ap.add_argument(
        "--n-stream",
        type=int,
        default=100,
        help="每个 env 参与候选与采样的 stream 数（只看前 n_stream 条）",
    )
    ap.add_argument(
        "--source-timesteps",
        type=int,
        default=-1,
        help="用于统计的时间步数；dataset 风格要求 >0 且能被 horizon 整除；为兼容，<0 将回退到不超过 T 的最大整倍数",
    )
    ap.add_argument(
        "--stability-coeff", type=float, default=1.0, help="与数据集一致，默认 1.0"
    )
    ap.add_argument(
        "--dataset-type",
        choices=["ad", "dpt", "idt"],
        default="ad",
        help="影响线性归一化细节：AD/IDT 与 DPT 写法略有不同",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="可选：随机种子。默认使用 config['env_split_seed']，以与数据集实现复现一致。",
    )
    args = ap.parse_args()

    # 绝对路径
    env_cfg_abs = os.path.abspath(args.env_config)
    alg_cfg_abs = os.path.abspath(args.alg_config)
    traj_dir_abs = os.path.abspath(args.traj_dir)
    out_root_abs = os.path.abspath(args.out)

    # 适配 include 相对路径
    _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs)

    # 读取配置
    cfg = get_config(env_cfg_abs)
    cfg.update(get_config(alg_cfg_abs))

    # 随机性
    seed_to_use = (
        args.seed if args.seed is not None else int(cfg.get("env_split_seed", 0))
    )
    random.seed(seed_to_use)
    np.random.seed(seed_to_use % (2**32 - 1))

    # 数据 & 输出路径
    h5_path = os.path.join(traj_dir_abs, get_traj_file_name(cfg) + ".hdf5")
    dataset_tag = compute_dataset_tag(traj_dir_abs, cfg)
    dataset_tag_safe = sanitize_tag(dataset_tag)  # <---- 用于拼到文件名
    out_dir_by_dataset = os.path.join(out_root_abs, dataset_tag)
    os.makedirs(out_dir_by_dataset, exist_ok=True)
    print(
        f"[INFO] 输出目录：{out_dir_by_dataset}  （dataset_tag='{dataset_tag}', filename_tag='{dataset_tag_safe}'）"
    )

    env_indices = resolve_env_indices_dataset_style(cfg, args.env_index)
    if args.max_envs and args.max_envs > 0:
        env_indices = env_indices[: args.max_envs]
    print(f"[INFO] 将绘制的 env 实例索引：{env_indices}")

    with h5py.File(h5_path, "r") as f:
        for i in env_indices:
            rewards_TS = f[str(i)]["rewards"][()]  # (T, S)
            T, S = rewards_TS.shape
            horizon = cfg["horizon"]

            returns_SE, E, use_T, S0 = returns_first_n_streams_dataset_style(
                rewards_TS, horizon, args.source_timesteps, args.n_stream
            )

            sel_idx, metric, probs_sorted, sorted_index_lst = (
                select_streams_like_dataset(
                    returns_SE=returns_SE,
                    max_reward=cfg["max_reward"],
                    n_select=min(args.n_stream, S0),
                    stability_coeff=args.stability_coeff,
                    dataset_type=args.dataset_type,
                )
            )

            sel_unique = np.unique(sel_idx)
            all_idx = np.arange(S0, dtype=int)
            not_sel_idx = np.setdiff1d(all_idx, sel_unique, assume_unique=True)

            print(
                f"Env {i}: selected={len(sel_idx)} (unique={sel_unique.size}), "
                f"not_selected={not_sel_idx.size}, candidates={S0}"
            )
            print(f"The selected streams (with duplicates) are {list(sel_idx)}")
            print(f"The unique selected streams are {list(sel_unique)}")
            print("----------------------------------------------")

            # 1) All candidates
            title_all = f"Env {i} — {cfg['env']} — All candidates (n={S0}, T={use_T}, episodes={E})"
            out_grid_all = os.path.join(
                out_dir_by_dataset,
                f"returns_all_candidates_{dataset_tag_safe}_env{i}.png",
            )
            plot_grid(returns_SE, all_idx, out_grid_all, title_all)

            # 2) Selected (with duplicates)
            title_sel = (
                f"Env {i} — {cfg['env']} — Selected "
                f"(selections={len(sel_idx)}, unique={sel_unique.size}, from {S0} candidates)"
            )
            out_grid_sel = os.path.join(
                out_dir_by_dataset, f"returns_selected_{dataset_tag_safe}_env{i}.png"
            )
            plot_grid(returns_SE, sel_idx, out_grid_sel, title_sel)

            # 3) Not selected
            title_not = f"Env {i} — {cfg['env']} — Not selected (n={not_sel_idx.size}, from {S0} candidates)"
            out_grid_not = os.path.join(
                out_dir_by_dataset,
                f"returns_not_selected_{dataset_tag_safe}_env{i}.png",
            )
            plot_grid(returns_SE, not_sel_idx, out_grid_not, title_not)

            # 4) Mean curves
            title_mean = (
                f"Env {i} — {cfg['env']} — Mean return: all vs selected vs not-selected"
            )
            out_mean3 = os.path.join(
                out_dir_by_dataset,
                f"mean_all_selected_not_{dataset_tag_safe}_env{i}.png",
            )
            plot_mean_three_groups(returns_SE, sel_idx, out_mean3, title_mean)

            print(
                f"[OK] env {i}: 保存：\n"
                f"  - {out_grid_all}\n"
                f"  - {out_grid_sel}\n"
                f"  - {out_grid_not}\n"
                f"  - {out_mean3}"
            )


if __name__ == "__main__":
    main()

```

## gridworld/plot_sampled_streams_softmax.py
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Softmax 版本的流采样可视化（与数据集实现完全一致）

- metric: stability + improvement
  stability = 1 + mean(negative diffs) / (max_reward - min(return_stream))
  improvement = (mean(return_stream) + (max-min))/2/max_reward
- 概率: 对按 metric 升序排列后的 metric 做 softmax(metric/T)，再线性缩放到 [0,1]
- 采样: 阈值扫描（从高到低），直到凑满 n_stream，允许重复
- 随机源: 使用 random.uniform，与数据集一致（并用 env_split_seed 初始化）

输出4张图：All/Selected/Not-selected 的网格，以及三组均值曲线对比
"""
import os, argparse, random
import numpy as np
import h5py
import matplotlib.pyplot as plt

from utils import get_config, get_traj_file_name  # 同目录 utils.py


# ---------- 与数据集风格一致的 returns/采样 ----------

def compute_use_T_like_dataset(T, horizon, source_timesteps):
    """
    数据集里直接传入 source_timesteps 并 reshape(E, horizon)，因此要求可被 horizon 整除。
    这里保持一致：如果 <0 则回退到 <=T 的最大整倍数（给可视化兜底）；否则必须整除。
    """
    if source_timesteps is None:
        raise ValueError("source_timesteps 不能为空；请显式设置。")
    if source_timesteps < 0:
        use_T = (T // horizon) * horizon
        print(
            f"[WARN] dataset 风格通常要求 source_timesteps>0 且可被 horizon 整除；"
            f"收到 {source_timesteps}，这里退回到 use_T={use_T}（与 horizon 对齐）。"
        )
        return use_T
    use_T = min(int(source_timesteps), T)
    if use_T % horizon != 0:
        raise ValueError(
            f"dataset 风格要求 source_timesteps 可被 horizon 整除；"
            f"当前 source_timesteps={source_timesteps}, horizon={horizon}."
        )
    return use_T


def returns_first_n_streams_dataset_style(rewards_TS, horizon, source_timesteps, n_stream):
    """
    按数据集逻辑：只看前 n_stream 条流，取前 use_T 个时间步，按 (E, horizon) reshape 后逐 episode 求和。
    返回：
      returns_SE: 形状 (S0, E)，S0=min(n_stream, S)
      E: episode 数
      use_T: 实际使用的时间步
      S0: 实际使用的流数
    """
    T, S = rewards_TS.shape
    S0 = min(int(n_stream), S)
    use_T = compute_use_T_like_dataset(T, horizon, source_timesteps)
    E = use_T // horizon
    ret_SE = rewards_TS[:use_T, :S0].reshape(E, horizon, S0).sum(axis=1).T  # (S0, E)
    return ret_SE, E, use_T, S0


def select_streams_softmax_dataset_style(
    returns_SE,
    max_reward,
    n_select,
    stability_coeff=1.0,
    temperature=0.125,   # 与数据集默认一致
):
    """
    完全对齐你的数据集实现：
    1) 升序排序 metric
    2) softmax(metric/T) -> 归一化 -> 线性缩放到 [0,1]
    3) 从高到低做阈值扫描（允许重复），直到抽满 n_select
    返回：sel_idx(含重复)、metric、probs_sorted(升序)、sorted_index_lst(升序索引)
    """
    S, E = returns_SE.shape
    metric_lst = []

    for j in range(S):
        r = returns_SE[j]                   # (E,)
        diff = np.diff(r)
        neg = diff[diff < 0]               # 严格小于0（与你的实现一致）

        denom = (max_reward - float(np.min(r)))  # 与数据集一致的分母
        # 保持原实现语义，不做额外数值防护
        stability = 1.0 + (np.mean(neg) if neg.size > 0 else 0.0) / denom
        improvement = (float(np.mean(r)) + (float(np.max(r)) - float(np.min(r)))) / 2.0 / max_reward

        metric = stability_coeff * stability + 1.0 * improvement
        metric_lst.append(float(metric))

    # 升序排序
    sorted_index_lst = sorted(range(S), key=lambda i: metric_lst[i])
    metric_array_sorted = np.array([metric_lst[i] for i in sorted_index_lst], dtype=float)

    # softmax(metric/T) 后线性缩放到 [0,1]
    softmax_vals = np.exp(metric_array_sorted / float(temperature))
    softmax_vals /= np.sum(softmax_vals)
    new_softmax_vals = softmax_vals - np.min(softmax_vals)
    if np.max(new_softmax_vals) > 0:
        probabilities_lst = new_softmax_vals / np.max(new_softmax_vals)
    else:
        probabilities_lst = np.zeros_like(new_softmax_vals)

    # 打印与数据集一致
    print(f"Scaled softmax probability (sorted from small to large) = {probabilities_lst}")

    # 阈值扫描（从高到低），允许重复，直到凑满 n_select
    temp_slice = []
    S_scan = min(S, n_select)  # 与原始结构保持一致
    while len(temp_slice) < n_select:
        for potato in reversed(range(S_scan)):  # 从高到低扫描
            selected_stream = sorted_index_lst[potato]
            selected_stream_prob = probabilities_lst[potato]
            if random.uniform(0, 1) <= selected_stream_prob:
                temp_slice.append(selected_stream)
            if len(temp_slice) >= n_select:
                break

    sel_idx = np.array(temp_slice[:n_select], dtype=np.int32)
    probs_sorted = probabilities_lst  # 升序概率
    return sel_idx, np.array(metric_lst, dtype=float), probs_sorted, sorted_index_lst


# ---------- 绘图 ----------

def plot_grid(returns_SE, selected_idx, out_path, title, cols=10):
    if len(selected_idx) == 0:
        print("[WARN] 没有可绘制的 stream，输出占位图：", title)
        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        ax.axis("off")
        ax.text(0.5, 0.5, "No streams in this group", ha="center", va="center")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(out_path, dpi=240)
        plt.close(fig)
        return

    selR = returns_SE[selected_idx]
    rows = int(np.ceil(len(selected_idx) / cols))
    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 2.1, rows * 1.7), sharex=True, sharey=True
    )
    axes = np.ravel(axes)
    for k, ax in enumerate(axes):
        if k < len(selected_idx):
            ax.plot(selR[k])
            ax.set_title(f"s{selected_idx[k]}", fontsize=8)
        else:
            ax.axis("off")
        ax.tick_params(labelsize=6)
    fig.suptitle(title, fontsize=12)
    fig.text(0.5, 0.02, "episode", ha="center")
    fig.text(0.02, 0.5, "return", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.03, 0.04, 1, 0.96])
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


def plot_mean_three_groups(returns_SE, selected_idx, out_path, title):
    all_mean = returns_SE.mean(axis=0)
    S0 = returns_SE.shape[0]
    sel_unique = np.unique(selected_idx) if len(selected_idx) > 0 else np.array([], dtype=int)
    not_sel = np.setdiff1d(np.arange(S0), sel_unique, assume_unique=True)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(all_mean, label=f"all candidates (n={S0})", linewidth=2)

    if sel_unique.size > 0:
        ax.plot(
            returns_SE[sel_unique].mean(axis=0),
            label=f"selected (n={sel_unique.size})",
            linewidth=2,
        )
    else:
        ax.plot(np.zeros_like(all_mean), label="selected (n=0)", linewidth=2)

    if not_sel.size > 0:
        ax.plot(
            returns_SE[not_sel].mean(axis=0),
            label=f"not selected (n={not_sel.size})",
            linewidth=2,
            linestyle="--",
        )
    else:
        ax.plot(
            np.zeros_like(all_mean),
            label="not selected (n=0)",
            linewidth=2,
            linestyle="--",
        )

    ax.set_xlabel("episode")
    ax.set_ylabel("return")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=240)
    plt.close(fig)


# ---------- env 选择：与数据集一致（先打乱再切分） ----------

def resolve_env_indices_dataset_style(cfg, which):
    if which in ("train", "test", "all"):
        if cfg["env"] == "darkroom":
            n_total = cfg["grid_size"] ** 2
        elif cfg["env"] == "darkroompermuted":
            n_total = 120
        elif cfg["env"] == "darkkeytodoor":
            n_total = round((cfg["grid_size"] ** 4) / 4)  # 若有自定义，可按需调整；darkroom 用不到
        else:
            raise ValueError("Unsupported env")

        idx = list(range(n_total))
        rng = random.Random(int(cfg.get("env_split_seed", 0)))
        rng.shuffle(idx)

        n_train = round(n_total * cfg["train_env_ratio"])
        if which == "train":
            return idx[:n_train]
        elif which == "test":
            return idx[n_train:]
        else:
            return idx
    return [int(x) for x in which.split(",") if x != ""]


# ---------- 工作目录自适应：为 include 相对路径兜底 ----------

def _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs):
    candidates = []
    for p in (env_cfg_abs, alg_cfg_abs):
        if p:
            d = os.path.abspath(os.path.join(os.path.dirname(p), "..", ".."))
            candidates.append(d)
    for root in candidates:
        if os.path.isdir(os.path.join(root, "cfg")) and os.path.isfile(
            os.path.join(root, "utils.py")
        ):
            try:
                os.chdir(root)
                print(f"[INFO] 工作目录已切换到 {root} 以适配 include 相对路径。")
            except Exception as e:
                print(f"[WARN] 切换工作目录到 {root} 失败：{e}")
            break


# ---------- 根据数据集目录名生成子目录标签 ----------

def compute_dataset_tag(traj_dir_abs, cfg):
    tag = os.path.basename(os.path.normpath(traj_dir_abs))
    if tag.lower() in {"", ".", "datasets", "dataset", "data", "trajectories", "traj"}:
        tag = get_traj_file_name(cfg)
    return tag


def sanitize_tag(tag: str) -> str:
    safe_chars = []
    for c in tag:
        if c.isalnum() or c in "._-":
            safe_chars.append(c)
        else:
            safe_chars.append("_")
    return "".join(safe_chars)


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-config", "-ec", required=True,
                    help="eg. cfg/env/darkroom.yaml 或 gridworld/cfg/env/darkroom.yaml")
    ap.add_argument("--alg-config", "-ac", required=True,
                    help="eg. cfg/alg/ppo_dr.yaml 或 gridworld/cfg/alg/ppo_dr.yaml")
    ap.add_argument("--traj-dir", "-t", default="./datasets")
    ap.add_argument("--out", "-o", default="./plots")
    ap.add_argument("--env-index", "-g", default="train",
                    help='绘制的 env 组索引："0" 或 "0,1,2" 或 "train"/"test"/"all"（与数据集一致）')
    ap.add_argument("--max-envs", type=int, default=1,
                    help="最多绘制多少个 env（在 --env-index 选出的列表里取前K个；0 表示不限制）。默认 1。")
    ap.add_argument("--n-stream", type=int, default=100,
                    help="每个 env 参与候选与采样的 stream 数（只看前 n_stream 条）")
    ap.add_argument("--source-timesteps", type=int, default=-1,
                    help="用于统计的时间步数；数据集风格要求 >0 且能被 horizon 整除；<0 将回退到不超过 T 的最大整倍数")
    ap.add_argument("--stability-coeff", type=float, default=1.0,
                    help="与数据集一致，默认 1.0")
    ap.add_argument("--temperature", type=float, default=0.125,
                    help="Softmax 温度（与数据集默认一致为 0.125）")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子；默认使用 config['env_split_seed']，以与数据集实现复现一致。")
    args = ap.parse_args()

    # 绝对路径
    env_cfg_abs = os.path.abspath(args.env_config)
    alg_cfg_abs = os.path.abspath(args.alg_config)
    traj_dir_abs = os.path.abspath(args.traj_dir)
    out_root_abs = os.path.abspath(args.out)

    # 适配 include 相对路径
    _maybe_chdir_for_includes(env_cfg_abs, alg_cfg_abs)

    # 读取配置
    cfg = get_config(env_cfg_abs)
    cfg.update(get_config(alg_cfg_abs))

    # 随机性（与数据集一致：使用 random 作为主要随机源）
    seed_to_use = args.seed if args.seed is not None else int(cfg.get("env_split_seed", 0))
    random.seed(seed_to_use)
    np.random.seed(seed_to_use % (2**32 - 1))

    # 数据 & 输出路径
    h5_path = os.path.join(traj_dir_abs, get_traj_file_name(cfg) + ".hdf5")
    dataset_tag = compute_dataset_tag(traj_dir_abs, cfg)
    dataset_tag_safe = sanitize_tag(dataset_tag)
    out_dir_by_dataset = os.path.join(out_root_abs, dataset_tag)
    os.makedirs(out_dir_by_dataset, exist_ok=True)
    print(f"[INFO] 输出目录：{out_dir_by_dataset}  （dataset_tag='{dataset_tag}', filename_tag='{dataset_tag_safe}'）")

    # env 选择（与数据集一致的打乱&切分方式）
    env_indices = resolve_env_indices_dataset_style(cfg, args.env_index)
    if args.max_envs and args.max_envs > 0:
        env_indices = env_indices[: args.max_envs]
    print(f"[INFO] 将绘制的 env 实例索引：{env_indices}")

    with h5py.File(h5_path, "r") as f:
        for i in env_indices:
            rewards_TS = f[str(i)]["rewards"][()]  # (T, S)
            T, S = rewards_TS.shape
            horizon = cfg["horizon"]

            returns_SE, E, use_T, S0 = returns_first_n_streams_dataset_style(
                rewards_TS, horizon, args.source_timesteps, args.n_stream
            )

            sel_idx, metric, probs_sorted, sorted_index_lst = select_streams_softmax_dataset_style(
                returns_SE=returns_SE,
                max_reward=cfg["max_reward"],
                n_select=min(args.n_stream, S0),
                stability_coeff=args.stability_coeff,
                temperature=args.temperature,
            )

            # 与数据集一致的打印
            print(f'Env {i}: there are {len(sel_idx)} streams randomly selected!')
            print(f'The selected streams are {list(sel_idx)}')
            print('----------------------------------------------')

            sel_unique = np.unique(sel_idx)
            all_idx = np.arange(S0, dtype=int)
            not_sel_idx = np.setdiff1d(all_idx, sel_unique, assume_unique=True)

            print(
                f"Env {i}: selected={len(sel_idx)} (unique={sel_unique.size}), "
                f"not_selected={not_sel_idx.size}, candidates={S0}"
            )

            # 1) All candidates
            title_all = f"Env {i} — {cfg['env']} — All candidates (n={S0}, T={use_T}, episodes={E})"
            out_grid_all = os.path.join(
                out_dir_by_dataset,
                f"returns_all_candidates_{dataset_tag_safe}_env{i}.png",
            )
            plot_grid(returns_SE, all_idx, out_grid_all, title_all)

            # 2) Selected (with duplicates)
            title_sel = (
                f"Env {i} — {cfg['env']} — Selected "
                f"(selections={len(sel_idx)}, unique={sel_unique.size}, from {S0} candidates)"
            )
            out_grid_sel = os.path.join(
                out_dir_by_dataset, f"returns_selected_{dataset_tag_safe}_env{i}.png"
            )
            plot_grid(returns_SE, sel_idx, out_grid_sel, title_sel)

            # 3) Not selected
            title_not = f"Env {i} — {cfg['env']} — Not selected (n={not_sel_idx.size}, from {S0} candidates)"
            out_grid_not = os.path.join(
                out_dir_by_dataset,
                f"returns_not_selected_{dataset_tag_safe}_env{i}.png",
            )
            plot_grid(returns_SE, not_sel_idx, out_grid_not, title_not)

            # 4) Mean curves
            title_mean = f"Env {i} — {cfg['env']} — Mean return: all vs selected vs not-selected"
            out_mean3 = os.path.join(
                out_dir_by_dataset,
                f"mean_all_selected_not_{dataset_tag_safe}_env{i}.png",
            )
            plot_mean_three_groups(returns_SE, sel_idx, out_mean3, title_mean)

            print(
                f"[OK] env {i}: 保存：\n"
                f"  - {out_grid_all}\n"
                f"  - {out_grid_sel}\n"
                f"  - {out_grid_not}\n"
                f"  - {out_mean3}"
            )


if __name__ == "__main__":
    main()

```

## gridworld/train.py
```python
import shutil
from datetime import datetime
from glob import glob
import os
import os.path as path
from modulefinder import ModuleFinder

import yaml
import argparse
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import ADDataset, DPTDataset, IDTDataset
from env import SAMPLE_ENVIRONMENT
from model import MODEL
from utils import get_config, get_data_loader, log_in_context, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from accelerate import Accelerator
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import make_env


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg-config",
        "-ac",
        required=False,
        default="./cfg/alg/ppo_dr.yaml",
        help="Algorithm config",
    )
    parser.add_argument(
        "--env-config",
        "-ec",
        required=False,
        default="./cfg/env/darkroom.yaml",
        help="Environment config",
    )
    parser.add_argument(
        "--model-config",
        "-mc",
        required=False,
        default="./cfg/model/ad_dr.yaml",
        help="Model config",
    )
    parser.add_argument(
        "--log-dir", "-l", required=False, default="./runs", help="Log directory"
    )
    parser.add_argument(
        "--traj-dir",
        "-t",
        required=False,
        default="./datasets",
        help="Trajectory directory",
    )
    parser.add_argument(
        "--no-backup",
        "-nb",
        required=False,
        default=False,
        help="Save code",
        action="store_true",
    )
    parser.add_argument("--override", "-o", default="")
    parser.add_argument(
        "--resume",
        required=False,
        default=False,
        help="Resume train",
        action="store_true",
    )
    parser.add_argument(
        "--mixed-precision",
        "-m",
        required=False,
        default="fp32",
        help="fp32 or fp16 or bf16",
    )
    parser.add_argument(
        "--disable-tqdm", "-d", required=False, default=False, action="store_true"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))
    config.update(get_config(args.model_config))

    # Override options
    for option in args.override.split("|"):
        if not option:
            continue
        address, value = option.split("=")
        keys = address.split(".")
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f"Warning: {address} is not defined in config file.")
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    if config["dynamics"]:
        log_dir = path.join(
            args.log_dir,
            f"{config['model']}-{config['env']}-dynamics{config['dynamics']}-str{config['dynamics_strength']}-seed{config['env_split_seed']}",
        )
    else:
        log_dir = path.join(
            args.log_dir,
            f"{config['model']}-{config['env']}-dynamics{config['dynamics']}-seed{config['env_split_seed']}",
        )

    writer = SummaryWriter(log_dir, flush_secs=15)

    # Prevent overwriting
    config["log_dir"] = log_dir
    config_save_path = path.join(config["log_dir"], "config.yaml")
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, "r") as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f"WARNING: {log_dir} already exists. Skipping...")
        exit(0)

    config["traj_dir"] = args.traj_dir
    config["mixed_precision"] = args.mixed_precision

    # Save config
    os.makedirs(config["log_dir"], mode=0o755, exist_ok=True)
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    print(f"Config saved to {config_save_path}")

    # Save code
    if not args.no_backup:
        code_dir = path.join(
            config["log_dir"], "code_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        mf = ModuleFinder([os.getcwd()])
        mf.run_script(__file__)
        for name, module in mf.modules.items():
            if module.__file__ is None:
                continue
            rel_path = path.relpath(module.__file__)
            new_path = path.join(code_dir, rel_path)
            new_dirname = path.dirname(new_path)
            os.makedirs(new_dirname, mode=0o750, exist_ok=True)
            shutil.copy2(rel_path, new_path)
        print(f"Code saved to {code_dir}")

    # Define accelerator
    if args.mixed_precision == "bf16" or args.mixed_precision == "fp16":
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
    elif args.mixed_precision == "fp32":
        accelerator = Accelerator(mixed_precision="no")
    else:
        raise ValueError(f"Unsupported mixed precision: {args.mixed_precision}")

    config["device"] = accelerator.device

    # Define model
    model_name = config["model"]
    model = MODEL[model_name](config)
    if config.get("use_lora", False):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA enabled: trainable params {trainable}/{total}")

    # Get datasets and dataloaders
    load_start_time = datetime.now()
    print(f"Data loading started at {load_start_time}")

    if config["model"] == "DPT":
        train_dataset = DPTDataset(
            config,
            args.traj_dir,
            "train",
            config["train_n_stream"],
            config["train_source_timesteps"],
        )
        test_dataset = DPTDataset(
            config, args.traj_dir, "test", 1, config["train_source_timesteps"]
        )
    elif config["model"] == "AD":
        train_dataset = ADDataset(
            config,
            args.traj_dir,
            "train",
            config["train_n_stream"],
            config["train_source_timesteps"],
        )
        test_dataset = ADDataset(
            config, args.traj_dir, "test", 1, config["train_source_timesteps"]
        )
    elif config["model"] == "IDT":
        train_dataset = IDTDataset(
            config,
            args.traj_dir,
            "train",
            config["train_n_stream"],
            config["train_source_timesteps"],
        )
        test_dataset = IDTDataset(
            config, args.traj_dir, "test", 1, config["train_source_timesteps"]
        )
    else:
        raise ValueError(f'Unsupported model: {config["model"]}')

    train_dataloader = get_data_loader(
        train_dataset,
        batch_size=config["train_batch_size"],
        config=config,
        shuffle=True,
    )
    train_dataloader = next_dataloader(train_dataloader)

    test_dataloader = get_data_loader(
        test_dataset, batch_size=config["test_batch_size"], config=config, shuffle=False
    )

    load_end_time = datetime.now()
    print()
    print(f"Data loading ended at {load_end_time}")
    print(f"Elapsed time: {load_end_time - load_start_time}")

    # Define optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer, config["num_warmup_steps"], config["train_timesteps"]
    )
    step = 0

    # Resume checkpoint
    if args.resume:
        ckpt_paths = sorted(glob(path.join(config["log_dir"], "ckpt-*.pt")))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_sched.load_state_dict(ckpt["lr_sched"])
            step = ckpt["step"]
            print(f"Checkpoint loaded from {ckpt_path}")

    # Define environments for evaluation
    env_name = config["env"]
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    train_env_args = train_env_args[:10]
    test_env_args = test_env_args[:10]
    env_args = train_env_args + test_env_args

    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in env_args])
    elif env_name == "darkroompermuted":
        envs = SubprocVecEnv([make_env(config, perm_idx=arg) for arg in env_args])
    elif env_name == "darkkeytodoor":
        envs = SubprocVecEnv(
            [make_env(config, key=arg[:2], goal=arg[2:]) for arg in env_args]
        )
    else:
        raise NotImplementedError(f"Environment {env_name} is not supported")

    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
        model, optimizer, train_dataloader, lr_sched
    )

    # Main training loop
    start_time = datetime.now()
    print(f"Training started at {start_time}")

    with tqdm(
        total=config["train_timesteps"],
        position=0,
        leave=True,
        disable=args.disable_tqdm,
    ) as pbar:
        pbar.update(step)

        while True:
            batch = next(train_dataloader)

            step += 1

            with accelerator.autocast():
                output = model(batch)

            if config["dynamics"]:
                loss = (
                    output["loss_action"]
                    + (output["loss_reward"] + output["loss_next_state"])
                    * config["dynamics_strength"]
                )
            else:
                loss = output["loss_action"]

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            pbar.set_postfix(loss=loss.item())

            if step % config["summary_interval"] == 0:

                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar(
                    "train/loss_action", output["loss_action"].item(), step
                )
                writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], step)
                writer.add_scalar("train/acc_action", output["acc_action"].item(), step)

                if config["dynamics"]:
                    writer.add_scalar(
                        "train/loss_reward", output["loss_reward"].item(), step
                    )
                    writer.add_scalar(
                        "train/acc_reward", output["acc_reward"].item(), step
                    )
                    writer.add_scalar(
                        "train/loss_next_state", output["loss_next_state"].item(), step
                    )
                    writer.add_scalar(
                        "train/acc_next_state", output["acc_next_state"].item(), step
                    )

            ############ Evaluation ############
            if step % config["eval_interval"] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f"Evaluating started at {eval_start_time}")

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_loss_reward = 0.0
                    test_acc_reward = 0.0
                    test_loss_next_state = 0.0
                    test_acc_next_state = 0.0
                    test_cnt = 0

                    for j, batch in enumerate(test_dataloader):
                        output = model(batch)
                        cnt = len(batch["states"])
                        test_loss_action += output["loss_action"].item() * cnt
                        test_acc_action += output["acc_action"].item() * cnt

                        if config["dynamics"]:
                            test_loss_reward += output["loss_reward"].item() * cnt
                            test_acc_reward += output["acc_reward"].item() * cnt
                            test_loss_next_state += (
                                output["loss_next_state"].item() * cnt
                            )
                            test_acc_next_state += output["acc_next_state"].item() * cnt

                        test_cnt += cnt

                writer.add_scalar("test/loss_action", test_loss_action / test_cnt, step)
                writer.add_scalar("test/acc_action", test_acc_action / test_cnt, step)

                if config["dynamics"]:
                    writer.add_scalar(
                        "test/loss_reward", test_loss_reward / test_cnt, step
                    )
                    writer.add_scalar(
                        "test/acc_reward", test_acc_reward / test_cnt, step
                    )
                    writer.add_scalar(
                        "test/loss_next_state", test_loss_next_state / test_cnt, step
                    )
                    writer.add_scalar(
                        "test/acc_next_state", test_acc_next_state / test_cnt, step
                    )

                eval_end_time = datetime.now()
                print()
                print(f"Evaluating ended at {eval_end_time}")
                print(f"Elapsed time: {eval_end_time - eval_start_time}")
                model.train()
                torch.cuda.empty_cache()
            ####################################

            ############ Generation ############
            if step % config["gen_interval"] == 0:
                model.eval()
                gen_start_time = datetime.now()
                print(f"Generation started at {gen_start_time}")

                with torch.no_grad():
                    output = model.evaluate_in_context(
                        envs, config["train_source_timesteps"]
                    )

                    train_rewards = output["reward_episode"][: len(train_env_args)]
                    test_rewards = output["reward_episode"][len(train_env_args) :]

                    log_in_context(
                        values=train_rewards,
                        max_reward=config["max_reward"],
                        success=None,
                        episode_length=config["horizon"],
                        tag="train_gen/reward_episode",
                        title="",
                        xlabel="In-context steps",
                        ylabel="Reward",
                        step=step,
                        writer=writer,
                    )

                    log_in_context(
                        values=test_rewards,
                        max_reward=config["max_reward"],
                        success=None,
                        episode_length=config["horizon"],
                        tag="test_gen/reward_episode",
                        title="",
                        xlabel="In-context steps",
                        ylabel="Reward",
                        step=step,
                        writer=writer,
                    )

                gen_end_time = datetime.now()
                print()
                print(f"Generation ended at {gen_end_time}")
                print(f"Elapsed time: {gen_end_time - gen_start_time}")
                model.train()
                torch.cuda.empty_cache()
            ####################################

            pbar.update(1)

            # LOGGING
            if step % config["ckpt_interval"] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob(path.join(config["log_dir"], "ckpt-*.pt")))
                for ckpt_path in ckpt_paths:
                    os.remove(ckpt_path)

                new_ckpt_path = path.join(config["log_dir"], f"ckpt-{step}.pt")

                torch.save(
                    {
                        "step": step,
                        "config": config,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_sched": lr_sched.state_dict(),
                    },
                    new_ckpt_path,
                )
                print(f"\nCheckpoint saved to {new_ckpt_path}")

            if step >= config["train_timesteps"]:
                break

    writer.flush()
    envs.close()

    end_time = datetime.now()
    print()
    print(f"Training ended at {end_time}")
    print(f"Elapsed time: {end_time - start_time}")

```

## gridworld/utils.py
```python
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

from env import map_dark_states
from functools import partial
import matplotlib.pyplot as plt


def get_traj_file_name(config):
    if config["env"] == 'metaworld':
        task = config['task']
    else:
        task = config['env']

    path = f'history_{task}_{config["alg"]}_alg-seed{config["alg_seed"]}'

    return path


def get_config(config_path):
    with open(config_path, 'r') as f:
        new_config = yaml.full_load(f)
    config = {}
    if 'include' in new_config:
        include_config = get_config(new_config['include'])
        config.update(include_config)
        del new_config['include']
    config.update(new_config)
    return config


def ad_collate_fn(batch, grid_size):
    res = {}
    res['query_states'] = torch.tensor(np.array([item['query_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['target_actions'] = torch.tensor(np.array([item['target_actions'] for item in batch]), requires_grad=False, dtype=torch.long)
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = F.one_hot(torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.long), num_classes=5)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    if 'target_next_states' in batch[0].keys():
        res['target_next_states'] = map_dark_states(torch.tensor(np.array([item['target_next_states'] for item in batch]), dtype=torch.long, requires_grad=False), grid_size=grid_size)
        res['target_rewards'] = torch.tensor(np.array([item['target_rewards'] for item in batch]), dtype=torch.long, requires_grad=False)
        
    return res


def dpt_collate_fn(batch, grid_size):
    res = {}
    res['query_states'] = torch.tensor(np.array([item['query_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['target_actions'] = torch.tensor(np.array([item['target_actions'] for item in batch]), requires_grad=False, dtype=torch.long)
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = F.one_hot(torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.long), num_classes=5)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    if 'target_next_states' in batch[0].keys():
        res['query_actions'] = torch.tensor(np.array([item['query_actions'] for item in batch]), requires_grad=False, dtype=torch.long)
        res['target_next_states'] = map_dark_states(torch.tensor(np.array([item['target_next_states'] for item in batch]), dtype=torch.long, requires_grad=False), grid_size=grid_size)
        res['target_rewards'] = torch.tensor(np.array([item['target_rewards'] for item in batch]), dtype=torch.long, requires_grad=False)
        
    return res


def idt_collate_fn(batch):
    res = {}
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False)
    res['actions'] = torch.tensor(np.array([item['actions'] for item in batch]), dtype=torch.long, requires_grad=False)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.long, requires_grad=False)
    res['return_to_go'] = torch.tensor(np.array([item['return_to_go'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False)

    return res


def get_data_loader(dataset, batch_size, config, shuffle=True):
    if config['model'] == 'AD':
        collate_fn = partial(ad_collate_fn, grid_size=config['grid_size'])
    elif config['model'] == 'DPT':
        collate_fn = partial(dpt_collate_fn, grid_size=config['grid_size'])
    elif config['model'] == 'IDT':
        collate_fn = idt_collate_fn

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn, num_workers=config['num_workers'], pin_memory=True, persistent_workers=True)
    

def log_in_context(values: np.ndarray, max_reward: int, episode_length: int, tag: str, title: str, xlabel: str, ylabel: str, step: int, success=None, writer=None) -> None:
    steps = np.arange(1, len(values[0])+1) * episode_length
    mean_value = values.mean(axis=0)
    
    plt.plot(steps, mean_value)
    
    if success is not None:
        success_rate = success.astype(np.float32).mean(axis=0)

        for i, (xi, yi) in enumerate(zip(steps, mean_value)):
            if (i+1) % 10 == 0:
                plt.annotate(f'{success_rate[i]:.2f}', (xi, yi))
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.ylim(-max_reward * 0.05, max_reward * 1.05)
    writer.add_figure(f'{tag}/mean', plt.gcf(), global_step=step)
    plt.close()
            
            
def next_dataloader(dataloader: DataLoader):
    """
    Makes the dataloader never end when the dataset is exhausted.
    This is done to remove the notion of an 'epoch' and to count only the amount
    of training steps.
    """
    while True:
        for batch in dataloader:
            yield batch
```

