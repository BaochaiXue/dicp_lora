# All Code from metaworld

## metaworld/alg/__init__.py
```python
from .ppo import PPOWrapper
from .sac import SACWrapper
from .utils import HistoryLoggerCallback

ALGORITHM = {
    'PPO': PPOWrapper,
    'SAC': SACWrapper,
}```

## metaworld/alg/ppo.py
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
                                         tensorboard_log=log_dir)```

## metaworld/alg/sac.py
```python
import torch
from stable_baselines3 import SAC


class SACWrapper(SAC):
    def __init__(self, config, env, seed, log_dir):
        policy = config['policy']
        lr = config['source_lr']
        buffer_size = config['buffer_size']
        learning_starts = config['learning_starts']
        batch_size = config['batch_size']
        train_freq = config['train_freq']
        gradient_steps = config['gradient_steps']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = env
        seed = seed

        super(SACWrapper, self).__init__(policy=policy,
                                         env=env,
                                         learning_rate=lr,
                                         buffer_size = buffer_size,
                                         learning_starts=learning_starts,
                                         batch_size=batch_size,
                                         train_freq=train_freq,
                                         gradient_steps=gradient_steps,
                                         verbose=0,
                                         seed=seed,
                                         device=device,
                                         tensorboard_log=log_dir)```

## metaworld/alg/utils.py
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
        self.already_success = []
        self.success = []

        self.history = history

        self.episode_rewards = []
        self.episode_success = []

    def _on_step(self) -> bool:
        self.states.append(self.locals["new_obs"][:, list(range(18, 36))])
        self.next_states.append(self.locals["new_obs"][:, list(range(18))])
        
        success = [info['success'] for info in self.locals['infos']]
        self.success.append(success)
        self.episode_success.append(success)
        
        self.actions.append(self.locals["actions"])
        self.rewards.append(self.locals["rewards"].copy())
        self.dones.append(self.locals["dones"])
        
        self.episode_rewards.append(self.locals['rewards'])
        
        if self.locals['dones'][0]:
            mean_reward = np.mean(np.mean(self.episode_rewards, axis=0))
            self.logger.record('rollout/mean_reward', mean_reward)
            self.episode_rewards = []
            
            mean_success_rate = np.mean((np.sum(self.episode_success, axis=0) > 0.0))
            self.logger.record('rollout/mean_success_rate', mean_success_rate)
            self.episode_success = []
            
        return True

    def _on_training_end(self):
        if self.env_name == 'metaworld':
            self.history[self.env_idx] = {
                'states': np.array(self.states, dtype=np.float32),
                'actions': np.array(self.actions, dtype=np.float32),
                'rewards': np.array(self.rewards, dtype=np.float32),
                'next_states': np.array(self.next_states, dtype=np.float32),
                'dones': np.array(self.dones, dtype=np.bool_),
                'success': np.array(self.success, dtype=np.float32)
            }
        else:
            raise ValueError('Invalid environment')
```

## metaworld/collect_data.py
```python
import os
from datetime import datetime

import yaml

import sys
sys.path.append(os.path.dirname(sys.path[0]))

from alg import ALGORITHM, HistoryLoggerCallback
import argparse
import multiprocessing
from utils import get_config, get_traj_file_name
import h5py

from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers.time_limit import TimeLimit
import metaworld


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=True, help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=True, help="Environment config")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Directory for history saving")
    parser.add_argument('--override', '-o', default='')
    args = parser.parse_args()
    return args


def worker(config, env_cls, task_instance, traj_dir, env_idx, history, file_name):
    
    env = DummyVecEnv([make_env(config, env_cls, task_instance)] * config['n_stream'])
    
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


def make_env(config, env_cls, task):
    def _init():
            env = env_cls()
            env.set_task(task)
            return TimeLimit(env, max_episode_steps=config['horizon'])
    return _init


if __name__ == '__main__':
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))    

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
    
    task = config['task']
    
    ml1 = metaworld.ML1(env_name=task, seed=config['mw_seed'])
        
    file_name = get_traj_file_name(config)
    
    # Collcet train task histories
    name, env_cls = list(ml1.train_classes.items())[0]
    task_instances = ml1.train_tasks
    path = f'{args.traj_dir}/{task}/'
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    start_time = datetime.now()
    print(f'Collecting train task histories started at {start_time}')
    
    with h5py.File(os.path.join(path, f'{file_name}.hdf5'), 'a') as f:
        start_idx = 0
        
        while f'{start_idx}' in f.keys():
            start_idx += 1
            
        with multiprocessing.Manager() as manager:
            
            while start_idx < len(task_instances):
                history = manager.dict()

                instances = task_instances[start_idx:start_idx+config['n_process']]
                
                with multiprocessing.Pool(processes=config['n_process']) as pool:
                    pool.starmap(worker, [(config, env_cls, task_instance, path, start_idx+i, history, file_name) for i, task_instance in enumerate(instances)])

                # Save the history dictionary
                for i in range(start_idx, start_idx+len(instances)):
                    env_group = f.create_group(f'{i}')
                    for key, value in history[i].items():
                        env_group.create_dataset(key, data=value)
            
                start_idx += len(instances)

    end_time = datetime.now()
    print()
    print(f'Collecting train task histories ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')


    # Collcet test task histories
    name, env_cls = list(ml1.test_classes.items())[0]
    task_instances = ml1.test_tasks[:10]
    path = f'{args.traj_dir}/{task}/test/'
        
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    start_time = datetime.now()
    print(f'Collecting test task histories started at {start_time}')
    
    print()
    
    with h5py.File(f'{path}/{file_name}.hdf5', 'a') as f:
        start_idx = 0
        
        while f'{start_idx}' in f.keys():
            start_idx += 1
            
        with multiprocessing.Manager() as manager:
            
            while start_idx < len(task_instances):
                history = manager.dict()

                instances = task_instances[start_idx:start_idx+config['n_process']]
                
                with multiprocessing.Pool(processes=config['n_process']) as pool:
                    pool.starmap(worker, [(config, env_cls, task_instance, path, start_idx+i, history, file_name) for i, task_instance in enumerate(instances)])

                # Save the history dictionary
                for i in range(start_idx, start_idx+len(instances)):
                    env_group = f.create_group(f'{i}')
                    for key, value in history[i].items():
                        env_group.create_dataset(key, data=value)
            
                start_idx += len(instances)

    end_time = datetime.now()
    print()
    print(f'Collecting test task histories ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')```

## metaworld/dataset.py
```python
from torch.utils.data import Dataset
import numpy as np
from utils import get_traj_file_name
import h5py
from einops import rearrange, repeat



class ADDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_seed=None, n_stream=None, source_timesteps=None):
        self.env = config['env']
        self.n_transit = config['n_transit']
        self.config = config
        
        states = []
        actions = []
        rewards = []
        next_states = []
        
        if mode == 'train':
            file_path = f'{traj_dir}/{get_traj_file_name(config)}.hdf5'
        elif mode == 'test':
            file_path = f'{traj_dir}/test/{get_traj_file_name(config)}.hdf5'
        else:
            raise ValueError('Invalid mode')

        with h5py.File(file_path, 'r') as f:
            for i in range(n_seed):
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :11])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])                    
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :11])
                
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
        
        if self.config['dynamics']:
            traj.update({
                'target_next_states': self.next_states[history_idx, transition_idx + self.n_transit - 1],
                'target_rewards': self.rewards[history_idx, transition_idx + self.n_transit - 1]
            })

        return traj
    
    
class IDTDataset(Dataset):
    def __init__(self, config, traj_dir, mode='train', n_seed=50, n_stream=None, source_timesteps=None):
        self.config = config
        self.env = config['env']
        self.n_transit = config['n_transit']

        states = []
        actions = []
        rewards = []
        next_states = []
        
        if mode == 'train':
            file_path = f'{traj_dir}/{get_traj_file_name(config)}.hdf5'
        elif mode == 'test':
            file_path = f'{traj_dir}/test/{get_traj_file_name(config)}.hdf5'
        else:
            raise ValueError('Invalid mode')

        with h5py.File(file_path, 'r') as f:
            for i in range(n_seed):
                states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :config['dim_obs']])
                actions.append(f[f'{i}']['actions'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])                    
                rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
                next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :config['dim_obs']])
                
        self.states = np.concatenate(states, axis=0)
        self.actions = np.concatenate(actions, axis=0)
        self.rewards = np.concatenate(rewards, axis=0)
        self.next_states = np.concatenate(next_states, axis=0)
        self.returns_to_go = self.get_returns_to_go(self.rewards)
                
        self.sort_episodes()
        
        self.returns_to_go = self.relabel_returns_to_go(self.returns_to_go)

    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)

    def __getitem__(self, i):
        history_idx = i // (len(self.states[0]) - self.n_transit + 1)
        transition_idx = i % (len(self.states[0]) - self.n_transit + 1)
        
        traj = {
            'states': self.states[history_idx, transition_idx:transition_idx + self.n_transit],
            'actions': self.actions[history_idx, transition_idx:transition_idx + self.n_transit],
            'rewards': self.rewards[history_idx, transition_idx:transition_idx + self.n_transit],
            'returns_to_go': self.returns_to_go[history_idx, transition_idx:transition_idx + self.n_transit],
            'next_states': self.next_states[history_idx, transition_idx:transition_idx + self.n_transit],
        }

        return traj
    
    def get_returns_to_go(self, rewards):
        episode_rewards = rewards.reshape(-1, rewards.shape[1] // self.config['horizon'], self.config['horizon'])
        return np.flip(np.flip(episode_rewards, axis=-1).cumsum(axis=-1), axis=-1).reshape(-1, rewards.shape[1])
    
    def sort_episodes(self):
        returns_to_go = rearrange(self.returns_to_go, 'traj (epi time) -> traj epi time', time=self.config['horizon'])        
        sorted_episode_idx = np.argsort(returns_to_go[:, :, 0])
        sorted_episode_idx = repeat(sorted_episode_idx, 'traj epi -> traj epi time', time=self.config['horizon'])
        
        returns_to_go = np.take_along_axis(returns_to_go, sorted_episode_idx, axis=1)
        self.returns_to_go = rearrange(returns_to_go, 'traj epi time -> traj (epi time)')
        
        rewards = rearrange(self.rewards, 'traj (epi time) -> traj epi time', time=self.config['horizon'])
        rewards = np.take_along_axis(rewards, sorted_episode_idx, axis=1)
        self.rewards = rearrange(rewards, 'traj epi time -> traj (epi time)')
        
        actions = rearrange(self.actions, 'traj (epi time) dim -> traj epi time dim', time=self.config['horizon'])
        actions = np.take_along_axis(actions, 
                                     repeat(sorted_episode_idx, 'traj epi time -> traj epi time dim', dim=self.actions.shape[-1]),
                                     axis=1)
        self.actions = rearrange(actions, 'traj epi time dim -> traj (epi time) dim')
        
        sorted_episode_idx = repeat(sorted_episode_idx, 'traj epi time -> traj epi time dim', dim=self.states.shape[-1])
        
        states = rearrange(self.states, 'traj (epi time) dim -> traj epi time dim', time=self.config['horizon'])
        states = np.take_along_axis(states, sorted_episode_idx, axis=1)
        self.states = rearrange(states, 'traj epi time dim -> traj (epi time) dim')
        
        next_states = rearrange(self.next_states, 'traj (epi time) dim -> traj epi time dim', time=self.config['horizon'])
        next_states = np.take_along_axis(next_states, sorted_episode_idx, axis=1)
        self.next_states = rearrange(next_states, 'traj epi time dim -> traj (epi time) dim')
    
    def relabel_returns_to_go(self, rtg):
        max_episode_rtg = rtg.max(axis=-1) # (num_traj, )
        max_episode_rtg = repeat(max_episode_rtg, 'traj -> traj epi', epi=rtg.shape[1] // self.config['horizon'])
        
        episode_rtg = rtg.reshape(-1, rtg.shape[1] // self.config['horizon'], self.config['horizon'])
        
        episode_offset = max_episode_rtg - episode_rtg[:, :, 0]
        offset = repeat(episode_offset, 'traj epi -> traj epi time', time=self.config['horizon'])
        
        return (episode_rtg + offset).reshape(-1, rtg.shape[1])```

## metaworld/evaluate.py
```python
from datetime import datetime

import os.path as path

import argparse
from glob import glob

import torch

from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import metaworld
from gymnasium.wrappers.time_limit import TimeLimit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', '-c', required=True, help="Checkpoint directory")
    parser.add_argument('--sample-size', '-k', required=False, type=int, default=10, help="Sample size for planning")
    parser.add_argument('--seed', '-s', required=False, type=int, default=0, help="Torch seed")

    args = parser.parse_args()
    return args


def make_env(config, env_cls, task):
    def _init():
            env = env_cls()
            env.set_task(task)
            return TimeLimit(env, max_episode_steps=config['horizon'])
    return _init


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

    ml1 = metaworld.ML1(env_name=config['task'], seed=config['mw_seed'])
    
    test_envs = []

    for task_name, env_cls in ml1.test_classes.items():
        task_instances = [task for task in ml1.test_tasks if task.env_name == task_name]
        for i in range(50):
            test_envs.append(make_env(config, env_cls, task_instances[i]))
    
    envs = SubprocVecEnv(test_envs)
    model.set_obs_space(envs.observation_space)
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if config['task'] == 'reach-v2':
        eval_episodes = 100
    elif config['task'] == 'push-v2':
        eval_episodes = 300
    elif config['task'] == 'pick-place-v2' or config['task'] == 'peg-insert-side-v2':
        eval_episodes = 2000
    else:
        eval_episodes = 200
    
    start_time = datetime.now()
    print(f'Generation started at {start_time}')
    
    with torch.no_grad():
        if config['dynamics']:
            output = model.evaluate_in_context(vec_env=envs,
                                               eval_timesteps=eval_episodes * config['horizon'],
                                               sample_size=args.sample_size)
            
        else:
            output = model.evaluate_in_context(vec_env=envs,
                                               eval_timesteps=eval_episodes * config['horizon'])
            
    end_time = datetime.now()
    print()
    print(f'Generation ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    
    # Clean up
    envs.close()
    
    if config['dynamics']:
        path = path.join(args.ckpt_dir, f'eval_result_k{args.sample_size}')
    else:
        path = path.join(args.ckpt_dir, 'eval_result')
    
    reward_episode = output['reward_episode']
    success = output['success']
    
    with open(f'{path}_reward.npy', 'wb') as f:
        np.save(f, reward_episode)
    with open(f'{path}_success.npy', 'wb') as f:
        np.save(f, success)```

## metaworld/model/__init__.py
```python
from .ad import AD
from .idt import IDT

MODEL = {
    "AD": AD,
    "IDT": IDT,
}```

## metaworld/model/ad.py
```python
import torch
import torch.nn as nn
from .tiny_llama.model import Transformer
from einops import pack, rearrange, repeat
import numpy as np


class AD(torch.nn.Module):
    def __init__(self, config):
        super(AD, self).__init__()

        self.config = config
        self.device = config['device']
        self.n_transit = config['n_transit']
        self.max_seq_length = config['n_transit']
        self.mixed_precision = config['mixed_precision']

        if 'task' in config and (config['task'] == "hammer-v2" or config['task'] == "stick-push-v2" or config['task'] == "stick-pull-v2"):
            assert config['dim_obs'] == 18
            self.obs_dim_idx = list(range(18))
        else:
            assert config['dim_obs'] == 11
            self.obs_dim_idx = list(range(11))
        
        self.transformer = Transformer(config)
        
        self.embed_context = nn.Linear(config['dim_obs'] * 2 + config['dim_actions'] + 1, config['tf_n_embd'])
        self.embed_query_state = nn.Linear(config['dim_obs'], config['tf_n_embd'])
        
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        if config['learn_var']:
            self.pred_actions = nn.Linear(config['tf_n_embd'], 2 * config['dim_actions'])
            self.loss_fn_gaussian = nn.GaussianNLLLoss(full=True, reduction='mean')
        else:
            self.pred_actions = nn.Linear(config['tf_n_embd'], config['dim_actions'])
        
        if config['dynamics']:
            self.embed_query_action = nn.Linear(config['dim_actions'], config['tf_n_embd'])
            self.pred_rewards = nn.Linear(config['tf_n_embd'], 1)

            if config['learn_transition']:
                self.pred_next_states = nn.Linear(config['tf_n_embd'], config['dim_obs'])
        
    def forward(self, x):
        query_states = x['query_states'].to(self.device)  # (batch_size, dim_obs)
        target_actions = x['target_actions'].to(self.device)  # (batch_size, dim_actions)
        states = x['states'].to(self.device)  # (batch_size, n_transit-1, dim_obs)
        actions = x['actions'].to(self.device)  # (batch_size, n_transit-1, dim_actions)
        next_states = x['next_states'].to(self.device)  # (batch_size, n_transit-1, dim_obs)
        rewards = x['rewards'].to(self.device)  # (batch_size, n_transit-1)
        rewards = rearrange(rewards, 'b n -> b n 1')

        query_states_embed = self.embed_query_state(query_states)
        query_states_embed = rearrange(query_states_embed, 'b d -> b 1 d')
        
        context, _ = pack([states, actions, rewards, next_states], 'b n *')
        
        context_embed = self.embed_context(context)
        context_embed, _ = pack([context_embed, query_states_embed], 'b * d')
        
        if self.config['dynamics']:
            query_actions_embed = self.embed_query_action(target_actions)
            query_actions_embed = rearrange(query_actions_embed, 'b d -> b 1 d')
            context_embed, _ = pack([context_embed, query_actions_embed], 'b * d')
            
        transformer_output = self.transformer(context_embed,
                                              max_seq_length=self.max_seq_length,
                                              dtype=self.mixed_precision)

        result = {}
        
        if self.config['learn_var']:
            dist = self.pred_actions(transformer_output[:, self.n_transit-1])
            mean = dist[:, :self.config['dim_actions']]
            var = torch.exp(dist[:, self.config['dim_actions']:])
            result['loss_action'] = self.loss_fn_gaussian(mean, target_actions, var)
        else:            
            predicted_actions = self.pred_actions(transformer_output[:, self.n_transit-1])
            result['loss_action'] = self.loss_fn(predicted_actions, target_actions)
        
        if self.config['dynamics']:
            predicted_rewards = self.pred_rewards(transformer_output[:, -1])[:, 0].clip(0, 10)
            target_rewards = x['target_rewards'].to(self.device)  # (batch_size, )
            result['loss_reward'] = self.loss_fn(predicted_rewards, target_rewards)
            
            if self.config['learn_transition']:
                predicted_states = self.pred_next_states(transformer_output[:, -1]).clip(self.obs_low, self.obs_high)
                target_states = x['target_next_states'].to(self.device)  # (batch_size, dim_obs)
                result['loss_next_state'] = self.loss_fn(predicted_states, target_states)
            
        return result

    def evaluate_in_context(self, vec_env, eval_timesteps, sample_size=1, beam_start=50, sample=True):
        outputs = {}
        outputs['reward_episode'] = []
        outputs['success'] = []

        reward_episode = np.zeros(vec_env.num_envs)
        success = np.zeros(vec_env.num_envs)
        
        # Get inital states embeddings
        query_states = vec_env.reset()[:, self.obs_dim_idx]  # (n_envs, obs_dim)
        query_states = torch.tensor(query_states, device=self.device, requires_grad=False, dtype=torch.float)
        query_states = rearrange(query_states, 'e d -> e 1 d')
        query_states_embed = self.embed_query_state(query_states)
        transformer_input = query_states_embed
        
        for step in range(eval_timesteps):
            query_states_prev = query_states.clone().detach()

            position=step % self.config['horizon']
            if self.config['dynamics'] and sample_size > 1 and step >= self.n_transit and position > beam_start:
                actions = self.greedy_search(x=transformer_input.clone().detach(),
                                             sample_size=sample_size)
            else:
                output = self.transformer(transformer_input,
                                          max_seq_length=self.max_seq_length,
                                          dtype='fp32')
                
                if self.config['learn_var']:
                    dist = self.pred_actions(output[:, -1])
                    mean = dist[:, :self.config['dim_actions']]
                    std = torch.exp(dist[:, self.config['dim_actions']:] / 2)
                    actions = (std * torch.randn_like(mean) + mean)
                elif sample:
                    mean = self.pred_actions(output[:, -1])
                    std = torch.ones_like(mean)
                    actions = (std * torch.randn_like(mean) + mean)
                else:
                    actions = self.pred_actions(output[:, -1])
                        
            query_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())

            actions = rearrange(actions, 'e d -> e 1 d')

            reward_episode += rewards
            rewards = torch.tensor(rewards, device=self.device, requires_grad=False, dtype=torch.float)
            rewards = rearrange(rewards, 'e -> e 1 1')

            query_states = torch.tensor(query_states[:, self.obs_dim_idx], device=self.device, requires_grad=False, dtype=torch.float)
            query_states = rearrange(query_states, 'e d -> e 1 d')
            
            success += np.array([info['success'] for info in infos])
            
            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                outputs['success'].append(success > 0.0)
                success = np.zeros(vec_env.num_envs)
                
                states_next = torch.tensor(np.stack([info['terminal_observation'][self.obs_dim_idx] for info in infos]), device=self.device, dtype=torch.float)
                states_next = rearrange(states_next, 'e d -> e 1 d')
            else:
                states_next = query_states.clone().detach()
            
            query_states_embed = self.embed_query_state(query_states)

            context, _ = pack([query_states_prev, actions, rewards, states_next], 'e i *')
            context_embed = self.embed_context(context)
            
            if transformer_input.size(1) > 1:
                context_embed, _ = pack([transformer_input[:, :-1], context_embed], 'e * h')
                context_embed = context_embed[:, -(self.n_transit-1):]
                
            transformer_input, _ = pack([context_embed, query_states_embed], 'e * h')
            
        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        outputs['success'] = np.maximum.accumulate(np.stack(outputs['success'], axis=1), axis=-1)

        return outputs
    
    def greedy_search(self, x, sample_size=5):
        batch_size = x.size(0)
        
        output = self.transformer(x,
                                  max_seq_length=self.max_seq_length,
                                  dtype="fp32")
        
        if self.config['learn_var']:
            dist_actions = self.pred_actions(output[:, -1])
            mean_actions = dist_actions[:, :self.config['dim_actions']]
            std_actions = torch.exp(dist_actions[:, self.config['dim_actions']:] / 2)
        else:
            mean_actions = self.pred_actions(output[:, -1])
            std_actions = torch.ones_like(mean_actions)
        
        mean_actions = rearrange(mean_actions, 'b a -> b 1 a')
        std_actions = rearrange(std_actions, 'b a -> b 1 a')
        
        sampled_actions = torch.randn((batch_size, sample_size-1, self.config['dim_actions']), device=self.device) * std_actions + mean_actions
        sampled_actions, _ = pack([mean_actions, sampled_actions], 'b * a')  # (batch_size, sample_size, dim_actions), Use mean as one action sample
        
        # Query sampled actions
        embed_actions = self.embed_query_action(sampled_actions)  # (batch_size, sample_size, hidden)
        embed_actions = rearrange(embed_actions, 'b k h -> b k 1 h')
        x = repeat(x, 'b n h -> b k n h', k=sample_size)
        x, _ = pack([x, embed_actions], 'b k * h')
        
        output = self.transformer(rearrange(x, 'b k n h -> (b k) n h'),
                                  max_seq_length=self.max_seq_length,
                                  dtype="fp32")
        
        output = rearrange(output, '(b k) n h -> b k n h', k=sample_size)

        rewards = self.pred_rewards(output[:, :, -1]).clip(0, 10)
        rewards = rearrange(rewards, 'b k 1 -> b k')
        rewards, indicies = rewards.sort(dim=-1, descending=True)
        beam = torch.gather(sampled_actions, 1, repeat(indicies, 'b k -> b k a', a=sampled_actions.size(2)))
        beam = rearrange(beam, 'b k a -> b k 1 a')
        
        return beam[:, 0, 0]
    
    def set_obs_space(self, obs_space):
        self.obs_low = torch.tensor(obs_space.low[:self.config['dim_obs']], device=self.device, requires_grad=False, dtype=torch.float)
        self.obs_high = torch.tensor(obs_space.high[:self.config['dim_obs']], device=self.device, requires_grad=False, dtype=torch.float)
    
    def set_action_space(self, action_space):
        self.action_low = torch.tensor(action_space.low, device=self.device, requires_grad=False, dtype=torch.float)
        self.action_high = torch.tensor(action_space.high, device=self.device, requires_grad=False, dtype=torch.float)```

## metaworld/model/idt.py
```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .tiny_llama.model import Transformer
from einops import pack, rearrange, repeat


class IDT(nn.Module):
    def __init__(self, config):
        super(IDT, self).__init__()
        self.config = config
        self.device = config['device']
        self.low_per_high = config['low_per_high']
        self.n_transit = config['n_transit']
        
        if 'task' in config and (config['task'] == "hammer-v2" or config['task'] == "stick-push-v2" or config['task'] == "stick-pull-v2"):
            assert config['dim_obs'] == 18
            self.obs_dim_idx = list(range(18))
        else:
            assert config['dim_obs'] == 11
            self.obs_dim_idx = list(range(11))
        
        assert self.config['horizon'] % self.low_per_high == 0
        
        self.reviewing_decisions = ReviewingDecisions(config)
        self.h_decision_transformer = HDecisionTransformer(config)
        self.decisions_to_go = DecisionsToGo(config)
    
        self.loss_fn = nn.MSELoss(reduction='mean')
        
        if config['learn_var']:
            self.loss_fn_gaussian = nn.GaussianNLLLoss(full=True, reduction='mean')
        
    def forward(self, x):
        states = x['states'].to(self.device)
        actions = x['actions'].to(self.device)
        next_states = x['next_states'].to(self.device)
        rewards = x['rewards'].to(self.device)
        returns_to_go = x['returns_to_go'].to(self.device)
        
        states = rearrange(states, 'b (high low) d -> b high low d', low=self.low_per_high)
        actions = rearrange(actions, 'b (high low) d -> b high low d', low=self.low_per_high)
        next_states = rearrange(next_states, 'b (high low) d -> b high low d', low=self.low_per_high)
        rewards = rearrange(rewards, 'b (high low) -> b high low 1', low=self.low_per_high)
        returns_to_go = rearrange(returns_to_go, 'b (high low) -> b high low 1', low=self.low_per_high)
        
        z_dists = self.reviewing_decisions(states, actions, rewards, next_states) # (batch_size, n_high, 2 * dim_z)
        
        h_states = states[:, :, 0]
        h_returns_to_go = returns_to_go[:, :, 0]
        
        z_pred = self.h_decision_transformer(h_returns_to_go, h_states, z_dists)
        
        z = self.get_gaussian_sample(z_pred[:, :, :self.config['dim_z']], z_pred[:, :, self.config['dim_z']:])
                
        pred_states, pred_actions, pred_rewards = self.decisions_to_go(z, states, actions, rewards)
        
        output = {}
        
        if self.config['learn_var']:
            mean = pred_actions[:, :, :, :self.config['dim_actions']]
            var = torch.exp(pred_actions[:, :, :, self.config['dim_actions']:])
            
            output['loss_action'] = self.loss_fn_gaussian(mean, actions, var)
        else:
            output['loss_action'] = self.loss_fn(pred_actions, actions)
        
        if self.config['dynamics']:
            output['loss_reward'] = self.loss_fn(pred_rewards.clip(0, 10), rewards)

            if self.config['learn_transition']:
                output['loss_next_state'] = self.loss_fn(pred_states.clip(self.obs_low, self.obs_high), next_states)

        return output
    
    def evaluate_in_context(self, vec_env, eval_timesteps, sample_size=1, beam_start=50, sample=True):
        outputs = {}
        outputs['reward_episode'] = []
        outputs['success'] = []

        reward_episode = np.zeros(vec_env.num_envs)
        success_episode = np.zeros(vec_env.num_envs)
        
        # Get inital states embeddings
        states = vec_env.reset()[:, self.obs_dim_idx]
        states = torch.tensor(states, device=self.device, requires_grad=False, dtype=torch.float)
        states = rearrange(states, 'e d -> e 1 d')
                
        return_to_go = np.array([10 * self.config['horizon'] for _ in range(vec_env.num_envs)])
        return_to_go = torch.tensor(return_to_go, device=self.device, requires_grad=False, dtype=torch.float)
        return_to_go = rearrange(return_to_go, 'e -> e 1 1')
        
        z_dists = None
        
        step = 0
        while step < eval_timesteps:
            z_pred = self.h_decision_transformer(return_to_go, states, z_dists)[:, -1:] # (batch_size, 1, 2 * dim_z)
            z = self.get_gaussian_sample(z_pred[:, :, :self.config['dim_z']], z_pred[:, :, self.config['dim_z']:])[:, 0] # (batch_size, n_low, dim_z)
            
            reward_low, success_low, next_states, dones, history_states, history_actions, history_rewards, history_next_states = self.decisions_to_go.predict_actions(
                vec_env, z, states[:, -1:],
                sample_size=sample_size if step >= self.n_transit and step % self.config['horizon'] > beam_start else 1,
                sample=sample)
            
            reward_episode += reward_low
            success_episode += success_low
            
            # update z
            history_states = rearrange(history_states, 'e low d -> e 1 low d')
            history_actions = rearrange(history_actions, 'e low d -> e 1 low d')
            history_rewards = rearrange(history_rewards, 'e low d -> e 1 low d')
            history_next_states = rearrange(history_next_states, 'e low d -> e 1 low d')
            z_reviewed = self.reviewing_decisions(history_states, history_actions, history_rewards, history_next_states)
            if z_dists is None:
                z_dists = z_reviewed
            else:
                z_dists, _ = pack([z_dists, z_reviewed], 'b * d')
            
            # update state
            states, _ = pack([states, next_states], 'e * d')
            
            if states.size(1) > self.n_transit // self.low_per_high:
                states = states[:, 1:]
                return_to_go = return_to_go[:, 1:]
                z_dists = z_dists[:, 1:]
            
            if dones[0]:
                outputs['reward_episode'].append(reward_episode)
                reward_episode = np.zeros(vec_env.num_envs)
                
                outputs['success'].append(success_episode > 0.0)
                success_episode = np.zeros(vec_env.num_envs)
                
                next_return_to_go = np.array([10 * self.config['horizon'] for _ in range(vec_env.num_envs)])
                next_return_to_go = torch.tensor(next_return_to_go, device=self.device, requires_grad=False, dtype=torch.float)
                next_return_to_go = rearrange(next_return_to_go, 'e -> e 1 1')
            else:
                next_return_to_go = return_to_go[:, -1, 0] - torch.tensor(reward_low, device=self.device, dtype=torch.float)
                next_return_to_go = rearrange(next_return_to_go, 'e -> e 1 1')
            
            # update rtg
            return_to_go, _ = pack([return_to_go, next_return_to_go], 'e * d')
            
            step += self.low_per_high
        
        outputs['reward_episode'] = np.stack(outputs['reward_episode'], axis=1)
        outputs['success'] = np.maximum.accumulate(np.stack(outputs['success'], axis=1), axis=-1)
        
        return outputs
    
    def get_gaussian_sample(self, mean, logvar):
        std = logvar.div(2).exp()
        std = repeat(std, 'b high d -> b high low d', low=self.low_per_high)
        mean = repeat(mean, 'b high d -> b high low d', low=self.low_per_high)
        eps = torch.randn_like(mean)
        return eps * std + mean
    
    def set_obs_space(self, obs_space):
        self.obs_low = torch.tensor(obs_space.low[:self.config['dim_obs']], device=self.device, requires_grad=False, dtype=torch.float)
        self.obs_high = torch.tensor(obs_space.high[:self.config['dim_obs']], device=self.device, requires_grad=False, dtype=torch.float)
    
    def set_action_space(self, action_space):
        self.action_low = torch.tensor(action_space.low, device=self.device, requires_grad=False, dtype=torch.float)
        self.action_high = torch.tensor(action_space.high, device=self.device, requires_grad=False, dtype=torch.float)
    

class ReviewingDecisions(nn.Module):
    def __init__(self, config):
        super(ReviewingDecisions, self).__init__()
        self.config = config
        self.low_per_high = config['low_per_high']
        
        self.transformer = Transformer(config)

        self.embed_state = nn.Linear(config['dim_obs'], config['tf_n_embd'])
        self.embed_action = nn.Linear(config['dim_actions'], config['tf_n_embd'])
        self.embed_reward = nn.Linear(1, config['tf_n_embd'])
        self.embed_next_state = nn.Linear(config['dim_obs'], config['tf_n_embd'])

        self.predict_z = nn.Linear(config['tf_n_embd'], 2 * config['dim_z'])

    def forward(self, states, actions, rewards, next_states):                
        states_embed = self.embed_state(states)
        actions_embed = self.embed_action(actions)
        rewards_embed = self.embed_reward(rewards)
        next_states_embed = self.embed_next_state(next_states)
        
        transformer_input = states_embed + actions_embed + rewards_embed + next_states_embed
        
        transformer_output = self.transformer(rearrange(transformer_input, 'b high low h -> (b high) low h'),
                                              max_seq_length=self.low_per_high,
                                              dtype=self.config['mixed_precision'])
        
        transformer_output = rearrange(transformer_output, '(b high) low h -> b high low h', b=states.size(0))
        z_dists = self.predict_z(transformer_output[:, :, -1])

        return z_dists


class HDecisionTransformer(nn.Module):
    def __init__(self, config):
        super(HDecisionTransformer, self).__init__()
        self.config = config
        self.transformer = Transformer(config)

        self.embed_return = nn.Linear(1, config['tf_n_embd'])
        self.embed_state = nn.Linear(config['dim_obs'], config['tf_n_embd'])
        self.embed_z = nn.Linear(2 * config['dim_z'], config['tf_n_embd'])

        self.pred_z_dist = nn.Linear(config['tf_n_embd'], 2 * config['dim_z'])

    def forward(self, returns_to_go, states, z_dists=None):
        if z_dists is None:
            z_dists = torch.zeros(states.size(0), 1, 2 * self.config['dim_z'], device=states.device)
        elif z_dists.size(1) < returns_to_go.size(1):
            z_dists = F.pad(z_dists, (0, 0, 0, 1))
            
        returns_embed = self.embed_return(returns_to_go)
        states_embed = self.embed_state(states)
        z_embed = self.embed_z(z_dists)
        
        returns_embed = rearrange(returns_embed, 'b n h -> b n 1 h')
        states_embed = rearrange(states_embed, 'b n h -> b n 1 h')
        z_embed = rearrange(z_embed, 'b n h -> b n 1 h')
        
        transformer_input, _ = pack([returns_embed, states_embed, z_embed], 'b n * h')

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

        if 'task' in config and (config['task'] == "hammer-v2" or config['task'] == "stick-push-v2" or config['task'] == "stick-pull-v2"):
            self.obs_dim_idx = list(range(18))
        else:
            self.obs_dim_idx = list(range(11))

        self.transformer = Transformer(config)

        self.embed_z = nn.Linear(config['dim_z'], config['tf_n_embd'])

        self.embed_state = nn.Linear(config['dim_obs'], config['tf_n_embd'])
        self.embed_action = nn.Linear(config['dim_actions'], config['tf_n_embd'])
        self.embed_reward = nn.Linear(1, config['tf_n_embd'])

        if config['learn_var']:
            self.pred_action = nn.Linear(config['tf_n_embd'], 2 * config['dim_actions'])
        else:
            self.pred_action = nn.Linear(config['tf_n_embd'], config['dim_actions'])
        
        if config['dynamics']:
            self.pred_next_state = nn.Linear(config['tf_n_embd'], config['dim_obs'])
            self.pred_reward = nn.Linear(config['tf_n_embd'], 1)

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

        pred_actions = self.pred_action(transformer_output[:, :, :, 1])
        
        if self.config['dynamics']:
            pred_states = self.pred_next_state(transformer_output[:, :, :, 2])
            pred_rewards = self.pred_reward(transformer_output[:, :, :, 2])
        else:
            pred_states = None
            pred_rewards = None

        return pred_states, pred_actions, pred_rewards
    
    def predict_actions(self, vec_env, z, states, sample_size=1, sample=True):
        reward_low = np.zeros(vec_env.num_envs)
        success_low = np.zeros(vec_env.num_envs)

        z_embed = self.embed_z(z)
        states_embed = self.embed_state(states)
        
        assert z_embed.size(1) == self.low_per_high
        assert states_embed.size(1) == 1
        
        history_states = states
        
        transformer_input, _ = pack([z_embed[:, :1], states_embed], 'e * h')
        
        for step in range(self.low_per_high):
            
            if self.config['dynamics'] and sample_size > 1:
                actions = self.greedy_search(x=transformer_input.clone().detach(),
                                             z_embed=z_embed.clone().detach(),
                                             sample_size=sample_size)
            else:
                transformer_output = self.transformer(transformer_input,
                                                      max_seq_length=4*self.config['low_per_high'],
                                                      dtype='fp32')
                
                
                if self.config['learn_var']:
                    dist = self.pred_action(transformer_output[:, -1]) # (batch_size, dim_actions)
                    mean = dist[:, :self.config['dim_actions']]
                    std = torch.exp(dist[:, self.config['dim_actions']:] / 2)
                    actions = (std * torch.randn_like(mean) + mean)
                elif sample:
                    mean = self.pred_action(transformer_output[:, -1])
                    std = torch.ones_like(mean)
                    actions = (std * torch.randn_like(mean) + mean)
                else:
                    actions = self.pred_action(transformer_output[:, -1])
            
            next_states, rewards, dones, infos = vec_env.step(actions.cpu().numpy())
            
            actions_embed = self.embed_action(actions)
            
            reward_low += rewards
            rewards = torch.tensor(rewards, device=self.config['device'], requires_grad=False, dtype=torch.float)
            rewards = rearrange(rewards, 'e -> e 1 1')
            rewards_embed = self.embed_reward(rewards)
            
            next_states = torch.tensor(next_states[:, self.obs_dim_idx], device=self.config['device'], requires_grad=False, dtype=torch.float)
            next_states = rearrange(next_states, 'e d -> e 1 d')
            next_states_embed = self.embed_state(next_states)
            
            success_low += np.array([info['success'] for info in infos])
            
            if step < self.low_per_high - 1:
                history_states, _ = pack([history_states, next_states], 'e * d')
                
            if step == 0:
                history_actions = actions
                history_rewards = rewards
                history_next_states = next_states
            else:
                history_actions, _ = pack([history_actions, actions], 'e * d')
                history_rewards, _ = pack([history_rewards, rewards], 'e * d')
                history_next_states, _ = pack([history_next_states, next_states], 'e * d')
            
            if step < self.low_per_high - 1:
                transformer_input, _ = pack([transformer_input, actions_embed, rewards_embed, z_embed[:, step+1:step+2], next_states_embed], 'e * h')
            
        return reward_low, success_low, next_states, dones, history_states, history_actions, history_rewards, history_next_states
    
    def greedy_search(self, x, z_embed, sample_size):
        output = self.transformer(x,
                                  max_seq_length=4*self.config['low_per_high'],
                                  dtype='fp32') 
            
        if self.config['learn_var']:
            dist_actions = self.pred_action(output[:, -1])
            mean_actions = dist_actions[:, :self.config['dim_actions']]
            std_actions = torch.exp(dist_actions[:, self.config['dim_actions']:] / 2)
        else:
            mean_actions = self.pred_action(output[:, -1])
            std_actions = torch.ones_like(mean_actions)
        
        mean_actions = rearrange(mean_actions, 'b a -> b 1 a')
        std_actions = rearrange(std_actions, 'b a -> b 1 a')
        
        sampled_actions = torch.randn((x.size(0), sample_size-1, self.config['dim_actions']), device=x.device) * std_actions + mean_actions
        sampled_actions, _ = pack([mean_actions, sampled_actions], 'b * a')  # (batch_size, sample_size, dim_actions), Use mean as one action sample
        
        # Query sampled actions
        embed_actions = self.embed_action(sampled_actions)  # (batch_size, sample_size, hidden)
        embed_actions = rearrange(embed_actions, 'b k h -> b k 1 h')
        x = repeat(x, 'b n h -> b k n h', k=sample_size)
        x, _ = pack([x, embed_actions], 'b k * h')

        output = self.transformer(rearrange(x, 'b k n h -> (b k) n h'),
                                  max_seq_length=4*self.config['low_per_high'],
                                  dtype="fp32")
        
        output = rearrange(output, '(b k) n h -> b k n h', k=sample_size)
        
        # Predict rewards
        rewards = self.pred_reward(output[:, :, -1]).clip(0, 10)

        # Sort actions according to rewards
        cum_rewards = rearrange(rewards, 'b k 1 -> b k')
        cum_rewards, indicies = cum_rewards.sort(dim=-1, descending=True)
        beam = torch.gather(sampled_actions, 1, repeat(indicies, 'b k -> b k a', a=sampled_actions.size(2)))
        beam = rearrange(beam, 'b k a -> b k 1 a')
        
        return beam[:, 0, 0]```

## metaworld/model/tiny_llama/__init__.py
```python
```

## metaworld/model/tiny_llama/config.py
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

## metaworld/model/tiny_llama/fused_rotary_embedding.py
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

## metaworld/model/tiny_llama/model.py
```python
import math
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
# from flash_attn import flash_attn_func
from lightning_utilities.core.imports import RequirementCache
from xformers.ops import SwiGLU

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

## metaworld/model/tiny_llama/rmsnorm.py
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

## metaworld/model/tiny_llama/utils.py
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
    return "16-mixed" if training else "16-true"```

## metaworld/train.py
```python
import shutil
from datetime import datetime
from glob import glob
import os
import os.path as path
import sys
sys.path.append(path.dirname(sys.path[0]))
from modulefinder import ModuleFinder

import yaml
import argparse
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import ADDataset, IDTDataset
from model import MODEL
from utils import get_config, get_data_loader, log_in_context, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from accelerate import Accelerator
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers.time_limit import TimeLimit
import metaworld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=False, default='./cfg/alg/ppo_ml1.yaml', help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=False, default='./cfg/env/ml1.yaml', help="Environment config")
    parser.add_argument('--model-config', '-mc', required=False, default='./cfg/model/idt_ml1.yaml', help="Model config")
    parser.add_argument('--log-dir', '-l', required=False, default='./runs', help="Log directory")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Trajectory directory")
    parser.add_argument('--no-backup', '-nb', required=False, default=False, help="Save code", action='store_true')
    parser.add_argument('--override', '-o', default='')
    parser.add_argument('--resume', required=False, default=False, help="Resume train", action='store_true')
    parser.add_argument('--mixed-precision', '-m', required=False, default='fp32')
    parser.add_argument('--disable-tqdm', '-d', required=False, default=False, action='store_true')

    args = parser.parse_args()
    return args


def make_env(config, env_cls, task):
    def _init():
            env = env_cls()
            env.set_task(task)
            return TimeLimit(env, max_episode_steps=config['horizon'])
    return _init


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))
    config.update(get_config(args.model_config))

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

    log_dir = path.join(args.log_dir, f"{config['model']}-ml1-{config['task']}-dynamics{config['dynamics']}-var{config['learn_var']}")
    writer = SummaryWriter(log_dir, flush_secs=15)
    
    # Prevent overwriting
    config['log_dir'] = log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f'WARNING: {log_dir} already exists. Skipping...')
        exit(0)

    traj_dir = path.join(args.traj_dir, config['task'])    
    config['traj_dir'] = traj_dir
    config['device'] = device
    config['mixed_precision'] = args.mixed_precision

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Config saved to {config_save_path}')

    # Save code
    if not args.no_backup:
        code_dir = path.join(config['log_dir'], 'code_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
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
        print(f'Code saved to {code_dir}')
    
    # Define model
    model_name = config['model']
    model = MODEL[model_name](config).to(device)

    # Get datasets and dataloaders
    load_start_time = datetime.now()
    print(f'Data loading started at {load_start_time}')
    
    if config['model'] == 'AD':
        train_dataset = ADDataset(config, traj_dir, 'train', config['train_n_seed'], config['train_n_stream'], config['train_source_timesteps'])
        test_dataset = ADDataset(config, traj_dir, 'test', 1, 1, config['train_source_timesteps'])
    elif config['model'] == 'IDT':
        train_dataset = IDTDataset(config, traj_dir, 'train', config['train_n_seed'], config['train_n_stream'], config['train_source_timesteps'])
        test_dataset = IDTDataset(config, traj_dir, 'test', 1, 1, config['train_source_timesteps'])
    else:
        raise ValueError(f'Unsupported model: {config["model"]}')
    
    train_dataloader = get_data_loader(train_dataset, batch_size=config['train_batch_size'], config=config, shuffle=True)
    train_dataloader = next_dataloader(train_dataloader)
    test_dataloader = get_data_loader(test_dataset, batch_size=config['test_batch_size'], config=config, shuffle=False)

    load_end_time = datetime.now()
    print()
    print(f'Data loading ended at {load_end_time}')
    print(f'Elapsed time: {load_end_time - load_start_time}')

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    lr_sched = get_cosine_schedule_with_warmup(optimizer, config['num_warmup_steps'], config['train_timesteps'])
    step = 0
    
    # Resume checkpoint
    if args.resume:
        ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_sched.load_state_dict(ckpt['lr_sched'])
            step = ckpt['step']
            print(f'Checkpoint loaded from {ckpt_path}')
    
    # Define environments for evaluation
    ml1 = metaworld.ML1(env_name=config['task'], seed=config['mw_seed'])
    
    train_envs = []
    test_envs = []
    
    for task_name, env_cls in ml1.train_classes.items():
        task_instances = [task for task in ml1.train_tasks if task.env_name == task_name]
        for i in range(config['n_train_envs_per_task']):
            train_envs.append(make_env(config, env_cls, task_instances[i]))
    
    for task_name, env_cls in ml1.test_classes.items():
        task_instances = [task for task in ml1.test_tasks if task.env_name == task_name]
        for i in range(config['n_test_envs_per_task']):
            test_envs.append(make_env(config, env_cls, task_instances[i]))

    envs = train_envs + test_envs

    envs = SubprocVecEnv(envs)
    model.set_obs_space(envs.observation_space)
    model.set_action_space(envs.action_space)
    
    # Wrap everything into an accelerator
    if args.mixed_precision == 'bf16' or args.mixed_precision == 'fp16':
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
    elif args.mixed_precision == 'fp32':
        accelerator = Accelerator(mixed_precision='no')
    else:
        raise ValueError(f'Unsupported mixed precision: {args.mixed_precision}')
    
    # Main training loop
    start_time = datetime.now()
    print(f'Training started at {start_time}')
    
    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=args.disable_tqdm) as pbar:
        pbar.update(step)

        while True:
            with accelerator.autocast():
                batch = next(train_dataloader)
            
            step += 1
            
            with accelerator.autocast():
                output = model(batch)

            if config['dynamics']:
                if config['learn_transition']:
                    loss = output['loss_action'] + (output['loss_reward'] + output['loss_next_state']) * config['dynamics_strength']
                else:
                    loss = output['loss_action'] + output['loss_reward'] * config['dynamics_strength']
            else:
                loss = output['loss_action']

            optimizer.zero_grad()            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            pbar.set_postfix(loss=loss.item())

            if step % config['summary_interval'] == 0:

                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'].item(), step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)

                if config['dynamics']:
                    writer.add_scalar('train/loss_reward', output['loss_reward'].item(), step)
                    if config['learn_transition']:
                        writer.add_scalar('train/loss_next_state', output['loss_next_state'].item(), step)

            ############ Evaluation ############
            if step % config['eval_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f'Evaluating started at {eval_start_time}')

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_loss_reward = 0.0
                    test_loss_next_state = 0.0
                    test_cnt = 0

                    for j, batch in enumerate(test_dataloader):
                        with accelerator.autocast():
                            output = model(batch)
                        cnt = len(batch['states'])
                        test_loss_action += output['loss_action'].item() * cnt
                 
                        if config['dynamics']:
                            test_loss_reward += output['loss_reward'].item() * cnt
                            if config['learn_transition']:
                                test_loss_next_state += output['loss_next_state'].item() * cnt

                        test_cnt += cnt

                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)

                if config['dynamics']:
                    writer.add_scalar('test/loss_reward', test_loss_reward / test_cnt, step)
                    if config['learn_transition']:
                        writer.add_scalar('test/loss_next_state', test_loss_next_state / test_cnt, step)


                eval_end_time = datetime.now()
                print()
                print(f'Evaluating ended at {eval_end_time}')
                print(f'Elapsed time: {eval_end_time - eval_start_time}')
                model.train()
                torch.cuda.empty_cache()
            ####################################

            ############ Generation ############
            if step % config['gen_interval'] == 0:
                model.eval()
                gen_start_time = datetime.now()
                print(f'Generation started at {gen_start_time}')

                with torch.no_grad():
                    output = model.evaluate_in_context(envs, config['test_source_timesteps'])
                    
                    train_rewards = output['reward_episode'][:len(train_envs)]
                    test_rewards = output['reward_episode'][len(train_envs):]
                    
                    if 'success' in output.keys():
                        train_success = output['success'][:len(train_envs)]
                        test_success = output['success'][len(train_envs):]
                        
                        writer.add_scalar('train/success_rate', train_success.max(axis=1).mean(), step)
                        writer.add_scalar('test/success_rate', test_success.max(axis=1).mean(), step)
                        
                    else:
                        train_success = None
                        test_success = None
                    

                    log_in_context(values=train_rewards,
                                   max_reward=config['max_reward'],
                                   success=train_success,
                                   episode_length = config['horizon'],
                                   tag='train_gen/reward_episode',
                                   title='',
                                   xlabel='In-context steps',
                                   ylabel='Reward',
                                   step=step,
                                   writer=writer)

                    log_in_context(values=test_rewards,
                                   max_reward=config['max_reward'],
                                   success=test_success,
                                   episode_length = config['horizon'],
                                   tag='test_gen/reward_episode',
                                   title='',
                                   xlabel='In-context steps',
                                   ylabel='Reward',
                                   step=step,
                                   writer=writer)

                gen_end_time = datetime.now()
                print()
                print(f'Generation ended at {gen_end_time}')
                print(f'Elapsed time: {gen_end_time - gen_start_time}')
                model.train()
                torch.cuda.empty_cache()
            ####################################

            pbar.update(1)

            # LOGGING
            if step % config['ckpt_interval'] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                for ckpt_path in ckpt_paths:
                    os.remove(ckpt_path)

                new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')

                torch.save({
                    'step': step,
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')


            if step >= config['train_timesteps']:
                break

    writer.flush()
    envs.close()

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')```

## metaworld/utils.py
```python
import yaml
import torch
from torch.utils.data import DataLoader
import numpy as np
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


def ad_collate_fn(batch):
    res = {}
    res['query_states'] = torch.tensor(np.array([item['query_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['target_actions'] = torch.tensor(np.array([item['target_actions'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['actions'] = torch.tensor(np.array([item['actions'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), requires_grad=False, dtype=torch.float)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), requires_grad=False, dtype=torch.float)
    
    if 'target_next_states' in batch[0].keys():        
        res['target_next_states'] = torch.tensor(np.array([item['target_next_states'] for item in batch]), dtype=torch.float, requires_grad=False)
        res['target_rewards'] = torch.tensor(np.array([item['target_rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    
    return res


def idt_collate_fn(batch):
    res = {}
    res['states'] = torch.tensor(np.array([item['states'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['actions'] = torch.tensor(np.array([item['actions'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['returns_to_go'] = torch.tensor(np.array([item['returns_to_go'] for item in batch]), dtype=torch.float, requires_grad=False)
    res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), dtype=torch.float, requires_grad=False)

    return res


def get_data_loader(dataset, batch_size, config, shuffle=True):
    if config['model'] == 'AD':
        collate_fn = ad_collate_fn
    elif config['model'] == 'IDT':
        collate_fn = idt_collate_fn
    else:
        print(config['model'])
        raise ValueError('Invalid model')
    
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
            yield batch```

