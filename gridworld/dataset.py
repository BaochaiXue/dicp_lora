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