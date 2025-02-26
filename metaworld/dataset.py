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
        
        return (episode_rtg + offset).reshape(-1, rtg.shape[1])