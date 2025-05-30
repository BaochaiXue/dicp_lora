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
