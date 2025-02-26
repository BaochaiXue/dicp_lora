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
        
        return beam[:, 0, 0]