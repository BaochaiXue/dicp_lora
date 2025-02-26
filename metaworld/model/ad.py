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
        self.action_high = torch.tensor(action_space.high, device=self.device, requires_grad=False, dtype=torch.float)