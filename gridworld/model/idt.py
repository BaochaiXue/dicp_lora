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