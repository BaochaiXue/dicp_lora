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