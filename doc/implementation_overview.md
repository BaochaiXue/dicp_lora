# Implementation Overview

This document describes how Distillation for In-Context Planning (DICP) is implemented in this repository. The code follows the paper *Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning* and provides training utilities for discrete gridworld tasks and the Meta‑World benchmark.

## Directory Layout

- `gridworld/` – environments, dataset loaders and models for 2‑D grid tasks.
- `metaworld/` – code for continuous control tasks using the same training pipeline.
- `reference_material/` – additional source code and papers referenced by the project.

## Data Collection

Trajectories are generated with RL algorithms (e.g. PPO) and saved per environment. `collect_data.py` spawns several workers, each running a wrapped agent that logs states, actions, rewards and next states. The resulting HDF5 file uses the naming convention produced by `get_traj_file_name`:

```python
path = f"history_{task}_{config['alg']}_alg-seed{config['alg_seed']}"
```
【F:gridworld/utils.py†L12-L27】

## Dataset Format

Each dataset class slices windows of length `n_transit` from recorded trajectories. For example, `ADDataset` loads and concatenates history tensors across environments:

```python
states.append(f[f'{i}']['states'][()].transpose(1,0,2)[:n_stream,:source_timesteps])
actions.append(f[f'{i}']['actions'][()].transpose(1,0)[:n_stream,:source_timesteps])
rewards.append(f[f'{i}']['rewards'][()].transpose(1,0)[:n_stream,:source_timesteps])
next_states.append(f[f'{i}']['next_states'][()].transpose(1,0,2)[:n_stream,:source_timesteps])
```
【F:gridworld/dataset.py†L43-L55】

The dataset length is the number of possible windows:

```python
return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
```
【F:gridworld/dataset.py†L57-L58】

During batching the collate functions in `utils.py` map fields to tensors and one‑hot encodings:

```python
res['actions'] = F.one_hot(torch.tensor(np.array([item['actions'] for item in batch]), dtype=torch.long), num_classes=5)
res['rewards'] = torch.tensor(np.array([item['rewards'] for item in batch]), dtype=torch.float)
```
【F:gridworld/utils.py†L40-L57】

## Token Design

### GridWorld

Gridworld states are 2‑D coordinates with size `grid_size × grid_size`. They are converted to integer tokens using `map_dark_states`:

```python
def map_dark_states(states, grid_size):
    return torch.sum(states * torch.tensor((grid_size, 1), device=states.device, requires_grad=False), dim=-1)
```
【F:gridworld/env/darkroom.py†L13-L15】

`map_dark_states_inverse` performs the reverse mapping to coordinates. Actions are integers in `[0, num_action)` and rewards are binary. Each transition token is the concatenation of state, one‑hot action, reward and next state. When dynamics modelling is enabled, the target action token is followed by predicted reward and next state tokens.

### Meta‑World

For continuous control tasks the state and action vectors are kept as floating point tensors. They are embedded with linear layers before entering the transformer.

## Models

All policies use a TinyLlama transformer backbone (`model/tiny_llama`). The `AD` model predicts the next action given `n_transit-1` past transitions and a query state:

```python
self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, config['tf_n_embd'])
self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], config['tf_n_embd'])
logits_actions = self.pred_action(transformer_output[:, self.n_transit-1])
```
【F:gridworld/model/ad.py†L22-L32】【F:gridworld/model/ad.py†L53-L60】

`DPT` extends this architecture with additional query tokens and supports beam‑search planning. `IDT` builds a hierarchical sequence of high‑level decisions and low‑level actions.

## Training and Evaluation

`train.py` ties everything together. It loads configuration files, builds datasets, and iteratively updates the model using `Accelerator`:

```python
with accelerator.autocast():
    output = model(batch)
loss = output['loss_action']
accelerator.backward(loss)
accelerator.clip_grad_norm_(model.parameters(), 1)
optimizer.step()
```
【F:gridworld/train.py†L193-L214】

Evaluation uses `evaluate.py`, which restores the latest checkpoint and calls `evaluate_in_context` on a vectorised environment:

```python
ckpt_paths = sorted(glob(path.join(args.ckpt_dir, 'ckpt-*.pt')))
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'])
```
【F:gridworld/evaluate.py†L32-L44】

Beam search is optional when dynamics are modelled.

---
