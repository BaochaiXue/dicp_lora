# Technical Overview of the DICP Repository

## Repository Layout

- `gridworld/` and `metaworld/` contain training code for discrete grid environments and the Meta-World robotics tasks respectively.
- `environment.yml` lists conda and pip dependencies needed to run the code.
- The project uses Python 3.8 and packages such as PyTorch, stable-baselines3 and Transformers.

## Dataset Handling

### GridWorld Datasets

`gridworld/dataset.py` defines classes for loading trajectory data from HDF5 files. For example, `ADDataset` reads saved states, actions, and rewards for selected environments and exposes them as PyTorch datasets:
```
    with h5py.File(f'{traj_dir}/{get_traj_file_name(config)}.hdf5', 'r') as f:
        for i in env_idx:
            states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
            actions.append(f[f'{i}']['actions'][()].transpose(1, 0)[:n_stream, :source_timesteps])
            rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
            next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
```
【F:gridworld/dataset.py†L45-L50】

The dataset length scales with the available transitions:
```
    def __len__(self):
        return (len(self.states[0]) - self.n_transit + 1) * len(self.states)
```
【F:gridworld/dataset.py†L57-L58】

### Meta-World Datasets

`metaworld/dataset.py` follows a similar pattern, reading recorded trajectories per seed:
```
with h5py.File(file_path, 'r') as f:
    for i in range(n_seed):
        states.append(f[f'{i}']['states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :11])
        actions.append(f[f'{i}']['actions'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps])
        rewards.append(f[f'{i}']['rewards'][()].transpose(1, 0)[:n_stream, :source_timesteps])
        next_states.append(f[f'{i}']['next_states'][()].transpose(1, 0, 2)[:n_stream, :source_timesteps, :11])
```
【F:metaworld/dataset.py†L27-L32】

Returns-to-go are computed and sorted to facilitate training:
```
returns_to_go = rearrange(self.returns_to_go, 'traj (epi time) -> traj epi time', time=self.config['horizon'])
sorted_episode_idx = np.argsort(returns_to_go[:, :, 0])
```
【F:metaworld/dataset.py†L74-L76】

## Environment Implementations

The repository provides small grid-based environments. `gridworld/env/darkroom.py` defines a simple 2D navigation task with a goal location:
```
class Darkroom(gym.Env):
    def __init__(self, config, **kwargs):
        ...
        self.goal = kwargs['goal']
        self.horizon = config['horizon']
        self.observation_space = spaces.Box(low=0, high=self.grid_size-1, shape=(self.dim_obs,), dtype=np.int32)
        self.action_space = spaces.Discrete(self.num_action)
```
【F:gridworld/env/darkroom.py†L51-L62】

The step function updates the agent position and returns a reward when it reaches the goal:
```
    reward = 1 if np.array_equal(s, self.goal) else 0
    self.current_step += 1
    done = self.current_step >= self.horizon
    info = {}
    return s.copy(), reward, done, done, info
```
【F:gridworld/env/darkroom.py†L97-L101】

## Algorithm Wrappers

Data collection uses wrappers around stable-baselines3 algorithms. `gridworld/alg/ppo.py` shows how PPO parameters are configured from YAML:
```
class PPOWrapper(PPO):
    def __init__(self, config, env, seed, log_dir):
        policy = config['policy']
        n_steps = config['n_steps']
        batch_size = config['batch_size']
        n_epochs = config['n_epochs']
        lr = config['source_lr']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
【F:gridworld/alg/ppo.py†L1-L24】

## Training Pipeline

Scripts such as `gridworld/train.py` parse configuration files and manage logging:
```
parser.add_argument('--alg-config', '-ac', required=False, default='./cfg/alg/ppo_dr.yaml', help="Algorithm config")
parser.add_argument('--env-config', '-ec', required=False, default='./cfg/env/darkroom.yaml', help="Environment config")
parser.add_argument('--model-config', '-mc', required=False, default='./cfg/model/ad_dr.yaml', help="Model config")
```
【F:gridworld/train.py†L29-L33】

During setup the config is stored and code is optionally backed up:
```
os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
with open(config_save_path, 'w') as f:
    yaml.dump(config, f)
...
if not args.no_backup:
    code_dir = path.join(config['log_dir'], 'code_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    mf = ModuleFinder([os.getcwd()])
    mf.run_script(__file__)
```
【F:gridworld/train.py†L94-L113】

## Models

The repository defines several transformer-based policies:
- `AD` (Action Distillation) predicts the next action and optionally the dynamics.
- `DPT` (Distilled Planning Transformer) supports beam search planning.
- `IDT` (In-Context Decision Transformer) uses a hierarchical structure with low- and high-level decisions.

`AD` embeds context transitions and queries using a TinyLlama transformer:
```
self.embed_context = nn.Linear(config['dim_states'] * 2 + config['num_actions'] + 1, config['tf_n_embd'])
self.embed_query_state = nn.Embedding(config['grid_size'] * config['grid_size'], config['tf_n_embd'])
self.pred_action = nn.Linear(config['tf_n_embd'], config['num_actions'])
```
【F:gridworld/model/ad.py†L22-L28】

## Usage

The top-level `README.md` explains how to collect data, train models and evaluate checkpoints:
```
python collect_data.py -ac [algorithm config]  -ec [environment config] -t [trajectory directory]
...
python train.py -ac [algorithm config]  -ec [environment config] -mc [model config] -t [trajectory directory] -l [log directory]
...
python evaluate.py -c [checkpoint directory] -k [beam size]
```
【F:README.md†L41-L59】

This repository implements Distillation for In-Context Planning across discrete gridworlds and robotic manipulation tasks using transformer-based models and reinforcement learning algorithms.


## Data Collection Workflow

The `collect_data.py` scripts in both `gridworld/` and `metaworld/` spawn multiple environment instances in parallel. Each environment is wrapped with `HistoryLoggerCallback` which records observations and rewards for every step:
```python
class HistoryLoggerCallback(BaseCallback):
    def __init__(self, env_name, env_idx, history=None):
        ...
        self.states.append(self.locals["obs_tensor"].cpu().numpy())
        self.next_states.append(self.locals["new_obs"])
        self.actions.append(self.locals["actions"])
```
【F:gridworld/alg/utils.py†L1-L24】

Data from each worker is concatenated into an HDF5 file, organised by environment index. For DPT training an additional pass annotates optimal actions after trajectories are saved.

## Dataloaders and Collation

`gridworld/utils.py` provides collate functions converting raw dictionaries into tensors and one-hot encodings before batching:
```python
res['actions'] = F.one_hot(torch.tensor(np.array([item['actions'] for item in batch]), dtype=torch.long), num_classes=5)
res['next_states'] = torch.tensor(np.array([item['next_states'] for item in batch]), dtype=torch.float)
if 'target_next_states' in batch[0].keys():
    res['target_next_states'] = map_dark_states(...)
```
【F:gridworld/utils.py†L35-L65】

These utilities are selected based on the model type and ensure consistent tensor shapes across environments.

## Training Loop Details

During training `train.py` wraps the model, dataloader and optimizer with `Accelerator` to support mixed precision. Each step runs in an `autocast` context and gradients are clipped:
```python
with accelerator.autocast():
    output = model(batch)
loss = output['loss_action']
optimizer.zero_grad()
accelerator.backward(loss)
accelerator.clip_grad_norm_(model.parameters(), 1)
optimizer.step()
```
【F:gridworld/train.py†L193-L214】

Evaluation and generation occur at fixed intervals to log performance and save checkpoints.

## Evaluation Scripts

`evaluate.py` loads the latest checkpoint and runs inference on a pool of environments. When dynamics are enabled the evaluation horizon doubles and optional beam search is used:
```python
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
```
【F:gridworld/evaluate.py†L60-L80】

## Additional Environment Variants

`DarkroomPermuted` randomises the mapping between actions and transitions, while `DarkKeyToDoor` requires finding a key before reaching the goal. Sampling utilities enumerate train and test splits using the configured random seed.

