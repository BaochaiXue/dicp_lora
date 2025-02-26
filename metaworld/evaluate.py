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
        np.save(f, success)