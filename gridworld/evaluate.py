from datetime import datetime

import argparse
from glob import glob

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import torch
import os.path as path

from env import SAMPLE_ENVIRONMENT, make_env
from model import MODEL
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', '-c', required=True, help="Checkpoint directory")
    parser.add_argument('--beam-k', '-k', required=False, type=int, default=10, help="Beam_k for planning")
    parser.add_argument('--seed', '-s', required=False, type=int, default=0, help="Torch seed")

    args = parser.parse_args()
    return args


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

    env_name = config['env']
    _, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    
    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in test_env_args])
    elif env_name == "darkkeytodoor":
        envs = SubprocVecEnv([make_env(config, key=arg[:2], goal=arg[2:]) for arg in test_env_args])
    elif env_name == 'darkroompermuted':
        envs = SubprocVecEnv([make_env(config, perm_idx=arg) for arg in test_env_args])
    else:
        raise NotImplementedError(f'Environment {env_name} is not supported')
    
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
    start_time = datetime.now()
    print(f'Generation started at {start_time}')
    
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

    end_time = datetime.now()
    print()
    print(f'Generation ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    
    envs.close()
    
    with open(path, 'wb') as f:
        np.save(f, test_rewards)