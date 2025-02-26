import os
from datetime import datetime

import yaml

import sys
sys.path.append(os.path.dirname(sys.path[0]))

from env import SAMPLE_ENVIRONMENT, make_env, Darkroom, DarkroomPermuted, DarkKeyToDoor
from alg import ALGORITHM, HistoryLoggerCallback
import argparse
import multiprocessing
from utils import get_config, get_traj_file_name
import h5py
import numpy as np

from stable_baselines3.common.vec_env import DummyVecEnv


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=True, help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=True, help="Environment config")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Directory for history saving")
    parser.add_argument('--override', '-o', default='')
    args = parser.parse_args()
    return args


def worker(arg, config, traj_dir, env_idx, history, file_name):
    
    if config['env'] == 'darkroom':
        env = DummyVecEnv([make_env(config, goal=arg)] * config['n_stream'])
    elif config['env'] == 'darkroompermuted':
        env = DummyVecEnv([make_env(config, perm_idx=arg)] * config['n_stream'])
    elif config['env'] == 'darkkeytodoor':
        env = DummyVecEnv([make_env(config, key=arg[:2], goal=arg[2:])] * config['n_stream'])
    else:
        raise ValueError('Invalid environment')
    
    alg_name = config['alg']
    seed = config['alg_seed'] + env_idx
    
    # Initialize algorithm
    alg = ALGORITHM[alg_name](config, env, seed, traj_dir)

    callback = HistoryLoggerCallback(config['env'], env_idx, history)

    log_name = f'{file_name}_{env_idx}'

    # Execute learning algorithm
    alg.learn(total_timesteps=config['total_source_timesteps'],
              callback=callback,
              log_interval=1,
              tb_log_name=log_name,
              reset_num_timesteps=True,
              progress_bar=False)
    
    env.close()


if __name__ == '__main__':
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))

    # Ensure the log directory exists
    if not os.path.exists(args.traj_dir):
        os.makedirs(args.traj_dir, exist_ok=True)

    # Override options
    for option in args.override.split('|'):
        if not option:
            continue
        address, value = option.split('=')
        keys = address.split('.')
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f'Warning: {address} is not defined in config file.')
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)
    
    train_args, test_args = SAMPLE_ENVIRONMENT[config['env']](config, shuffle=False)

    total_args = train_args + test_args
    n_envs = len(total_args)

    file_name = get_traj_file_name(config)
    path = f'{args.traj_dir}/{file_name}.hdf5'
    
    start_time = datetime.now()
    print(f'Training started at {start_time}')

    with multiprocessing.Manager() as manager:
        history = manager.dict()

        # Create a pool with a maximum of n_workers
        with multiprocessing.Pool(processes=config['n_process']) as pool:
            # Map the worker function to the environments with the other arguments
            pool.starmap(worker, [(total_args[i], config, args.traj_dir, i, history, file_name) for i in range(n_envs)])

        # Save the history dictionary
        with h5py.File(path, 'w-') as f:
            for i in range(n_envs):
                env_group = f.create_group(f'{i}')
                for key, value in history[i].items():
                    env_group.create_dataset(key, data=value)
    
    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')
    
    start_time = datetime.now()
    print(f'Annotating optimal actions for DPT started at {start_time}')
    
    with h5py.File(path, 'a') as f:
        for i in range(n_envs):            
            if config['env'] == 'darkroom':
                states = f[f'{i}']['states'][()].transpose(1, 0, 2)
                actions = f[f'{i}']['actions'][()].transpose(1, 0)
                rewards = f[f'{i}']['rewards'][()].transpose(1, 0)
                env = Darkroom(config, goal=total_args[i])
                optimal_actions = np.zeros_like(actions)
                
                for stream_idx in range(states.shape[0]):
                    for step_idx in range(states.shape[1]):
                        optimal_actions[stream_idx, step_idx] = env.get_optimal_action(states[stream_idx, step_idx])
            
                group = f[f'{i}']
                group.create_dataset('optimal_actions', data=optimal_actions)
                
            elif config['env'] == 'darkroompermuted':
                states = f[f'{i}']['states'][()].transpose(1, 0, 2)
                actions = f[f'{i}']['actions'][()].transpose(1, 0)
                rewards = f[f'{i}']['rewards'][()].transpose(1, 0)
                env = DarkroomPermuted(config, perm_idx=i)
                optimal_actions = np.zeros_like(actions)
                
                for stream_idx in range(states.shape[0]):
                    for step_idx in range(states.shape[1]):
                        optimal_actions[stream_idx, step_idx] = env.get_optimal_action(states[stream_idx, step_idx])
                
                group = f[f'{i}']
                group.create_dataset('optimal_actions', data=optimal_actions)
                
            elif config['env'] == 'darkkeytodoor':
                states = f[f'{i}']['states'][()].transpose(1, 0, 2).reshape(100, -1, config['horizon'], 2)
                actions = f[f'{i}']['actions'][()].transpose(1, 0).reshape(100, -1, config['horizon'])
                rewards = f[f'{i}']['rewards'][()].transpose(1, 0).reshape(100, -1, config['horizon'])
                env = DarkKeyToDoor(config, key=total_args[i][:2], goal=total_args[i][2:])
                optimal_actions = np.zeros_like(actions)
                
                for stream_idx in range(states.shape[0]):
                    for episode_idx in range(states.shape[1]):
                        have_key=False
                        for step_idx in range(states.shape[2]):
                            optimal_actions[stream_idx, episode_idx, step_idx] = env.get_optimal_action(states[stream_idx, episode_idx, step_idx], have_key)
                            if not have_key and rewards[stream_idx, episode_idx, step_idx] > 0:
                                have_key = True
                        
                group = f[f'{i}']
                group.create_dataset('optimal_actions', data=optimal_actions.reshape(100, -1))
                
            else:
                raise ValueError('Invalid environment')
                
    end_time = datetime.now()
    print()
    print(f'Annotating ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')