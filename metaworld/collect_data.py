import os
from datetime import datetime

import yaml

import sys
sys.path.append(os.path.dirname(sys.path[0]))

from alg import ALGORITHM, HistoryLoggerCallback
import argparse
import multiprocessing
from utils import get_config, get_traj_file_name
import h5py

from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers.time_limit import TimeLimit
import metaworld


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=True, help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=True, help="Environment config")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Directory for history saving")
    parser.add_argument('--override', '-o', default='')
    args = parser.parse_args()
    return args


def worker(config, env_cls, task_instance, traj_dir, env_idx, history, file_name):
    
    env = DummyVecEnv([make_env(config, env_cls, task_instance)] * config['n_stream'])
    
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


def make_env(config, env_cls, task):
    def _init():
            env = env_cls()
            env.set_task(task)
            return TimeLimit(env, max_episode_steps=config['horizon'])
    return _init


if __name__ == '__main__':
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn')

    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))    

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
    
    task = config['task']
    
    ml1 = metaworld.ML1(env_name=task, seed=config['mw_seed'])
        
    file_name = get_traj_file_name(config)
    
    # Collcet train task histories
    name, env_cls = list(ml1.train_classes.items())[0]
    task_instances = ml1.train_tasks
    path = f'{args.traj_dir}/{task}/'
    
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    start_time = datetime.now()
    print(f'Collecting train task histories started at {start_time}')
    
    with h5py.File(os.path.join(path, f'{file_name}.hdf5'), 'a') as f:
        start_idx = 0
        
        while f'{start_idx}' in f.keys():
            start_idx += 1
            
        with multiprocessing.Manager() as manager:
            
            while start_idx < len(task_instances):
                history = manager.dict()

                instances = task_instances[start_idx:start_idx+config['n_process']]
                
                with multiprocessing.Pool(processes=config['n_process']) as pool:
                    pool.starmap(worker, [(config, env_cls, task_instance, path, start_idx+i, history, file_name) for i, task_instance in enumerate(instances)])

                # Save the history dictionary
                for i in range(start_idx, start_idx+len(instances)):
                    env_group = f.create_group(f'{i}')
                    for key, value in history[i].items():
                        env_group.create_dataset(key, data=value)
            
                start_idx += len(instances)

    end_time = datetime.now()
    print()
    print(f'Collecting train task histories ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')


    # Collcet test task histories
    name, env_cls = list(ml1.test_classes.items())[0]
    task_instances = ml1.test_tasks[:10]
    path = f'{args.traj_dir}/{task}/test/'
        
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        
    start_time = datetime.now()
    print(f'Collecting test task histories started at {start_time}')
    
    print()
    
    with h5py.File(f'{path}/{file_name}.hdf5', 'a') as f:
        start_idx = 0
        
        while f'{start_idx}' in f.keys():
            start_idx += 1
            
        with multiprocessing.Manager() as manager:
            
            while start_idx < len(task_instances):
                history = manager.dict()

                instances = task_instances[start_idx:start_idx+config['n_process']]
                
                with multiprocessing.Pool(processes=config['n_process']) as pool:
                    pool.starmap(worker, [(config, env_cls, task_instance, path, start_idx+i, history, file_name) for i, task_instance in enumerate(instances)])

                # Save the history dictionary
                for i in range(start_idx, start_idx+len(instances)):
                    env_group = f.create_group(f'{i}')
                    for key, value in history[i].items():
                        env_group.create_dataset(key, data=value)
            
                start_idx += len(instances)

    end_time = datetime.now()
    print()
    print(f'Collecting test task histories ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')