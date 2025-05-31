import shutil
from datetime import datetime
from glob import glob
import os
import os.path as path
import sys
sys.path.append(path.dirname(sys.path[0]))
from modulefinder import ModuleFinder

import yaml
import argparse
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import ADDataset, IDTDataset
from model import MODEL
from utils import get_config, get_data_loader, log_in_context, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from accelerate import Accelerator
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium.wrappers.time_limit import TimeLimit
import metaworld

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg-config', '-ac', required=False, default='./cfg/alg/ppo_ml1.yaml', help="Algorithm config")
    parser.add_argument('--env-config', '-ec', required=False, default='./cfg/env/ml1.yaml', help="Environment config")
    parser.add_argument('--model-config', '-mc', required=False, default='./cfg/model/idt_ml1.yaml', help="Model config")
    parser.add_argument('--log-dir', '-l', required=False, default='./runs', help="Log directory")
    parser.add_argument('--traj-dir', '-t', required=False, default='./datasets', help="Trajectory directory")
    parser.add_argument('--no-backup', '-nb', required=False, default=False, help="Save code", action='store_true')
    parser.add_argument('--override', '-o', default='')
    parser.add_argument('--resume', required=False, default=False, help="Resume train", action='store_true')
    parser.add_argument('--mixed-precision', '-m', required=False, default='fp32')
    parser.add_argument('--disable-tqdm', '-d', required=False, default=False, action='store_true')

    args = parser.parse_args()
    return args


def make_env(config, env_cls, task):
    def _init():
            env = env_cls()
            env.set_task(task)
            return TimeLimit(env, max_episode_steps=config['horizon'])
    return _init


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))
    config.update(get_config(args.model_config))

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

    log_dir = path.join(args.log_dir, f"{config['model']}-ml1-{config['task']}-dynamics{config['dynamics']}-var{config['learn_var']}")
    writer = SummaryWriter(log_dir, flush_secs=15)
    
    # Prevent overwriting
    config['log_dir'] = log_dir
    config_save_path = path.join(config['log_dir'], 'config.yaml')
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, 'r') as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f'WARNING: {log_dir} already exists. Skipping...')
        exit(0)

    traj_dir = path.join(args.traj_dir, config['task'])    
    config['traj_dir'] = traj_dir
    config['device'] = device
    config['mixed_precision'] = args.mixed_precision

    # Save config
    os.makedirs(config['log_dir'], mode=0o755, exist_ok=True)
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f)
    print(f'Config saved to {config_save_path}')

    # Save code
    if not args.no_backup:
        code_dir = path.join(config['log_dir'], 'code_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        mf = ModuleFinder([os.getcwd()])
        mf.run_script(__file__)
        for name, module in mf.modules.items():
            if module.__file__ is None:
                continue
            rel_path = path.relpath(module.__file__)
            new_path = path.join(code_dir, rel_path)
            new_dirname = path.dirname(new_path)
            os.makedirs(new_dirname, mode=0o750, exist_ok=True)
            shutil.copy2(rel_path, new_path)
        print(f'Code saved to {code_dir}')
    
    # Define model
    model_name = config['model']
    model = MODEL[model_name](config).to(device)
    if config.get('use_lora', False):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA enabled: trainable params {trainable}/{total}")

    # Get datasets and dataloaders
    load_start_time = datetime.now()
    print(f'Data loading started at {load_start_time}')
    
    if config['model'] == 'AD':
        train_dataset = ADDataset(config, traj_dir, 'train', config['train_n_seed'], config['train_n_stream'], config['train_source_timesteps'])
        test_dataset = ADDataset(config, traj_dir, 'test', 1, 1, config['train_source_timesteps'])
    elif config['model'] == 'IDT':
        train_dataset = IDTDataset(config, traj_dir, 'train', config['train_n_seed'], config['train_n_stream'], config['train_source_timesteps'])
        test_dataset = IDTDataset(config, traj_dir, 'test', 1, 1, config['train_source_timesteps'])
    else:
        raise ValueError(f'Unsupported model: {config["model"]}')
    
    train_dataloader = get_data_loader(train_dataset, batch_size=config['train_batch_size'], config=config, shuffle=True)
    train_dataloader = next_dataloader(train_dataloader)
    test_dataloader = get_data_loader(test_dataset, batch_size=config['test_batch_size'], config=config, shuffle=False)

    load_end_time = datetime.now()
    print()
    print(f'Data loading ended at {load_end_time}')
    print(f'Elapsed time: {load_end_time - load_start_time}')

    # Define optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    lr_sched = get_cosine_schedule_with_warmup(optimizer, config['num_warmup_steps'], config['train_timesteps'])
    step = 0
    
    # Resume checkpoint
    if args.resume:
        ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            lr_sched.load_state_dict(ckpt['lr_sched'])
            step = ckpt['step']
            print(f'Checkpoint loaded from {ckpt_path}')
    
    # Define environments for evaluation
    ml1 = metaworld.ML1(env_name=config['task'], seed=config['mw_seed'])
    
    train_envs = []
    test_envs = []
    
    for task_name, env_cls in ml1.train_classes.items():
        task_instances = [task for task in ml1.train_tasks if task.env_name == task_name]
        for i in range(config['n_train_envs_per_task']):
            train_envs.append(make_env(config, env_cls, task_instances[i]))
    
    for task_name, env_cls in ml1.test_classes.items():
        task_instances = [task for task in ml1.test_tasks if task.env_name == task_name]
        for i in range(config['n_test_envs_per_task']):
            test_envs.append(make_env(config, env_cls, task_instances[i]))

    envs = train_envs + test_envs

    envs = SubprocVecEnv(envs)
    model.set_obs_space(envs.observation_space)
    model.set_action_space(envs.action_space)
    
    # Wrap everything into an accelerator
    if args.mixed_precision == 'bf16' or args.mixed_precision == 'fp16':
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
    elif args.mixed_precision == 'fp32':
        accelerator = Accelerator(mixed_precision='no')
    else:
        raise ValueError(f'Unsupported mixed precision: {args.mixed_precision}')
    
    # Main training loop
    start_time = datetime.now()
    print(f'Training started at {start_time}')
    
    with tqdm(total=config['train_timesteps'], position=0, leave=True, disable=args.disable_tqdm) as pbar:
        pbar.update(step)

        while True:
            with accelerator.autocast():
                batch = next(train_dataloader)
            
            step += 1
            
            with accelerator.autocast():
                output = model(batch)

            if config['dynamics']:
                if config['learn_transition']:
                    loss = output['loss_action'] + (output['loss_reward'] + output['loss_next_state']) * config['dynamics_strength']
                else:
                    loss = output['loss_action'] + output['loss_reward'] * config['dynamics_strength']
            else:
                loss = output['loss_action']

            optimizer.zero_grad()            
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            pbar.set_postfix(loss=loss.item())

            if step % config['summary_interval'] == 0:

                writer.add_scalar('train/loss', loss.item(), step)
                writer.add_scalar('train/loss_action', output['loss_action'].item(), step)
                writer.add_scalar('train/lr', lr_sched.get_last_lr()[0], step)

                if config['dynamics']:
                    writer.add_scalar('train/loss_reward', output['loss_reward'].item(), step)
                    if config['learn_transition']:
                        writer.add_scalar('train/loss_next_state', output['loss_next_state'].item(), step)

            ############ Evaluation ############
            if step % config['eval_interval'] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f'Evaluating started at {eval_start_time}')

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_loss_reward = 0.0
                    test_loss_next_state = 0.0
                    test_cnt = 0

                    for j, batch in enumerate(test_dataloader):
                        with accelerator.autocast():
                            output = model(batch)
                        cnt = len(batch['states'])
                        test_loss_action += output['loss_action'].item() * cnt
                 
                        if config['dynamics']:
                            test_loss_reward += output['loss_reward'].item() * cnt
                            if config['learn_transition']:
                                test_loss_next_state += output['loss_next_state'].item() * cnt

                        test_cnt += cnt

                writer.add_scalar('test/loss_action', test_loss_action / test_cnt, step)

                if config['dynamics']:
                    writer.add_scalar('test/loss_reward', test_loss_reward / test_cnt, step)
                    if config['learn_transition']:
                        writer.add_scalar('test/loss_next_state', test_loss_next_state / test_cnt, step)


                eval_end_time = datetime.now()
                print()
                print(f'Evaluating ended at {eval_end_time}')
                print(f'Elapsed time: {eval_end_time - eval_start_time}')
                model.train()
                torch.cuda.empty_cache()
            ####################################

            ############ Generation ############
            if step % config['gen_interval'] == 0:
                model.eval()
                gen_start_time = datetime.now()
                print(f'Generation started at {gen_start_time}')

                with torch.no_grad():
                    output = model.evaluate_in_context(envs, config['test_source_timesteps'])
                    
                    train_rewards = output['reward_episode'][:len(train_envs)]
                    test_rewards = output['reward_episode'][len(train_envs):]
                    
                    if 'success' in output.keys():
                        train_success = output['success'][:len(train_envs)]
                        test_success = output['success'][len(train_envs):]
                        
                        writer.add_scalar('train/success_rate', train_success.max(axis=1).mean(), step)
                        writer.add_scalar('test/success_rate', test_success.max(axis=1).mean(), step)
                        
                    else:
                        train_success = None
                        test_success = None
                    

                    log_in_context(values=train_rewards,
                                   max_reward=config['max_reward'],
                                   success=train_success,
                                   episode_length = config['horizon'],
                                   tag='train_gen/reward_episode',
                                   title='',
                                   xlabel='In-context steps',
                                   ylabel='Reward',
                                   step=step,
                                   writer=writer)

                    log_in_context(values=test_rewards,
                                   max_reward=config['max_reward'],
                                   success=test_success,
                                   episode_length = config['horizon'],
                                   tag='test_gen/reward_episode',
                                   title='',
                                   xlabel='In-context steps',
                                   ylabel='Reward',
                                   step=step,
                                   writer=writer)

                gen_end_time = datetime.now()
                print()
                print(f'Generation ended at {gen_end_time}')
                print(f'Elapsed time: {gen_end_time - gen_start_time}')
                model.train()
                torch.cuda.empty_cache()
            ####################################

            pbar.update(1)

            # LOGGING
            if step % config['ckpt_interval'] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob(path.join(config['log_dir'], 'ckpt-*.pt')))
                for ckpt_path in ckpt_paths:
                    os.remove(ckpt_path)

                new_ckpt_path = path.join(config['log_dir'], f'ckpt-{step}.pt')

                torch.save({
                    'step': step,
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_sched': lr_sched.state_dict(),
                }, new_ckpt_path)
                print(f'\nCheckpoint saved to {new_ckpt_path}')


            if step >= config['train_timesteps']:
                break

    writer.flush()
    envs.close()

    end_time = datetime.now()
    print()
    print(f'Training ended at {end_time}')
    print(f'Elapsed time: {end_time - start_time}')