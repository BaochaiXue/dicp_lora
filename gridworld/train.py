import shutil
from datetime import datetime
from glob import glob
import os
import os.path as path
from modulefinder import ModuleFinder

import yaml
import argparse
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from dataset import ADDataset, DPTDataset, IDTDataset
from env import SAMPLE_ENVIRONMENT
from model import MODEL
from utils import get_config, get_data_loader, log_in_context, next_dataloader
from transformers import get_cosine_schedule_with_warmup

import multiprocessing
from tqdm import tqdm
from accelerate import Accelerator
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import make_env


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg-config",
        "-ac",
        required=False,
        default="./cfg/alg/ppo_dr.yaml",
        help="Algorithm config",
    )
    parser.add_argument(
        "--env-config",
        "-ec",
        required=False,
        default="./cfg/env/darkroom.yaml",
        help="Environment config",
    )
    parser.add_argument(
        "--model-config",
        "-mc",
        required=False,
        default="./cfg/model/ad_dr.yaml",
        help="Model config",
    )
    parser.add_argument(
        "--log-dir", "-l", required=False, default="./runs", help="Log directory"
    )
    parser.add_argument(
        "--traj-dir",
        "-t",
        required=False,
        default="./datasets",
        help="Trajectory directory",
    )
    parser.add_argument(
        "--no-backup",
        "-nb",
        required=False,
        default=False,
        help="Save code",
        action="store_true",
    )
    parser.add_argument("--override", "-o", default="")
    parser.add_argument(
        "--resume",
        required=False,
        default=False,
        help="Resume train",
        action="store_true",
    )
    parser.add_argument(
        "--mixed-precision",
        "-m",
        required=False,
        default="fp32",
        help="fp32 or fp16 or bf16",
    )
    parser.add_argument(
        "--disable-tqdm", "-d", required=False, default=False, action="store_true"
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    args = parse_arguments()

    # Load and update config
    config = get_config(args.env_config)
    config.update(get_config(args.alg_config))
    config.update(get_config(args.model_config))

    # Override options
    for option in args.override.split("|"):
        if not option:
            continue
        address, value = option.split("=")
        keys = address.split(".")
        here = config
        for key in keys[:-1]:
            if key not in here:
                here[key] = {}
            here = here[key]
        if keys[-1] not in here:
            print(f"Warning: {address} is not defined in config file.")
        here[keys[-1]] = yaml.load(value, Loader=yaml.FullLoader)

    if config["dynamics"]:
        log_dir = path.join(
            args.log_dir,
            f"{config['model']}-{config['env']}-dynamics{config['dynamics']}-str{config['dynamics_strength']}-seed{config['env_split_seed']}",
        )
    else:
        log_dir = path.join(
            args.log_dir,
            f"{config['model']}-{config['env']}-dynamics{config['dynamics']}-seed{config['env_split_seed']}",
        )

    writer = SummaryWriter(log_dir, flush_secs=15)

    # Prevent overwriting
    config["log_dir"] = log_dir
    config_save_path = path.join(config["log_dir"], "config.yaml")
    try:
        # Try to open config file to bypass NFS cache
        with open(config_save_path, "r") as f:
            f.read(1)
            config_exists = True
    except FileNotFoundError:
        config_exists = False

    if config_exists and not args.resume:
        print(f"WARNING: {log_dir} already exists. Skipping...")
        exit(0)

    config["traj_dir"] = args.traj_dir
    config["mixed_precision"] = args.mixed_precision

    # Save config
    os.makedirs(config["log_dir"], mode=0o755, exist_ok=True)
    with open(config_save_path, "w") as f:
        yaml.dump(config, f)
    print(f"Config saved to {config_save_path}")

    # Save code
    if not args.no_backup:
        code_dir = path.join(
            config["log_dir"], "code_" + datetime.now().strftime("%Y%m%d_%H%M%S")
        )
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
        print(f"Code saved to {code_dir}")

    # Define accelerator
    if args.mixed_precision == "bf16" or args.mixed_precision == "fp16":
        accelerator = Accelerator(mixed_precision=args.mixed_precision)
    elif args.mixed_precision == "fp32":
        accelerator = Accelerator(mixed_precision="no")
    else:
        raise ValueError(f"Unsupported mixed precision: {args.mixed_precision}")

    config["device"] = accelerator.device

    # Define model
    model_name = config["model"]
    model = MODEL[model_name](config)
    if config.get("use_lora", False):
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"LoRA enabled: trainable params {trainable}/{total}")

    # Get datasets and dataloaders
    load_start_time = datetime.now()
    print(f"Data loading started at {load_start_time}")

    if config["model"] == "DPT":
        train_dataset = DPTDataset(
            config,
            args.traj_dir,
            "train",
            config["train_n_stream"],
            config["train_source_timesteps"],
        )
        test_dataset = DPTDataset(
            config, args.traj_dir, "test", 1, config["train_source_timesteps"]
        )
    elif config["model"] == "AD":
        train_dataset = ADDataset(
            config,
            args.traj_dir,
            "train",
            config["train_n_stream"],
            config["train_source_timesteps"],
        )
        test_dataset = ADDataset(
            config, args.traj_dir, "test", 1, config["train_source_timesteps"]
        )
    elif config["model"] == "IDT":
        train_dataset = IDTDataset(
            config,
            args.traj_dir,
            "train",
            config["train_n_stream"],
            config["train_source_timesteps"],
        )
        test_dataset = IDTDataset(
            config, args.traj_dir, "test", 1, config["train_source_timesteps"]
        )
    else:
        raise ValueError(f'Unsupported model: {config["model"]}')

    train_dataloader = get_data_loader(
        train_dataset,
        batch_size=config["train_batch_size"],
        config=config,
        shuffle=True,
    )
    train_dataloader = next_dataloader(train_dataloader)

    test_dataloader = get_data_loader(
        test_dataset, batch_size=config["test_batch_size"], config=config, shuffle=False
    )

    load_end_time = datetime.now()
    print()
    print(f"Data loading ended at {load_end_time}")
    print(f"Elapsed time: {load_end_time - load_start_time}")

    # Define optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config["lr"],
        betas=(config["beta1"], config["beta2"]),
        weight_decay=config["weight_decay"],
    )
    lr_sched = get_cosine_schedule_with_warmup(
        optimizer, config["num_warmup_steps"], config["train_timesteps"]
    )
    step = 0

    # Resume checkpoint
    if args.resume:
        ckpt_paths = sorted(glob(path.join(config["log_dir"], "ckpt-*.pt")))
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[-1]
            ckpt = torch.load(ckpt_path)
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            lr_sched.load_state_dict(ckpt["lr_sched"])
            step = ckpt["step"]
            print(f"Checkpoint loaded from {ckpt_path}")

    # Define environments for evaluation
    env_name = config["env"]
    train_env_args, test_env_args = SAMPLE_ENVIRONMENT[env_name](config)
    train_env_args = train_env_args[:10]
    test_env_args = test_env_args[:10]
    env_args = train_env_args + test_env_args

    if env_name == "darkroom":
        envs = SubprocVecEnv([make_env(config, goal=arg) for arg in env_args])
    elif env_name == "darkroompermuted":
        envs = SubprocVecEnv([make_env(config, perm_idx=arg) for arg in env_args])
    elif env_name == "darkkeytodoor":
        envs = SubprocVecEnv(
            [make_env(config, key=arg[:2], goal=arg[2:]) for arg in env_args]
        )
    else:
        raise NotImplementedError(f"Environment {env_name} is not supported")

    model, optimizer, train_dataloader, lr_sched = accelerator.prepare(
        model, optimizer, train_dataloader, lr_sched
    )

    # Main training loop
    start_time = datetime.now()
    print(f"Training started at {start_time}")

    with tqdm(
        total=config["train_timesteps"],
        position=0,
        leave=True,
        disable=args.disable_tqdm,
    ) as pbar:
        pbar.update(step)

        while True:
            batch = next(train_dataloader)

            step += 1

            with accelerator.autocast():
                output = model(batch)

            if config["dynamics"]:
                loss = (
                    output["loss_action"]
                    + (output["loss_reward"] + output["loss_next_state"])
                    * config["dynamics_strength"]
                )
            else:
                loss = output["loss_action"]

            optimizer.zero_grad()
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            if not accelerator.optimizer_step_was_skipped:
                lr_sched.step()

            pbar.set_postfix(loss=loss.item())

            if step % config["summary_interval"] == 0:

                writer.add_scalar("train/loss", loss.item(), step)
                writer.add_scalar(
                    "train/loss_action", output["loss_action"].item(), step
                )
                writer.add_scalar("train/lr", lr_sched.get_last_lr()[0], step)
                writer.add_scalar("train/acc_action", output["acc_action"].item(), step)

                if config["dynamics"]:
                    writer.add_scalar(
                        "train/loss_reward", output["loss_reward"].item(), step
                    )
                    writer.add_scalar(
                        "train/acc_reward", output["acc_reward"].item(), step
                    )
                    writer.add_scalar(
                        "train/loss_next_state", output["loss_next_state"].item(), step
                    )
                    writer.add_scalar(
                        "train/acc_next_state", output["acc_next_state"].item(), step
                    )

            ############ Evaluation ############
            if step % config["eval_interval"] == 0:
                torch.cuda.empty_cache()
                model.eval()
                eval_start_time = datetime.now()
                print(f"Evaluating started at {eval_start_time}")

                with torch.no_grad():
                    test_loss_action = 0.0
                    test_acc_action = 0.0
                    test_loss_reward = 0.0
                    test_acc_reward = 0.0
                    test_loss_next_state = 0.0
                    test_acc_next_state = 0.0
                    test_cnt = 0

                    for j, batch in enumerate(test_dataloader):
                        output = model(batch)
                        cnt = len(batch["states"])
                        test_loss_action += output["loss_action"].item() * cnt
                        test_acc_action += output["acc_action"].item() * cnt

                        if config["dynamics"]:
                            test_loss_reward += output["loss_reward"].item() * cnt
                            test_acc_reward += output["acc_reward"].item() * cnt
                            test_loss_next_state += (
                                output["loss_next_state"].item() * cnt
                            )
                            test_acc_next_state += output["acc_next_state"].item() * cnt

                        test_cnt += cnt

                writer.add_scalar("test/loss_action", test_loss_action / test_cnt, step)
                writer.add_scalar("test/acc_action", test_acc_action / test_cnt, step)

                if config["dynamics"]:
                    writer.add_scalar(
                        "test/loss_reward", test_loss_reward / test_cnt, step
                    )
                    writer.add_scalar(
                        "test/acc_reward", test_acc_reward / test_cnt, step
                    )
                    writer.add_scalar(
                        "test/loss_next_state", test_loss_next_state / test_cnt, step
                    )
                    writer.add_scalar(
                        "test/acc_next_state", test_acc_next_state / test_cnt, step
                    )

                eval_end_time = datetime.now()
                print()
                print(f"Evaluating ended at {eval_end_time}")
                print(f"Elapsed time: {eval_end_time - eval_start_time}")
                model.train()
                torch.cuda.empty_cache()
            ####################################

            ############ Generation ############
            if step % config["gen_interval"] == 0:
                model.eval()
                gen_start_time = datetime.now()
                print(f"Generation started at {gen_start_time}")

                with torch.no_grad():
                    output = model.evaluate_in_context(
                        envs, config["train_source_timesteps"]
                    )

                    train_rewards = output["reward_episode"][: len(train_env_args)]
                    test_rewards = output["reward_episode"][len(train_env_args) :]

                    log_in_context(
                        values=train_rewards,
                        max_reward=config["max_reward"],
                        success=None,
                        episode_length=config["horizon"],
                        tag="train_gen/reward_episode",
                        title="",
                        xlabel="In-context steps",
                        ylabel="Reward",
                        step=step,
                        writer=writer,
                    )

                    log_in_context(
                        values=test_rewards,
                        max_reward=config["max_reward"],
                        success=None,
                        episode_length=config["horizon"],
                        tag="test_gen/reward_episode",
                        title="",
                        xlabel="In-context steps",
                        ylabel="Reward",
                        step=step,
                        writer=writer,
                    )

                gen_end_time = datetime.now()
                print()
                print(f"Generation ended at {gen_end_time}")
                print(f"Elapsed time: {gen_end_time - gen_start_time}")
                model.train()
                torch.cuda.empty_cache()
            ####################################

            pbar.update(1)

            # LOGGING
            if step % config["ckpt_interval"] == 0:
                # Remove old checkpoints
                ckpt_paths = sorted(glob(path.join(config["log_dir"], "ckpt-*.pt")))
                for ckpt_path in ckpt_paths:
                    os.remove(ckpt_path)

                new_ckpt_path = path.join(config["log_dir"], f"ckpt-{step}.pt")

                torch.save(
                    {
                        "step": step,
                        "config": config,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_sched": lr_sched.state_dict(),
                    },
                    new_ckpt_path,
                )
                print(f"\nCheckpoint saved to {new_ckpt_path}")

            if step >= config["train_timesteps"]:
                break

    writer.flush()
    envs.close()

    end_time = datetime.now()
    print()
    print(f"Training ended at {end_time}")
    print(f"Elapsed time: {end_time - start_time}")
