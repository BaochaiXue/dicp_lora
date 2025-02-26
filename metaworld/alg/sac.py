import torch
from stable_baselines3 import SAC


class SACWrapper(SAC):
    def __init__(self, config, env, seed, log_dir):
        policy = config['policy']
        lr = config['source_lr']
        buffer_size = config['buffer_size']
        learning_starts = config['learning_starts']
        batch_size = config['batch_size']
        train_freq = config['train_freq']
        gradient_steps = config['gradient_steps']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        env = env
        seed = seed

        super(SACWrapper, self).__init__(policy=policy,
                                         env=env,
                                         learning_rate=lr,
                                         buffer_size = buffer_size,
                                         learning_starts=learning_starts,
                                         batch_size=batch_size,
                                         train_freq=train_freq,
                                         gradient_steps=gradient_steps,
                                         verbose=0,
                                         seed=seed,
                                         device=device,
                                         tensorboard_log=log_dir)