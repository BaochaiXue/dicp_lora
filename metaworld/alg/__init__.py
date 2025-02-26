from .ppo import PPOWrapper
from .sac import SACWrapper
from .utils import HistoryLoggerCallback

ALGORITHM = {
    'PPO': PPOWrapper,
    'SAC': SACWrapper,
}