import random
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    # На Mac M2 будет mps (ускорение) если доступно
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")