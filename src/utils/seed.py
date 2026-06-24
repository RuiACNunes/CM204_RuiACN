"""[VIAL-infra] Reprodutibilidade: fixa a seed global de random/numpy/torch.

Chamada no início de todo script (ver SPEC_CM204.md §9).
"""
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
