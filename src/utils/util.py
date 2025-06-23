import os
import torch
import random
import numpy as np

from typing import Optional


def set_seed(seed: int,
             use_cudnn: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = use_cudnn

def set_device(index: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available() and index is not None:
        device = torch.device('cuda', index)
    else:
        device = torch.device('cpu')
    
    return device