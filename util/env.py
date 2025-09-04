import torch
import numpy as np
from typing import Literal

_device = None 

def get_device() -> torch.device | Literal["cpu"]:
    # returns cpu string or torch device for cuda or mps. 
    return _device

def set_device(dev):
    global _device
    _device = dev

def init_work(worker_id, seed):
    np.random.seed(seed + worker_id)
