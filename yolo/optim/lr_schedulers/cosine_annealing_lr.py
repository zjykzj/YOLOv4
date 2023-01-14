# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午4:44
@file: cosine_annealing_lr.py.py
@author: zj
@description: 
"""

import torch.optim as optim
from torch.optim.optimizer import Optimizer


def build_cosine_annealing_lr(optimizer: Optimizer, max_epoch: int = 90, minimal_lr: float = 1e-8):
    assert isinstance(optimizer, Optimizer)
    return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=minimal_lr)
