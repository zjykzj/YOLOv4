# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午4:42
@file: multistep_lr.py
@author: zj
@description: 
"""
from typing import List

import torch.optim as optim
from torch.optim.optimizer import Optimizer


def build_multi_step_lr(optimizer: Optimizer, milestones: List = None, gamma: float = 0.1):
    if milestones is None:
        milestones = [30, 60, 80]
    assert isinstance(optimizer, Optimizer)

    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
