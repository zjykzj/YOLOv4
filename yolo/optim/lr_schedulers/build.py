# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午4:42
@file: build.py
@author: zj
@description: 
"""
from typing import Dict

from torch.optim.optimizer import Optimizer

from .multistep_lr import build_multi_step_lr
from .cosine_annealing_lr import build_cosine_annealing_lr


def adjust_learning_rate(cfg: Dict, optimizer: Optimizer, epoch: int, step: int, len_epoch: int) -> None:
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    lr = float(cfg['OPTIMIZER']['LR'])

    warmup_epoch = int(cfg['LR_SCHEDULER']['WARMUP_EPOCH'])
    # Warmup
    if epoch < warmup_epoch:
        lr = lr * float(1 + step + epoch * len_epoch) / (warmup_epoch * len_epoch)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def build_lr_scheduler(cfg: Dict, optimizer: Optimizer):
    assert isinstance(optimizer, Optimizer)

    lr_scheduler_type = cfg['LR_SCHEDULER']['TYPE']
    is_warmup = cfg['LR_SCHEDULER']['IS_WARMUP']
    warmup_epoch = int(cfg['LR_SCHEDULER']['WARMUP_EPOCH'])

    if 'MultiStepLR' == lr_scheduler_type:
        milestones = list(cfg['LR_SCHEDULER']['MILESTONES'])
        assert isinstance(milestones, (list, tuple)), print(type(milestones), milestones)
        if is_warmup:
            milestones = [stone - warmup_epoch for stone in milestones]
        gamma = float(cfg['LR_SCHEDULER']['GAMMA'])
        lr_scheduler = build_multi_step_lr(optimizer, milestones=milestones, gamma=gamma)
    elif 'CosineAnnealingLR' == lr_scheduler_type:
        max_epoch = int(cfg['TRAIN']['MAX_EPOCHS'])
        if is_warmup:
            max_epoch -= warmup_epoch
        minimal_lr = float(cfg['LR_SCHEDULER']['MINIMAL_LR'])
        lr_scheduler = build_cosine_annealing_lr(optimizer,
                                                 max_epoch=max_epoch, minimal_lr=minimal_lr)
    else:
        raise ValueError(f"{lr_scheduler_type} does not support.")

    return lr_scheduler
