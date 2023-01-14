# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午10:58
@file: build.py
@author: zj
@description: 
"""
from typing import Dict

from torch import nn
from torch.nn import Module

from .sgd import build_sgd


def build_optimizer(cfg: Dict, model: Module):
    optimizer_type = cfg['OPTIMIZER']['TYPE']
    lr = cfg['OPTIMIZER']['LR']
    momentum = cfg['OPTIMIZER']['MOMENTUM']
    weight_decay = cfg['OPTIMIZER']['DECAY']

    # optimizer setup
    groups = filter_weight(cfg, model)

    if 'SGD' == optimizer_type:
        optimizer = build_sgd(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"{optimizer_type} does not support.")

    optimizer.zero_grad()
    return optimizer


def filter_weight(cfg: Dict, module: Module):
    """
    1. Avoid bias of all layers and normalization layer for weight decay.
    2. And filter all layers which require_grad=False
    refer to
    1. [Allow to set 0 weight decay for biases and params in batch norm #1402](https://github.com/pytorch/pytorch/issues/1402)
    2. [Weight decay in the optimizers is a bad idea (especially with BatchNorm)](https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994)
    """
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                if cfg['OPTIMIZER']['NO_BIAS'] is True:
                    group_no_decay.append(m.bias)
                else:
                    group_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                if cfg['OPTIMIZER']['NO_BIAS'] is True:
                    group_no_decay.append(m.bias)
                else:
                    group_decay.append(m.bias)
        elif isinstance(m, (nn.modules.batchnorm._BatchNorm, nn.GroupNorm, nn.LayerNorm)):
            if cfg['OPTIMIZER']['NO_NORM'] is True:
                if m.weight is not None:
                    group_no_decay.append(m.weight)
                if m.bias is not None:
                    group_no_decay.append(m.bias)
            else:
                if m.weight is not None:
                    group_decay.append(m.weight)
                if m.bias is not None:
                    group_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)

    new_group_decay = filter(lambda p: p.requires_grad, group_decay)
    new_group_no_decay = filter(lambda p: p.requires_grad, group_no_decay)
    groups = [dict(params=new_group_decay), dict(params=new_group_no_decay, weight_decay=0.)]
    return groups
