# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午5:01
@file: sgd.py
@author: zj
@description: 
"""
from typing import List

import torch.optim as optim


def build_sgd(groups: List, lr: float = 0.1, momentum: float = 0.9, weight_decay: float = 1e-5):
    return optim.SGD(groups, lr=lr, momentum=momentum, weight_decay=weight_decay)
