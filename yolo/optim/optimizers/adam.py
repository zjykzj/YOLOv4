# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午5:01
@file: sgd.py
@author: zj
@description: 
"""
from typing import List

import torch.optim as optim


def build_adam(groups: List, lr: float = 0.1):
    return optim.Adam(groups, lr=lr, betas=(0.9, 0.999), eps=1e-08)
