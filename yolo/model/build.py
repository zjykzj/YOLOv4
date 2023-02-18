# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:00
@file: build.py
@author: zj
@description:
"""

from argparse import Namespace
from typing import Dict

import torch

from .yolov4 import YOLOv4
from .yololoss import YOLOLoss


def build_model(args: Namespace, cfg: Dict, device=None):
    if args.channels_last:
        memory_format = torch.channels_last
    else:
        memory_format = torch.contiguous_format

    model = YOLOv4(cfg['MODEL'], device=device)
    model = model.to(memory_format=memory_format, device=device)

    return model


def build_criterion(cfg: Dict, device=None):
    criterion = YOLOLoss(cfg['MODEL'], ignore_thresh=float(cfg['CRITERION']['IGNORE_THRESH']), device=device).to(device)
    return criterion
