# -*- coding: utf-8 -*-

"""
@date: 2022/12/14 上午11:19
@file: build.py
@author: zj
@description: 
"""

from typing import Dict
from argparse import Namespace

import torch

from .cocodataset import COCODataset
from .transform import Transform


def build_data(args: Namespace, cfg: Dict):
    # 创建转换器
    train_transform = Transform(cfg, is_train=True)
    val_transform = Transform(cfg, is_train=False)

    # 创建数据集
    train_dataset = COCODataset(root=args.data,
                                name='train2017',
                                img_size=cfg['TRAIN']['IMGSIZE'],
                                model_type=cfg['MODEL']['TYPE'],
                                is_train=True,
                                transform=train_transform,
                                max_num_labels=cfg['DATA']['MAX_NUM_LABELS'],
                                is_mosaic=cfg['DATA']['AUGMENTATION']
                                )
    val_dataset = COCODataset(root=args.data,
                              name='val2017',
                              img_size=cfg['TEST']['IMGSIZE'],
                              model_type=cfg['MODEL']['TYPE'],
                              is_train=False,
                              transform=val_transform,
                              max_num_labels=cfg['DATA']['MAX_NUM_LABELS'],
                              )

    # 创建采样器
    train_sampler = None
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # 创建加载器
    collate_fn = torch.utils.data.default_collate
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['DATA']['BATCH_SIZE'], shuffle=(train_sampler is None),
        num_workers=cfg['DATA']['WORKERS'], pin_memory=True, sampler=train_sampler, collate_fn=collate_fn)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=None)

    return train_sampler, train_loader, val_loader
