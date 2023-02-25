# -*- coding: utf-8 -*-

from __future__ import division
from typing import Dict

import yaml
import torch.cuda

import argparse
from argparse import Namespace

from yolo.data.transform import Transform
from yolo.data.cocodataset import COCODataset
from yolo.model.yolov4 import YOLOv4
from yolo.engine.build import validate

"""
操作流程：

1. 解析命令行参数 + 配置文件
2. 创建模型，初始化权重
3. 创建COCO评估器
4. 开始评估
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-c', '--cfg', type=str, default='config/yolov4_default.cfg',
                        help='config file. see readme')
    parser.add_argument('-ckpt', '--checkpoint', type=str,
                        help='pytorch checkpoint file path')
    return parser.parse_args()


def data_init(args: Namespace, cfg: Dict):
    val_transform = Transform(cfg, is_train=False)
    val_dataset = COCODataset(root=args.data,
                              name='val2017',
                              img_size=cfg['TEST']['IMGSIZE'],
                              model_type=cfg['MODEL']['TYPE'],
                              is_train=False,
                              transform=val_transform
                              )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=False, num_workers=0, pin_memory=True, sampler=None)

    return val_loader


def main():
    """
    YOLOv3 trainer. See README for details.
    """
    args = parse_args()
    print("Setting Arguments.. : ", args)

    # Parse config settings
    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)

    print("successfully loaded config file: ", args.cfg)

    # Initiate model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # YOLOv3还是通过模型定义方式获取YOLO模型！！！
    model = YOLOv4(cfg['MODEL'], device=device).to(device)

    # 预训练权重加载，共两种方式
    if args.checkpoint:
        print("=> loading checkpoint '{}'".format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=device)

        state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)

    val_loader = data_init(args, cfg)

    print("Begin evaluating ...")
    conf_thresh = cfg['TEST']['CONFTHRE']
    nms_thresh = float(cfg['TEST']['NMSTHRE'])
    ap50_95, ap50 = validate(val_loader, model, conf_thresh, nms_thresh, device=device)


if __name__ == '__main__':
    main()
