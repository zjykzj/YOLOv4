# -*- coding: utf-8 -*-

"""
@date: 2023/1/29 上午10:42
@file: model.py
@author: zj
@description: 
"""

import time
import torch

from yolo.model.yolov4 import YOLOv4

if __name__ == '__main__':
    cfg_file = 'config/yolov4_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    # 创建YOLOv3
    model = YOLOv4(cfg['MODEL'])
    # model = ConvBNAct(in_ch=3, out_ch=3, kernel_size=1, stride=1)
    print(model)

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    model = model.to(device)
    model.eval()

    from yolo.util.utils import init_seed

    init_seed()

    t0 = time.time()
    # data = torch.randn((1, 3, 416, 416))
    # data = torch.randn((1, 3, 608, 608))
    data = torch.randn((1, 3, 640, 640))
    # print(data.reshape(-1)[:20])
    t1 = time.time()

    outputs = model(data.to(device))
    # print(outputs.shape)
    t2 = time.time()

    print('-----------------------------------')
    print('           Preprocess : %f' % (t1 - t0))
    print('      Model Inference : %f' % (t2 - t1))
    print('-----------------------------------')
    print(outputs.reshape(-1)[:20])
