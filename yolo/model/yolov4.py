# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 上午11:20
@file: darknet.py
@author: zj
@description: 
"""
from typing import Dict
from collections import OrderedDict

import os

import torch
from torch import nn

from darknet.darknet import ConvBNAct, ResBlock, DownSample
from .yololayer import YOLOLayer

from yolo.util import logging

logger = logging.get_logger(__name__)


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        self.stem = ConvBNAct(in_ch=3, out_ch=32, kernel_size=3, stride=1)

        self.stage1 = DownSample(in_ch=32, out_ch=64, kernel_size=3, stride=2, num_blocks=1)
        self.stage2 = DownSample(in_ch=64, out_ch=128, kernel_size=3, stride=2, num_blocks=2)
        self.stage3 = DownSample(in_ch=128, out_ch=256, kernel_size=3, stride=2, num_blocks=8)
        self.stage4 = DownSample(in_ch=256, out_ch=512, kernel_size=3, stride=2, num_blocks=8)
        self.stage5 = DownSample(in_ch=512, out_ch=1024, kernel_size=3, stride=2, num_blocks=4)

    def forward(self, x):
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x3 = self.stage3(x)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)

        return x3, x4, x5


class SPPBlock(nn.Module):

    def __init__(self):
        super(SPPBlock, self).__init__()
        self.conv1 = nn.Sequential(
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act="leaky_relu"),
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act="leaky_relu"),
            ConvBNAct(in_ch=1024, out_ch=1024, kernel_size=1, stride=1, act="leaky_relu"),
        )

        self.max_pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.max_pool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        self.conv2 = ConvBNAct(in_ch=2048, out_ch=512, kernel_size=1, stride=1, act='leaky_relu')

    def forward(self, x):
        x = self.conv1(x)
        m1 = self.max_pool1(x)
        m2 = self.max_pool2(x)
        m3 = self.max_pool1(x)

        x = torch.cat([m3, m2, m1, x], dim=1)
        x = self.conv2(x)
        return x


class FPNBlock(nn.Module):

    def __init__(self):
        super(FPNBlock, self).__init__()

        self.module1 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act='leaky_relu')
        )

        self.module2 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv8 = ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1, act='leaky_relu')

        self.module3 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1, act='leaky_relu')
        )

        self.module4 = nn.Sequential(
            ConvBNAct(in_ch=256, out_ch=128, kernel_size=1, stride=1),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )
        self.conv15 = ConvBNAct(in_ch=256, out_ch=128, kernel_size=1, stride=1, act='leaky_relu')

        self.module5 = nn.Sequential(
            ConvBNAct(in_ch=256, out_ch=128, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=128, out_ch=256, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=256, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=128, out_ch=256, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=128, kernel_size=1, stride=1, act='leaky_relu'),
        )

    def forward(self, x, last_x1, last_x2):
        x = self.module1(x)

        x1 = self.module2(x)
        last_x1 = self.conv8(last_x1)
        assert x1.shape[2:] == last_x1.shape[2:]
        x1 = torch.cat((x1, last_x1), dim=1)
        x1 = self.module3(x1)

        x2 = self.module4(x1)
        last_x2 = self.conv15(last_x2)
        assert x2.shape[2:] == last_x2.shape[2:]
        x2 = torch.cat((x2, last_x2), dim=1)

        return x, x1, x2


class PANBlock(nn.Module):

    def __init__(self):
        super(PANBlock, self).__init__()

        self.conv1 = ConvBNAct(in_ch=128, out_ch=256, kernel_size=3, stride=2, act='leaky_relu')
        self.module1 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=256, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1, act='leaky_relu'),
        )

        self.conv8 = ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=2, act='leaky_relu')
        self.module2 = nn.Sequential(
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act='leaky_relu'),
        )

    def forward(self, x1, x2, x3):
        x = self.conv1(x1)
        assert x.shape[2:] == x2.shape[2:]
        x = torch.cat((x, x2), dim=1)
        x2 = self.module1(x)

        x = self.conv8(x2)
        assert x.shape[2:] == x3.shape[2:]
        x3 = torch.cat((x, x3), dim=1)
        x3 = self.module2(x3)

        return x1, x2, x3


class Neck(nn.Module):
    """
    SPP + FPN + PAN
    """

    def __init__(self):
        super(Neck, self).__init__()

        self.spp = SPPBlock()
        self.fpn = FPNBlock()
        self.pan = PANBlock()

    def forward(self, x1, x2, x3):
        x1 = self.spp(x1)

        x1, x2, x3 = self.fpn(x1, x2, x3)
        x1, x2, x3 = self.pan(x1, x2, x3)

        return x1, x2, x3


class Head(nn.Module):

    def __init__(self, cfg: Dict):
        super(Head, self).__init__()

        # 输出通道数：坐标（x/y/w/h）+ 坐标框置信度 + 类别数
        output_channels = (4 + 1 + cfg['N_CLASSES']) * 3

        self.yolo1 = nn.Sequential(
            ConvBNAct(in_ch=128, out_ch=256, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=output_channels, kernel_size=3, stride=1, act='linear'),
        )

        self.yolo2 = nn.Sequential(
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=output_channels, kernel_size=1, stride=1, act='linear'),
        )

        self.yolo3 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=1024, out_ch=output_channels, kernel_size=1, stride=1, act='leaky_relu'),
        )

    def forward(self, x1, x2, x3):
        """
        x1: [B, 128, H/8, W/8]
        x2: [B, 256, H/16, W/16]
        x3: [B, 512, H/32, W/32]
        """
        assert input1.shape[1] == 128
        x1 = self.yolo1(input1)

        x = self.conv3(input1)
        x = torch.cat([x, input2], dim=1)
        assert x.shape[1] == 512
        x2 = self.yolo2(x)

        x = self.conv11()
        assert x3.shape[1] == 1024
        x3 = self.yolo3(x3)

        return x1, x2, x3


class YOLOv4(nn.Module):

    def __init__(self, cfg: Dict):
        super(YOLOv4, self).__init__()
        assert cfg['TYPE'] == 'YOLOv4'

        self.backbone = Backbone()
        self.neck = Neck()
        self.head = Head(cfg)

        self._init(ckpt_path=cfg['BACKBONE_PRETRAINED'])

    def _init(self, ckpt_path=None):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                # nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.weight, 0.01)
                nn.init.constant_(m.bias, 0)
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            logger.info(f'Loading pretrained darknet53: {ckpt_path}')

            ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
            ckpt = OrderedDict({key: ckpt[key] for key in list(filter(lambda x: 'backbone' in x, ckpt.keys()))})
            ckpt = OrderedDict({key.replace("module.backbone.", ""): value for key, value in ckpt.items()})

            self.backbone.load_state_dict(ckpt, strict=True)

    def forward(self, x):
        # x: [B, 3, H, W]
        x1, x2, x3 = self.backbone(x)
        # x1: [B, 256, H/8, W/8]
        # x2: [B, 512, H/16, W/16]
        # x3: [B, 1024, H/32, W/32]
        x1, x2, x3 = self.neck(x3, x2, x1)
        # x1: [B, 512, H/32, W/32]
        # x2: [B, 256, H/16, W/16]
        # x3: [B, 128, H/8, W/8]
        x1, x2, x3 = self.head(x3, x2, x1)
        # x1: [B, H/32 * W/32 * 3, 85]
        # x2: [B, H/16 * W/16 * 3, 85]
        # x3: [B, H/8 * W/8 * 3, 85]

        # res: [B, (H*W + 2H*2W + 4H*4W) / (32*32) * 3, 85]
        #     =[B, H*W*63 / (32*32), 85
        if self.training:
            return [x1, x2, x3]
        else:
            return torch.cat((x1, x2, x3), 1)


if __name__ == '__main__':
    cfg_file = 'config/yolov3_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    # 创建YOLOv3
    model = YOLOv3(cfg['MODEL'])
    # model = ConvBNAct(in_ch=3, out_ch=3, kernel_size=1, stride=1)
    print(model)
    model.eval()

    import random
    import numpy as np

    seed = 1  # seed必须是int，可以自行设置
    random.seed(seed)
    np.random.seed(seed)  # numpy产生的随机数一致
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 让显卡产生的随机数一致
        torch.cuda.manual_seed_all(seed)  # 多卡模式下，让所有显卡生成的随机数一致？这个待验证
        # CUDA中的一些运算，如对sparse的CUDA张量与dense的CUDA张量调用torch.bmm()，它通常使用不确定性算法。
        # 为了避免这种情况，就要将这个flag设置为True，让它使用确定的实现。
        torch.backends.cudnn.deterministic = True

        # 设置这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。
        # 但是由于噪声和不同的硬件条件，即使是同一台机器，benchmark都可能会选择不同的算法。为了消除这个随机性，设置为 False
        torch.backends.cudnn.benchmark = False

    # data = torch.randn((1, 3, 416, 416))
    # data = torch.randn((1, 3, 608, 608))
    data = torch.randn((1, 3, 640, 640))
    print(data.reshape(-1)[:20])
    outputs = model(data)
    print(outputs.shape)

    print(outputs.reshape(-1)[:20])
