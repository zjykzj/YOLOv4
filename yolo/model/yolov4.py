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

from darknet.darknet import ConvBNAct, CSPDownSample0, CSPDownSample
from .yololayer import YOLOLayer

from yolo.util import logging

logger = logging.get_logger(__name__)


class Backbone(nn.Module):

    def __init__(self):
        super(Backbone, self).__init__()
        self.stem = ConvBNAct(in_ch=3, out_ch=32, kernel_size=3, stride=1, act='mish')

        self.stage1 = CSPDownSample0(in_ch=32, out_ch=64, kernel_size=3, stride=2, act='mish')
        self.stage2 = CSPDownSample(in_ch=64, out_ch=128, kernel_size=3, stride=2, num_blocks=2, act='mish')
        self.stage3 = CSPDownSample(in_ch=128, out_ch=256, kernel_size=3, stride=2, num_blocks=8, act='mish')
        self.stage4 = CSPDownSample(in_ch=256, out_ch=512, kernel_size=3, stride=2, num_blocks=8, act='mish')
        self.stage5 = CSPDownSample(in_ch=512, out_ch=1024, kernel_size=3, stride=2, num_blocks=4, act='mish')

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
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act="leaky_relu"),
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
            ConvBNAct(in_ch=256, out_ch=128, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=128, out_ch=256, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=128, kernel_size=1, stride=1, act='leaky_relu'),
        )

    def forward(self, x3, x4, x5):
        """
        x3: [B, 256, H/8, H/8] --> f1
        x4: [B, 512, H/16, H/16] --> f2
        x5: [B, 512, H/32, H/32] --> f3
        """
        f3 = self.module1(x5)

        f2 = self.module2(x5)
        x4 = self.conv8(x4)
        assert f2.shape[2:] == x4.shape[2:]
        f2 = torch.cat((x4, f2), dim=1)
        f2 = self.module3(f2)

        f1 = self.module4(f2)
        x3 = self.conv15(x3)
        assert f1.shape[2:] == x3.shape[2:]
        f1 = torch.cat((x3, f1), dim=1)
        f1 = self.module5(f1)

        return f1, f2, f3


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
        )

        self.conv7 = ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=2, act='leaky_relu')
        self.module2 = nn.Sequential(
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=1024, out_ch=512, kernel_size=1, stride=1, act='leaky_relu'),
        )

    def forward(self, f1, f2, f3):
        # f1: [B, 128, H/8, W/8]
        # f2: [B, 256, H/16, W/16]
        # f3: [B, 512, H/32, W/32]
        p1 = f1

        p2 = self.conv1(f1)
        assert p2.shape[2:] == f2.shape[2:]
        p2 = torch.cat((p2, f2), dim=1)
        p2 = self.module1(p2)

        p3 = self.conv7(p2)
        assert p3.shape[2:] == f3.shape[2:]
        p3 = torch.cat((p3, f3), dim=1)
        p3 = self.module2(p3)

        return p1, p2, p3


class Neck(nn.Module):
    """
    SPP + FPN + PAN
    """

    def __init__(self):
        super(Neck, self).__init__()

        self.spp = SPPBlock()
        self.fpn = FPNBlock()
        self.pan = PANBlock()

    def forward(self, x3, x4, x5):
        """
        x3: [B, 256, H/8, W/8]
        x4: [B, 512, H/16, W/16]
        x5: [B, 1024, H/32, W/32]
        """
        spp_output = self.spp(x5)
        # spp_output: [B, 512, H/32, W/32]

        f1, f2, f3 = self.fpn(x3, x4, spp_output)
        # f1: [B, 128, H/8, W/8]
        # f2: [B, 256, H/16, W/16]
        # f3: [B, 512, H/32, W/32]
        p1, p2, p3 = self.pan(f1, f2, f3)
        # p1: [B, 128, H/8, W/8]
        # p2: [B, 256, H/16, W/16]
        # p3: [B, 512, H/32, W/32]

        return p1, p2, p3


class Head(nn.Module):

    def __init__(self, cfg: Dict):
        super(Head, self).__init__()

        # 输出通道数：坐标（x/y/w/h）+ 坐标框置信度 + 类别数
        output_channels = (4 + 1 + cfg['N_CLASSES']) * 3

        self.yolo1 = nn.Sequential(
            ConvBNAct(in_ch=128, out_ch=256, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=256, out_ch=output_channels, kernel_size=3, stride=1, act='linear'),
            YOLOLayer(cfg, layer_no=0)
        )

        self.yolo2 = nn.Sequential(
            ConvBNAct(in_ch=256, out_ch=512, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=512, out_ch=output_channels, kernel_size=1, stride=1, act='linear'),
            YOLOLayer(cfg, layer_no=1)
        )

        self.yolo3 = nn.Sequential(
            ConvBNAct(in_ch=512, out_ch=1024, kernel_size=3, stride=1, act='leaky_relu'),
            ConvBNAct(in_ch=1024, out_ch=output_channels, kernel_size=1, stride=1, act='leaky_relu'),
            YOLOLayer(cfg, layer_no=2)
        )

    def forward(self, p1, p2, p3):
        """
        p1: [B, 128, H/8, W/8]
        p2: [B, 256, H/16, W/16]
        p3: [B, 512, H/32, W/32]
        """
        assert p1.shape[1] == 128
        x1 = self.yolo1(p1)

        assert p2.shape[1] == 256
        x2 = self.yolo2(p2)

        assert p3.shape[1] == 512
        x3 = self.yolo3(p3)

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
            logger.info(f'Loading pretrained cspdarknet53: {ckpt_path}')

            ckpt = torch.load(ckpt_path, map_location='cpu')['state_dict']
            ckpt = OrderedDict({key: ckpt[key] for key in list(filter(lambda x: 'backbone' in x, ckpt.keys()))})
            ckpt = OrderedDict({key.replace("module.backbone.", ""): value for key, value in ckpt.items()})

            self.backbone.load_state_dict(ckpt, strict=True)

    def forward(self, x):
        # x: [B, 3, H, W]
        x3, x4, x5 = self.backbone(x)
        # x3: [B, 256, H/8, W/8]
        # x4: [B, 512, H/16, W/16]
        # x5: [B, 1024, H/32, W/32]
        p1, p2, p3 = self.neck(x3, x4, x5)
        # p1: [B, 128, H/8, W/8]
        # p2: [B, 256, H/16, W/16]
        # p3: [B, 512, H/32, W/32]
        x1, x2, x3 = self.head(p1, p2, p3)
        # x1: [B, H/16 * W/16 * 3, 85]
        # x2: [B, H/8 * W/8 * 3, 85]
        # x3: [B, H/32 * W/32 * 3, 85]

        # res: [B, (H*W + 2H*2W + 4H*4W) / (32*32) * 3, 85]
        #     =[B, H*W*63 / (32*32), 85
        if self.training:
            return [x1, x2, x3]
        else:
            return torch.cat((x1, x2, x3), 1)


if __name__ == '__main__':
    cfg_file = 'config/yolov4_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    # 创建YOLOv3
    model = YOLOv4(cfg['MODEL'])
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
