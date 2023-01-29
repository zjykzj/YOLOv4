# -*- coding: utf-8 -*-

"""
@date: 2023/1/29 上午10:42
@file: model.py
@author: zj
@description: 
"""

import time
import torch
import torch.nn as nn

from yolo.util.utils import init_seed
from yolo.model.yolov4 import YOLOv4, CSPDownSample0, Backbone, SPPBlock, FPNBlock

from darknet.darknet import ConvBNAct


class TinyNet(nn.Module):

    def __init__(self):
        super(TinyNet, self).__init__()
        # self.conv = ConvBNAct(in_ch=3, out_ch=32, kernel_size=3, stride=1, act='mish')

        # self.stem = ConvBNAct(in_ch=3, out_ch=32, kernel_size=3, stride=1, act='mish')
        # self.stage1 = CSPDownSample0(in_ch=32, out_ch=64, kernel_size=3, stride=2, act='mish')

        self.backbone = Backbone()
        self.spp = SPPBlock()
        self.fpn = FPNBlock()

        self._init()

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

    def forward(self, x):
        # x = self.conv(x)
        # x = self.stem(x)
        # x = self.stage1(x)
        x3, x4, x5 = self.backbone(x)
        spp_output = self.spp(x5)
        # spp_output: [B, 512, H/32, W/32]

        f1, f2, f3 = self.fpn(x3, x4, spp_output)
        return f1, f2, f3


def main():
    cfg_file = 'config/yolov4_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    init_seed()
    data = torch.randn((1, 3, 640, 640))

    # 创建YOLOv3
    model = YOLOv4(cfg['MODEL'])
    # model = ConvBNAct(in_ch=3, out_ch=3, kernel_size=1, stride=1)
    print(model)

    # if torch.cuda.is_available():
    #     device = torch.device('cuda:0')
    # else:
    device = torch.device('cpu')
    # model = model.to(device)
    model.eval()

    # init_seed()

    t0 = time.time()
    # data = torch.randn((1, 3, 416, 416))
    # data = torch.randn((1, 3, 608, 608))
    print(data.reshape(-1)[:20])
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


def main2():
    init_seed()

    model = TinyNet()
    model.eval()

    import torch

    data = torch.randn(1, 3, 640, 640)
    print(data.reshape(-1)[:20])

    # output = model(data)
    # print(output.reshape(-1)[:20])

    x3, x4, x5 = model(data)
    print(x3.reshape(-1)[:20])


if __name__ == '__main__':
    # main()
    main2()
