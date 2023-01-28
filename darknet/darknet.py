# -*- coding: utf-8 -*-

"""
@date: 2023/1/7 下午12:26
@file: darknet.py
@author: zj
@description: CSPDarknet53. Refer to https://github.com/Tianxiaomo/pytorch-YOLOv4/blob/master/models.py
"""

import torch
from torch import nn


class ConvBNAct(nn.Module):

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int, act='leaky_relu'):
        super().__init__()
        pad = (kernel_size - 1) // 2
        # H_out = floor((H_in + 2 * Pad - Dilate * (Kernel - 1) - 1) / Stride + 1)
        #       = floor((H_in + 2 * (Kernel - 1) // 2 - Dilate * (Kernel - 1) - 1) / Stride + 1)

        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=out_ch,
                              kernel_size=(kernel_size, kernel_size),
                              stride=(stride, stride),
                              padding=pad,
                              bias=False)
        self.norm = nn.BatchNorm2d(out_ch)

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'leaky_relu':
            self.act = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        elif act == 'mish':
            self.act = nn.Mish(inplace=False)
        elif act == 'linear':
            self.act = nn.Identity()
        else:
            raise ValueError(f"{act} does not support.")

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)

        return x


class ResBlock(nn.Module):

    def __init__(self, ch, num_blocks=1, shortcut=True, act="mish"):

        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(num_blocks):
            self.module_list.append(nn.Sequential(
                # 1x1卷积，不改变通道数和空间尺寸
                ConvBNAct(ch, ch, 1, 1, act=act),
                # 3x3卷积，不改变通道数和空间尺寸
                ConvBNAct(ch, ch, 3, 1, act=act)
            ))

    def forward(self, x):
        for module in self.module_list:
            h = x
            h = module(h)
            x = x + h if self.shortcut else h
        return x


class CSPDownSample0(nn.Module):

    def __init__(self, in_ch=32, out_ch=64, kernel_size=3, stride=2, act='mish'):
        super(CSPDownSample0, self).__init__()
        self.base = ConvBNAct(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, act=act)
        self.part1 = ConvBNAct(in_ch=out_ch, out_ch=out_ch, kernel_size=1, stride=1, act=act)
        self.part2 = nn.Sequential(
            ConvBNAct(in_ch=out_ch, out_ch=out_ch, kernel_size=1, stride=1, act=act),
            ConvBNAct(in_ch=out_ch, out_ch=out_ch // 2, kernel_size=1, stride=1, act=act),
            ConvBNAct(in_ch=out_ch // 2, out_ch=out_ch, kernel_size=3, stride=1, act=act),
            ConvBNAct(in_ch=out_ch, out_ch=out_ch, kernel_size=1, stride=1, act=act),
        )
        self.transition = ConvBNAct(in_ch=out_ch * 2, out_ch=out_ch, kernel_size=1, stride=1, act=act)

    def forward(self, x):
        x = self.base(x)

        x1 = self.part1(x)
        x2 = self.part2(x)

        x = torch.cat([x2, x1], dim=1)
        x = self.transition(x)

        return x


class CSPDownSample(nn.Module):

    def __init__(self, in_ch=64, out_ch=128, kernel_size=3, stride=2, num_blocks=1, shortcut=True, act='mish'):
        super(CSPDownSample, self).__init__()
        self.base = ConvBNAct(in_ch=in_ch, out_ch=out_ch, kernel_size=kernel_size, stride=stride, act=act)
        self.part1 = ConvBNAct(in_ch=out_ch, out_ch=out_ch // 2, kernel_size=1, stride=1, act=act)
        self.part2 = nn.Sequential(
            ConvBNAct(in_ch=out_ch, out_ch=out_ch // 2, kernel_size=1, stride=1, act=act),
            ResBlock(out_ch // 2, num_blocks=num_blocks, shortcut=shortcut, act=act),
            ConvBNAct(in_ch=out_ch // 2, out_ch=out_ch // 2, kernel_size=1, stride=1, act=act)
        )
        self.transition = ConvBNAct(in_ch=out_ch, out_ch=out_ch, kernel_size=1, stride=1, act=act)

    def forward(self, x):
        x = self.base(x)

        x1 = self.part1(x)
        x2 = self.part2(x)

        x = torch.cat([x2, x1], dim=1)
        x = self.transition(x)

        return x


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
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        return x


class CSPDarknet53(nn.Module):

    def __init__(self, num_classes=1000):
        super(CSPDarknet53, self).__init__()
        self.backbone = Backbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x)
        x = x.reshape(x.shape[:2])

        x = self.classifier(x)
        return x


if __name__ == '__main__':
    m = CSPDarknet53()
    # data = torch.randn(1, 3, 224, 224)
    data = torch.randn(1, 3, 256, 256)

    output = m(data)
    print(output.shape)
