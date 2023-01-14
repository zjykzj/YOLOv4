# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 下午4:39
@file: yololayer.py
@author: zj
@description: 
"""

import numpy as np

import torch
from torch import nn


class YOLOLayer(nn.Module):
    """
    操作流程：

    1. 卷积操作
        执行1x1卷积操作，通道数为(锚点框数量 * (类别数 + 5))，默认为3*(80+5)=255
    2. 数据格式转换
        改变特征数据形状： [B, 锚点框数量*(类别数+5), F_H, F_W] -> [B, 锚点框数量, 类别数+5, F_H, F_W]
        改变特征数据维度： [B, 锚点框数量, 类别数+5, F_H, F_W] -> [B, 锚点框数量, F_H, F_W, 类别数+5]
    3. 预测框坐标、目标置信度以及分类概率计算
        b_x = sigmoid(t_x) + c_x
        b_y = sigmoid(t_y) + c_y
        b_w = p_w * exp(t_w)
        b_h = p_h * exp(t_h)

        针对预测框左上角坐标（前2位xy）、目标置信度（第4位）以及分类概率（后80位）执行sigmoid操作，将数值压缩到(0, 1)之间）
            预测框坐标： sigmoid([B, 锚点框数量, F_H, F_W, :4])
            目标置信度＋分类概率： sigmoid([B, 锚点框数量, F_H, F_W, 4:])
        计算网格坐标
            x_shift: [0, 1, 2, ..., F_W - 1] -> [B, n_anchors, F_H, F_W]
            y_shift: [0, 1, 2, ..., F_H - 1] -> [B, n_anchors, F_H, F_W]
        预测框坐标b_x/b_y分别加上每个网格的左上角坐标c_x/c_y
            b_x: [B, 锚点框数量, F_H, F_W, 0] + [B, n_anchors, F_H, F_W]
            b_y: [B, 锚点框数量, F_H, F_W, 1] + [B, n_anchors, F_H, F_W]
        计算锚点框
            w_anchors: [n_anchors] -> [1, n_anchors, 1, 1] -> [B, n_anchors, F_H, F_W]
            h_anchors: [n_anchors] -> [1, n_anchors, 1, 1] -> [B, n_anchors, F_H, F_W]
        预测框坐标b_w/b_h分别进行指数运算后乘以锚点框宽高w_anchors/h_anchors
            b_w: exp([B, 锚点框数量, F_H, F_W, 2]) * [B, n_anchors, F_H, F_W]
            b_h: exp([B, 锚点框数量, F_H, F_W, 3]) * [B, n_anchors, F_H, F_W]
    4. 计算实际的预测框坐标
        坐标转换, 将预测框坐标转换回原始图像大小
            [B, 锚点框数量, F_H, F_W, :4] * 输入特征相对于原始图像空间尺寸的缩放倍数
        整理所有网格计算得到的预测框坐标
        [B, n_anchors, F_H, F_W, n_ch] -> [B, n_anchors * F_H * F_W, n_ch]
    """

    def __init__(self, cfg, layer_no, in_ch):
        super(YOLOLayer, self).__init__()

        # 预先设定的缩放倍数
        strides = [32, 16, 8]
        # 当前YOLOLayer使用的缩放倍数
        self.stride = strides[layer_no]
        self.layer_no = layer_no

        # 预定义的所有锚点框
        self.anchors = cfg['ANCHORS']
        # 获取当前YOLO层使用的锚点框
        self.anch_mask = cfg['ANCHOR_MASK'][layer_no]
        # 获取当前YOLO层使用的锚点框个数，默认为3
        self.n_anchors = len(self.anch_mask)
        # 按照指定倍数进行缩放
        self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
        self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]

        # 数据集类别数
        self.n_classes = cfg['N_CLASSES']
        # 1x1卷积操作，计算特征图中每个网格的预测框（锚点框数量*(类别数+4(xywh)+1(置信度))）
        self.conv = nn.Conv2d(in_channels=in_ch,
                              out_channels=self.n_anchors * (self.n_classes + 5),
                              kernel_size=(1, 1), stride=(1, 1), padding=0)

    def forward(self, x):
        output = self.conv(x)

        # 批量大小
        batchsize = output.shape[0]
        # 特征图空间尺寸
        fsize = output.shape[2]
        # 输出通道数
        # n_ch = 4(xywh) + 1(conf) + n_classes
        n_ch = 5 + self.n_classes
        dtype = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        # [B, C_out, F_H, F_W] -> [B, n_anchors, n_ch, F_H, F_W]
        # C_out = n_anchors * (5 + n_classes)
        output = output.view(batchsize, self.n_anchors, n_ch, fsize, fsize)
        # [B, n_anchors, n_ch, F_H, F_W] -> [B, n_anchors, F_H, F_W, n_ch]
        output = output.permute(0, 1, 3, 4, 2)  # .contiguous()

        # logistic activation for xy, obj, cls
        # 针对预测坐标(xy)和预测分类结果执行sigmoid运算，将数值归一化到(0, 1)之间
        output[..., np.r_[:2, 4:n_ch]] = torch.sigmoid(output[..., np.r_[:2, 4:n_ch]])

        # calculate pred - xywh obj cls
        # 网格坐标
        # [0, 1, 2, ..., F_W - 1] -> [B, n_anchors, F_H, F_W]
        x_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32), output.shape[:4]))
        # [0, 1, 2, ..., F_H - 1] -> [F_H, 1] -> [B, n_anchors, F_H, F_W]
        y_shift = dtype(np.broadcast_to(np.arange(fsize, dtype=np.float32).reshape(fsize, 1), output.shape[:4]))

        # [n_anchors, 2]
        masked_anchors = np.array(self.masked_anchors)

        # [n_anchors] -> [1, n_anchors, 1, 1] -> [B, n_anchors, F_H, F_W]
        w_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 0], (1, self.n_anchors, 1, 1)), output.shape[:4]))
        # [n_anchors] -> [1, n_anchors, 1, 1] -> [B, n_anchors, F_H, F_W]
        h_anchors = dtype(np.broadcast_to(np.reshape(
            masked_anchors[:, 1], (1, self.n_anchors, 1, 1)), output.shape[:4]))

        pred = output.clone()
        # 预测框坐标x0加上每个网格的左上角坐标x
        # b_x = sigmoid(t_x) + c_x
        pred[..., 0] += x_shift
        # 预测框坐标y0加上每个网格的左上角坐标y
        # b_y = sigmoid(t_y) + c_y
        pred[..., 1] += y_shift
        # 计算预测框长/宽的实际长度
        # b_w = exp(t_w) * p_w
        pred[..., 2] = torch.exp(pred[..., 2]) * w_anchors
        # b_h = exp(t_h) * p_h
        pred[..., 3] = torch.exp(pred[..., 3]) * h_anchors

        if self.training:
            res = dict({
                'layer_no': self.layer_no,
                # output: [B, n_anchors*(5+n_classes), F_H, F_W]
                # 5 = xywh+conf
                'output': output,
                # pred[..., :4]: [B, n_anchors, F_H, F_W, 4]
                # 4 = xywh
                'pred': pred[..., :4]
            })
            return res
        else:
            # 推理阶段，不计算损失
            # 将预测框坐标按比例返回到原图大小
            pred[..., :4] *= self.stride
            # [B, n_anchors, F_H, F_W, n_ch] -> [B, n_anchors * F_H * F_W, n_ch]
            # return pred.view(batchsize, -1, n_ch).data
            return pred.reshape(batchsize, -1, n_ch)
