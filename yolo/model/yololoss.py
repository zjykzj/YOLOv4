# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 下午5:51
@file: yololoss.py
@author: zj
@description: 
"""
from typing import Dict
import numpy as np

import torch
from torch import nn


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.
    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.
    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    from: https://github.com/chainer/chainercv
    """
    # bboxes_a: [N_a, 4]
    # bboxes_b: [N_b, 4]
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    # top left
    if xyxy:
        # xyxy: x_top_left, y_top_left, x_bottom_right, y_bottom_right
        # 计算交集矩形的左上角坐标
        # torch.max([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        # torch.max: 双重循环
        #   第一重循环 for i in range(N_a)，遍历boxes_a, 获取边界框i，大小为[2]
        #       第二重循环　for j in range(N_b)，遍历bboxes_b，获取边界框j，大小为[2]
        #           分别比较i[0]/j[0]和i[1]/j[1]，获取得到最大的x/y
        #   遍历完成后，获取得到[N_a, N_b, 2]
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        # bottom right
        # 计算交集矩形的右下角坐标
        # torch.min([N_a, 1, 2], [N_b, 2]) -> [N_a, N_b, 2]
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        # 计算bboxes_a的面积
        # x_bottom_right/y_bottom_right - x_top_left/y_top_left = w/h
        # prod([N, w/h], 1) = [N], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        # x_center/y_center -> x_top_left, y_top_left
        tl = torch.max((bboxes_a[:, None, :2] - bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2))
        # bottom right
        # x_center/y_center -> x_bottom_right/y_bottom_right
        br = torch.min((bboxes_a[:, None, :2] + bboxes_a[:, None, 2:] / 2),
                       (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2))

        # prod([N_a, w/h], 1) = [N_a], 每个item表示边界框的面积w*h
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    # 判断符合条件的结果：x_top_left/y_top_left < x_bottom_right/y_bottom_right
    # [N_a, N_b, 2] < [N_a, N_b, 2] = [N_a, N_b, 2]
    # prod([N_a, N_b, 2], 2) = [N_a, N_b], 数值为1/0
    en = (tl < br).type(tl.type()).prod(dim=2)
    # 首先计算交集w/h: [N_a, N_b, 2] - [N_a, N_b, 2] = [N_a, N_b, 2]
    # 然后计算交集面积：prod([N_a, N_b, 2], 2) = [N_a, N_b]
    # 然后去除不符合条件的交集面积
    # [N_a, N_b] * [N_a, N_b](数值为1/0) = [N_a, N_b]
    # 大小为[N_a, N_b]，表示bboxes_a的每个边界框与bboxes_b的每个边界框之间的IoU
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())

    # 计算IoU
    # 首先计算所有面积
    # area_a[:, None] + area_b - area_i =
    # [N_a, 1] + [N_b] - [N_a, N_b] = [N_a, N_b]
    # 然后交集面积除以所有面积，计算IoU
    # [N_a, N_b] / [N_a, N_b] = [N_a, N_b]
    return area_i / (area_a[:, None] + area_b - area_i)


class YOLOLoss(nn.Module):
    """
    操作流程：


    """

    def __init__(self, cfg: Dict, ignore_thresh=0.7):
        super(YOLOLoss, self).__init__()
        self.cfg = cfg
        # 阈值
        self.ignore_thresh = ignore_thresh

        # 特征图相对于
        # [3]
        self.strides = [32, 16, 8]  # fixed
        # 预设的锚点框列表，保存了所有的锚点框长宽
        # [9, 2]
        self.anchors = cfg['ANCHORS']
        # 数据集类别数
        # COCO: 80
        self.n_classes = cfg['N_CLASSES']

        # 损失函数，work for ???
        self.l2_loss = nn.MSELoss(size_average=False)
        self.bce_loss = nn.BCELoss(size_average=False)

    def forward(self, outputs, target):
        assert isinstance(target, dict)
        # [B, K(每幅图片拥有的真值标签框数目), 5(cls_id + bbox)]
        labels = target['padded_labels'].cuda()
        # print(labels)
        # print("img_info:", target['img_info'])

        assert isinstance(outputs, list)
        loss_list = []
        for output_dict in outputs:
            # 逐图像进行损失计算
            assert isinstance(output_dict, dict)
            # 获取当前特征层下标
            layer_no = output_dict['layer_no']
            # 获取当前YOLO层特征数据使用的锚点框
            # [3, 3] -> [3]
            self.anch_mask = self.cfg['ANCHOR_MASK'][layer_no]
            # 当前YOLO层特征数据使用的锚点框个数，默认为3
            self.n_anchors = len(self.anch_mask)

            # 第N个YOLO层使用的步长，也就是输入图像大小和使用的特征数据之间的缩放比率
            self.stride = self.strides[layer_no]
            # 按比例缩放锚点框长／宽
            # [9, 2]
            self.all_anchors_grid = [(w / self.stride, h / self.stride) for w, h in self.anchors]
            # 采集指定YOLO使用的锚点
            # [3, 2]
            self.masked_anchors = [self.all_anchors_grid[i] for i in self.anch_mask]
            # [9, 4]
            self.ref_anchors = np.zeros((len(self.all_anchors_grid), 4))
            # 赋值，锚点框宽／高
            self.ref_anchors[:, 2:] = np.array(self.all_anchors_grid)
            self.ref_anchors = torch.FloatTensor(self.ref_anchors)

            # 获取特征层数据 [B, n_anchors * (xywh+conf+n_classes), F_H, F_W]
            output = output_dict['output']
            # 获取预测边界框　[B, n_anchors, F_H, F_W, 4(xywh)]
            pred = output_dict['pred']

            # 图像批量数目
            batchsize = output.shape[0]
            # 特征数据的空间尺寸
            fsize = output.shape[2]

            # 特征层最终输出的通道维度大小
            n_ch = 5 + self.n_classes
            assert output.shape[-1] == n_ch

            # 数值类型以及对应设备
            dtype = torch.cuda.FloatTensor if output.is_cuda else torch.FloatTensor

            # target assignment
            # 创建掩码，作用母鸡？？？
            # [B, n_anchors, F_H, F_W, 4+n_classes]
            tgt_mask = torch.zeros(batchsize, self.n_anchors,
                                   fsize, fsize, 4 + self.n_classes).type(dtype)
            # [B, n_anchors, F_H, F_W]
            # 哪个预测框参与计算
            obj_mask = torch.ones(batchsize, self.n_anchors, fsize, fsize).type(dtype)
            # [B, n_anchors, F_H, F_W, 2]
            # 这个应该是作用于预测框的w/h
            tgt_scale = torch.zeros(batchsize, self.n_anchors, fsize, fsize, 2).type(dtype)

            # [B, n_anchors, F_H, F_W, n_ch]
            # n_ch = 4(xywh) + 1(conf) + n_classes
            # 实际用于损失计算的标签
            target = torch.zeros(batchsize, self.n_anchors, fsize, fsize, n_ch).type(dtype)

            labels = labels.cpu().data
            # [N, K, 5] -> [N, K] -> [N]
            # 计算有效的真值标签框数目
            # 首先判断是否bbox的xywh有大于0，然后求和计算每幅图像拥有的目标个数
            nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

            # xc: [B, K]
            # B: 批量大小
            # K: 真值框数目
            # xc(x_center): 取值在(0, 1)之间
            # # xc * fsize：计算实际坐标
            # truth_x_all = labels[:, :, 1] * fsize
            # # yc: [B, K]
            # truth_y_all = labels[:, :, 2] * fsize
            # # w: [B, K]
            # truth_w_all = labels[:, :, 3] * fsize
            # # h: [B, K]
            # truth_h_all = labels[:, :, 4] * fsize
            # xc * fsize：计算实际坐标
            # xc / stride：真值标签框坐标缩放指定倍数，匹配当前特征数据空间尺寸
            # print(labels[:, :, 1])
            # print(labels[:, :, 2])
            #
            # 将真值标签框的坐标映射到缩放后的特征数据中
            # xc: [B, K]
            truth_x_all = labels[:, :, 1] / self.stride
            # yc: [B, K]
            truth_y_all = labels[:, :, 2] / self.stride
            # w: [B, K]
            truth_w_all = labels[:, :, 3] / self.stride
            # h: [B, K]
            truth_h_all = labels[:, :, 4] / self.stride
            # xc/yc转换成INT16格式i/j，映射到指定网格中
            truth_i_all = truth_x_all.to(torch.int16).numpy()
            truth_j_all = truth_y_all.to(torch.int16).numpy()
            # print(truth_x_all)
            # print(truth_y_all)

            # 逐图像处理
            for b in range(batchsize):
                # 获取该幅图像定义的真值标签框个数
                n = int(nlabel[b])
                if n == 0:
                    # 如果为0，说明该图像没有符合条件的真值标签框，那么跳过损失计算
                    continue
                # 去除空的边界框，获取真正的标注框坐标
                truth_box = dtype(np.zeros((n, 4)))
                # 重新赋值，在数据类定义中，前n个就是真正的真值边界框
                # 赋值宽和高
                truth_box[:n, 2] = truth_w_all[b, :n]
                truth_box[:n, 3] = truth_h_all[b, :n]
                # 真值标签框的x_center，也就是第i个网格
                truth_i = truth_i_all[b, :n]
                # 真值标签框的y_center，也就是第j个网格
                truth_j = truth_j_all[b, :n]

                # calculate iou between truth and reference anchors
                # 首先计算真值边界框和锚点框之间的IoU
                # 注意：此时truth_box和ref_anchors的x_center/y_center坐标都是0/0，所以
                # x_center/y_center/w/h可以看成x_top_left/y_top_left/x_bottom_right/y_bottom_right
                # 设置xyxy=True，进行IoU计算
                # ([n, 4], [9, 4]) -> [n, 9]
                # 计算所有锚点框与真值标注框之间IoU的目的是为了找到真值标注框与哪个锚点框最匹配，
                # 如果真值标注框与锚点框的最大IoU超过阈值，并且该锚点框作用于该层特征数据，那么该真值标注框对应网格中使用？？？
                anchor_ious_all = bboxes_iou(truth_box.cpu(), self.ref_anchors, xyxy=True)
                # 找出和真值边界框之间的IoU最大的锚点框的下标
                # [n, 9] -> [n]
                best_n_all = np.argmax(anchor_ious_all, axis=1)
                # 求余操作，3的余数，作用？？？
                # [n] -> [n]
                best_n = best_n_all % 3
                # (best_n_all == self.anch_mask[0]): [n] == 第一个锚点框下标
                # (beat_n_all == self.anch_mask[1]): [n] == 第二个锚点框下标
                # (beat_n_all == self.anch_mask[1]): [n] == 第三个锚点框下标
                # [n] | [n] | [n] = [n]
                # 计算每个真值标注框最匹配的锚点框作用在当前层特征数据的掩码
                best_n_mask = ((best_n_all == self.anch_mask[0]) | (
                        best_n_all == self.anch_mask[1]) | (best_n_all == self.anch_mask[2]))

                # 赋值x_center和y_center
                truth_box[:n, 0] = truth_x_all[b, :n]
                truth_box[:n, 1] = truth_y_all[b, :n]

                # 计算预测框和真值边界框的IoU
                # ([n_anchors*F_H*F_W, 4], [n, 4]) -> [B*n_anchors*F_H*F_W, n]
                # 预测框坐标：xc/yc是相对于指定网格的比率计算，w/h是相对于特征图空间尺寸的对数运算
                # 真值标注框：xc/yc是相对于输入模型图像的比率计算，w/h是相对于输入模型图像的比率计算，也就是说，参照物是特征图空间尺寸
                pred_ious = bboxes_iou(pred[b].reshape(-1, 4), truth_box, xyxy=False)
                # pred[b].view(-1, 4), truth_box, xyxy=False)
                # 计算每个预测框与重叠最大的真值标签框的IoU
                # pred_best_iou: [n_anchors*F_H*F_W]
                # 所有的预测框
                pred_best_iou, _ = pred_ious.max(dim=1)
                # 计算掩码，IoU比率要大于忽略阈值。也就是说，如果IoU大于忽略阈值（也就是说预测框坐标与真值标注框坐标非常接近），那么该预测框不参与损失计算
                # pred_best_iou: [n_anchors*F_H*F_W]，取值为true/false
                pred_best_iou = (pred_best_iou > self.ignore_thresh)
                # 改变形状，[n_anchors*F_H*F_W] -> [n_anchors, F_H, F_W]
                pred_best_iou = pred_best_iou.view(pred[b].shape[:3])
                # set mask to zero (ignore) if pred matches truth
                # RuntimeError: Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.
                # obj_mask[b] = 1 - pred_best_iou
                # 目标掩码，预测框与真值标注框之间的IoU大于忽略阈值的不参与计算
                # [n_anchors, F_H, F_W]
                obj_mask[b] = ~pred_best_iou
                # obj_mask: 取值为True/False
                # True表示该预测框参与损失计算

                if sum(best_n_mask) == 0:
                    # 如果真值边界框和当前层使用的锚点框之间不存在最佳匹配，那么不计算损失
                    # 目标：不同层的特征数据负责不同大小的边界框预测
                    continue

                # 遍历真值标签框
                for ti in range(best_n.shape[0]):
                    # 该真值标签框是否和本层特征使用的锚点框最佳匹配
                    if best_n_mask[ti] == 1:
                        # 如果是，那么计算预测框损失
                        # 获取第ti个真值标签框对应的网格位置
                        i, j = truth_i[ti], truth_j[ti]
                        # 该真值标签框最佳匹配的锚点框
                        # ??? 为什么要这样呢，明明有些锚点框不作用于当前特征层数据
                        a = best_n[ti]
                        # print(b, a, j, i, n, ti)
                        # print(truth_i)
                        # print(truth_j)
                        # b: 第b张图像
                        # a: 第a个锚点框，对应第a个预测框
                        # j: 第j列网格
                        # i: 第i行网格
                        # 目标掩码：第[b, a, j, i]个预测框的掩码设置为1，表示参与损失计算
                        # obj_mask: [B, n_anchors, F_H, F_W]
                        #
                        # obj_mask经过了两次设置，
                        # 1. 第一次设置是计算预测框与真值标签框
                        obj_mask[b, a, j, i] = 1
                        # 坐标以及分类掩码：因为采用多标签训练方式，实际损失计算采用二元逻辑回归损失
                        # tgt_mask: [B, n_anchors, F_H, F_W, 4+n_classes]
                        tgt_mask[b, a, j, i, :] = 1
                        # target: [B, n_anchors, F_H, F_W, n_ch]
                        # 每个真值标注框对应一个预测框
                        #
                        # truth_x_all: [B, K]
                        # 计算第b张图像第ti个真值标签框的xc相对于其所属网格的大小
                        # 设置对应网格中真值标签框xc的大小
                        target[b, a, j, i, 0] = truth_x_all[b, ti] - \
                                                truth_x_all[b, ti].to(torch.int16).to(torch.float)
                        # truth_y_all: [B, K]
                        # 计算第b张图像第ti个真值标签框的yc相对于其所属网格的大小
                        target[b, a, j, i, 1] = truth_y_all[b, ti] - \
                                                truth_y_all[b, ti].to(torch.int16).to(torch.float)
                        # truth_w_all: [B, K]
                        # truth_w_all[b, ti]: 第b张图像第ti个真值标签框的w。
                        # 注意：w为真值标签框宽与实际输入图像宽的比率　乘以　当前特征数据宽，也就是说，经过了倍数缩放
                        #
                        # best_n: [n]
                        # best_n[ti]: 第ti个真值标签框对应的锚点框下标
                        # self.masked_anchors: [3, 2] 该层特征使用的锚点框列表。注意：其w/h经过了倍数缩放
                        # torch.Tensor(self.masked_anchors)[best_n[ti], 0]: 第ti个真值框匹配的锚点框的w
                        #
                        # log(w_truth / w_anchor):
                        # 计算第b张图像第ti个真值标签框的宽与对应锚点框的宽的比率的对数
                        target[b, a, j, i, 2] = torch.log(
                            truth_w_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 0] + 1e-16)
                        target[b, a, j, i, 3] = torch.log(
                            truth_h_all[b, ti] / torch.Tensor(self.masked_anchors)[best_n[ti], 1] + 1e-16)
                        # 该预测框的目标置信度设置为1，说明该预测框有效
                        target[b, a, j, i, 4] = 1
                        # 该b张第ti个真值标签框的类下标参与计算
                        target[b, a, j, i, 5 + labels[b, ti, 0].to(torch.int16).numpy()] = 1

                        # tgt_scale: [B, n_anchors, F_H, F_W, 2]
                        # ???
                        tgt_scale[b, a, j, i, :] = torch.sqrt(
                            2 - truth_w_all[b, ti] * truth_h_all[b, ti] / fsize / fsize)

            # loss calculation
            # 掩码的目的是为了屏蔽不符合要求的预测框
            # 首先过滤掉不符合条件的置信度
            output[..., 4] *= obj_mask
            # 然后过滤掉不符合条件的坐标以及类别
            output[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            # 针对w/h，某些预测框还需要乘以？？？
            output[..., 2:4] *= tgt_scale

            # 掩码分两部分：
            # 一部分是预测框数据，另一部分是对应标签
            target[..., 4] *= obj_mask
            target[..., np.r_[0:4, 5:n_ch]] *= tgt_mask
            target[..., 2:4] *= tgt_scale

            # 加权二值交叉熵损失
            bceloss = nn.BCELoss(weight=tgt_scale * tgt_scale, size_average=False)  # weighted BCEloss
            # 计算预测框xc/yc的损失
            loss_xy = bceloss(output[..., :2], target[..., :2])
            # 计算预测框w/h的损失
            loss_wh = self.l2_loss(output[..., 2:4], target[..., 2:4]) / 2
            # 计算目标置信度损失
            loss_obj = self.bce_loss(output[..., 4], target[..., 4])
            # 计算各个类别的分类概率损失
            loss_cls = self.bce_loss(output[..., 5:], target[..., 5:])
            # 计算统一损失
            loss_l2 = self.l2_loss(output, target)

            # 最终损失 = xc/yc损失 + w/h损失 + obj损失 + 分类损失
            loss = loss_xy + loss_wh + loss_obj + loss_cls

            # loss_xy + loss_wh + loss_obj + loss_cls + loss_xy + loss_wh + loss_obj + loss_cls + loss_l2 =
            # 2*loss_xy + 2*loss_wh + 2*loss_obj + 2*loss_cls + loss_l2
            # 因为loss_wh = self.l2_loss(...) / 2, 所以上式等同于
            # 2*bceloss + self.l2_loss + 2*self.bce_loss + 2*self.bce_loss + self.l2_loss

            # loss_list.append([loss, loss_xy, loss_wh, loss_obj, loss_cls, loss_l2])
            # loss_list.append(loss + loss_xy + loss_wh + loss_obj + loss_cls + loss_l2)
            loss_list.append(loss)

        return sum(loss_list)


if __name__ == '__main__':
    cfg_file = 'config/yolov3_default.cfg'
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    m = YOLOLoss(cfg['MODEL'], 0, ignore_thresh=0.70)
    print(m)

    output = torch.randn(10, 3 * (5 + 80), 20, 20)
    pred = torch.abs(torch.randn(10, 3, 20, 20, 4) * 200)
    labels = torch.randn(10, 50, 5)
    labels[..., 1:] = torch.abs(labels[..., 1:] * 200)
    labels[..., 0] = torch.abs((labels[..., 0] * 80).type(torch.int))

    loss = m(output, pred, labels)
    print(loss)
