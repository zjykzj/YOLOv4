# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午5:33
@file: utils.py
@author: zj
@description: 
"""
import os.path

import torch
import shutil

import numpy as np


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', output_dir='./'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ckpt_path = os.path.join(output_dir, filename)
    torch.save(state, ckpt_path)
    if is_best:
        shutil.copyfile(ckpt_path, os.path.join(output_dir, 'model_best.pth.tar'))


def synchronize():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def nms(bbox, thresh, score=None, limit=None):
    """Suppress bounding boxes according to their IoUs and confidence scores.
    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.
    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    from: https://github.com/chainer/chainercv
    """

    if len(bbox) == 0:
        return np.zeros((0,), dtype=np.int32)

    if score is not None:
        # 按照置信度大小进行排序
        order = score.argsort()[::-1]
        bbox = bbox[order]
    # 计算预测框面积
    # bbox[:, 2:] - bbox[:, :2]:
    #   (x2 - x1)/(y2 - y1)
    # bbox_area: [N_bbox]
    bbox_area = np.prod(bbox[:, 2:] - bbox[:, :2], axis=1)

    selec = np.zeros(bbox.shape[0], dtype=bool)
    for i, b in enumerate(bbox):
        # 计算每个预测框与其他预测框的IoU
        # 首先计算交集面积，
        # top-left: [2(top, left)]
        tl = np.maximum(b[:2], bbox[selec, :2])
        # bottom-right: [2(bottom, right)]
        br = np.minimum(b[2:], bbox[selec, 2:])
        area = np.prod(br - tl, axis=1) * (tl < br).all(axis=1)

        iou = area / (bbox_area[i] + bbox_area[selec] - area)
        if (iou >= thresh).any():
            # 如果iou大于阈值，说明该预测框与前面已确认的预测框高度重叠，需要舍弃
            continue

        selec[i] = True
        if limit is not None and np.count_nonzero(selec) >= limit:
            # 是否对每一个类别的预测框数目进行约束
            break

    selec = np.where(selec)[0]
    if score is not None:
        selec = order[selec]
    return selec.astype(np.int32)


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    Postprocess for the output of YOLO model
    perform box transformation, specify the class for each detection,
    and perform class-wise non-maximum suppression.
    Args:
        prediction (torch tensor): The shape is :math:`(N, B, 4)`.
            :math:`N` is the number of predictions,
            :math:`B` the number of boxes. The last axis consists of
            :math:`xc, yc, w, h` where `xc` and `yc` represent a center
            of a bounding box.
        num_classes (int):
            number of dataset classes.
        conf_thre (float):
            confidence threshold ranging from 0 to 1,
            which is defined in the config file.
        nms_thre (float):
            IoU threshold of non-max suppression ranging from 0 to 1.

    Returns:
        output (list of torch tensor):

    """
    # 输入prediction: [B, N_bbox, 4+1+80]
    # [x_center, y_center, box_w, box_h] -> [x_topleft, y_topleft, x_rightbottom, y_rightbottom]
    box_corner = prediction.new(prediction.shape)
    # 计算左上角坐标x0 = x_c - w/2
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # 计算左上角坐标y0 = y_c - h/2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # 计算右下角坐标x1 = x_c + w/2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # 计算右下角坐标y1 = y_c + h/2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    # 最大的预测数目，按照置信度进行排序后过滤
    max_num_preds = 300

    # 保存每张图像的预测结果
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        # 计算每幅图像的预测结果
        #
        # Filter out confidence scores below threshold
        # 计算每个预测框对应的最大类别概率
        # [N_bbox, num_classes] -> ｛分类概率，对应下标｝
        class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # [N_bbox]
        class_pred = class_pred[0]
        # 置信度掩码 [N_bbox] -> [N_bbox]
        # Pr(Class_i | Object) * Pr(Object) = Pr(Class_i)
        # 类别概率 * 置信度 = 置信度
        conf_mask = (image_pred[:, 4] * class_pred >= conf_thre)
        conf_mask = conf_mask.squeeze()
        # 过滤不符合置信度阈值的预测框
        image_pred = image_pred[conf_mask]

        # # 如果此时输出个数超过最大限制，那么再次进行过滤，按照置信度进行排序，去前面N个
        # if len(image_pred) > max_num_preds:
        #     class_pred = class_pred[conf_mask]
        #     conf_mask = torch.argsort(image_pred[:, 4] * class_pred)
        #     conf_mask = conf_mask[:max_num_preds]
        #     image_pred = image_pred[conf_mask]

        # If none are remaining => process next image
        # 如果所有预测框都已经舍弃，继续下一张图片的预测框计算
        if not image_pred.size(0):
            continue
        # Get detections with higher confidence scores than the threshold
        # 也就是说，每个预测框对应不同类别后可以存在多个，它们具有相同的坐标，但是存在不同的类别
        # 后续按照类别进行NMS过滤，相同类别的边界框进行过滤
        # (image_pred[:, 5:] * image_pred[:, 4][:, None] >= conf_thre)得到一个二维矩阵：[N_bbox, 80]
        # nonzero()得到一个二维矩阵：[N_nonzero, 2]
        # N_nonzero表示二维矩阵[N_bbox, 80]中每一行不为0的数目
        # [N_nonzero, 0]表示行下标
        # [N_nonzero, 1]表示列下标
        # 也就是说，计算每个预测框对应的置信度大于置信度阈值的类别有多少
        ind = (image_pred[:, 5:] * image_pred[:, 4][:, None] >= conf_thre).nonzero()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        # 获取预测结果
        # image_pred[ind[:, 0], :5]: 选择置信度大于等于阈值的预测框，得到预测框的预测坐标 + 置信度
        # image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1): 选择置信度大于等于阈值的预测框，得到每个预测框的分类概率
        # ind[:, 1].float().unsqueeze(1): 选择置信度大于等于阈值的预测框，得到预测框的分类下标
        # [N_ind, 5] + [N_ind, 1] + [N_ind, 1] = [N_ind, 7]
        detections = torch.cat((
            # xc/yc/w/h/conf_pred
            image_pred[ind[:, 0], :5],
            # cls_pred
            image_pred[ind[:, 0], 5 + ind[:, 1]].unsqueeze(1),
            # cls_ind
            ind[:, 1].float().unsqueeze(1)
        ), 1)
        # Iterate through all predicted classes
        # 按照类别进行NMS阈值过滤
        #
        # 统计所有预测框对应的类别列表
        # detections[:, -1]：得到预测框的分类下标
        # .unique()：去除重复的分类下标
        unique_labels = detections[:, -1].cpu().unique()
        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
        for c in unique_labels:
            # 逐个类别进行NMS过滤
            # 计算特定类别的预测框
            # Get the detections with the particular class
            # 获取特定类别的预测框列表
            # [[xc, yc, w, h, conf_pred, cls_pred, cls_ind], ...]
            detections_class = detections[detections[:, -1] == c]
            nms_in = detections_class.cpu().numpy()
            # 输入
            # nms_in[:, :4]: 特定类别的预测框坐标
            # nms_thre: NMS阈值
            # nms_in[:, 4] * nms_in[:, 5]:
            #   Pr(Object) * Pr(Class_i | Object) = Pr(Class_i)
            #   属于该类别的置信度
            nms_out_index = nms(
                nms_in[:, :4], nms_thre, score=nms_in[:, 4] * nms_in[:, 5])

            # 获取过滤后的预测框
            detections_class = detections_class[nms_out_index]
            if output[i] is None:
                # 第一次预测，必然为空，所以直接赋值
                # 第i张图片的预测结果为None，直接赋值
                output[i] = detections_class
            else:
                # 后续不为空，进行连接操作
                # 第i张图片的预测结果不为None，连接操作
                output[i] = torch.cat((output[i], detections_class))

    # 返回所有图片的预测框
    return output


def bbox2yolobox(labels):
    """
    [x1, y1, x2, y2, cls_id] -> [x_c, y_c, w, h, cls_id]
    """
    assert len(labels) > 0 and len(labels[0]) == 5
    # x1/y1/w/h -> x1/y1/x2/y2
    x1 = labels[:, 1]
    y1 = labels[:, 2]
    x2 = labels[:, 3]
    y2 = labels[:, 4]

    # x1/y1/x2/y2 -> xc/yc/w/h
    labels[:, 1] = ((x1 + x2) / 2)
    labels[:, 2] = ((y1 + y2) / 2)
    labels[:, 3] = x2 - x1
    labels[:, 4] = y2 - y1
    return labels


def label2yolobox(labels):
    """
    Transform coco labels to yolo box labels
    Args:
        labels (numpy.ndarray): label data whose shape is :math:`(N, 5)`.
            Each label consists of [class, x, y, w, h] where \
                class (float): class index.
                x, y, w, h (float) : coordinates of \
                    left-top points, width, and height of a bounding box.
                    Values range from 0 to width or height of the image.
        info_img : tuple of h, w, nh, nw, dx, dy.
            h, w (int): original shape of the image
            nh, nw (int): shape of the resized image without padding
            dx, dy (int): pad size
        maxsize (int): target image size after pre-processing
        lrflip (bool): horizontal flip flag

    Returns:
        labels:label data whose size is :math:`(N, 5)`.
            Each label consists of [class, xc, yc, w, h] where
                class (float): class index.
                xc, yc (float) : center of bbox whose values range from 0 to 1.
                w, h (float) : size of bbox whose values range from 0 to 1.
    """
    # x1/y1/w/h -> x1/y1/x2/y2
    x1 = labels[:, 1]
    y1 = labels[:, 2]
    x2 = (labels[:, 1] + labels[:, 3])
    y2 = (labels[:, 2] + labels[:, 4])

    # x1/y1/x2/y2 -> xc/yc/w/h
    labels[:, 1] = ((x1 + x2) / 2)
    labels[:, 2] = ((y1 + y2) / 2)
    return labels


def yolobox2xywh(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            src_h, src_w (int): original shape of the image
            dst_h, dst_w (int): resized shape of the image
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # (原始高，原始宽，缩放后高，缩放后宽，ROI区域左上角x0，ROI区域左上角y0)
    src_h, src_w, dst_h, dst_w = info_img
    # 预测框左上角和右下角坐标
    y1, x1, y2, x2 = box

    # 计算预测框高，缩放到原始图像
    box_h = (y2 - y1) / dst_h * src_h
    # 计算预测框宽，缩放到原始图像
    box_w = (x2 - x1) / dst_w * src_w
    # 预测框左上角坐标，先将坐标系恢复到缩放后图像，然后缩放到原始图像
    y1 = y1 / dst_h * src_h
    x1 = x1 / dst_w * src_w

    # [左上角y1，左上角x1，右下角y2，右下角x2]
    label = [x1, y1, box_w, box_h]
    return label


def yolobox2yxyx(box, info_img):
    """
    Transform yolo box labels to yxyx box labels.
    Args:
        box (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system after pre-processing.
        info_img : tuple of h, w, nh, nw, dx, dy.
            src_h, src_w (int): original shape of the image
            dst_h, dst_w (int): resized shape of the image
    Returns:
        label (list): box data with the format of [y1, x1, y2, x2]
            in the coordinate system of the input image.
    """
    # (原始高，原始宽，缩放后高，缩放后宽，ROI区域左上角x0，ROI区域左上角y0)
    src_h, src_w, dst_h, dst_w = info_img
    # 预测框左上角和右下角坐标
    y1, x1, y2, x2 = box

    # 预测框左上角/右下角坐标，先将坐标系恢复到缩放后图像，然后缩放到原始图像
    # src_h / dst_h = src_y1 / dst_y1
    # => src_y1 = dst_y1 * src_h / dst_h
    y1 = y1 * src_h / dst_h
    y2 = y2 * src_h / dst_h
    x1 = x1 * src_w / dst_w
    x2 = x2 * src_w / dst_w

    # [左上角y1，左上角x1，右下角y2，右下角x2]
    label = [y1, x1, y2, x2]
    return label
