# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午2:35
@file: transform.py
@author: zj
@description: 
"""
from typing import Dict, List

import cv2

from numpy import ndarray
import numpy as np


def resize_and_pad(src_img, bboxes, dst_size, jitter_ratio=0.0, random_replacing=False):
    """
    src_img: [H, W, 3]
    bboxes: [K, 4] x1/y1/b_w/b_h
    """
    src_h, src_w = src_img.shape[:2]

    dh = jitter_ratio * src_h
    dw = jitter_ratio * src_w
    new_ratio = (src_w + np.random.uniform(low=-dw, high=dw)) / (src_h + np.random.uniform(low=-dh, high=dh))
    if new_ratio < 1:
        # 高大于宽
        # 设置目标大小为高，等比例缩放宽，剩余部分进行填充
        dst_h = dst_size
        dst_w = new_ratio * dst_size
    else:
        # 宽大于等于高
        # 设置目标大小为宽，等比例缩放高，剩余部分进行填充
        dst_w = dst_size
        dst_h = dst_size / new_ratio
    dst_w = int(dst_w)
    dst_h = int(dst_h)

    # 计算ROI填充到结果图像的左上角坐标
    if random_replacing:
        dx = int(np.random.uniform(dst_size - dst_w))
        dy = int(np.random.uniform(dst_size - dst_h))
    else:
        # 等比例进行上下或者左右填充
        dx = (dst_size - dst_w) // 2
        dy = (dst_size - dst_h) // 2

    # 先进行图像缩放，然后创建目标图像，填充ROI区域
    resized_img = cv2.resize(src_img, (dst_w, dst_h))
    padded_img = np.zeros((dst_size, dst_size, 3), dtype=np.uint8) * 127
    padded_img[dy:dy + dst_h, dx:dx + dst_w, :] = resized_img

    if len(bboxes) > 0:
        # 进行缩放以及填充后需要相应的修改坐标位置
        # x_left_top
        bboxes[:, 0] = bboxes[:, 0] / src_w * dst_w + dx
        # y_left_top
        bboxes[:, 1] = bboxes[:, 1] / src_h * dst_h + dy
        # 对于宽/高而言，仅需缩放对应比例即可，不需要增加填充坐标
        # box_w
        bboxes[:, 2] = bboxes[:, 2] / src_w * dst_w
        # box_h
        bboxes[:, 3] = bboxes[:, 3] / src_h * dst_h

    img_info = [src_h, src_w, dst_h, dst_w, dx, dy, dst_size]
    return padded_img, bboxes, img_info


def left_right_flip(img, bboxes):
    dst_img = np.flip(img, axis=2).copy()

    if len(bboxes) > 0:
        h, w = img.shape[:2]
        # 左右翻转，所以宽/高不变，变换左上角坐标(x1, y1)和右上角坐标(x2, y1)进行替换
        x2 = bboxes[:, 0] + bboxes[:, 2]
        # y1/2/h不变，仅变换x1 = w - x2
        bboxes[:, 0] = w - x2

    return dst_img, bboxes


def rand_scale(s):
    """
    calculate random scaling factor
    Args:
        s (float): range of the random scale.
    Returns:
        random scaling factor (float) whose range is
        from 1 / s to s .
    """
    scale = np.random.uniform(low=1, high=s)
    if np.random.rand() > 0.5:
        return scale
    return 1 / scale


def color_dithering(src_img, hue, saturation, exposure):
    """
    src_img: 图像 [H, W, 3]
    hue: 色调
    saturation: 饱和度
    exposure: 曝光度
    """
    dhue = np.random.uniform(low=-hue, high=hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)

    img = cv2.cvtColor(src_img, cv2.COLOR_RGB2HSV)
    img = np.asarray(img, dtype=np.float32) / 255.
    img[:, :, 1] *= dsat
    img[:, :, 2] *= dexp
    H = img[:, :, 0] + dhue

    if dhue > 0:
        H[H > 1.0] -= 1.0
    else:
        H[H < 0.0] += 1.0

    img[:, :, 0] = H
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    img = np.asarray(img, dtype=np.float32)

    return img


class Transform(object):

    def __init__(self, cfg: Dict, is_train: bool = True):
        self.is_train = is_train

        # 空间抖动
        self.jitter_ratio = cfg['AUGMENTATION']['JITTER']
        # 随机放置
        self.random_placing = cfg['AUGMENTATION']['RANDOM_PLACING']
        # 左右翻转
        self.is_flip = cfg['AUGMENTATION']['RANDOM_HORIZONTAL_FLIP']
        # 颜色抖动
        self.color_jitter = cfg['AUGMENTATION']['COLOR_DITHERING']
        self.hue = cfg['AUGMENTATION']['HUE']
        self.saturation = cfg['AUGMENTATION']['SATURATION']
        self.exposure = cfg['AUGMENTATION']['EXPOSURE']

    def __call__(self, img: ndarray, bboxes: List, img_size: int):
        # BGR -> RGB
        img = img[:, :, ::-1]
        if self.is_train:
            # 首先进行缩放+填充+空间抖动
            img, bboxes, img_info = resize_and_pad(img, bboxes, img_size, self.jitter_ratio, self.random_placing)
            assert np.all(bboxes <= img_size), print(img_info, '\n', bboxes)
            # 然后进行左右翻转
            if self.is_flip and np.random.randn() > 0.5:
                img, bboxes = left_right_flip(img, bboxes)
            # 最后进行颜色抖动
            if self.color_jitter:
                img = color_dithering(img, self.hue, self.saturation, self.exposure)
        else:
            # 进行缩放+填充，不执行空间抖动
            img, bboxes, img_info = resize_and_pad(img, bboxes, img_size, jitter_ratio=0., random_replacing=False)
            assert np.all(bboxes <= img_size), print(img_info, '\n', bboxes)

        return img, bboxes, img_info
