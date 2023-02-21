# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午2:35
@file: transform.py
@author: zj
@description: 
"""
import copy
import random
from typing import Dict, List

import cv2
import numpy as np
import torch
from numpy import ndarray


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


def rect_intersection(a, b):
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])

    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    return [minx, miny, maxx, maxy]


def crop_and_pad(src_img: ndarray, bboxes: ndarray, jitter_ratio: float = 0.3, ):
    # Base info
    src_h, src_w = src_img.shape[:2]

    jitter_h, jitter_w = np.array(np.array([src_h, src_w]) * jitter_ratio, dtype=np.int)
    crop_left = random.randint(-jitter_w, jitter_w)
    crop_right = random.randint(-jitter_w, jitter_w)
    crop_top = random.randint(-jitter_h, jitter_h)
    crop_bottom = random.randint(-jitter_h, jitter_h)

    crop_h = src_h - crop_top - crop_bottom
    crop_w = src_w - crop_left - crop_right

    # x1,y1,x2,y2
    crop_rect = [crop_left, crop_top, crop_left + crop_w, crop_top + crop_h]
    img_rect = [0, 0, src_w, src_h]

    # intersection_rect = rect_intersection(img_rect, crop_rect)
    intersection_rect = rect_intersection(crop_rect, img_rect)
    intersection_rect_w = intersection_rect[2] - intersection_rect[0]
    intersection_rect_h = intersection_rect[3] - intersection_rect[1]
    # x1,y1,x2,y2
    dst_intersection_rect = [max(0, -crop_left), max(0, -crop_top),
                             max(0, -crop_left) + intersection_rect_w,
                             max(0, -crop_top) + intersection_rect_h]
    assert (dst_intersection_rect[3] - dst_intersection_rect[1]) == (intersection_rect[3] - intersection_rect[1])
    assert (dst_intersection_rect[2] - dst_intersection_rect[0]) == (intersection_rect[2] - intersection_rect[0])

    # Image Crop and Pad
    crop_img = np.zeros([crop_h, crop_w, 3])
    crop_img[:, :, ] = np.mean(src_img, axis=(0, 1))
    # crop_img[dst_y1:dst_y2, dst_x1:dst_x2] = src_img[y1:y2, x1:x2]
    crop_img[dst_intersection_rect[1]:dst_intersection_rect[3], dst_intersection_rect[0]:dst_intersection_rect[2]] = \
        src_img[intersection_rect[1]:intersection_rect[3], intersection_rect[0]:intersection_rect[2]]

    # BBoxes Crop and Pad
    # 如果真值边界框数目为0，那么返回
    if len(bboxes) != 0:
        # [x1, y1, x2, y2, cls_id]
        assert len(bboxes[0]) == 5
        # 随机打乱真值边界框
        np.random.shuffle(bboxes)
        # 原始图像的边界框坐标基于抖动调整坐标系
        bboxes[:, 0] -= crop_left
        bboxes[:, 2] -= crop_left
        bboxes[:, 1] -= crop_top
        bboxes[:, 3] -= crop_top

        # 设置x0, x1的最大最小值
        # 精度截断
        bboxes[:, 0] = np.clip(bboxes[:, 0], 0, crop_w)
        bboxes[:, 2] = np.clip(bboxes[:, 2], 0, crop_w)
        # 设置y0，y1的最大最小值
        # 精度截断
        bboxes[:, 1] = np.clip(bboxes[:, 1], 0, crop_h)
        bboxes[:, 3] = np.clip(bboxes[:, 3], 0, crop_h)

        # 找出x0==x1，取值为0或者sx 或者y0==y1，取值为0或者xy的边界框
        # 也就是说，边界框经过抖动和截断后变成了一条线
        out_box = list(np.where(((bboxes[:, 1] == crop_h) & (bboxes[:, 3] == crop_h)) |
                                ((bboxes[:, 0] == crop_w) & (bboxes[:, 2] == crop_w)) |
                                ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                                ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
        list_box = list(range(bboxes.shape[0]))
        # 移除这种边界框
        for i in out_box:
            list_box.remove(i)
        # 获取剩余边界框
        bboxes = bboxes[list_box]

    crop_info = [crop_left, crop_right, crop_top, crop_bottom, crop_w, crop_h]
    return crop_img, bboxes, crop_info


def left_right_flip(img, bboxes, is_flip=True):
    assert len(img.shape) == 3 and img.shape[2] == 3

    is_flip = is_flip and np.random.randn() > 0.5
    if is_flip:
        # [H, W, C]
        img = np.flip(img, axis=1).copy()
        h, w = img.shape[:2]

        if len(bboxes) > 0:
            # 左右翻转，所以y值不变，变换x值
            temp = w - bboxes[:, 0]
            bboxes[:, 0] = w - bboxes[:, 2]
            bboxes[:, 2] = temp

    return img, bboxes, is_flip


def image_resize(img, bboxes, dst_size):
    sized_img = cv2.resize(img, (dst_size, dst_size), cv2.INTER_LINEAR)

    img_h, img_w = img.shape[:2]

    if len(bboxes) > 0:
        # 转换抖动图像上的边界框坐标到网络输入图像的边界框坐标
        # dst_bbox / bbox = dst_size / img_size
        # dst_bbox = bbox * (dst_size / img_size)
        bboxes[:, 0] *= (dst_size / img_w)
        bboxes[:, 2] *= (dst_size / img_w)
        bboxes[:, 1] *= (dst_size / img_h)
        bboxes[:, 3] *= (dst_size / img_h)

    return sized_img, bboxes


def rand_uniform_strong(min, max):
    """
    随机均匀增强
    """
    if min > max:
        swap = min
        min = max
        max = swap
    return random.random() * (max - min) + min


def rand_scale(s):
    """
    随机缩放，放大或者缩小
    """
    scale = rand_uniform_strong(1, s)
    if random.randint(0, 1) % 2:
        return scale
    return 1. / scale


def color_dithering(img, hue, saturation, exposure, is_jitter=True):
    """
    img: 图像 [H, W, 3]
    hue: 色调
    saturation: 饱和度
    exposure: 曝光度
    """
    if is_jitter:
        # 色度、饱和度、曝光度
        dhue = rand_uniform_strong(-hue, hue)
        dsat = rand_scale(saturation)
        dexp = rand_scale(exposure)

        src_dtype = img.dtype
        img = img.astype(np.float32)

        # HSV augmentation
        # 先转换到HSV颜色空间，然后手动调整饱和度、亮度和色度，最后转换成为RGB颜色空间
        # cv2.COLOR_BGR2HSV, cv2.COLOR_RGB2HSV, cv2.COLOR_HSV2BGR, cv2.COLOR_HSV2RGB
        if dsat != 1 or dexp != 1 or dhue != 0:
            if img.shape[2] >= 3:
                hsv_src = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # RGB to HSV
                # Solve: https://github.com/Tianxiaomo/pytorch-YOLOv4/issues/427
                hsv = list(cv2.split(hsv_src))
                hsv[1] *= dsat
                hsv[2] *= dexp
                hsv[0] += 179 * dhue
                hsv_src = cv2.merge(hsv)
                # HSV to RGB (the same as previous)
                img = np.clip(cv2.cvtColor(hsv_src, cv2.COLOR_HSV2RGB), 0, 255)
            else:
                img *= dexp

        img.astype(src_dtype)
    return img


def filter_truth(bboxes, dx, dy, sx, sy, xd, yd):
    assert dx >= 0 and dy >= 0
    # 图像抖动后的边界框坐标
    bboxes[:, 0] -= dx
    bboxes[:, 2] -= dx
    bboxes[:, 1] -= dy
    bboxes[:, 3] -= dy

    assert sx > 0 and sy > 0
    # 边界框大小不能超出裁剪区域
    bboxes[:, 0] = np.clip(bboxes[:, 0], 0, sx)
    bboxes[:, 2] = np.clip(bboxes[:, 2], 0, sx)

    bboxes[:, 1] = np.clip(bboxes[:, 1], 0, sy)
    bboxes[:, 3] = np.clip(bboxes[:, 3], 0, sy)

    # 过滤截断后不存在的边界框
    out_box = list(np.where(((bboxes[:, 1] == sy) & (bboxes[:, 3] == sy)) |
                            ((bboxes[:, 0] == sx) & (bboxes[:, 2] == sx)) |
                            ((bboxes[:, 1] == 0) & (bboxes[:, 3] == 0)) |
                            ((bboxes[:, 0] == 0) & (bboxes[:, 2] == 0)))[0])
    list_box = list(range(bboxes.shape[0]))
    for i in out_box:
        list_box.remove(i)
    bboxes = bboxes[list_box]

    assert xd >= 0 and yd >= 0
    # mosaic后的图像左上角坐标
    bboxes[:, 0] += xd
    bboxes[:, 2] += xd
    bboxes[:, 1] += yd
    bboxes[:, 3] += yd

    return bboxes


def blend_mosaic(out_img, img, bboxes, cut_x, cut_y, mosaic_idx, crop_info):
    crop_left, crop_right, crop_top, crop_bottom, crop_w, crop_h, is_flip = crop_info[:7]
    if is_flip:
        tmp = crop_left
        crop_left = crop_right
        crop_right = tmp
    img_h, img_w = img.shape[:2]

    # left_shift / top_shift / right_shift / bottom_shift > 0
    # dst_left / crop_left = dst_w / crop_w
    # dst_left = crop_left * (dst_w / crop_w)
    left_shift = int(min(cut_x, max(0, (-int(crop_left) * img_w / crop_w))))
    top_shift = int(min(cut_y, max(0, (-int(crop_top) * img_h / crop_h))))
    right_shift = int(min((img_w - cut_x), max(0, (-int(crop_right) * img_w / crop_w))))
    bottom_shift = int(min((img_h - cut_y), max(0, (-int(crop_bottom) * img_h / crop_h))))

    left_shift = min(left_shift, img_w - cut_x)
    top_shift = min(top_shift, img_h - cut_y)
    right_shift = min(right_shift, cut_x)
    bottom_shift = min(bottom_shift, cut_y)

    if mosaic_idx == 0:
        # 左上角贴图，大小：[h, w]=[cut_y, cut_x]，左上角坐标：[x, y]=[0, 0]
        # 原图裁剪左上角坐标：[x, y]=[left_shift, top_shift]
        bboxes = filter_truth(bboxes, left_shift, top_shift, cut_x, cut_y, 0, 0)
        out_img[:cut_y, :cut_x] = img[top_shift:top_shift + cut_y, left_shift:left_shift + cut_x]
    if mosaic_idx == 1:
        # 右上角贴图，大小：[h, w]=[cut_y, w-cut_x]，左上角坐标：[x,y]=[cut_x, 0]
        # 原图裁剪左上角坐标：[x, y]=[cut_x-right_shift, top_shift]
        bboxes = filter_truth(bboxes, cut_x - right_shift, top_shift, img_w - cut_x, cut_y, cut_x, 0)
        out_img[:cut_y, cut_x:] = img[top_shift:top_shift + cut_y, cut_x - right_shift:img_w - right_shift]
    if mosaic_idx == 2:
        # 左下角
        bboxes = filter_truth(bboxes, left_shift, cut_y - bottom_shift, cut_x, img_h - cut_y, 0, cut_y)
        out_img[cut_y:, :cut_x] = img[cut_y - bottom_shift:img_h - bottom_shift, left_shift:left_shift + cut_x]
    if mosaic_idx == 3:
        # 右下角
        bboxes = filter_truth(bboxes, cut_x - right_shift, cut_y - bottom_shift,
                              img_w - cut_x, img_h - cut_y, cut_x, cut_y)
        out_img[cut_y:, cut_x:] = img[cut_y - bottom_shift:img_h - bottom_shift,
                                  cut_x - right_shift:img_w - right_shift]

    return out_img, bboxes


def xywh2xyxy(bboxes):
    if len(bboxes) == 0:
        return bboxes

    dst_bboxes = copy.deepcopy(bboxes)
    # x2 = x1 + w
    dst_bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
    # y2 = y1 + h
    dst_bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]

    return dst_bboxes


def xyxy2yolobox(bboxes):
    # [x1, y1, x2, y2] -> [xc, yc, w, h]
    if len(bboxes) == 0:
        return bboxes

    dst_bboxes = copy.deepcopy(bboxes)
    dst_bboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2
    dst_bboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3]) / 2
    dst_bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    dst_bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]

    return dst_bboxes


class Transform(object):
    """
    图像+标签转换

    训练阶段：颜色通道转换（BGR->RGB）、随机裁剪+填充、随机翻转、图像缩放、颜色抖动、mosaic
    验证阶段：
    """

    def __init__(self, cfg: Dict, is_train: bool = True):
        self.is_train = is_train

        # crop
        self.jitter_ratio = cfg['AUGMENTATION']['JITTER']
        # flip
        self.is_flip = cfg['AUGMENTATION']['RANDOM_HORIZONTAL_FLIP']
        # color jitter
        self.color_jitter = cfg['AUGMENTATION']['COLOR_DITHERING']
        self.hue = cfg['AUGMENTATION']['HUE']
        self.saturation = cfg['AUGMENTATION']['SATURATION']
        self.exposure = cfg['AUGMENTATION']['EXPOSURE']
        # mosaic
        self.is_mosaic = cfg['AUGMENTATION']['IS_MOSAIC']
        self.min_offset = cfg['AUGMENTATION']['MIN_OFFSET']
        # 单张图片预设的最大真值边界框数目
        self.max_num_labels = cfg['DATA']['MAX_NUM_LABELS']

    def _get_train_item(self, img_list: List[ndarray], bboxes_list: List[ndarray], img_size: int):
        # 指定结果图像的宽/高
        out_img = np.zeros([img_size, img_size, 3])
        # 在输出图像上的真值边界框可以有多个
        out_bboxes = []

        # 进行随机裁剪，随机生成裁剪图像的起始坐标
        # 坐标x取值在[0.2*w, 0.8*w]之间，坐标y同理
        cut_x = random.randint(int(img_size * self.min_offset), int(img_size * (1 - self.min_offset)))
        cut_y = random.randint(int(img_size * self.min_offset), int(img_size * (1 - self.min_offset)))

        for idx, (img, bboxes) in enumerate(zip(img_list, bboxes_list)):
            assert len(bboxes) == 0 or bboxes.shape[1] == 5
            assert len(img.shape) == 3 and img.shape[2] == 3
            bboxes = xywh2xyxy(bboxes)

            # BGR -> RGB
            img = img[:, :, ::-1]
            # 随机裁剪 + 填充
            img, bboxes, crop_info = crop_and_pad(img, bboxes, self.jitter_ratio)
            # 随机翻转
            img, bboxes, is_flip = left_right_flip(img, bboxes, is_flip=self.is_flip)
            crop_info.append(is_flip)
            # 图像缩放
            img, bboxes = image_resize(img, bboxes, img_size)
            # 最后进行颜色抖动
            img = color_dithering(img, self.hue, self.saturation, self.exposure, is_jitter=self.color_jitter)

            if self.is_mosaic:
                assert len(img_list) == 4 and len(bboxes_list) == 4
                out_img, bboxes = blend_mosaic(out_img, img, bboxes, cut_x, cut_y, idx, crop_info)
                out_bboxes.append(bboxes)
            else:
                assert len(img_list) == 1 and len(bboxes_list) == 1
                out_img = img
                out_bboxes = bboxes

        if self.is_mosaic:
            out_bboxes = np.concatenate(out_bboxes, axis=0)

        return out_img, out_bboxes

    def _get_val_item(self, img_list: List[ndarray], bboxes_list: List[ndarray], img_size: int):
        assert len(img_list) == 1 and len(bboxes_list) == 1
        img = img_list[0]
        bboxes = bboxes_list[0]
        # bbox: [x1, y1, x2, y2, cls_id]
        assert len(bboxes[0]) == 5

        # BGR -> RGB
        img = img[:, :, ::-1]
        # [x, y, w, h] -> [x1, y1, x2, y2]
        bboxes = xywh2xyxy(bboxes)

        return img, bboxes

    def __call__(self, img_list: List[ndarray], bboxes_list: List[ndarray], img_size: int):
        """
        bboxes_list: [bboxes, ...]
        bboxes: [[x1, y1, x2, y2, cls_id], ...]
        """
        if self.is_train:
            out_img, out_bboxes = self._get_train_item(img_list, bboxes_list, img_size)
        else:
            out_img, out_bboxes = self._get_val_item(img_list, bboxes_list, img_size)

        # 每幅图像设置固定个数的真值边界框，不足的填充为0
        dst_bboxes = np.zeros((self.max_num_labels, 5))
        if len(out_bboxes) > 0:
            out_bboxes = np.stack(out_bboxes)
            # [x1, y1, x2, y2] -> [x_c, y_c, w, h]
            out_bboxes = xyxy2yolobox(out_bboxes)
            assert np.all(out_bboxes[:, :4] <= img_size), print(out_bboxes)
            dst_bboxes[range(len(out_bboxes))] = out_bboxes[:]
        dst_bboxes = torch.from_numpy(dst_bboxes)

        # return img, padded_labels, labels, info_img
        # img: [3, H, W]
        # padded_labels: [K, 5]
        target = dict({
            'padded_labels': dst_bboxes,
            'img_info': list()
        })

        # 数据预处理
        out_img = torch.from_numpy(out_img).permute(2, 0, 1).contiguous() / 255
        return out_img, target
