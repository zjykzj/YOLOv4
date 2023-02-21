# -*- coding: utf-8 -*-

"""
@date: 2023/2/20 下午2:46
@file: transform.py
@author: zj
@description: 
"""

import os
import cv2
import copy
import random

import numpy as np

from yolo.data.transform import Transform
from yolo.data.transform import crop_and_pad, left_right_flip, color_dithering, image_resize, xywh2xyxy, blend_mosaic
from yolo.data.cocodataset import COCODataset


def load_config(cfg_file='./config/yolov4_default.cfg'):
    assert os.path.isfile(cfg_file)
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    return cfg


def draw(img, bboxes):
    img = copy.deepcopy(img).astype(np.uint8)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]
        if x1 == x2 or y1 == y2:
            continue
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=1)
    return img


def test_crop_and_pad(coco):
    index = 100
    img, bboxes, img_id = coco.get_img_and_labels(index)
    bboxes = xywh2xyxy(bboxes)
    print(img.shape, bboxes.shape, img_id)
    draw_src = draw(img, bboxes)
    cv2.imwrite("draw_src.jpg", draw_src)

    for i in range(5):
        crop_img, crop_bboxes, crop_info = crop_and_pad(copy.deepcopy(img), copy.deepcopy(bboxes), jitter_ratio=0.3)
        print(crop_img.shape, crop_bboxes.shape, img_id)
        draw_dst = draw(crop_img, crop_bboxes)
        cv2.imwrite(f"draw_dst_{i}.jpg", draw_dst)


def test_left_right_flip(coco):
    index = 100
    img, bboxes, img_id = coco.get_img_and_labels(index)
    bboxes = xywh2xyxy(bboxes)
    print(img.shape, bboxes.shape, img_id)
    draw_src = draw(img, bboxes)
    cv2.imwrite("draw_src.jpg", draw_src)

    for i in range(5):
        dst_img, crop_bboxes, is_flip = left_right_flip(copy.deepcopy(img), copy.deepcopy(bboxes), is_flip=True)
        print(dst_img.shape, crop_bboxes.shape, img_id, is_flip)
        draw_dst = draw(dst_img, crop_bboxes)
        cv2.imwrite(f"draw_dst_{i}.jpg", draw_dst)


def test_color_dithering(coco):
    index = 100
    img, bboxes, img_id = coco.get_img_and_labels(index)
    bboxes = xywh2xyxy(bboxes)
    print(img.shape, bboxes.shape, img_id)
    draw_src = draw(img, bboxes)
    cv2.imwrite("draw_src.jpg", draw_src)

    for i in range(5):
        dst_img = color_dithering(copy.deepcopy(img), 0.1, 1.5, 1.5, is_jitter=True)
        draw_dst = draw(dst_img, bboxes)
        cv2.imwrite(f"draw_dst_{i}.jpg", draw_dst)


def test_image_resize(coco):
    index = 100
    img, bboxes, img_id = coco.get_img_and_labels(index)
    bboxes = xywh2xyxy(bboxes)
    print(img.shape, bboxes.shape, img_id)
    draw_src = draw(img, bboxes)
    cv2.imwrite("draw_src.jpg", draw_src)

    for i in range(5):
        img_size = (random.randint(0, 9) % 10 + 10) * 32
        dst_img, dst_bboxes = image_resize(copy.deepcopy(img), copy.deepcopy(bboxes), img_size)
        draw_dst = draw(dst_img, dst_bboxes)
        cv2.imwrite(f"draw_dst_{i}.jpg", draw_dst)


def test_mosaic(coco):
    index = 100
    src_img, src_bboxes, img_id = coco.get_img_and_labels(index)
    src_bboxes = xywh2xyxy(src_bboxes)
    print(src_img.shape, src_bboxes.shape, img_id)
    draw_src = draw(src_img, src_bboxes)
    cv2.imwrite("draw_src.jpg", draw_src)

    img_size = (random.randint(0, 9) % 10 + 10) * 32
    min_offset = 0.2

    # 指定结果图像的宽/高
    out_img = np.zeros([img_size, img_size, 3])
    # 在输出图像上的真值边界框可以有多个
    out_bboxes = []

    # 进行随机裁剪，随机生成裁剪图像的起始坐标
    # 坐标x取值在[0.2*w, 0.8*w]之间，坐标y同理
    cut_x = random.randint(int(img_size * min_offset), int(img_size * (1 - min_offset)))
    cut_y = random.randint(int(img_size * min_offset), int(img_size * (1 - min_offset)))

    img_list = list()
    bboxes_list = list()
    img_list.append(copy.deepcopy(src_img))
    bboxes_list.append(copy.deepcopy(src_bboxes))

    for k in range(3):
        img, bboxes, _ = coco.get_img_and_labels()
        bboxes = xywh2xyxy(bboxes)
        img_list.append(img)
        bboxes_list.append(bboxes)

        draw_src = draw(img, bboxes)
        cv2.imwrite(f"draw_src_{k + 1}.jpg", draw_src)

    for idx, (img, bboxes) in enumerate(zip(img_list, bboxes_list)):
        assert len(img_list) == 4 and len(bboxes_list) == 4
        img_h, img_w = img.shape[:2]
        crop_info = [0, 0, 0, 0, img_w, img_h, False]
        # 缩放到指定大小
        img, bboxes = image_resize(img, bboxes, img_size)
        # mosaic
        out_img, bboxes = blend_mosaic(out_img, img, bboxes, cut_x, cut_y, idx, crop_info)
        out_bboxes.append(bboxes)

    out_bboxes = np.concatenate(out_bboxes, axis=0)
    draw_dst = draw(out_img, out_bboxes)
    cv2.imwrite(f"draw_dst.jpg", draw_dst)


def test_combine_v1(coco):
    index = 100
    img, bboxes, img_id = coco.get_img_and_labels(index)
    bboxes = xywh2xyxy(bboxes)
    print(img.shape, bboxes.shape, img_id)
    draw_src = draw(img, bboxes)
    cv2.imwrite("draw_src.jpg", draw_src)

    for i in range(5):
        crop_img, crop_bboxes, crop_info = crop_and_pad(copy.deepcopy(img), copy.deepcopy(bboxes), jitter_ratio=0.3)
        print(crop_img.shape, crop_bboxes.shape, img_id)

        dst_img, crop_bboxes, is_flip = left_right_flip(crop_img, crop_bboxes, is_flip=True)
        print(dst_img.shape, crop_bboxes.shape, img_id, is_flip)

        dst_img = color_dithering(dst_img, 0.1, 1.5, 1.5, is_jitter=True)

        img_size = (random.randint(0, 9) % 10 + 10) * 32
        dst_img, dst_bboxes = image_resize(dst_img, crop_bboxes, img_size)
        draw_dst = draw(dst_img, dst_bboxes)
        cv2.imwrite(f"draw_dst_{i}.jpg", draw_dst)


def test_combine_v2(coco):
    index = 100
    src_img, src_bboxes, img_id = coco.get_img_and_labels(index)
    src_bboxes = xywh2xyxy(src_bboxes)
    print(src_img.shape, src_bboxes.shape, img_id)
    draw_src = draw(src_img, src_bboxes)
    cv2.imwrite("draw_src.jpg", draw_src)

    img_size = (random.randint(0, 9) % 10 + 10) * 32
    min_offset = 0.2

    # 指定结果图像的宽/高
    out_img = np.zeros([img_size, img_size, 3])
    # 在输出图像上的真值边界框可以有多个
    out_bboxes = []

    # 进行随机裁剪，随机生成裁剪图像的起始坐标
    # 坐标x取值在[0.2*w, 0.8*w]之间，坐标y同理
    cut_x = random.randint(int(img_size * min_offset), int(img_size * (1 - min_offset)))
    cut_y = random.randint(int(img_size * min_offset), int(img_size * (1 - min_offset)))

    img_list = list()
    bboxes_list = list()
    img_list.append(copy.deepcopy(src_img))
    bboxes_list.append(copy.deepcopy(src_bboxes))

    for k in range(3):
        img, bboxes, _ = coco.get_img_and_labels()
        bboxes = xywh2xyxy(bboxes)
        img_list.append(img)
        bboxes_list.append(bboxes)

        draw_src = draw(img, bboxes)
        cv2.imwrite(f"draw_src_{k + 1}.jpg", draw_src)

    for idx, (img, bboxes) in enumerate(zip(img_list, bboxes_list)):
        assert len(img_list) == 4 and len(bboxes_list) == 4

        crop_img, crop_bboxes, crop_info = crop_and_pad(copy.deepcopy(img), copy.deepcopy(bboxes), jitter_ratio=0.3)
        print(crop_img.shape, crop_bboxes.shape, img_id)

        dst_img, crop_bboxes, is_flip = left_right_flip(crop_img, crop_bboxes, is_flip=True)
        crop_info.append(is_flip)
        print(dst_img.shape, crop_bboxes.shape, img_id, is_flip)

        dst_img = color_dithering(dst_img, 0.1, 1.5, 1.5, is_jitter=True)

        dst_img, dst_bboxes = image_resize(dst_img, crop_bboxes, img_size)
        draw_dst = draw(dst_img, dst_bboxes)
        cv2.imwrite(f"draw_dst_{idx}.jpg", draw_dst)

        # mosaic
        out_img, bboxes = blend_mosaic(out_img, dst_img, dst_bboxes, cut_x, cut_y, idx, crop_info)
        out_bboxes.append(bboxes)

    out_bboxes = np.concatenate(out_bboxes, axis=0)
    draw_dst = draw(out_img, out_bboxes)
    cv2.imwrite(f"draw_dst.jpg", draw_dst)


if __name__ == '__main__':
    cfg = load_config()
    # cfg['AUGMENTATION']['IS_MOSAIC'] = False

    train_transform = Transform(cfg, is_train=True)

    data = './COCO/'
    train_dataset = COCODataset(root=data,
                                name='train2017',
                                img_size=cfg['TRAIN']['IMGSIZE'],
                                model_type=cfg['MODEL']['TYPE'],
                                is_train=True,
                                transform=train_transform,
                                )
    # test_crop_and_pad(train_dataset)
    # test_left_right_flip(train_dataset)
    # test_color_dithering(train_dataset)
    # test_image_resize(train_dataset)
    # test_mosaic(train_dataset)
    # test_combine_v1(train_dataset)
    test_combine_v2(train_dataset)
