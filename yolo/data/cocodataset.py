# -*- coding: utf-8 -*-

"""
@date: 2023/1/3 下午5:59
@file: cocodataset.py
@author: zj
@description: 
"""

import cv2
import random
import os.path

import numpy as np
from pycocotools.coco import COCO

import torch
from torch import Tensor
from torch.utils.data import Dataset

from yolo.util.utils import bbox2yolobox


def get_coco_label_names():
    """
    COCO label names and correspondence between the model's class index and COCO class index.
    Returns:
        coco_label_names (tuple of str) : all the COCO label names including background class.
        coco_class_ids (list of int) : index of 80 classes that are used in 'instance' annotations
        coco_cls_colors (np.ndarray) : randomly generated color vectors used for box visualization

    """
    coco_label_names = ('background',  # class zero
                        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
                        'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
                        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
                        'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                        'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
                        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
                        'couch', 'potted plant', 'bed', 'mirror', 'dining table', 'window', 'desk',
                        'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
                        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
                        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                        )
    coco_class_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                      21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
                      46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67,
                      70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    coco_cls_colors = np.random.randint(128, 255, size=(80, 3))

    return coco_label_names, coco_class_ids, coco_cls_colors


class COCODataset(Dataset):

    def __init__(self, root, name: str = 'train2017', img_size: int = 416, min_size: int = 1,
                 model_type: str = 'YOLO', is_train: bool = True, transform=None, num_classes=80):
        self.root = root
        self.name = name
        self.img_size = img_size
        self.min_size = min_size
        self.model_type = model_type
        self.is_train = is_train
        self.transform = transform
        self.num_classes = num_classes

        if 'train' in self.name:
            json_file = 'instances_train2017.json'
        elif 'val' in self.name:
            json_file = 'instances_val2017.json'
        else:
            raise ValueError(f"{name} does not match any files")
        annotation_file = os.path.join(self.root, 'annotations', json_file)
        self.coco = COCO(annotation_file)

        # 获取图片ID列表
        self.ids = self.coco.getImgIds()
        # 获取类别ID
        self.class_ids = sorted(self.coco.getCatIds())

    def __len__(self):
        return len(self.ids)

    def get_img_and_labels(self, index=None):
        if index is None:
            index = random.choice(range(len(self.ids)))

        # 获取ID
        img_id = self.ids[index]
        # 获取图像路径
        img_file = os.path.join(self.root, 'images', self.name, '{:012}'.format(img_id) + '.jpg')
        assert os.path.isfile(img_file), img_file
        img = cv2.imread(img_file)
        # 获取标注框信息
        anno_ids = self.coco.getAnnIds(imgIds=[int(img_id)], iscrowd=None)
        annotations = self.coco.loadAnns(anno_ids)
        bboxes = []
        for anno in annotations:
            if anno['bbox'][2] > self.min_size and anno['bbox'][3] > self.min_size:
                # bbox: [x, y, w, h]
                tmp_bbox = list(anno['bbox'])
                tmp_bbox.append(self.class_ids.index(anno['category_id']))
                bboxes.insert(0, tmp_bbox)
        bboxes = np.array(bboxes)
        if len(bboxes) > 0:
            bboxes = bboxes[np.where((bboxes[:, 4] < self.num_classes) & (bboxes[:, 4] >= 0))[0]]

        return img, bboxes, img_id

    def __getitem__(self, index):
        img, bboxes, img_id = self.get_img_and_labels(index)

        target = None
        if self.transform is not None:
            img_list = list()
            bboxes_list = list()
            img_list.append(img)
            bboxes_list.append(bboxes)

            if self.is_train and self.transform.is_mosaic:
                for _ in range(3):
                    img, bboxes, _ = self.get_img_and_labels()
                    while True:
                        if len(bboxes) > 0:
                            break
                        img, bboxes, _ = self.get_img_and_labels()

                    img_list.append(img)
                    bboxes_list.append(bboxes)

            img, target = self.transform(img_list, bboxes_list, self.img_size)
            assert isinstance(img, Tensor)
            assert isinstance(target, dict)

            bboxes = target['padded_labels']
            assert isinstance(bboxes, Tensor), bboxes
            assert len(bboxes) > 0 and len(bboxes[0]) == 5, bboxes
            assert np.alltrue(bboxes.numpy()[:, 4] < self.num_classes) and np.alltrue(bboxes.numpy()[:, 4] >= 0), bboxes

            img_info = target['img_info']
            img_info.append(img_id)
            img_info.append(index)
            target['img_info'] = img_info

        # print(padded_labels)
        return img, target

    def set_img_size(self, img_size):
        self.img_size = img_size

    def get_img_size(self):
        return self.img_size


if __name__ == '__main__':
    dataset = COCODataset("COCO", name='train2017', img_size=608, is_train=True)
    # dataset = COCODataset("COCO", name='val2017', img_size=416, is_train=False)

    # img, target = dataset.__getitem__(333)
    # img, target = dataset.__getitem__(57756)
    # img, target = dataset.__getitem__(87564)
    img, target = dataset.__getitem__(51264)
    print(img.shape)
    padded_labels = target['padded_labels']
    img_info = target['img_info']
    print(padded_labels.shape)
    print(img_info)
    print(padded_labels)
