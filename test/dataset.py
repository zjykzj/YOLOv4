# -*- coding: utf-8 -*-

"""
@date: 2023/2/21 下午4:57
@file: dataset.py
@author: zj
@description: 
"""

import os
import random

from tqdm import tqdm

from yolo.data.cocodataset import COCODataset
from yolo.data.transform import Transform


def load_config(cfg_file='./config/yolov4_default.cfg'):
    assert os.path.isfile(cfg_file)
    with open(cfg_file, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    return cfg


def test_dataset(is_train=True):
    transform = Transform(cfg, is_train=is_train)

    data = './COCO/'
    dataset = COCODataset(root=data,
                          name='train2017' if is_train else "val2017",
                          img_size=cfg['TRAIN']['IMGSIZE'],
                          model_type=cfg['MODEL']['TYPE'],
                          is_train=is_train,
                          transform=transform,
                          )

    # index = 220
    # index = 25
    # print(index)
    # img, target = dataset.__getitem__(index)

    print(is_train, len(dataset.ids))
    for index in tqdm(range(len(dataset.ids))):
        img, target = dataset.__getitem__(index)


if __name__ == '__main__':
    cfg = load_config()

    # test_dataset(is_train=False)
    test_dataset(is_train=True)
