# -*- coding: utf-8 -*-
import copy
from typing import List, Tuple, Dict

import os
import cv2
import glob
import yaml

from pathlib import Path

import argparse
from argparse import Namespace

import numpy as np
from numpy import ndarray

import torch.cuda
from torch import Tensor
from torch.nn import Module

from yolo.data.cocodataset import get_coco_label_names
from yolo.data.transform import Transform
from yolo.model.yolov4 import YOLOv4
from yolo.util.utils import yolobox2yxyx, postprocess


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch YOLOv4 Detection.")
    parser.add_argument('--cfg', type=str, default='config/yolov4_default.cfg')
    parser.add_argument('--ckpt', type=str,
                        help='path to the check point file')

    parser.add_argument('--source', type=str, default="./data/images/mountain.png",
                        help="Specify the image or directory to be detected")
    parser.add_argument('--dest', type=str, default="./runs/detect/",
                        help="Specify the directory where the image is saved")

    parser.add_argument('--conf-thre', type=float, default=-0.1,
                        help="Confidence threshold")
    parser.add_argument('--nms-thre', type=float, default=-0.1,
                        help="NMS threshold")

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        cfg = yaml.safe_load(f)
    return args, cfg


def image_preprocess(args: Namespace, cfg: Dict):
    if os.path.isfile(args.source):
        img_path_list = [args.source]
    else:
        assert os.path.isdir(args.source)
        img_path_list = glob.glob(os.path.join(args.source, "*.jpg")) + glob.glob(os.path.join(args.source, "*.png"))

    img_raw_list = list()
    img_name_list = list()
    dst_img_list = list()
    dst_target_list = list()
    if len(img_path_list) != 0:

        transform = Transform(cfg, is_train=False)
        imgsize = cfg['TEST']['IMGSIZE']

        for img_path in img_path_list:
            img_name = os.path.basename(img_path)
            img_name_list.append(img_name)

            image = cv2.imread(img_path)
            img_raw = image
            # img_raw = image.copy()[:, :, ::-1].transpose((2, 0, 1))

            out_img, target = transform([copy.deepcopy(image)], [np.array([])], imgsize)

            img_raw_list.append(img_raw)
            dst_img_list.append(out_img)
            dst_target_list.append(target)

    return dst_img_list, dst_target_list, img_raw_list, img_name_list


def model_init(args: Namespace, cfg: Dict):
    """
    ????????????????????????????????????
    """
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = YOLOv4(cfg['MODEL'], device=device).to(device)

    if args.ckpt:
        assert os.path.isfile(args.ckpt), '--ckpt must be specified'
        print("=> loading checkpoint '{}'".format(args.ckpt))
        checkpoint = torch.load(args.ckpt, map_location=device)

        state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
        model.load_state_dict(state_dict, strict=True)

    model.eval()
    return model, device


@torch.no_grad()
def process(args: Namespace, cfg: Dict, img_list: List[Tensor], model: Module, device: torch.device):
    """
    ???????????? + ????????????????????????????????? + IoU???????????????
    """
    conf_thre = cfg['TEST']['CONFTHRE'] if args.conf_thre < 0. else args.conf_thre
    nms_thre = cfg['TEST']['NMSTHRE'] if args.nms_thre < 0. else args.nms_thre

    outputs_list = list()
    for img in img_list:
        # img: [1, 3, 416, 416]
        # ?????????????????????????????????????????????????????????????????????????????????????????????+????????????
        outputs = model(img.unsqueeze(0).to(device)).cpu()
        # outputs: [B, N_bbox, 4(xywh)+1(conf)+num_classes]
        # ?????????????????????????????????????????????????????????????????????????????????+NMS IoU????????????
        outputs = postprocess(outputs, 80, conf_thre=conf_thre, nms_thre=nms_thre)
        # outputs: [B, Num_boxes, xc+yc+w+h+conf+cls_conf+cls_pred]
        outputs_list.append(outputs[0].numpy())

    return outputs_list


def increment_path(path, exist_ok=False, sep='', mkdir=False):
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

        # Method 1
        for n in range(2, 9999):
            p = f'{path}{sep}{n}{suffix}'  # increment path
            if not os.path.exists(p):  #
                break
        path = Path(p)

        # Method 2 (deprecated)
        # dirs = glob.glob(f"{path}{sep}*")  # similar paths
        # matches = [re.search(rf"{path.stem}{sep}(\d+)", d) for d in dirs]
        # i = [int(m.groups()[0]) for m in matches if m]  # indices
        # n = max(i) + 1 if i else 2  # increment number
        # path = Path(f"{path}{sep}{n}{suffix}")  # increment path

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory

    return path


def parse_info(outputs_list: List, target_list: List):
    coco_class_names, coco_class_ids, coco_class_colors = get_coco_label_names()

    # ??????????????????????????????????????????????????????????????????????????????
    bboxes_list = list()
    names_list = list()
    colors_list = list()

    for outputs, target in zip(outputs_list, target_list):
        img_info = target['img_info']

        bboxes = list()
        names = list()
        colors = list()

        # x1/y1: ???????????????
        # x2/y2: ???????????????
        # conf: ?????????
        # cls_conf: ???????????????
        # cls_pred: ????????????
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in outputs:
            cls_id = coco_class_ids[int(cls_pred)]
            print(int(x1), int(y1), int(x2), int(y2), float(conf), int(cls_pred))
            print('\t+ Label: %s, Conf: %.5f' %
                  (coco_class_names[cls_id], cls_conf.item()))
            box = yolobox2yxyx([y1, x1, y2, x2], img_info[:4])
            bboxes.append(box)
            names.append(f'{coco_class_names[cls_id]} {cls_conf.item():.2f}')
            colors.append(coco_class_colors[int(cls_pred)])

        bboxes_list.append(bboxes)
        names_list.append(names)
        colors_list.append(colors)

    return bboxes_list, names_list, colors_list


def show_bbox(save_dir: str, img_raw_list: List[ndarray], img_name_list: List[str],
              bboxes_list: List, names_list: List, colors_list: List):
    """
    ????????????????????????????????????
    1. ????????????
    2. ???????????????
    3. ???????????? + ????????????
    """
    line_width = 3
    txt_color = (255, 255, 255)

    for img_raw, img_name, bboxes, names, colors in zip(
            img_raw_list, img_name_list, bboxes_list, names_list, colors_list):
        im = img_raw
        lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  # line width

        for box, name, color in zip(bboxes, names, colors):
            # box: [y1, x1, y2, x2]
            # print(box, name, color)
            assert len(box) == 4, box
            color = tuple([int(x) for x in color])
            # p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
            p1, p2 = (int(box[1]), int(box[0])), (int(box[3]), int(box[2]))
            cv2.rectangle(im, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)

            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(name, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(im, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im,
                        name, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)

        im_path = os.path.join(save_dir, img_name)
        print(f"\t+ img path: {im_path}")
        cv2.imwrite(im_path, im)


def main():
    args, cfg = parse_args()
    print("args:\n", args)
    print("cfg:\n", cfg)

    img_list, target_list, img_raw_list, img_name_list = image_preprocess(args, cfg)
    assert len(img_list) > 0, "No images available!!!"

    model, device = model_init(args, cfg)
    outputs_list = process(args, cfg, img_list, model, device)

    # Directories
    dest_name = "exp"
    save_dir = increment_path(Path(args.dest) / dest_name, exist_ok=False)  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    bboxes_list, names_list, colors_list = parse_info(outputs_list, target_list)
    show_bbox(str(save_dir), img_raw_list, img_name_list, bboxes_list, names_list, colors_list)


if __name__ == '__main__':
    main()
