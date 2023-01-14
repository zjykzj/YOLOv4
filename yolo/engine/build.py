# -*- coding: utf-8 -*-

"""
@date: 2023/1/9 下午5:29
@file: build.py
@author: zj
@description: 
"""

import time
import json
import random
import tempfile

from tqdm import tqdm
from pycocotools.cocoeval import COCOeval

import torch.optim
import torch.utils.data
import torch.nn.parallel
import torch.distributed as dist
import torch.utils.data.distributed

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from yolo.optim.lr_schedulers.build import adjust_learning_rate
from yolo.util.metric import AverageMeter
from yolo.util.utils import postprocess, yolobox2label

from yolo.util import logging

logger = logging.get_logger(__name__)


def train(args, cfg, train_loader, model, criterion, optimizer, device=None, epoch=0):
    batch_time = AverageMeter()
    losses = AverageMeter()

    is_warmup = cfg['LR_SCHEDULER']['IS_WARMUP']
    warmup_epoch = cfg['LR_SCHEDULER']['WARMUP_EPOCH']
    accumulation_steps = cfg['TRAIN']['ACCUMULATION_STEPS']

    # switch to train mode
    model.train()
    end = time.time()

    assert hasattr(train_loader.dataset, 'set_img_size')
    optimizer.zero_grad()
    for i, (input, target) in enumerate(train_loader):
        if is_warmup and epoch < warmup_epoch:
            adjust_learning_rate(cfg, optimizer, epoch, i, len(train_loader))

        # compute output
        output = model(input.to(device))
        loss = criterion(output, target) / accumulation_steps

        # compute gradient and do SGD step
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        if (i + 1) % args.print_freq == 0:
            # Every print_freq iterations, check the loss, accuracy, and speed.
            # For best performance, it doesn't make sense to print these metrics every
            # iteration, since they incur an allreduce and some host<->device syncs.

            # Average loss and accuracy across processes for logging
            if args.distributed:
                reduced_loss = reduce_tensor(args, loss.data)
            else:
                reduced_loss = loss.data

            # to_python_float incurs a host<->device sync
            losses.update(to_python_float(reduced_loss), input.size(0))

            torch.cuda.synchronize()
            batch_time.update((time.time() - end) / args.print_freq)
            end = time.time()

            img_size = train_loader.dataset.get_img_size()
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            logger.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Speed {3:.3f} ({4:.3f})\t'
                        'Lr {5:.8f}\t'
                        'Loss {loss.val:.10f} ({loss.avg:.4f})\t'
                        'ImgSize: {6}x{6}\t'.format(
                (epoch + 1), (i + 1), len(train_loader),
                args.world_size * float(cfg['DATA']['BATCH_SIZE']) / batch_time.val,
                args.world_size * float(cfg['DATA']['BATCH_SIZE']) / batch_time.avg,
                current_lr,
                img_size,
                batch_time=batch_time,
                loss=losses))

            # 每隔10轮都重新指定输入图像大小
            img_size = (random.randint(0, 9) % 10 + 10) * 32
            train_loader.dataset.set_img_size(img_size)


@torch.no_grad()
def validate(val_loader, model, conf_threshold, nms_threshold, device=None):
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    ids = list()
    data_list = list()

    end = time.time()
    for i, (img, target) in enumerate(tqdm(val_loader)):
        assert isinstance(target, dict)
        img_info = [x.cpu().item() for x in target['img_info']]
        id_ = img_info[-2]

        # 从这里也判断出是单个推理
        id_ = int(id_)
        # 将原始图像下标挨个保存
        ids.append(id_)
        with torch.no_grad():
            # 模型推理，返回预测结果
            # img: [B, 3, 416, 416]
            outputs = model(img.to(device))
        # 后处理，进行置信度阈值过滤 + NMS阈值过滤
        # 输入outputs: [B, 预测框数目, 85(xywh + obj_confg + num_classes)]
        # 输出outputs: [B, 过滤后的预测框数目, 7(xyxy + obj_conf + cls_conf + cls_id)]
        outputs = postprocess(outputs, 80, conf_threshold, nms_threshold)
        # 从这里也可以看出是单张推理
        # 如果结果为空，那么不执行后续运算
        if outputs[0] is None:
            continue
        # 提取单张图片的运行结果
        # outputs: [N_ind, 7]
        outputs = outputs[0].cpu().data

        for output in outputs:
            x1 = float(output[0])
            y1 = float(output[1])
            x2 = float(output[2])
            y2 = float(output[3])
            # 分类标签
            label = val_loader.dataset.class_ids[int(output[6])]
            # 转换到原始图像边界框坐标
            box = yolobox2label((y1, x1, y2, x2), img_info[:6])
            # [y1, x1, y2, x2] -> [x1, y1, w, h]
            bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
            # 置信度 = 目标置信度 * 分类置信度
            score = float(output[4].data.item() * output[5].data.item())  # object score * class score
            # 保存计算结果
            A = {"image_id": id_, "category_id": label, "bbox": bbox,
                 "score": score, "segmentation": []}  # COCO json format
            data_list.append(A)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    logger.info('Time {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time))

    annType = ['segm', 'bbox', 'keypoints']

    # 计算完成所有测试图像的预测结果后
    # Evaluate the Dt (detection) json comparing with the ground truth
    if len(data_list) > 0:
        cocoGt = val_loader.dataset.coco
        # workaround: temporarily write data to json file because pycocotools can't process dict in py36.
        _, tmp = tempfile.mkstemp()
        json.dump(data_list, open(tmp, 'w'))
        cocoDt = cocoGt.loadRes(tmp)
        cocoEval = COCOeval(val_loader.dataset.coco, cocoDt, annType[1])
        cocoEval.params.imgIds = ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        # AP50_95, AP50
        return cocoEval.stats[0], cocoEval.stats[1]
    else:
        return 0, 0


def reduce_tensor(args, tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= args.world_size
    return rt
