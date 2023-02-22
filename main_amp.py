# -*- coding: utf-8 -*-

import os
import time
import argparse

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

from yolo.data.build import build_data
from yolo.model.build import build_model, build_criterion
from yolo.optim.optimizers.build import build_optimizer
from yolo.optim.lr_schedulers.build import build_lr_scheduler
from yolo.engine.build import validate, train
from yolo.util.utils import save_checkpoint, synchronize

from yolo.util import logging

logger = logging.get_logger(__name__)


def parse():
    parser = argparse.ArgumentParser(description='PyTorch YOLOv3 Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-c', "--cfg", default='config/yolov3_default.cfg', type=str, metavar='CFG',
                        help='path to config file (default: config/yolov3_default.cfg)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--deterministic', action='store_true')

    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', 0), type=int)
    parser.add_argument('--sync_bn', action='store_true',
                        help='enabling apex sync BN.')

    parser.add_argument('--opt-level', type=str)
    parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
    parser.add_argument('--loss-scale', type=str, default=None)
    parser.add_argument('--channels-last', type=bool, default=False)
    args = parser.parse_args()
    return args


def main():
    global best_ap50, best_ap50_95, args

    args = parse()
    # load cfg
    with open(args.cfg, 'r') as f:
        import yaml

        cfg = yaml.safe_load(f)

    logging.setup_logging(local_rank=args.local_rank, output_dir=cfg['TRAIN']['OUTPUT_DIR'])
    logger.info("opt_level = {}".format(args.opt_level))
    logger.info("keep_batchnorm_fp32 = {} {}".format(args.keep_batchnorm_fp32, type(args.keep_batchnorm_fp32)))
    logger.info("loss_scale = {} {}".format(args.loss_scale, type(args.loss_scale)))

    logger.info("\nCUDNN VERSION: {}\n".format(torch.backends.cudnn.version()))

    cudnn.benchmark = True
    best_ap50 = 0
    best_ap50_95 = 0
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        torch.manual_seed(args.local_rank)
        torch.set_printoptions(precision=10)

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1

    args.gpu = 1 if torch.cuda.is_available() else 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    # create model
    device = torch.device(f'cuda:{args.local_rank}' if args.world_size > 1 or args.gpu > 0 else 'cpu')
    logger.info("device: {}".format(device))
    model = build_model(args, cfg, device=device)

    # # Scale learning rate based on global batch size
    # cfg['OPTIMIZER']['LR'] = float(cfg['OPTIMIZER']['LR']) * float(
    #     cfg['DATA']['BATCH_SIZE'] * cfg['TRAIN']['ACCUMULATION_STEPS'] * args.world_size) / 64.
    # cfg['OPTIMIZER']['LR'] = float(cfg['OPTIMIZER']['LR']) * args.world_size / float(
    #     cfg['DATA']['BATCH_SIZE'] * cfg['TRAIN']['ACCUMULATION_STEPS'])
    optimizer = build_optimizer(cfg, model)
    # Initialize Amp.  Amp accepts either values or strings for the optional override arguments,
    # for convenient interoperation with argparse.
    model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
    lr_scheduler = build_lr_scheduler(cfg, optimizer)

    # For distributed training, wrap the model with apex.parallel.DistributedDataParallel.
    # This must be done AFTER the call to amp.initialize.  If model = DDP(model) is called
    # before model, ... = amp.initialize(model, ...), the call to amp.initialize may alter
    # the types of model's parameters in a way that disrupts or destroys DDP's allreduce hooks.
    if args.distributed:
        # By default, apex.parallel.DistributedDataParallel overlaps communication with
        # computation in the backward pass.
        # model = DDP(model)
        # delay_allreduce delays all communication to the end of the backward pass.
        model = DDP(model, delay_allreduce=True)

    # define loss function (criterion) and optimizer
    criterion = build_criterion(cfg, device=device)

    start_epoch = int(cfg['TRAIN']['START_EPOCH'])
    max_epochs = int(cfg['TRAIN']['MAX_EPOCHS'])

    # Optionally resume from a checkpoint
    if args.resume:
        # Use a local scope to avoid dangling references
        def resume():
            if os.path.isfile(args.resume):
                logger.info("=> loading checkpoint '{}'".format(args.resume))
                checkpoint = torch.load(args.resume, map_location=device)
                # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda(args.gpu))
                start_epoch = checkpoint['epoch']

                global best_ap50, best_ap50_95
                best_ap50 = checkpoint['ap50']
                best_ap50_95 = checkpoint['ap50_95']

                if args.distributed:
                    state_dict = {key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()}
                else:
                    state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)

                if hasattr(checkpoint, 'optimizer'):
                    optimizer.load_state_dict(checkpoint['optimizer'])
                if hasattr(checkpoint, 'lr_scheduler'):
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

                logger.info("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
            else:
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

        resume()

    # Data loading code
    train_sampler, train_loader, val_loader = build_data(args, cfg)

    conf_thresh = cfg['TEST']['CONFTHRE']
    nms_thresh = float(cfg['TEST']['NMSTHRE'])
    if args.evaluate and args.local_rank == 0:
        logger.info("Begin evaluating ...")
        # ap50_95, ap50 = evaluator.evaluate(model)
        validate(val_loader, model, conf_thresh, nms_thresh, device=device)
        return

    logger.info("\nargs: {}".format(args))
    logger.info("\ncfg: {}".format(cfg))

    is_warmup = cfg['LR_SCHEDULER']['IS_WARMUP']
    warmup_epoch = int(cfg['LR_SCHEDULER']['WARMUP_EPOCH'])

    # pytorch-accurate time
    synchronize()
    # Note: epoch begin from 0
    for epoch in range(start_epoch, max_epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        start = time.time()
        train(args, cfg, train_loader, model, criterion, optimizer, device=device, epoch=epoch)
        # pytorch-accurate time
        logger.info("One epoch train need: {:.3f}".format((time.time() - start)))
        synchronize()

        if is_warmup and epoch < warmup_epoch:
            pass
        else:
            lr_scheduler.step()

        # save checkpoint
        if args.local_rank == 0:
            # evaluate on validation set
            logger.info("Begin evaluating ...")
            start = time.time()
            ap50_95, ap50 = validate(val_loader, model, conf_thresh, nms_thresh, device=device)
            logger.info("One epoch validate need: {:.3f}".format((time.time() - start)))

            # save checkpoint
            is_best = ap50 > best_ap50
            if is_best:
                best_ap50 = ap50
                best_ap50_95 = ap50_95

            save_checkpoint({
                'epoch': epoch + 1,
                'ap50': ap50,
                'ap50_95': ap50_95,
                'best_ap50': best_ap50,
                'best_ap50_95': best_ap50_95,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }, is_best, output_dir=cfg['TRAIN']['OUTPUT_DIR'])

        synchronize()


if __name__ == '__main__':
    main()
