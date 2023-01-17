# CSPDarknet53

## Train

```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port "31226" main_amp.py -b 128 --workers 4 --lr 0.1 --weight-decay 1e-4 --epochs 120 --opt-level O1 ./imagenet/
```

## References

* [CSP DarkNet](https://paperswithcode.com/lib/timm/csp-darknet)
* [YOLOv3/darknet](https://github.com/zjykzj/YOLOv3/tree/master/darknet)