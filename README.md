# OTA: Optimal Transport Assignment for Object Detection

![GitHub](https://img.shields.io/github/license/Megvii-BaseDetection/LLA)

This project provides an implementation for "OTA: Optimal Transport Assignment for Object Detection" on PyTorch.

<img src="./ota.png" width="800" height="380">

## Results on COCO val set

| Model | Backbone | LR Sched. | mAP | Recall | AP50/AP75/APs/APm/APl | Download |
|:------| :----:   | :----: |:---:| :---:| :---:| :---:|
|  [RetinaNet](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/retinanet/retinanet.res50.fpn.coco.multiscale.1x) | Res50   | 1x       | 36.5 |  53.4  |  56.2/39.3/21.9/40.5/47.7  | - |
|  [Faster R-CNN](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x) | Res50   | 1x       | 38.1 |  52.2  |  58.9/41.0/22.5/41.5/48.9  | - |
|  [FCOS](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x) | Res50   | 1x       | 38.7 |  57.0   | 57.5/41.7/22.6/42.7/49.9   | - |
|  [FreeAnchor](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/free_anchor/free_anchor.res50.fpn.coco.800size.1x) | Res50   | 1x | 38.4 | 55.4  | 57.0/41.1/21.9/41.7/51.8      | - |
|  [ATSS](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/atss/atss.res50.fpn.coco.800size.1x) | Res50   | 1x    | 39.4 | 57.7    |  57.5/42.7/22.9/42.9/51.2   | - |
|  [PAA](https://github.com/kkhoot/PAA) | Res50   | 1x  | 40.4 |   -  |  -   | - |
|  OTA | Res50   | 1x       | 40.7  |  59.0 |  58.4/44.3/23.2/45.0/53.6    | weights |
