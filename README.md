# OTA: Optimal Transport Assignment for Object Detection

![GitHub](https://img.shields.io/github/license/Megvii-BaseDetection/OTA)

This project provides an implementation for our CVPR2021 paper "OTA: Optimal Transport Assignment for Object Detection" on PyTorch.
**Paper Link**: comming soon.

<img src="./ota.png" width="700" height="330">

## Requirements
* [cvpods](https://github.com/Megvii-BaseDetection/cvpods)

## Get Started

* install cvpods locally (requires cuda to compile)
```shell

python3 -m pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
python3 -m pip install -e cvpods

# Or,
pip install -r requirements.txt
python3 setup.py build develop
```

* prepare datasets
```shell
cd /path/to/cvpods/datasets
ln -s /path/to/your/coco/dataset coco
```

* Train & Test
```shell
git clone https://github.com/Megvii-BaseDetection/OTA.git
cd playground/detection/coco/ota.res50.fpn.coco.800size.1x  # for example

# Train
pods_train --num-gpus 8

# Test
pods_test --num-gpus 8 \
    MODEL.WEIGHTS /path/to/your/save_dir/ckpt.pth # optional
    OUTPUT_DIR /path/to/your/save_dir # optional

# Multi node training
## sudo apt install net-tools ifconfig
pods_train --num-gpus 8 --num-machines N --machine-rank 0/1/.../N-1 --dist-url "tcp://MASTER_IP:port"

```

### Results on COCO val set

| Model | Backbone | LR Sched. | mAP | Recall | AP50/AP75/APs/APm/APl | Download |
|:------| :----:   | :----: |:---:| :---:| :---:| :---:|
|  [RetinaNet](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/retinanet/retinanet.res50.fpn.coco.multiscale.1x) | R50   | 1x       | 36.5 |  53.4  |  56.2/39.3/21.9/40.5/47.7  | - |
|  [Faster R-CNN](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/rcnn/faster_rcnn.res50.fpn.coco.multiscale.1x) | R50   | 1x       | 38.1 |  52.2  |  58.9/41.0/22.5/41.5/48.9  | - |
|  [FCOS](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/fcos/fcos.res50.fpn.coco.800size.1x) | R50   | 1x       | 38.7 |  57.0   | 57.5/41.7/22.6/42.7/49.9   | - |
|  [FreeAnchor](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/free_anchor/free_anchor.res50.fpn.coco.800size.1x) | R50   | 1x | 38.4 | 55.4  | 57.0/41.1/21.9/41.7/51.8      | - |
|  [ATSS](https://github.com/Megvii-BaseDetection/cvpods/tree/master/playground/detection/coco/atss/atss.res50.fpn.coco.800size.1x) | R50   | 1x    | 39.4 | 57.7    |  57.5/42.7/22.9/42.9/51.2   | - |
|  [PAA\(w/. Voting\)](https://github.com/kkhoot/PAA) | R50   | 1x  | 40.4 |   -  |  -   | - |
|  [OTA](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.res50.fpn.coco.800size.1x) | R50   | 1x       | **40.7**  |  **59.0** |  **58.4**/**44.3**/**23.2**/**45.0**/**53.6**    | [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/ETpQpDF_5E5JlmNfK1h4zLABNH1St_BXLnkvbpKjAEB5Tg) |

### Results on COCO test-dev
| Model | Backbone | LR Sched. | Training Scale (ShortSide) |mAP | AP50/AP75/APs/APm/APl | Download |
|:------| :----:   | :----: |:---:| :---:| :---:| :---:|
|  [OTA](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.res101.fpn.coco.800size.1x) | R101   | 2x | 640~800 | 45.3 | 63.5/49.3/26.9/48.8/56.1   | [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EXRgFRfL2ZZHiuKEK2bNn5oBjKIlQwaeX0zH02wWomGLYQ?e=6Ctp5E) |
|  [OTA](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.fpn.coco.800size.1x) | X101     | 2x | 640~800 | 47.0 | 65.8/51.1/29.2/50.4/57.9 | [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/Ec2yTrxYDZFAgqWGEnNT6pwB870Frg641WRy7zctHyRzPw?e=YS1RC2) |
|  [OTA](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | X101-DCN | 2x | 640~800 | 49.2 |   67.6/53.5/30.0/52.5/62.3 | [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EYy9odfpIEhIszqrI_vbuzIBlPcW7YRZYmXaT9ws7FkRRg?e=ZYo8SO) |
|  [OTA*](https://github.com/Joker316701882/OTA/tree/main/playground/detection/coco/ota.x101.dcnv2.fpn.coco.800size.1x) | X101-DCN | 2x | 640~800 | 51.5 |   68.6/57.1/34.1/53.7/64.1 | [weights](https://megvii-my.sharepoint.cn/:u:/g/personal/gezheng_megvii_com/EYy9odfpIEhIszqrI_vbuzIBlPcW7YRZYmXaT9ws7FkRRg?e=ZYo8SO) |

\* stands for ATSS-style testing time augmentation. To enable testing time augmentation, add/modify the following code frac in the corresponding config.py

```python

TEST=dict(
    DETECTIONS_PER_IMAGE=300,
    AUG=dict(
        ENABLED=True,
        MAX_SIZE=3000,
        MIN_SIZES=(400, 500, 600, 640, 700, 900, 1000, 1100, 1200, 1300, 1400, 1800),
        EXTRA_SIZES=((800, 1333),),
        SCALE_FILTER=True,
        SCALE_RANGES=(
        [96, 10000], [96, 10000], [64, 10000], [64, 10000], [64, 10000], [0, 10000], [0, 10000], [0, 256], [0, 256], [0, 192], [0, 192], [0, 96], [0, 10000])
    )
),

```

## Acknowledgement
This repo is developed based on cvpods. Please check [cvpods](https://github.com/Megvii-BaseDetection/cvpods) for more details and features.

## License
This repo is released under the Apache 2.0 license. Please see the LICENSE file for more information.

