# OTA: Optimal Transport Assignment for Object Detection

This project provides an implementation for "OTA: Optimal Transport Assignment for Object Detection" on PyTorch.

## Results on COCO val set

| Model | Backbone | LR Sched. | mAP | Recall | AP50/AP75/APs/APm/APl | Download |
|:------| :----:   | :----: |:---:| :---:| :---:| :---:|
|  RetinaNet | Res50   | 1x       |  |   |     |    | weights |
|  FCOS | Res50   | 1x       |  |    |  |    | weights |
|  FreeAnchor | Res50   | 1x       |  |   |    |    | weights |
|  ATSS | Res50   | 1x       |  |     |   |   | weights |
|  PAA | Res50   | 1x       |  |     |    |  | weights |
|  OTA | Res50   | 1x       | 40.7  |  59.0 |  58.4/44.3/23.2/45.0/53.6   |    | weights |
