### Dataset

Specify the path in ```datasets/scannet.py```. Then images, depths, and poses can be returned. 

### Visualization

Wandb and some visualization scripts. 

### OpenSeg / LSeg

OpenSeg is not currently available and we will using LSeg first. The weight is released [here](https://github.com/isl-org/lang-seg).

### RegionCLIP

RegionCLIP-zeroshot is a package that can be installed. It also provides zero-shot inference pipeline. So transfering seems not to be tough. 

### Proposed method

Given paired images w.r.t. a scene, RegionCLIP provides pseudo 2D boxes while LSeg provides pixel-wise CLIP-like visual features. Using camera parameters, they can then be projected to 3D point clouds (e.g. 3D bboxes through GSS) and 3D feature clouds (similar to 3DVG-Transformer). The feature cloud can provide labels for each class. Finally, the supervision is 2-fold
- localization loss using pseudo boxes as gt annotations. 
- CLIP-like feature alignment loss for each box. 

### Ways to modify

The best way is to construct an end-to-end pipeline. It is simpler to use if implemented. But the training speed may suffer from the projection speed. 

An alternative way is to construct a dataset, which is almost the same as ScanNetV2 except that the training labels and boxes are replaced with projected LSeg's sem feature and RegionCLIP's boxes. It is much more **tractable** at each single step and the speed will remains the same as the original 3DETR. 

I suggest we start with the second and integrate later if the inference is fast. 

### Work to do

1. LSeg preprocessing (data_preprocessing/pseudo_label_util.py)
2. RegionCLIP+GSS preprocessing (data_preprocessing/pseudo_box_util.py)
3. Loss modification. 