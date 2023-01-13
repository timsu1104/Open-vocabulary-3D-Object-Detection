"""
Use RegionCLIP+GSS to extract boxes from ScanNet images. 
TODO: Refer to the zero-shot inference pipeline of RegionCLIP. Use GSS's proposal to find the best 3D box. 
"""

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.modeling import GeneralizedRCNNWithTTA