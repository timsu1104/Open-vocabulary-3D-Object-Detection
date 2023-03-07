"""
Lift 2D boxes to 3D.
The core function for cropping is PROJECTOR.compute_frustum_box. 
"""
import os, sys
import numpy as np
import multiprocessing as mp
mp.set_start_method('forkserver', force=True)
from time import time

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.projection import ProjectionHelper
from utils.box_3d_utils import nms_3d_faster, box_3d_iou, vv2cs, cs2vv
from utils.io_utils import load_pose, read_alignment, load_intrinsic, get_scene_list, load_depth, load_label, type2class, nyu40ids, get_scene_val_list

DATASET_ROOT_DIR = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/scannet/scannet_train_detection_data'
SCANNET_DIR = '/share/suzhengyuan/data/ScanNetv2/scan'
SCANNET_FRAMES_ROOT = "/data/suzhengyuan/ScanRefer/scannet_train_images/frames_square"
SCANNET_LABEL_ROOT = "/data1/lseg_data/data/scannet/pseudo_labels_maskclip"
# "/data1/lseg_data/data/scannet/pseudo_labels" # LSeg segmentation result
# SCANNET_2DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/2D_refined"
SCANNET_2DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/2D_refined"
SCANNET_3DBOXS = DATASET_ROOT_DIR # "/share/suzhengyuan/data/RegionCLIP_boxes/3D_MaskCLIP_woprior" # output path
VIEW = 'multi'
PSEUDO_FLAG = True
NMS_THRESH = 0.7
SIZE_NMS_THRESH = 0
USE_GSS = True
MATCH_THRESH = 0.3
# GSS_BASE = "/share/suzhengyuan/code/WyPR/gss/computed_proposal_scannet/SZ+V+SG-V+F"
GSS_BASE = "/home/zhengyuan/code/OVDet/third_party/gss/scannet_gss_unsup"

DEBUG=True # replace existing files
TEST = False # only run on scene0000_00

def gt_number(scan_name):
    """
    scan_name : str
    """
    number = np.zeros(len(type2class.keys()))
    box = np.load(os.path.join(SCANNET_3DBOXS, scan_name) + "_bbox.npy")
    for i, _c in enumerate(nyu40ids):
        if (box[:, -1] == _c).sum() > 0:
            number[i] += 1
    
    return number
    
if __name__ == '__main__':
    scene_list = get_scene_list() + get_scene_val_list()
    
    mp.freeze_support()
    start = time()
    p = mp.Pool(processes=mp.cpu_count() // 2)
    result = p.map(gt_number, scene_list)
    p.close()
    p.join()
    print("Done! {}".format(sum(result)))