import os
import numpy as np
from scipy.stats import mode
import multiprocessing as mp
from time import time

from utils.projection import ProjectionHelper
from utils.box_3d_utils import vv2cs, cs2vv
from utils.io_utils import get_scene_list, nyu40ids

DATASET_ROOT_DIR = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/scannet/scannet_train_detection_data'
SCANNET_DIR = '/share/suzhengyuan/data/ScanNetv2/scan'
SCANNET_LABEL_ROOT = DATASET_ROOT_DIR # "" # LSeg segmentation result
SCANNET_3DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/3D_GSS_gt" # output path
NMS_THRESH = 0.7
USE_GSS = True
GSS_BASE = "/home/zhengyuan/code/OVDet/third_party/gss/scannet_gss_unsup"

os.makedirs(SCANNET_3DBOXS, exist_ok=True)

SCANNET_META_PATH = os.path.join(SCANNET_DIR, '{}', '{}.txt')
SCANNET_LABEL_PATH = os.path.join(SCANNET_LABEL_ROOT, "{}_sem_label.npy")
SCANNET_3DBOX_PATH = os.path.join(SCANNET_3DBOXS, "{}_bbox.npy") # scene_id, mode
GSS_PATH = os.path.join(GSS_BASE, "{}_prop.npy") # scene_id

PROJECTOR = ProjectionHelper(0.1, 10.0, [240, 320])

def labeling(scan_name, replace = True):
    """
    scan_name : str
    """
    if not replace and os.path.isfile(SCANNET_3DBOX_PATH.format(scan_name)):
        boxes = np.load(SCANNET_3DBOX_PATH.format(scan_name))
        return boxes.shape[0]
    
    start = time()
    
    # Read data
    point_cloud = np.load(os.path.join(DATASET_ROOT_DIR, scan_name) + "_vert.npy")[:, :3]
    semantic_labels = np.load(SCANNET_LABEL_PATH.format(scan_name))
            
    boxes = np.load(GSS_PATH.format(scan_name)) # numBox, 7
    total_box = boxes.shape[0]
    boxes = cs2vv(boxes)
    box_pool = []
    for box in boxes:
        mask = np.prod((point_cloud >= box[:3]) * (point_cloud <= box[3:6]), axis=-1).astype('bool')
        selected_label = semantic_labels[mask]
        selected_label = selected_label[selected_label >= 3]
        if selected_label.shape[0] == 0: continue
        label = mode(selected_label, keepdims=False).mode
        if label in nyu40ids:
            # print(label)
            box[-1] = label
            box_pool.append(box)
    if len(box_pool) >= 2:
        boxes = np.stack(box_pool)
    elif len(box_pool) == 1:
        boxes = box_pool[0][None, :]
    else:
        boxes = np.zeros((0, 7))
        np.save(SCANNET_3DBOX_PATH.format(scan_name), boxes)
        print("Saving {}, {}/{} boxes in total, elapsed {}s.".format(scan_name + "_bbox.npy", boxes.shape[0], total_box, time() - start))
        return boxes.shape[0]

    boxes = vv2cs(boxes)
    
    np.save(SCANNET_3DBOX_PATH.format(scan_name), boxes)
    print("Saving {}, {}/{} boxes in total, elapsed {}s.".format(scan_name + "_bbox.npy", boxes.shape[0], total_box, time() - start))
    return boxes.shape[0]

if __name__ == '__main__':
    scene_list = get_scene_list()
    
    labeling(scene_list[0])
    
    start = time()
    p = mp.Pool(mp.cpu_count() // 2)
    result = p.map(labeling, scene_list[1:])
    p.close()
    p.join()
    print("Done! Elapsed {}s. Max number of box: {}".format(time() - start, max(result)))