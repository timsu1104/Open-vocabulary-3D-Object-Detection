"""
Lift 2D boxes to 3D.
The core function for cropping is PROJECTOR.compute_frustum_box. 
"""
import os
import numpy as np
import multiprocessing as mp
from time import time

from utils.projection import ProjectionHelper
from utils.box_3d_utils import nms_3d_faster, box_3d_iou, vv2cs, cs2vv
from utils.io_utils import load_pose, read_alignment, load_intrinsic, get_scene_list, load_depth, load_label

DATASET_ROOT_DIR = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/scannet/scannet_train_detection_data'
SCANNET_DIR = '/share/suzhengyuan/data/ScanNetv2/scan'
SCANNET_FRAMES_ROOT = "/data/suzhengyuan/ScanRefer/scannet_train_images/frames_square"
SCANNET_LABEL_ROOT = "/data1/lseg_data/data/scannet/pseudo_labels" # LSeg segmentation result
# SCANNET_2DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/2D_refined"
SCANNET_2DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/2D_refined"
SCANNET_3DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/3D_LSeg_woprior" # output path
VIEW = 'multi'
PSEUDO_FLAG = True
NMS_THRESH = 0.7
SIZE_NMS_THRESH = 0
USE_GSS = True
MATCH_THRESH = 0.3
GSS_BASE = "/share/suzhengyuan/code/WyPR/gss/computed_proposal_scannet/SZ+V+SG-V+F"
# "/home/zhengyuan/code/OVDet/third_party/gss/scannet_gss_unsup"

DEBUG=True # replace existing files
TEST = False # only run on scene0000_00

os.makedirs(SCANNET_3DBOXS, exist_ok=True)
SCANNET_META_PATH = os.path.join(SCANNET_DIR, '{}', '{}.txt')
SCANNET_LABEL_PATH = os.path.join(SCANNET_LABEL_ROOT, "{}.npy") # gt suffix: _sem_label
SCANNET_FRAMES = os.path.join(SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAMES, "{}") # name of the file
SCANNET_2DBOX_PATH = os.path.join(SCANNET_2DBOXS, "{}/{}") # scene_id, mode
SCANNET_3DBOX_PATH = os.path.join(SCANNET_3DBOXS, "{}_bbox.npy") # scene_id, mode
GSS_PATH = os.path.join(GSS_BASE, "{}_prop.npy") # scene_id

# projection
PROJECTOR = ProjectionHelper(0.1, 10.0, [240, 320])

def cat_box(box_list, l=7):
    if len(box_list) == 0:
        return np.zeros((0, l))
    elif len(box_list) == 1:
        return box_list[0]
    else:
        return np.concatenate(box_list, 0)

def compute_box(points, depths, camera_to_world, box_batch, label_batch, axis_align_matrix, intrinsic):
    """
        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)
        
        :return mask bool, (num_points, )
    """
    num_frames = len(camera_to_world)
    boxes = []
    intrinsic = PROJECTOR.resize_intrinsic(intrinsic)

    for i in range(num_frames):
        if box_batch[i].shape[0] == 0: # neglect box of zero size
            continue
        box = PROJECTOR.compute_frustum_box(
            points, 
            depths[i], 
            camera_to_world[i], 
            boxes=box_batch[i], 
            labels=label_batch if VIEW == 'multi' else label_batch[i], 
            axis_align_matrix=axis_align_matrix, 
            intrinsic=intrinsic, 
            view=VIEW
        )
        if isinstance(box, np.ndarray):
            boxes.append(box)
    
    boxes = cat_box(boxes)
        
    return boxes


def lifting(scan_name):
    """
    scan_name : str
    """
    if (not DEBUG) and os.path.isfile(os.path.join(SCANNET_3DBOXS, scan_name) + "_bbox.npy"):
        print(scan_name, "already exists. ")
        box = np.load(os.path.join(SCANNET_3DBOXS, scan_name) + "_bbox.npy")
        return box.shape[0]
    
    start = time()
    frame_list = list(map(lambda x: x.split(".")[0], sorted(os.listdir(SCANNET_FRAMES.format(scan_name, "color")))))
    
    # Read data
    point_cloud = np.load(os.path.join(DATASET_ROOT_DIR, scan_name) + "_vert.npy")[:, :3]
    semantic_labels = np.load(SCANNET_LABEL_PATH.format(scan_name), allow_pickle=True)
    if VIEW == 'single':
        if PSEUDO_FLAG:
            semantic_labels = np.stack([semantic_labels.item()[int(frame_id)].astype(np.int64) for frame_id in frame_list])
        else:
            semantic_labels = np.stack([load_label(SCANNET_FRAME_PATH.format(scan_name, "label-mapped", "{}.png".format(frame_id))) for frame_id in frame_list])
    elif PSEUDO_FLAG:
        point_cloud = semantic_labels[:, :3]
        semantic_labels = semantic_labels[:, 3]
        
    intrinsic = load_intrinsic(SCANNET_FRAMES.format(scan_name, 'intrinsic_depth.txt'))
    
    axis_align_matrix = read_alignment(SCANNET_META_PATH.format(scan_name, scan_name))
    orig_point_cloud = PROJECTOR.project_alignment(point_cloud, np.linalg.inv(axis_align_matrix))
            
    scene_depths = [load_depth(SCANNET_FRAME_PATH.format(scan_name, "depth", "{}.png".format(frame_id))) for frame_id in frame_list]
    scene_poses = [load_pose(SCANNET_FRAME_PATH.format(scan_name, "pose", "{}.txt".format(frame_id))) for frame_id in frame_list]
    boxes = [np.load(SCANNET_2DBOX_PATH.format(scan_name, "color/{}.npy".format(frame_id))) for frame_id in frame_list]
    
    # remove box at edge
    boxes = [PROJECTOR.get_edge_mask(box) for box in boxes]
    
    total_box = sum([b.shape[0] for b in boxes])
    
    io_time = time() - start
    start = time()
    
    # lifting
    sem_seg_labels = PROJECTOR.project_label(semantic_labels, PSEUDO_FLAG)
    boxes = compute_box(orig_point_cloud, scene_depths, scene_poses, boxes, sem_seg_labels, axis_align_matrix, intrinsic)
    if boxes.shape[0] == 0:
        np.save(SCANNET_3DBOX_PATH.format(scan_name), boxes)
        print("Saving {}, {}/{} boxes in total. IO time {}s. ".format(scan_name + "_bbox.npy", boxes.shape[0], total_box, io_time))
        return 0
    
    lift_time = time() - start
    start = time()
    
    # nms
    boxes = nms_3d_faster(boxes, NMS_THRESH, class_wise=True)
    
    nms_time = time() - start
    start = time()
    
    # Find the GSS proposal that is close to our proposal
    if USE_GSS:
        box_pool = np.load(GSS_PATH.format(scan_name)) # numBox, 7
        box_pool = cs2vv(box_pool)
        # box_pool = nms_3d_faster(box_pool, NMS_THRESH)
        labels = -100 * np.ones(box_pool.shape[0])
        tmp_score = np.zeros(box_pool.shape[0]) # used for label selection
        for box in boxes:
            iou = box_3d_iou(box, box_pool)
            if iou.max() < MATCH_THRESH: continue
            index = np.argmax(iou)
            if box[-2] > tmp_score[index]:
                labels[index] = box[-1]
                tmp_score[index] = box[-2]
        # box_pool[:, -1] = labels
        scale = box_pool[:, 3:6] - box_pool[:, 0:3]
        box_pool = np.concatenate([box_pool[:, :6], np.stack([tmp_score, labels, np.prod(scale, axis=-1), 2 * np.sum(scale * np.roll(scale, 1, axis=-1), axis=-1)], axis=1)], axis=-1)
        boxes = box_pool[labels != -100]
        if boxes.shape[0] == 0:
            np.save(SCANNET_3DBOX_PATH.format(scan_name), boxes)
            print("Saving {}, {}/{} boxes in total. Time {}s. ".format(scan_name + "_bbox.npy", boxes.shape[0], total_box, time() - start + io_time + nms_time))
            return 0
        boxes = nms_3d_faster(boxes, SIZE_NMS_THRESH, use_size_score=True, class_wise=True, size_typ="Volume")
    gss_time = time() - start

    boxes = vv2cs(boxes)
    boxes[:, [6, 7]] = boxes[:, [7, 6]]
    np.save(SCANNET_3DBOX_PATH.format(scan_name), boxes)
    print("Saving {}, {}/{} boxes in total, elapsed {}s (IO {}s, LIFT {}s, NMS {}s, GSS {}s).".format(scan_name + "_bbox.npy", boxes.shape[0], total_box, io_time + lift_time + nms_time + gss_time, io_time, lift_time, nms_time, gss_time))
    return boxes.shape[0]
    
if __name__ == '__main__':
    scene_list = get_scene_list()
    
    # test
    lifting(scene_list[0])
    if TEST: exit(0)
    print("[INFO] Testing Complete. Launching pipeline....")
    
    start = time()
    p = mp.Pool(mp.cpu_count())
    result = p.map(lifting, scene_list[1:])
    p.close()
    p.join()
    print("Done! Elapsed {}s. Box stats: Avg {}, Max {}".format(time() - start, sum(result) / len(result), max(result)))