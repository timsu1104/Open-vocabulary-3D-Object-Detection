"""
Lift 2D boxes to 3D.

Example Usage: 
python sunrgbd/lift_boxes.py -i 2D -o 3D_LSeg_obb --test

python sunrgbd/lift_boxes.py -i 2D_ov3detic -o 3D_LSeg_obb_ov3detic --test
"""
import os, sys
import numpy as np
import multiprocessing as mp
mp.set_start_method('forkserver', force=True)
from time import time
from PIL import Image
from argparse import ArgumentParser
from functools import partial

# HACK: add cwd
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.projection import SUNRGBD_Calibration, get_edge_mask
from utils.box_3d_utils import nms_3d_faster_obb, get_oriented_box_pca, get_iou_obb
from utils.constants import const_sunrgbd

# Configuration
PSEUDO_LABEL_ROOT = "/data1/lseg_data/data/sunrgbd/pseudo_labels"
PSEUDO_LABEL_PATH = os.path.join(PSEUDO_LABEL_ROOT, "{}.npy")

CALIB_PATH = os.path.join(const_sunrgbd.calib_data_dir, "{}.txt")
DEPTH_PATH = os.path.join(const_sunrgbd.depth_data_dir, "{}.png")
GT_LABEL_PATH = os.path.join(const_sunrgbd.seglabel_data_dir, "img-{}.png")

def cat_box(box_list, l=8):
    if len(box_list) == 0:
        return np.zeros((0, l))
    elif len(box_list) == 1:
        return box_list[0][None, :]
    else:
        return np.stack(box_list, 0)

def compute_box(boxes, labels, depths, calib):
    """
        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)
        
        :return mask bool, (num_points, )
    """
    
    v, u = np.indices(labels.shape)
    boxes_3d = []
    for box in boxes:
        x, y, w, h = box[:4]
        box_label = int(box[-1])
        mask = (u >= x) * (u <= x+w) * (v >= y) * (v <= y+h) * (labels == box_label) * (depths > 0)
        mask = mask.astype('bool')
        if mask.sum() > 1:
            uv_depth = np.stack([u[mask], v[mask], depths[mask]], -1)
            sub_cloud = calib.project_image_to_upright_depth(uv_depth)
            box_3d = np.concatenate([get_oriented_box_pca(sub_cloud), box[-2:]], -1)
            boxes_3d.append(box_3d)
    boxes_3d = cat_box(boxes_3d)
        
    return boxes_3d # M, 9


def lifting(scan_name, debug=False, opts=None):
    """
    scan_name : str e.g. 005051
    """
    assert opts is not None
    
    out_fn = os.path.join(const_sunrgbd.box_root, opts.output, scan_name + "_bbox.npy")
    in_fn = os.path.join(const_sunrgbd.box_root, opts.input, scan_name + ".npy")
    
    if (not debug) and os.path.isfile(out_fn):
        print(scan_name, "already exists. ")
        box = np.load(out_fn)
        return box.shape[0]
    
    start = time()
    
    # Read data
    if opts.use_gt:
        semantic_labels = np.array(Image.open(GT_LABEL_PATH.format(scan_name)))
    else:
        semantic_labels = np.load(PSEUDO_LABEL_PATH.format(scan_name)) + 1
    calibrater = SUNRGBD_Calibration(CALIB_PATH.format(scan_name))
    depth = np.array(Image.open(DEPTH_PATH.format(scan_name))) / 8000
    
    boxes = np.load(in_fn) 
    
    # remove box at edge
    boxes = get_edge_mask(boxes, semantic_labels.shape)
    total_box = boxes.shape[0]
    
    io_time = time() - start
    start = time()
    
    # lifting
    sem_seg_labels = SUNRGBD_Calibration.project_label(semantic_labels)
    boxes = compute_box(boxes, sem_seg_labels, depth, calibrater)
    if boxes.shape[0] == 0:
        np.save(out_fn, boxes)
        print("Saving {}, {}/{} boxes in total. IO time {}s. ".format(scan_name + "_bbox.npy", boxes.shape[0], total_box, io_time))
        return 0
    
    lift_time = time() - start
    start = time()
    
    # nms
    boxes = nms_3d_faster_obb(boxes, opts.thresh_nms, class_wise=True)
    
    nms_time = time() - start
    start = time()
    
    box_pool = np.zeros((0, 8))
    
    if not opts.no_gss:
        # Find the GSS proposal that is close to our proposal
        box_pool = np.load(os.path.join(const_sunrgbd.gss_root, opts.gss_prop, scan_name+"_prop.npy")) # numBox, 8
        aabb_f = box_pool.shape[1] == 7
        
        labels = -100 * np.ones(box_pool.shape[0])
        tmp_score = np.zeros(box_pool.shape[0]) # used for label selection
        for box in boxes:
            if aabb_f: box_pool[:, 6] = np.pi / 2
            iou = np.array([get_iou_obb(box, prop) for prop in box_pool])
            if iou.max() < opts.thresh_match: continue
            index = np.argmax(iou)
            if box[-2] > tmp_score[index]:
                labels[index] = box[-1]
                tmp_score[index] = box[-2]
        scale = box_pool[:, 3:6]
        if not aabb_f: box_pool = box_pool[:, :-1]
        box_pool = np.concatenate([box_pool, np.stack([tmp_score, labels, np.prod(scale, axis=-1), 2 * np.sum(scale * np.roll(scale, 1, axis=-1), axis=-1)], axis=1)], axis=-1)
        boxes = box_pool[labels != -100] # center, size, angle, score, label, size, area
        if boxes.shape[0] == 0:
            np.save(out_fn, boxes)
            print("Saving {}, {}/{} boxes in total. Time {}s. ".format(scan_name + "_bbox.npy", boxes.shape[0], total_box, time() - start + io_time + nms_time))
            return 0
    gss_time = time() - start
    
    boxes = nms_3d_faster_obb(boxes, opts.thresh_size, use_size_score=True, class_wise=True, size_typ="Volume")

    boxes[:, 3:6] /= 2
    boxes[:, [-4, -3]] = boxes[:, [-3, -4]] # labels, score, volume, area
    np.save(out_fn, boxes)
    print("Saving {}, {}/{} boxes in total, elapsed {}s (IO {}s, LIFT {}s, NMS {}s, GSS {}s).".format(scan_name + "_bbox.npy", boxes.shape[0], total_box, io_time + lift_time + nms_time + gss_time, io_time, lift_time, nms_time, gss_time))
    return boxes.shape[0]
    
if __name__ == '__main__':
    
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument('-i', "--input", default="2D", type=str)
    parser.add_argument('-o', "--output", default="3D_LSeg_woprior", type=str)
    parser.add_argument("--gss_prop", default="SZ+V-obb-V+F-obb", type=str)
    parser.add_argument("--thresh_nms", default=0.7, type=float)
    parser.add_argument("--thresh_size", default=0, type=float)
    parser.add_argument("--thresh_match", default=0.3, type=float)
    parser.add_argument("--test", default=False, action="store_true")
    parser.add_argument("--use_gt", default=False, action="store_true")
    parser.add_argument("--no_gss", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(os.path.join(const_sunrgbd.box_root, args.output), exist_ok=True)
    
    lift_func = partial(lifting, opts=args)
    lift_func_debug = partial(lifting, debug=True, opts=args)
    
    scene_list = const_sunrgbd.get_scan_names()
    
    # test
    lift_func_debug('005051')
    if args.test: exit(0)
    print("[INFO] Testing Complete. Launching pipeline....")
    
    mp.freeze_support()
    start = time()
    p = mp.Pool(mp.cpu_count() // 2)
    result = p.map(lift_func, scene_list)
    p.close()
    p.join()
    print("Done! Elapsed {}s. Box stats: Avg {}, Max {}".format(time() - start, sum(result) / len(result), max(result)))