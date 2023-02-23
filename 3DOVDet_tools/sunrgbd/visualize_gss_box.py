"""
Toy example to visualize boxes on pointcloud. 

Example Usage: 
python sunrgbd/visualize_gss_box.py --box $box_tag 
"""
import os, sys
import numpy as np
from time import time
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser

# HACK: add cwd
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.io_utils import write_oriented_bbox, write_ply_rgb, GT_MODE, write_bbox
from utils.constants import const_sunrgbd

if __name__ == '__main__':
    
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--box", default="SZ+V-obb-V+F-obb", type=str)
    parser.add_argument('-s', "--scene_id", default='005051', type=str)
    args = parser.parse_args()
    
    VIS_PATH = os.path.join(const_sunrgbd.vis_root, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{args.box}-{args.scene_id}', '{}.ply')
    scan_path = args.scene_id
    
    start = time()
    point_cloud = np.load(os.path.join(const_sunrgbd.root_dir+'_train', scan_path + "_pc.npz"))["pc"]
    gt_boxes = np.load(os.path.join(const_sunrgbd.root_dir+'_train', scan_path + "_bbox.npy"))
    os.makedirs(os.path.split(VIS_PATH)[0], exist_ok=True)
    prop = np.load(os.path.join(const_sunrgbd.gss_root, args.box, scan_path + "_prop.npy"))
    write_ply_rgb(point_cloud[..., :3], point_cloud[..., 3:6]*256, VIS_PATH.format('pointcloud'))
    
    if gt_boxes.shape[0] > 0:
        gt_boxes[:, 3:6] *= 2
        gt_boxes[:, 6] *= -1
        write_oriented_bbox(gt_boxes[:, :7], VIS_PATH.format('gt_boxes'))
        
    # if prop.shape[0] > 0:
    #     for i, box in enumerate(tqdm(prop)):
    #         write_bbox(box, GT_MODE, VIS_PATH.format(f'prop-{i}'))
            # write_oriented_bbox(gt_boxes[:, :7], VIS_PATH.format('gt_boxes'))
    prop[:, 3:6] *= 2
    prop[:, 6] *= -1
    write_oriented_bbox(prop[:, :7], VIS_PATH.format('prop_boxes'))
    

    print("Done! Elapsed {}s.".format(time() - start))