"""
Toy example to visualize boxes on pointcloud. 

Example Usage: 
python sunrgbd/visualize_box.py --box $box_tag 
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
from utils.io_utils import write_bbox, write_ply_rgb, GT_MODE
from utils.constants import const_sunrgbd

if __name__ == '__main__':
    
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--box", default="test", type=str)
    parser.add_argument('-s', "--scene_id", default='000000', type=str)
    args = parser.parse_args()
    
    VIS_PATH = os.path.join(const_sunrgbd.vis_root, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{args.box}-{args.scene_id}', '{}.ply')
    os.makedirs(VIS_PATH, exist_ok=True)
    scan_path = args.scene_id
    
    start = time()
    point_cloud = np.load(os.path.join(const_sunrgbd.root_dir, scan_path + "_pc.npz"))["pc"]
    boxes = np.load(os.path.join(const_sunrgbd.root_dir, scan_path + "_bbox.npy"))
    
    write_ply_rgb(point_cloud[..., :3], point_cloud[..., 3:6], VIS_PATH.format('pointcloud'))
    for i, box in enumerate(tqdm(boxes)):
        # box: 6 + 1
        box_label = box[7]
        box[6] = 0
        write_bbox(box[:7], GT_MODE, VIS_PATH.format(f'{i}-{const_sunrgbd.class2type[box_label]}-score{box[8]}-size{box[9]}-area{box[10]}'))

    print("Done! Elapsed {}s.".format(time() - start))