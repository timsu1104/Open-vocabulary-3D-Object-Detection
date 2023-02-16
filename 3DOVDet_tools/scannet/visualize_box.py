"""
Toy example to visualize RGB-D boxes on pointcloud. 
"""
import os
import numpy as np
from time import time
from datetime import datetime
from tqdm import tqdm

from utils.scannet_io_utils import read_mesh_vertices_rgb, read_alignment, write_bbox, write_ply_rgb, GT_MODE, class2type, align_mesh

# Visualization Configurations
SCANNET_3DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/3D_GSS_LSeg_multi" # output path
SCANNET_DIR = "/share/suzhengyuan/data/ScanNetv2/scan"
SCENE_ID = "scene0000_00"
VIS_ROOT = "/home/zhengyuan/packages/RegionCLIP/output/visualizations"

# constructions
BOX_TAG = os.path.split(SCANNET_3DBOXS)[1]
SCANNET_3DBOX_PATH = os.path.join(SCANNET_3DBOXS, "{}_bbox.npy") # scene_id, mode
SCANNET_PC_PATH = os.path.join(SCANNET_DIR, '{}', '{}_vh_clean_2.ply')
SCANNET_META_PATH = os.path.join(SCANNET_DIR, '{}', '{}.txt')
VIS_PATH = os.path.join(VIS_ROOT, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{BOX_TAG}-{SCENE_ID}', '{}.ply')
scan_name = SCENE_ID

os.makedirs(os.path.join(VIS_ROOT, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-{BOX_TAG}-{SCENE_ID}'), exist_ok=True)


if __name__ == '__main__':
    start = time()
    point_cloud, pc_color = read_mesh_vertices_rgb(SCANNET_PC_PATH.format(scan_name, scan_name))
    axis_align_matrix = read_alignment(SCANNET_META_PATH.format(scan_name, scan_name))
    assert axis_align_matrix is not None
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    mesh = align_mesh(SCANNET_PC_PATH.format(scan_name, scan_name), axis_align_matrix)
    mesh.write(VIS_PATH.format('mesh'))
    pts = np.ones((point_cloud.shape[0], 4))
    pts[:,0:3] = point_cloud[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    aligned_vertices = np.copy(point_cloud)
    point_cloud[:,0:3] = pts[:,0:3]
    boxes = np.load(SCANNET_3DBOX_PATH.format(scan_name))
    
    # write_ply_rgb(point_cloud, pc_color, VIS_PATH.format('pointcloud'))
    # print(boxes)
    for i, box in enumerate(tqdm(boxes)):
        # box: 6 + 1
        box_label = box[6]
        box[6] = 0
        write_bbox(box, GT_MODE, VIS_PATH.format(f'{i}-{class2type[box_label]}-score{box[7]}-size{box[8]}-area{box[9]}'))
    # biggest_box = np.zeros(7)
    # biggest_box[:3] = (point_cloud.max(0) + point_cloud.min(0)) / 2
    # biggest_box[3:6] = point_cloud.max(0) - point_cloud.min(0)
    # write_bbox(biggest_box, GT_MODE, VIS_PATH.format('axis'))

    print("Done! Elapsed {}s.".format(time() - start))