"""
Toy example to align RGB-D image with pointcloud and visualize the result. 
"""
import os
import numpy as np
from time import time
from datetime import datetime

from utils.io_utils import read_alignment, write_ply_rgb, align_mesh, load_intrinsic, load_depth, load_pose, load_image

# Visualization Configurations
SCANNET_FRAMES_ROOT = "/data/suzhengyuan/ScanRefer/scannet_train_images/frames_square"
SCANNET_DIR = "/share/suzhengyuan/data/ScanNetv2/scan"
SCENE_ID = "scene0000_00"
FRAME_ID = "1180"
VIS_ROOT = "/home/zhengyuan/packages/RegionCLIP/output/visualizations"

# constructions
SCANNET_PC_PATH = os.path.join(SCANNET_DIR, '{}', '{}_vh_clean_2.ply')
SCANNET_META_PATH = os.path.join(SCANNET_DIR, '{}', '{}.txt')
VIS_PATH = os.path.join(VIS_ROOT, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-depth-{SCENE_ID}', '{}.ply')
scan_name = SCENE_ID
frame_id = FRAME_ID
SCANNET_FRAMES = os.path.join(SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAMES, "{}") # name of the file

os.makedirs(os.path.join(VIS_ROOT, f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}-depth-{SCENE_ID}'), exist_ok=True)

def depth2xyz(cam_matrix, depth):
    # create xyz coordinates from image position
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    xyz = (np.linalg.inv(cam_matrix[:3, :3]).dot(uv1_points.T) * depth.ravel()).T
    return xyz

def project_alignment(point_cloud, axis_align_matrix):
    pts = np.ones((point_cloud.shape[0], 4))
    pts[:,0:3] = point_cloud[:,0:3]
    pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
    aligned_vertices = np.copy(point_cloud)
    aligned_vertices[:,0:3] = pts[:,0:3]
    return aligned_vertices

if __name__ == '__main__':
    start = time()
    axis_align_matrix = read_alignment(SCANNET_META_PATH.format(scan_name, scan_name))
    axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    mesh = align_mesh(SCANNET_PC_PATH.format(scan_name, scan_name), axis_align_matrix)
    mesh.write(VIS_PATH.format('mesh'))
    
    image = load_image(SCANNET_FRAME_PATH.format(scan_name, "color", "{}.jpg".format(frame_id)))
    depth = load_depth(SCANNET_FRAME_PATH.format(scan_name, "depth", "{}.png".format(frame_id)))
    pose = load_pose(SCANNET_FRAME_PATH.format(scan_name, "pose", "{}.txt".format(frame_id)))
    intrinsic = load_intrinsic(SCANNET_FRAMES.format(scan_name, 'intrinsic_depth.txt'))
    depth_size = (640, 480)  # intrinsic matrix is based on 640x480 depth maps.
    image_dims = (320, 240)
    resize_scale = (depth_size[0] / image_dims[0], depth_size[1] / image_dims[1])
    intrinsic[0] /= resize_scale[0]
    intrinsic[1] /= resize_scale[1]
    
    v, u = np.indices(depth.shape)
    points = np.ones((image_dims[0] * image_dims[1], 4))
    points = depth2xyz(u.ravel(), v.ravel(), depth.ravel(), intrinsic)
    points = np.matmul(points, pose[:3, :3].T) + pose[:3, 3]
    points = project_alignment(points, axis_align_matrix)
    
    write_ply_rgb(points, image.reshape(-1, 3), VIS_PATH.format('receptive_field'))

    print("Done! Elapsed {}s.".format(time() - start))