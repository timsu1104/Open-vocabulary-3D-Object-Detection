# Copyright (c) Facebook, Inc. and its affiliates.


""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import cv2
import numpy as np
from torch.utils.data import Dataset
import scipy.io as sio  # to load .mat files for depth points

import utils.pc_util as pc_util
from utils.random_cuboid import RandomCuboid
from utils.pc_util import shift_scale_points, scale_points
from utils.box_util import (
    flip_axis_to_camera_tensor,
    get_3d_box_batch_tensor,
    flip_axis_to_camera_np,
    get_3d_box_batch_np,
)


MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1
DATA_PATH_V1 = "/share/suzhengyuan/code/ScanRefer-3DVG/votenet/sunrgbd/sunrgbd_pc_bbox_50k_v1" ## Replace with path to dataset
RAW_DATA_PATH = "/share/suzhengyuan/code/ScanRefer-3DVG/votenet/sunrgbd/sunrgbd_trainval"
DATA_PATH_V2 = "" ## Not used in the codebase.

NUM_CLS = 10 # sunrgbd number of classes
MAX_NUM_2D_DET = 100 # maximum number of 2d boxes per image
MAX_NUM_PIXEL = 530*730 # maximum number of pixels per image


FEATURE_2D_PATH = ""
PSEUDO_BOX_PATH = ""
MAX_NUM_PSEUDO_BOX = 500 # 200

class SunrgbdDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 20
        self.clip_embed_length = 640
        self.num_angle_bin = 12
        self.max_num_obj = 64
        # self.type2class = {
        #     'bathtub': 0,
        #     'bed': 1,
        #     'bookshelf': 2,
        #     'box': 3,
        #     'chair': 4,
        #     'counter': 5,
        #     'desk': 6,
        #     'door': 7,
        #     'dresser': 8,
        #     'lamp': 9,
        #     'night_stand': 10,
        #     'pillow': 11,
        #     'sink': 12,
        #     'sofa': 13,
        #     'table': 14,
        #     'tv': 15,
        #     'toilet': 16
        # }
        self.type2class = {
            'toilet': 0,
            'bed': 1,
            'chair': 2,
            'bathtub': 3,
            'sofa': 4,
            'dresser': 5,
            'scanner': 6,
            'fridge': 7,
            'lamp': 8,
            'desk': 9,
            'table': 10,
            'stand': 11,
            'cabinet': 12,
            'counter': 13,
            'bin': 14,
            'bookshelf': 15,
            'pillow': 16,
            'microwave': 17,
            'sink': 18,
            'stool': 19
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        # self.type2onehotclass = {
        #     'bathtub': 0,
        #     'bed': 1,
        #     'bookshelf': 2,
        #     'box': 3,
        #     'chair': 4,
        #     'counter': 5,
        #     'desk': 6,
        #     'door': 7,
        #     'dresser': 8,
        #     'lamp': 9,
        #     'night_stand': 10,
        #     'pillow': 11,
        #     'sink': 12,
        #     'sofa': 13,
        #     'table': 14,
        #     'tv': 15,
        #     'toilet': 16
        # }
        # self.support_class = np.array([2, 5, 10, 11, 12, 14])        
        self.support_class = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        # self.support_class = np.array([9, 10, 11, 12, 13, 14, 15, 16])

    def angle2class(self, angle):
        """Convert continuous angle to discrete class
        [optinal] also small regression number from
        class center angle to current angle.

        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        returns class [0,1,...,N-1] and a residual number such that
            class*(2pi/N) + number = angle
        """
        num_class = self.num_angle_bin
        angle = angle % (2 * np.pi)
        assert angle >= 0 and angle <= 2 * np.pi
        angle_per_class = 2 * np.pi / float(num_class)
        shifted_angle = (angle + angle_per_class / 2) % (2 * np.pi)
        class_id = int(shifted_angle / angle_per_class)
        residual_angle = shifted_angle - (
            class_id * angle_per_class + angle_per_class / 2
        )
        return class_id, residual_angle

    def class2angle(self, pred_cls, residual, to_label_format=True):
        """Inverse function to angle2class"""
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format and angle > np.pi:
            angle = angle - 2 * np.pi
        return angle

    def class2angle_batch(self, pred_cls, residual, to_label_format=True):
        num_class = self.num_angle_bin
        angle_per_class = 2 * np.pi / float(num_class)
        angle_center = pred_cls * angle_per_class
        angle = angle_center + residual
        if to_label_format:
            mask = angle > np.pi
            angle[mask] = angle[mask] - 2 * np.pi
        return angle

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        return self.class2angle_batch(pred_cls, residual, to_label_format)

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    def my_compute_box_3d(self, center, size, heading_angle):
        R = pc_util.rotz(-1 * heading_angle)
        l, w, h = size
        x_corners = [-l, l, l, -l, -l, l, l, -l]
        y_corners = [w, w, -w, -w, w, w, -w, -w]
        z_corners = [h, h, h, h, -h, -h, -h, -h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] += center[0]
        corners_3d[1, :] += center[1]
        corners_3d[2, :] += center[2]
        return np.transpose(corners_3d)


class SunrgbdDetectionDataset(Dataset):
    def __init__(
        self,
        dataset_config,
        split_set="train",
        close_set=False,
        root_dir=None,
        meta_data_dir=None,
        pseudo_box_dir=None,
        gss_box_dir=None,
        feature_2d_dir=None,
        gss_feats_dir=None,
        feature_global_dir=None,
        pseudo_feature_dir=None,
        num_points=20000,
        use_color=False,
        use_image=False,
        use_height=False,
        use_v1=True,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
        use_pbox=False,
        use_gss=False,
        only_pbox=False,
        use_2d_feature=False
    ):
        assert num_points <= 50000
        assert split_set in ["train", "val", "trainval"]
        self.dataset_config = dataset_config
        self.use_v1 = use_v1

        if root_dir is None:
            root_dir = DATA_PATH_V1 if use_v1 else DATA_PATH_V2
            
        if pseudo_box_dir is None:
            pseudo_box_dir = PSEUDO_BOX_PATH
            
        if feature_2d_dir is None:
            feature_2d_dir = FEATURE_2D_PATH

        self.data_path = root_dir + "_%s" % (split_set) + '_ov3detic_filtered'
        self.raw_data_path = RAW_DATA_PATH
        self.pseudo_box_dir = pseudo_box_dir
        self.feature_2d_dir = feature_2d_dir
        self.gss_box_dir = gss_box_dir
        self.gss_feats_dir = gss_feats_dir
        self.feature_global_dir = feature_global_dir
        self.pseudo_feature_dir = pseudo_feature_dir

        if split_set in ["train", "val"]:
            self.scan_names = sorted(
                list(
                    set([os.path.basename(x)[0:6] for x in os.listdir(self.data_path)])
                )
            )
        elif split_set in ["trainval"]:
            # combine names from both
            sub_splits = ["train", "val"]
            all_paths = []
            for sub_split in sub_splits:
                data_path = self.data_path.replace("trainval", sub_split)
                basenames = sorted(
                    list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)]))
                )
                basenames = [os.path.join(data_path, x) for x in basenames]
                all_paths.extend(basenames)
            all_paths.sort()
            self.scan_names = all_paths
            
        # self.scan_names = self.scan_names[:100]

        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_image = use_image
        self.use_height = use_height
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(
            min_points=random_cuboid_min_points,
            aspect=0.75,
            min_crop=0.75,
            max_crop=1.0,
        )
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        self.max_num_obj = 64
        
        self.train = split_set == "train"
        self.close_set = close_set
        self.use_pbox = use_pbox
        self.use_gss = use_gss
        self.only_pbox = only_pbox
        self.use_2d_feature = use_2d_feature
        if use_pbox or only_pbox:
            self.max_num_obj = MAX_NUM_PSEUDO_BOX

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        if scan_name.startswith("/"):
            scan_path = scan_name
        else:
            scan_path = os.path.join(self.data_path, scan_name)
        point_cloud = np.load(scan_path + "_pc.npz")["pc"]  # Nx6
        bboxes = np.load(scan_path + "_bbox.npy")  # K,8
        
        if self.use_2d_feature:
            feature_2d = np.load(os.path.join(self.feature_2d_dir, scan_name) + "features.npy")
            if self.feature_global_dir is not None:
                feature_global = np.load(os.path.join(self.feature_global_dir, scan_name) + "features.npy")
            if self.use_pbox:
                assert self.pseudo_feature_dir is not None
                pseudo_feature_2d = np.load(os.path.join(self.pseudo_feature_dir, scan_name) + "features.npy")
            if self.use_gss:
                feature_gss = np.load(os.path.join(self.gss_feats_dir, scan_name) + "features.npy")
            
        # ADD: remove gt box in novel set
        mask = np.isin(bboxes[:, -1], self.dataset_config.support_class)
        if self.use_pbox:
            pseudo_bboxes = np.load(os.path.join(self.pseudo_box_dir, scan_name) + "_bbox.npy")[:, :8]
            # pseudo_mask = np.isin(pseudo_bboxes[:, -1], self.dataset_config.support_class)
        if self.use_gss:
            gss_bboxes = np.load(os.path.join(self.gss_box_dir, scan_name) + "_prop.npy")[:, :8]
            
        if not self.close_set:
            if self.train:
                bboxes = bboxes[mask]
                if self.use_2d_feature:
                    feature_2d = feature_2d[mask]
                if self.use_pbox:
                    # pseudo_bboxes = pseudo_bboxes[~pseudo_mask]
                    if self.use_2d_feature:
                        # feature_2d = np.concatenate([feature_2d, pseudo_feature_2d[~pseudo_mask]], 0)
                        feature_2d = np.concatenate([feature_2d, pseudo_feature_2d], 0)
                if self.use_gss and self.use_2d_feature:
                    feature_2d = np.concatenate([feature_2d, feature_gss], 0)
            # else: # TTA
            #     bboxes = bboxes[~mask]
            #     if self.use_2d_feature:
            #         feature_2d = feature_2d[~mask]
            #     if self.use_pbox:
            #         pseudo_bboxes = pseudo_bboxes[~pseudo_mask]
            #         if self.use_2d_feature:
            #             feature_2d = np.concatenate([feature_2d, pseudo_feature_2d[~pseudo_mask]], 0)
        
        confident_box_num = bboxes.shape[0]
        gt_box_num = bboxes.shape[0]
                    
        if self.use_pbox:
            bboxes = np.concatenate([bboxes, pseudo_bboxes], axis=0)
            confident_box_num += pseudo_bboxes.shape[0]
        if self.use_gss:
            bboxes = np.concatenate([bboxes, gss_bboxes], axis=0)
        if self.only_pbox:
            if self.use_gss:
                bboxes = np.load(os.path.join(self.gss_box_dir, scan_name) + "_prop.npy")[:, :7]
                confident_box_num = bboxes.shape[0]
                gt_box_num = bboxes.shape[0]
            else:
                bboxes = np.load(os.path.join(self.pseudo_box_dir, scan_name) + "_bbox.npy")[:, :8]
                confident_box_num = bboxes.shape[0]
                gt_box_num = bboxes.shape[0]
        
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        if self.use_2d_feature:
            if self.use_gss: assert feature_gss.shape[0] == gss_bboxes.shape[0]
            if self.use_pbox: assert pseudo_feature_2d.shape[0] == pseudo_bboxes.shape[0], f"{pseudo_feature_2d.shape} {pseudo_bboxes.shape}"
            assert feature_2d.shape[0] == bboxes.shape[0], f"{feature_2d.shape}, {bboxes.shape}"
            
            if bboxes.shape[0] > MAX_NUM_OBJ:
                bboxes = bboxes[:MAX_NUM_OBJ]
                feature_2d = feature_2d[:MAX_NUM_OBJ]
                if confident_box_num > MAX_NUM_OBJ:
                    confident_box_num = MAX_NUM_OBJ
        
                    
        if self.use_image:
            # Read camera parameters
            calib_lines = [line for line in open(os.path.join(self.raw_data_path, 'calib', scan_name+'.txt')).readlines()]
            calib_Rtilt = np.reshape(np.array([float(x) for x in calib_lines[0].rstrip().split(' ')]), (3,3), 'F')
            calib_K = np.reshape(np.array([float(x) for x in calib_lines[1].rstrip().split(' ')]), (3,3), 'F')
            # Read image
            full_img = np.array(cv2.imread(os.path.join(self.raw_data_path, 'image', scan_name+'.jpg')))
            full_img_height = full_img.shape[0]
            full_img_width = full_img.shape[1]
            full_img_1d = np.zeros((MAX_NUM_PIXEL*3), dtype=np.float32)
            full_img_1d[:full_img_height*full_img_width*3] = full_img.flatten()

        if not self.use_color:
            point_cloud = point_cloud[:, 0:3]
        else:
            assert point_cloud.shape[1] == 6
            point_cloud = point_cloud[:, 0:6]
            point_cloud[:, 3:] = point_cloud[:, 3:] - MEAN_COLOR_RGB

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate(
                [point_cloud, np.expand_dims(height, 1)], 1
            )  # (N,4) or (N,7)

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:
            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = np.pi - bboxes[:, 6]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 3) - np.pi / 6  # -30 ~ +30 degree
            rot_mat = pc_util.rotz(rot_angle)

            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

            # Augment RGB color
            if self.use_color:
                rgb_color = point_cloud[:, 3:6] + MEAN_COLOR_RGB
                rgb_color *= (
                    1 + 0.4 * np.random.random(3) - 0.2
                )  # brightness change for each channel
                rgb_color += (
                    0.1 * np.random.random(3) - 0.05
                )  # color shift for each channel
                rgb_color += np.expand_dims(
                    (0.05 * np.random.random(point_cloud.shape[0]) - 0.025), -1
                )  # jittering on each pixel
                rgb_color = np.clip(rgb_color, 0, 1)
                # randomly drop out 30% of the points' colors
                rgb_color *= np.expand_dims(
                    np.random.random(point_cloud.shape[0]) > 0.3, -1
                )
                point_cloud[:, 3:6] = rgb_color - MEAN_COLOR_RGB

            # Augment point cloud scale: 0.85x-1.15x
            scale_ratio = np.random.random() * 0.3 + 0.85
            scale_ratio = np.expand_dims(np.tile(scale_ratio, 3), 0)
            point_cloud[:, 0:3] *= scale_ratio
            bboxes[:, 0:3] *= scale_ratio
            bboxes[:, 3:6] *= scale_ratio

            if self.use_height:
                point_cloud[:, -1] *= scale_ratio[0, 0]

            if self.use_random_cuboid:
                point_cloud, bboxes, _, keep_boxes = self.random_cuboid_augmentor(
                    point_cloud, bboxes
                )
                if keep_boxes is not None:
                    gt_box_num = keep_boxes[:gt_box_num].sum()
                    confident_box_num = keep_boxes[:confident_box_num].sum()
                    feature_2d = feature_2d[keep_boxes]

        # ------------------------------- LABELS ------------------------------
        angle_classes = np.zeros((self.max_num_obj,), dtype=np.float32)
        angle_residuals = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_angles = np.zeros((self.max_num_obj,), dtype=np.float32)
        raw_sizes = np.zeros((self.max_num_obj, 3), dtype=np.float32)
        box_mask = np.zeros((self.max_num_obj))
        box_mask[0 : bboxes.shape[0]] = 1
        label_mask = np.zeros((self.max_num_obj))
        label_mask[0 : confident_box_num] = 1
        # max_bboxes = np.zeros((self.max_num_obj, 8))
        # max_bboxes[0 : bboxes.shape[0], :] = bboxes

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((self.max_num_obj, 6))

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            # semantic_class = bbox[7]
            raw_angles[i] = bbox[6] % 2 * np.pi
            box3d_size = bbox[3:6] * 2
            raw_sizes[i, :] = box3d_size
            angle_class, angle_residual = self.dataset_config.angle2class(bbox[6])
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            corners_3d = self.dataset_config.my_compute_box_3d(
                bbox[0:3], bbox[3:6], bbox[6]
            )
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    (zmin + zmax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                    zmax - zmin,
                ]
            )
            target_bboxes[i, :] = target_bbox
        
        if self.use_2d_feature:
            assert feature_2d.shape[0] == bboxes.shape[0], f"{feature_2d.shape}, {bboxes.shape}"
            clip_features = np.zeros((self.max_num_obj, 640))
            clip_features[:feature_2d.shape[0]] = feature_2d

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )

        point_cloud_dims_min = point_cloud.min(axis=0)
        point_cloud_dims_max = point_cloud.max(axis=0)

        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_centers = target_bboxes.astype(np.float32)[:, 0:3]
        box_centers_normalized = shift_scale_points(
            box_centers[None, ...],
            src_range=[
                point_cloud_dims_min[None, ...],
                point_cloud_dims_max[None, ...],
            ],
            dst_range=self.center_normalizing_range,
        )
        box_centers_normalized = box_centers_normalized.squeeze(0)
        box_centers_normalized = box_centers_normalized * box_mask[..., None]

        # re-encode angles to be consistent with VoteNet eval
        angle_classes = angle_classes.astype(np.int64)
        angle_residuals = angle_residuals.astype(np.float32)
        raw_angles = self.dataset_config.class2angle_batch(
            angle_classes, angle_residuals
        )

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        
        if not self.only_pbox:
            target_bboxes_semcls = np.zeros((64, )) # np.zeros((self.max_num_obj))
            target_bboxes_semcls[0 : gt_box_num] = bboxes[:gt_box_num, -1]  # from 0 to 9
            ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["gt_box_all"] = box_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["gt_angle_class_label"] = angle_classes
        ret_dict["gt_angle_residual_label"] = angle_residuals
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max
        if self.use_2d_feature:
            ret_dict["feature_2d"] = clip_features
            if self.feature_global_dir is not None:
                ret_dict["feature_global"] = feature_global # 640
        if self.use_image:
            ret_dict["image"] = full_img_1d
            ret_dict["image_height"] = full_img_height
            ret_dict["image_width"] = full_img_width
            ret_dict["calib_Rtilt"] = calib_Rtilt
            ret_dict["calib_K"] = calib_K
        return ret_dict
