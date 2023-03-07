# Copyright (c) Facebook, Inc. and its affiliates.

""" 
Modified from https://github.com/facebookresearch/votenet
Dataset for object bounding box regression.
An axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz)
where (cx,cy,cz) is the center point of the box, dx is the x-axis length of the box.
"""
import os
import sys

import numpy as np
import torch
import utils.pc_util as pc_util
from torch.utils.data import Dataset
from utils.box_util import (flip_axis_to_camera_np, flip_axis_to_camera_tensor,
                            get_3d_box_batch_np, get_3d_box_batch_tensor)
from utils.pc_util import scale_points, shift_scale_points
from utils.image_util import image_processor
from utils.random_cuboid import RandomCuboid



IGNORE_LABEL = -100
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DATASET_ROOT_DIR = "/share/suzhengyuan/code/ScanRefer-3DVG/votenet/matterport3d/matterport_detection_data"  ## Replace with path to dataset
DATASET_METADATA_DIR = "/share/suzhengyuan/code/ScanRefer-3DVG/votenet/matterport3d/meta_data" ## Replace with path to dataset
SCANNET_FRAMES_ROOT = "/share/suzhengyuan/data/matterport3d/v1/scans"
SCANNET_FRAMES = os.path.join(SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAMES, "{}") # name of the file
FEATURE_2D_PATH = ''
PSEUDO_BOX_PATH = ""
MAX_NUM_PSEUDO_BOX = 1000


class MatterportDatasetConfig(object):
    def __init__(self):
        self.num_semcls = 18
        self.clip_embed_length = 640
        self.num_angle_bin = 1
        self.max_num_obj = 64
        self.max_num_frame = 417

        self.type2class = {
            'chair': 0, 
            'door': 1, 
            'table': 2, 
            'picture': 3, 
            'cabinet': 4, 
            'cushion': 5, 
            'window': 6, 
            'sofa': 7, 
            'bed': 8, 
            'curtain': 9, 
            'chest_of_drawers': 10, 
            'plant': 11, 
            'sink': 12, 
            'stairs': 13, 
            'ceiling': 14, 
            'toilet': 15, 
            'stool': 16, 
            'towel': 17, 
            'mirror': 18, 
            'tv_monitor': 19, 
            'shower': 20, 
            'column': 21, 
            'bathtub': 22, 
            'counter': 23, 
            'fireplace': 24, 
            'lighting': 25, 
            'beam': 26, 
            'railing': 27, 
            'shelving': 28, 
            'blinds': 29, 
            'gym_equipment': 30, 
            'seating': 31, 
            'board_panel': 32, 
            'furniture': 33, 
            'appliances': 34, 
            'clothes': 35, 
            'objects': 36, 
            'misc': 37
        }
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        
        self.support_class = np.arange(19)

        # Semantic Segmentation Classes. Not used in 3DETR
        self.num_class_semseg = 20
        self.type2class_semseg = {
            'wall': 0, 
            'floor': 1, 
            'chair': 2, 
            'door': 3, 
            'table': 4, 
            'picture': 5, 
            'cabinet': 6, 
            'cushion': 7, 
            'window': 8, 
            'sofa': 9, 
            'bed': 10, 
            'curtain': 11, 
            'chest_of_drawers': 12, 
            'plant': 13, 
            'sink': 14, 
            'stairs': 15, 
            'ceiling': 16, 
            'toilet': 17, 
            'stool': 18, 
            'towel': 19, 
            'mirror': 20, 
            'tv_monitor': 21, 
            'shower': 22, 
            'column': 23, 
            'bathtub': 24, 
            'counter': 25, 
            'fireplace': 26, 
            'lighting': 27, 
            'beam': 28, 
            'railing': 29, 
            'shelving': 30, 
            'blinds': 31, 
            'gym_equipment': 32, 
            'seating': 33, 
            'board_panel': 34, 
            'furniture': 35, 
            'appliances': 36, 
            'clothes': 37, 
            'objects': 38, 
            'misc': 39
        }
        self.class2type_semseg = {
            self.type2class_semseg[t]: t for t in self.type2class_semseg
        }

    def angle2class(self, angle):
        raise ValueError("ScanNet does not have rotated bounding boxes.")

    def class2anglebatch_tensor(self, pred_cls, residual, to_label_format=True):
        zero_angle = torch.zeros(
            (pred_cls.shape[0], pred_cls.shape[1]),
            dtype=torch.float32,
            device=pred_cls.device,
        )
        return zero_angle

    def class2anglebatch(self, pred_cls, residual, to_label_format=True):
        zero_angle = np.zeros(pred_cls.shape[0], dtype=np.float32)
        return zero_angle

    def param2obb(
        self,
        center,
        heading_class,
        heading_residual,
        size_class,
        size_residual,
        box_size=None,
    ):
        heading_angle = self.class2angle(heading_class, heading_residual)
        if box_size is None:
            box_size = self.class2size(int(size_class), size_residual)
        obb = np.zeros((7,))
        obb[0:3] = center
        obb[3:6] = box_size
        obb[6] = heading_angle * -1
        return obb

    def box_parametrization_to_corners(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_tensor(box_center_unnorm)
        boxes = get_3d_box_batch_tensor(box_size, box_angle, box_center_upright)
        return boxes

    def box_parametrization_to_corners_np(self, box_center_unnorm, box_size, box_angle):
        box_center_upright = flip_axis_to_camera_np(box_center_unnorm)
        boxes = get_3d_box_batch_np(box_size, box_angle, box_center_upright)
        return boxes

    @staticmethod
    def rotate_aligned_boxes(input_boxes, rot_mat):
        centers, lengths = input_boxes[:, 0:3], input_boxes[:, 3:6]
        new_centers = np.dot(centers, np.transpose(rot_mat))

        dx, dy = lengths[:, 0] / 2.0, lengths[:, 1] / 2.0
        new_x = np.zeros((dx.shape[0], 4))
        new_y = np.zeros((dx.shape[0], 4))

        for i, crnr in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
            crnrs = np.zeros((dx.shape[0], 3))
            crnrs[:, 0] = crnr[0] * dx
            crnrs[:, 1] = crnr[1] * dy
            crnrs = np.dot(crnrs, np.transpose(rot_mat))
            new_x[:, i] = crnrs[:, 0]
            new_y[:, i] = crnrs[:, 1]

        new_dx = 2.0 * np.max(new_x, 1)
        new_dy = 2.0 * np.max(new_y, 1)
        new_lengths = np.stack((new_dx, new_dy, lengths[:, 2]), axis=1)

        return np.concatenate([new_centers, new_lengths], axis=1)


class MatterportDetectionDataset(Dataset):
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
        num_points=40000,
        use_color=False,
        use_image=False,
        use_height=False,
        augment=False,
        use_random_cuboid=True,
        random_cuboid_min_points=30000,
        use_pbox=False,
        use_gss=False,
        only_pbox=False,
        use_2d_feature=False
    ):

        self.dataset_config = dataset_config
        assert split_set in ["train", "val"]
        if root_dir is None:
            root_dir = DATASET_ROOT_DIR

        if meta_data_dir is None:
            meta_data_dir = DATASET_METADATA_DIR
            
        if pseudo_box_dir is None:
            pseudo_box_dir = PSEUDO_BOX_PATH
            
        if feature_2d_dir is None:
            feature_2d_dir = FEATURE_2D_PATH

        self.data_path = root_dir
        self.pseudo_box_dir = pseudo_box_dir
        self.feature_2d_dir = feature_2d_dir
        self.gss_box_dir = gss_box_dir
        self.gss_feats_dir = gss_feats_dir
        self.feature_global_dir = feature_global_dir
        self.pseudo_feature_dir = pseudo_feature_dir
        self.train = split_set == "train"
        self.close_set = close_set
        self.only_pbox = only_pbox
        all_scan_names = list(
            set(
                [
                    os.path.basename(x)[0:12]
                    for x in os.listdir(self.data_path)
                    if x.startswith("scene")
                ]
            )
        )
        if split_set == "all":
            self.scan_names = all_scan_names
        elif split_set in ["train", "val", "test"]:
            split_filenames = os.path.join(meta_data_dir, f"matterport_{split_set}.txt")
            with open(split_filenames, "r") as f:
                self.scan_names = f.read().splitlines()
            # remove unavailiable scans
            num_scans = len(self.scan_names)
            self.scan_names = [
                sname for sname in self.scan_names if sname in all_scan_names
            ]
            print(f"kept {len(self.scan_names)} scans out of {num_scans}")
        else:
            raise ValueError(f"Unknown split name {split_set}")

        self.num_points = num_points
        self.use_color = use_color
        self.use_image = use_image
        self.use_height = use_height
        self.augment = augment
        self.use_pbox = use_pbox
        self.use_gss = use_gss
        self.use_2d_feature = use_2d_feature
        self.use_random_cuboid = use_random_cuboid
        self.random_cuboid_augmentor = RandomCuboid(min_points=random_cuboid_min_points)
        self.center_normalizing_range = [
            np.zeros((1, 3), dtype=np.float32),
            np.ones((1, 3), dtype=np.float32),
        ]
        
        if self.use_image:
            self.img_processor = image_processor()
            self.max_num_frame = dataset_config.max_num_frame
            
        if use_pbox:
            self.dataset_config.max_num_obj = MAX_NUM_PSEUDO_BOX
        
        print("Dataset built.")

    def __len__(self):
        return len(self.scan_names)

    def __getitem__(self, idx):
        scan_name = self.scan_names[idx]
        mesh_vertices = np.load(os.path.join(self.data_path, scan_name) + "_vert.npy")
        
        instance_bboxes = np.load(os.path.join(self.data_path, scan_name) + "_bbox.npy")
                
        mask = np.isin(instance_bboxes[:, -1], self.dataset_config.support_class)
        if self.use_pbox:
            pseudo_bboxes = np.load(os.path.join(self.pseudo_box_dir, scan_name) + "_bbox.npy")[:, :7]
            # pseudo_mask = np.isin(pseudo_bboxes[:, -1], self.dataset_config.support_class)
        if self.use_gss:
            gss_bboxes = np.load(os.path.join(self.gss_box_dir, scan_name) + "_prop.npy")[:, :7]
        
        if self.use_2d_feature:
            feature_2d = np.load(os.path.join(self.feature_2d_dir, scan_name) + "features.npy")
            if self.feature_global_dir is not None:
                feature_global = np.load(os.path.join(self.feature_global_dir, scan_name) + "features.npy")
            if self.use_pbox:
                assert self.pseudo_feature_dir is not None
                pseudo_feature_2d = np.load(os.path.join(self.pseudo_feature_dir, scan_name) + "features.npy")
                assert pseudo_feature_2d.shape[0] == pseudo_bboxes.shape[0]
            if self.use_gss:
                feature_gss = np.load(os.path.join(self.gss_feats_dir, scan_name) + "features.npy")
        
        if not self.close_set:
            if self.train:
                instance_bboxes = instance_bboxes[mask]
                if self.use_2d_feature:
                    feature_2d = feature_2d[mask]
                if self.use_pbox:
                    # pseudo_bboxes = pseudo_bboxes[~pseudo_mask]
                    if self.use_2d_feature:
                        feature_2d = np.concatenate([feature_2d, pseudo_feature_2d], 0)
                if self.use_gss and self.use_2d_feature:
                    feature_2d = np.concatenate([feature_2d, feature_gss], 0)
            # else: # TTA
            #     instance_bboxes = instance_bboxes[~mask]
            #     if self.use_2d_feature:
            #         feature_2d = feature_2d[~mask]
            #     if self.use_pbox:
            #         pseudo_bboxes = pseudo_bboxes[~pseudo_mask]
            #         if self.use_2d_feature:
            #             feature_2d = np.concatenate([feature_2d, pseudo_feature_2d[~pseudo_mask]], 0)
        
        confident_box_num = instance_bboxes.shape[0]
        gt_box_num = instance_bboxes.shape[0]
        
        if self.use_pbox:
            instance_bboxes = np.concatenate([instance_bboxes, pseudo_bboxes], axis=0)
            confident_box_num += pseudo_bboxes.shape[0]
        if self.use_gss:
            instance_bboxes = np.concatenate([instance_bboxes, gss_bboxes], axis=0)
        if self.only_pbox:
            if self.use_gss:
                instance_bboxes = np.load(os.path.join(self.gss_box_dir, scan_name) + "_prop.npy")[:, :6]
                confident_box_num = instance_bboxes.shape[0]
                gt_box_num = instance_bboxes.shape[0]
            else:
                instance_bboxes = np.load(os.path.join(self.pseudo_box_dir, scan_name) + "_bbox.npy")[:, :7]
                confident_box_num = instance_bboxes.shape[0]
                gt_box_num = instance_bboxes.shape[0]
        
        MAX_NUM_OBJ = self.dataset_config.max_num_obj
        if self.use_2d_feature:
            if self.use_gss: assert feature_gss.shape[0] == gss_bboxes.shape[0]
            if self.use_pbox: assert pseudo_feature_2d.shape[0] == pseudo_bboxes.shape[0]
            assert feature_2d.shape[0] == instance_bboxes.shape[0], f"{feature_2d.shape}, {instance_bboxes.shape}"
            
            if instance_bboxes.shape[0] > MAX_NUM_OBJ:
                instance_bboxes = instance_bboxes[:MAX_NUM_OBJ]
                feature_2d = feature_2d[:MAX_NUM_OBJ]
                if confident_box_num > MAX_NUM_OBJ:
                    confident_box_num = MAX_NUM_OBJ
        
        if self.use_image:
            # load frames
            frame_list = list(map(lambda x: x.split(".")[0], sorted(os.listdir(SCANNET_FRAMES.format(scan_name, "color")))))
            scene_images = np.zeros((self.max_num_frame, 240, 320, 3))
            scene_poses = np.zeros((self.max_num_frame, 4, 4))
            for i, frame_id in enumerate(frame_list):
                scene_images[i] = self.img_processor.load_image(SCANNET_FRAME_PATH.format(scan_name, "color", "{}.jpg".format(frame_id)))
                scene_poses[i] = self.img_processor.load_pose(SCANNET_FRAME_PATH.format(scan_name, "pose", "{}.txt".format(frame_id)))
            scene_intrinsics = self.img_processor.load_intrinsic(SCANNET_FRAMES.format(scan_name, 'intrinsic_depth.txt'))
            scene_intrinsics[:2] /= 2

        if not self.use_color:
            point_cloud = mesh_vertices[:, 0:3]  # do not use color for now
            pcl_color = mesh_vertices[:, 3:6]
        else:
            point_cloud = mesh_vertices[:, 0:6]
            point_cloud[:, 3:] = (point_cloud[:, 3:] - MEAN_COLOR_RGB) / 256.0
            pcl_color = point_cloud[:, 3:]

        if self.use_height:
            floor_height = np.percentile(point_cloud[:, 2], 0.99)
            height = point_cloud[:, 2] - floor_height
            point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)], 1)

        # ------------------------------- LABELS ------------------------------
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6), dtype=np.float32)
        target_bboxes_mask = np.zeros((MAX_NUM_OBJ), dtype=np.float32)
        box_mask = np.zeros((MAX_NUM_OBJ))
        angle_classes = np.zeros((MAX_NUM_OBJ,), dtype=np.int64)
        angle_residuals = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)
        raw_sizes = np.zeros((MAX_NUM_OBJ, 3), dtype=np.float32)
        raw_angles = np.zeros((MAX_NUM_OBJ,), dtype=np.float32)

        if self.augment and self.use_random_cuboid:
            (
                point_cloud,
                instance_bboxes,
                per_point_labels,
                keep_boxes
            ) = self.random_cuboid_augmentor(
                point_cloud, instance_bboxes
            )
            if keep_boxes is not None:
                gt_box_num = keep_boxes[:gt_box_num].sum()
                confident_box_num = keep_boxes[:confident_box_num].sum()
                feature_2d = feature_2d[keep_boxes]
        #     semantic_labels = per_point_labels[0]

        point_cloud, choices = pc_util.random_sampling(
            point_cloud, self.num_points, return_choices=True
        )
        # semantic_labels = semantic_labels[choices]

        # sem_seg_labels = np.ones_like(semantic_labels) * IGNORE_LABEL

        # for _c in self.dataset_config.nyu40ids_semseg:
        #     sem_seg_labels[
        #         semantic_labels == _c
        #     ] = self.dataset_config.nyu40id2class_semseg[_c]

        pcl_color = pcl_color[choices]

        box_mask[0 : instance_bboxes.shape[0]] = 1
        target_bboxes_mask[0 : confident_box_num] = 1
        target_bboxes[0 : instance_bboxes.shape[0], :] = instance_bboxes[:, 0:6]

        # ------------------------------- DATA AUGMENTATION ------------------------------
        if self.augment:

            if np.random.random() > 0.5:
                # Flipping along the YZ plane
                point_cloud[:, 0] = -1 * point_cloud[:, 0]
                target_bboxes[:, 0] = -1 * target_bboxes[:, 0]

            if np.random.random() > 0.5:
                # Flipping along the XZ plane
                point_cloud[:, 1] = -1 * point_cloud[:, 1]
                target_bboxes[:, 1] = -1 * target_bboxes[:, 1]

            # Rotation along up-axis/Z-axis
            rot_angle = (np.random.random() * np.pi / 18) - np.pi / 36  # -5 ~ +5 degree
            rot_mat = pc_util.rotz(rot_angle)
            point_cloud[:, 0:3] = np.dot(point_cloud[:, 0:3], np.transpose(rot_mat))
            target_bboxes = self.dataset_config.rotate_aligned_boxes(
                target_bboxes, rot_mat
            )

        raw_sizes = target_bboxes[:, 3:6]
        point_cloud_dims_min = point_cloud.min(axis=0)[:3]
        point_cloud_dims_max = point_cloud.max(axis=0)[:3]

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
        box_centers_normalized = box_centers_normalized * target_bboxes_mask[..., None]
        mult_factor = point_cloud_dims_max - point_cloud_dims_min
        box_sizes_normalized = scale_points(
            raw_sizes.astype(np.float32)[None, ...],
            mult_factor=1.0 / mult_factor[None, ...],
        )
        box_sizes_normalized = box_sizes_normalized.squeeze(0)

        box_corners = self.dataset_config.box_parametrization_to_corners_np(
            box_centers[None, ...],
            raw_sizes.astype(np.float32)[None, ...],
            raw_angles.astype(np.float32)[None, ...],
        )
        box_corners = box_corners.squeeze(0)

        ret_dict = {}
        ret_dict["point_clouds"] = point_cloud.astype(np.float32)
        if self.use_image:
            ret_dict["images"] = scene_images.astype(np.float32)
            ret_dict["poses"] = scene_poses.astype(np.float32)
            ret_dict["intrinsics"] = scene_intrinsics.astype(np.float32)
            ret_dict["frame_length"] = np.array(len(frame_list)).astype(np.int64)
        ret_dict["gt_box_corners"] = box_corners.astype(np.float32)
        ret_dict["gt_box_centers"] = box_centers.astype(np.float32)
        ret_dict["gt_box_centers_normalized"] = box_centers_normalized.astype(
            np.float32
        )
        ret_dict["gt_angle_class_label"] = angle_classes.astype(np.int64)
        ret_dict["gt_angle_residual_label"] = angle_residuals.astype(np.float32)
        if not self.only_pbox:
            target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
            target_bboxes_semcls[0 : gt_box_num] = instance_bboxes[:gt_box_num, -1]
            ret_dict["gt_box_sem_cls_label"] = target_bboxes_semcls.astype(np.int64)
        
        ret_dict["gt_box_present"] = target_bboxes_mask.astype(np.float32)
        ret_dict["gt_box_all"] = box_mask.astype(np.float32)
        ret_dict["scan_idx"] = np.array(idx).astype(np.int64)
        ret_dict["pcl_color"] = pcl_color
        if self.use_2d_feature:
            clip_features = np.zeros((MAX_NUM_OBJ, 640))
            clip_features[:feature_2d.shape[0]] = feature_2d
            ret_dict["feature_2d"] = clip_features
            if self.feature_global_dir is not None:
                ret_dict["feature_global"] = feature_global # 640
        ret_dict["gt_box_sizes"] = raw_sizes.astype(np.float32)
        ret_dict["gt_box_sizes_normalized"] = box_sizes_normalized.astype(np.float32)
        ret_dict["gt_box_angles"] = raw_angles.astype(np.float32)
        ret_dict["point_cloud_dims_min"] = point_cloud_dims_min.astype(np.float32)
        ret_dict["point_cloud_dims_max"] = point_cloud_dims_max.astype(np.float32)
        return ret_dict
