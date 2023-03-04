#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.

CUDA_VISIBLE_DEVICES=6 python main.py \
--dataset_name sunrgbd \
--max_epoch 90 \
--nqueries 128 \
--base_lr 7e-4 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--loss_sem_cls_weight 0 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir outputs/sunrgbd_quick \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction
