### Proposed method


### Work to do

1. Baseline speed (Current: 10d), loss weighting, deadlock when using multiple GPU
2. LSeg label on SUN RGB-D

Need help: 
1. baseline speedup

### Current data available

1. Unsupervised GSS data: `/share/suzhengyuan/code/WyPR/gss/computed_proposal_sunrgbd/SZ+V-V+F`
2. RegionCLIP 2D boxes: `/share/suzhengyuan/data/RegionCLIP_boxes_sunrgbd/2D`
3. LSeg labels:`/data1/lseg_data/data/sunrgbd/pseudo_labels`



### Experiments


rm exp/sunrgbd/openset_main/*
CUDA_VISIBLE_DEVICES=2,4,6,7 python -u main.py \
--dataset_name sunrgbd \
--max_epoch 400 \
--nqueries 128 \
--base_lr 7e-4 \
--ngpus 4 \
--eval_every_epoch 20 \
--batchsize_per_gpu 24 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--use_image \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--loss_2dalignment_weight 1e-2 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir exp/sunrgbd/openset_main \
--use_pbox \
--pseudo_label_dir /share/suzhengyuan/data/RegionCLIP_boxes_sunrgbd/3D_maskclip_nyu38 \
--use_2d_feature \
--feature_2d_dir /share/suzhengyuan/data/RegionCLIP_features/gt_sunrgbd_fixed \
--pseudo_feats_dir /share/suzhengyuan/data/RegionCLIP_features/maskclip_sunrgbd_fixed \
--gss_feats_dir /share/suzhengyuan/data/RegionCLIP_features/gss_sunrgbd_fixed \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_sunrgbd.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > log/sunrgbd/openset_main.log

CUDA_VISIBLE_DEVICES=2 python -u main.py \
--dataset_name sunrgbd \
--max_epoch 400 \
--nqueries 128 \
--base_lr 7e-4 \
--eval_every_epoch 20 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--use_image \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--loss_2dalignment_weight 1e-2 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir exp/sunrgbd/test \
--use_pbox \
--pseudo_label_dir /share/suzhengyuan/data/RegionCLIP_boxes_sunrgbd/3D_maskclip_nyu38 \
--use_2d_feature \
--feature_2d_dir /share/suzhengyuan/data/RegionCLIP_features/gt_sunrgbd_fixed \
--pseudo_feats_dir /share/suzhengyuan/data/RegionCLIP_features/maskclip_sunrgbd_fixed \
--gss_feats_dir /share/suzhengyuan/data/RegionCLIP_features/gss_sunrgbd_fixed \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_sunrgbd.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction



--use_gss \
--gss_box_dir /share/suzhengyuan/code/WyPR/gss/computed_proposal_sunrgbd/SZ+V-obb-V+F-obb \











CUDA_VISIBLE_DEVICES=4,5,6,7 python -u main.py \
--dataset_name scannet \
--max_epoch 400 \
--nqueries 256 \
--ngpus 4 \
--batchsize_per_gpu 12 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--loss_2dalignment_weight 0 \
--loss_glob_alignment_weight 0 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir exp/scannet/openset_baseline_3detr_orig \
--clip_embed_path /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_3detr.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction

rm exp/scannet/openset_alignonly/*
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u main.py \
--dataset_name scannet \
--max_epoch 400 \
--nqueries 256 \
--ngpus 4 \
--batchsize_per_gpu 12 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--loss_2dalignment_weight 1e-2 \
--loss_sem_cls_weight 0 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir exp/scannet/openset_alignonly \
--use_2d_feature \
--feature_2d_dir /share/suzhengyuan/data/RegionCLIP_features/scannet_gt \
--feature_global_dir /share/suzhengyuan/data/RegionCLIP_features/scannet_2dglobal \
--clip_embed_path /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > log/scannet/openset_alignonly.log


screen
conda activate scanqa
cd ~/code/OVDet
rm exp/scannet/openset_main/*
CUDA_VISIBLE_DEVICES=4,5,6,7 python -u main.py \
--dataset_name scannet \
--max_epoch 400 \
--nqueries 256 \
--ngpus 4 \
--dist_url tcp://localhost:12343 \
--batchsize_per_gpu 12 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--loss_2dalignment_weight 1e-2 \
--loss_sem_cls_weight 0 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir exp/scannet/openset_main \
--use_pbox \
--pseudo_label_dir /share/suzhengyuan/data/RegionCLIP_boxes/3D_MaskCLIP \
--use_2d_feature \
--feature_2d_dir /share/suzhengyuan/data/RegionCLIP_features/scannet_gt \
--pseudo_feats_dir /share/suzhengyuan/data/RegionCLIP_features/maskclip_scannet \
--gss_feats_dir /share/suzhengyuan/data/RegionCLIP_features/scannet_gss \
--clip_embed_path /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > log/scannet/openset_main.log

CUDA_VISIBLE_DEVICES=0,3 python -u main.py \
--dataset_name scannet \
--max_epoch 400 \
--nqueries 256 \
--ngpus 2 \
--dist_url tcp://localhost:12323 \
--batchsize_per_gpu 12 \
--matcher_giou_cost 2 \
--matcher_cls_cost 1 \
--matcher_center_cost 0 \
--matcher_objectness_cost 0 \
--loss_giou_weight 1 \
--loss_no_object_weight 0.25 \
--loss_2dalignment_weight 1e-2 \
--loss_sem_cls_weight 0 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir exp/scannet/openset_main_keep1000 \
--use_pbox \
--pseudo_label_dir /share/suzhengyuan/data/RegionCLIP_boxes/3D_MaskCLIP \
--use_2d_feature \
--feature_2d_dir /share/suzhengyuan/data/RegionCLIP_features/scannet_gt \
--pseudo_feats_dir /share/suzhengyuan/data/RegionCLIP_features/maskclip_scannet \
--gss_feats_dir /share/suzhengyuan/data/RegionCLIP_features/scannet_gss \
--clip_embed_path /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > log/scannet/openset_main_keep1000.log




--use_gss \
--gss_box_dir /share/suzhengyuan/code/WyPR/gss/computed_proposal_scannet/SZ+V+maskclip-V+F \













### Examine the model

CUDA_VISIBLE_DEVICES=6 python main.py --dataset_name sunrgbd --nqueries 128 --test_only --test_ckpt /home/zhengyuan/code/OVDet/exp/sunrgbd/openset_baseline/checkpoint.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_sunrgbd.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > test_sunrgbd_all.log 2>&1 &


CUDA_VISIBLE_DEVICES=6 python main.py --dataset_name scannet --nqueries 256 --test_only --test_ckpt /home/zhengyuan/code/OVDet/exp/scannet/openset_traintime_maskclip_sem_new/checkpoint.pth \
--clip_embed_path /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_3detr.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > test_scannet_maskclip.log 2>&1 &

CUDA_VISIBLE_DEVICES=6 python main.py --dataset_name scannet --nqueries 256 --test_only --test_ckpt /home/zhengyuan/code/OVDet/exp/scannet/openset_baseline_new/checkpoint.pth \
--clip_embed_path /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_3detr.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > test_scannet_baseline.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name scannet --nqueries 256 --test_only --test_ckpt /home/zhengyuan/code/OVDet/exp/scannet/openset_baseline_3detr_orig/checkpoint.pth \
--clip_embed_path /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_3detr.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > test_scannet_baseline_orig.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python main.py --dataset_name scannet --nqueries 128 --test_only --test_ckpt /home/zhengyuan/code/OVDet/exp/scannet/openset_baseline_new_2d5e-3/checkpoint.pth \
--clip_embed_path /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_3detr.pth \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_scannet.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction > test_scannet_baseline_5e-3.log 2>&1 &