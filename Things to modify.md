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

Open-set Baseline

```bash
CUDA_VISIBLE_DEVICES=7 python main.py \
--dataset_name sunrgbd \
--max_epoch 200 \
--nqueries 128 \
--base_lr 2e-5 \
--matcher_giou_cost 3 \
--matcher_cls_cost 1 \
--matcher_center_cost 5 \
--matcher_objectness_cost 5 \
--use_image \
--loss_giou_weight 0 \
--loss_no_object_weight 0.1 \
--loss_2dalignment_weight 2e-4 \
--save_separate_checkpoint_every_epoch -1 \
--checkpoint_dir exp/sunrgbd/openset_baseline \
MODEL.WEIGHTS /home/zhengyuan/packages/RegionCLIP/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH /home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_sunrgbd.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG /home/zhengyuan/packages/RegionCLIP/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction
```