# Open-set 3D Detection

This is the code repository of a project, which has been stuck for a long time and is not expected to be updated. 

### Preparation

Follow the installation guide of [3DETR](https://github.com/facebookresearch/3detr), then install [RegionCLIP](https://github.com/microsoft/RegionCLIP). 

### Training

Launch the baseline model by

```[language=bash]
python main.py \
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
MODEL.WEIGHTS $RegionCLIP_PATH/pretrained_ckpt/regionclip/regionclip_pretrained-cc_rn50x4.pth \
MODEL.CLIP.TEXT_EMB_PATH $RegionCLIP_PATH/datasets/custom_concepts/concepts_sunrgbd.pth \
MODEL.CLIP.OFFLINE_RPN_CONFIG $RegionCLIP_PATH/configs/LVISv1-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml \
MODEL.CLIP.TEXT_EMB_DIM 640 \
MODEL.RESNETS.DEPTH 200 \
MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION 18 \
MODEL.CLIP.CROP_REGION_TYPE GT \
MODEL.ROI_HEADS.NAME CLIPRes5ROIHeads_FeatureExtraction
```

### Acknowledgement

This repository is based on [3DETR](https://github.com/facebookresearch/3detr), a simple yet effective transformer Model for 3D Object Detection. 
