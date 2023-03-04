import torch
from torch import nn
import numpy as np
from detectron2.modeling.meta_arch import CLIPFastRCNN
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.structures import Boxes, Instances, pairwise_iou, ImageList

def region_clip_config_setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.region_clip_config_file)
    cfg.merge_from_list(args.opts)
    
    # offline_cfg = get_cfg()
    # offline_cfg.merge_from_file(cfg.MODEL.CLIP.OFFLINE_RPN_CONFIG)
    # if cfg.MODEL.CLIP.OFFLINE_RPN_LSJ_PRETRAINED: # large-scale jittering (LSJ) pretrained RPN
    #     offline_cfg.MODEL.BACKBONE.FREEZE_AT = 0 # make all fronzon layers to "SyncBN"
    #     offline_cfg.MODEL.RESNETS.NORM = "SyncBN" # 5 resnet layers
    #     offline_cfg.MODEL.FPN.NORM = "SyncBN" # fpn layers
    #     offline_cfg.MODEL.RPN.CONV_DIMS = [-1, -1] # rpn layers
    # if cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH:
    #     offline_cfg.MODEL.RPN.NMS_THRESH = cfg.MODEL.CLIP.OFFLINE_RPN_NMS_THRESH  # 0.9
    # if cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST:
    #     offline_cfg.MODEL.RPN.POST_NMS_TOPK_TEST = cfg.MODEL.CLIP.OFFLINE_RPN_POST_NMS_TOPK_TEST # 1000
        
    # cfg.freeze()
    return cfg

class RegionCLIP(nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        cfg = region_clip_config_setup(args)
    
        self.model = CLIPFastRCNN(cfg)
        DetectionCheckpointer(self.model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        
        print("\nLoading new checkpoint")
        cfg.MODEL.CLIP.CROP_REGION_TYPE = 'RPN'
        cfg.freeze()
        self.rpn_model = CLIPFastRCNN(cfg)
        DetectionCheckpointer(self.rpn_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
    
    @property
    def device(self):
        return self.model.device
        
    @torch.no_grad()
    def inference(self, *args, **kwargs):
        return self.model.inference(*args, **kwargs)

    @torch.no_grad()
    def inference_rpn(self, image, pred_boxes):
        """
        pred_boxes: (N, 4), XYXY
        
        Return: objectness, N
        """
        assert pred_boxes.size(0) > 0
        batched_inputs = [{
            'image': image.permute(2, 0, 1).contiguous(),
            'instances': Instances((image.shape[0], image.shape[1]))
        }]
        images = self.rpn_model.offline_preprocess_image(batched_inputs)
        features = self.rpn_model.offline_backbone(images.tensor)
        proposals, _ = self.rpn_model.offline_proposal_generator(images, features, None)
        rpn_proposal: Boxes = proposals[0].proposal_boxes # M, 4
        rpn_obj_logits = proposals[0].objectness_logits # M,
        
        # print([proposals[0].objectness_logits.abs().max() for x in proposals])
        # print(batched_inputs[0]["image"].max(), batched_inputs[0]["image"].min())
        # print(images[0].max(), images[0].min())
        # exit(0)
        
        assert rpn_obj_logits.shape[0] == rpn_proposal.tensor.shape[0]
        if rpn_proposal.tensor.shape[0] > 0:
            ious = pairwise_iou(Boxes(pred_boxes), rpn_proposal) # N, M
            matched_idx = ious.argmax(-1) # N, 
            objectness = rpn_obj_logits[matched_idx]
            return objectness
        else:
            return None

def build_RegionCLIP(args, dataset_config):
    
    model = RegionCLIP(args)
    model.eval()
    
    return model, None