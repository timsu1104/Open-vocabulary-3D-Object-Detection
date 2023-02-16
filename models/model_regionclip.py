from detectron2.modeling.meta_arch import CLIPFastRCNN
from detectron2.config import get_cfg
from detectron2.checkpoint import DetectionCheckpointer

def region_clip_config_setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.region_clip_config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def build_RegionCLIP(args, dataset_config):
    cfg = region_clip_config_setup(args)
    model = CLIPFastRCNN(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False
    )
    model.eval()
    return model, None