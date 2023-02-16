# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr
from .model_regionclip import build_RegionCLIP

MODEL_FUNCS = {
    "3detr": build_3detr,
    "regionclip": build_RegionCLIP
}

def build_model(args, dataset_config, model_name=None):
    if model_name is None:
        model_name = args.model_name
    model, processor = MODEL_FUNCS[model_name](args, dataset_config)
    return model, processor
