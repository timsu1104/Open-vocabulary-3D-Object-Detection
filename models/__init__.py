# Copyright (c) Facebook, Inc. and its affiliates.
from .model_3detr import build_3detr, load_text_embed
from .pointMLP_ULIP.pointmlp import pointMLP_ULIP
import torch, torch.nn as nn

MODEL_FUNCS = {
    "3detr": build_3detr,
}

def build_model(args, dataset_config):
    model, processor = MODEL_FUNCS[args.model_name](args, dataset_config)
    return model, processor

def build_ULIP(args, dataset_config):
    text_embedding=load_text_embed(args)
    model = pointMLP_ULIP(num_classes=dataset_config.num_semcls+1, text_embedding=text_embedding)
    checkpoint = torch.load(args.ulip_ckpt_path)
    model = nn.DataParallel(model)
    checkpoint['net']['module.classifier.weight'] = text_embedding
    model.load_state_dict(checkpoint['net'])
    model.eval()
    return model