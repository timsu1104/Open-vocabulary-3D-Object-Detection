# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils.SLIP_utils as utils

class CLIPLoss(nn.Module):
    def __init__(self, text_embedding):
        super().__init__()
        self.labels = None
        self.last_local_batch_size = None
        
        self.text_embed = text_embedding
        self.logit_scale = 1 / 0.07

    def forward(self, pc_embed, labels):
        """
        pc_embed: torch.tensor, (B, C=640)
        labels: torch.tensor, (B, )
        """
        text_embed = torch.index_select(self.text_embed, dim=0, index=labels)
        logit_scale = self.logit_scale
        local_batch_size = pc_embed.size(0)

        if local_batch_size != self.last_local_batch_size:
            self.labels = local_batch_size * utils.get_rank() + torch.arange(
                local_batch_size, device=pc_embed.device
            )
            self.last_local_batch_size = local_batch_size

        # normalized features
        pc_embed = F.normalize(pc_embed, dim=-1, p=2)
        text_embed = F.normalize(text_embed, dim=-1, p=2)

        # gather features from all GPUs
        image_embed_all, text_embed_all = \
            utils.all_gather_batch([pc_embed, text_embed])

        # cosine similarity as logits
        logits_per_cloud = logit_scale * pc_embed @ text_embed_all.t()
        logits_per_text = logit_scale * text_embed @ image_embed_all.t()

        loss = (F.cross_entropy(logits_per_cloud, self.labels) + \
            F.cross_entropy(logits_per_text, self.labels)) / 2

        return loss