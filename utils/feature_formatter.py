import numpy as np
import torch
import os

from tqdm import tqdm

class FeatureFormatter():
    def __init__(self, output_path, scene_list, text_embedding=None, glob=False) -> None:
        self.feats = []
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.scene_list = scene_list
        self.text_embedding = text_embedding.cuda() if text_embedding is not None else None
        
        self.correct = 0
        self.total = 0
        self.glob = glob
    
    def step(self, batch_data_label):
        """
        box_prediction = {
            "sem_cls_logits": cls_logits,
            "center_normalized": center_normalized.contiguous(),
            "center_unnormalized": center_unnormalized,
            "size_normalized": size_normalized,
            "size_unnormalized": size_unnormalized,
            "angle_logits": angle_logits,
            "angle_residual": angle_residual,
            "angle_residual_normalized": angle_residual_normalized,
            "angle_continuous": angle_continuous,
            "objectness_prob": objectness_prob,
            "sem_cls_prob": semcls_prob,
            "box_corners": box_corners,
        }
        box: center, size, label, score, objectness prob, idx
        """
        clip_feats = batch_data_label["batch_clip_logits"]
        scan_idx = batch_data_label["scan_idx"][:, None]
        self.feats.extend(list(zip(scan_idx, clip_feats)))
        
        # if not self.glob:
        #     for clip_feat, label, mask in zip(clip_feats, batch_data_label["gt_box_sem_cls_label"], batch_data_label["gt_box_present"].bool()):
        #         cls_logits = clip_feat @ self.text_embedding.transpose(0, 1)
        #         cls_prob = torch.softmax(cls_logits[:, :-1], -1)
        #         cls_pred = torch.argmax(cls_prob, -1)
        #         gt_label = label[mask]
        #         assert cls_pred.size() == gt_label.size(), f"{cls_pred.size()} {gt_label.size()}"
                
        #         self.correct += torch.sum(cls_pred == gt_label).item()
        #         self.total += gt_label.size(0)
    
    def save_one_feats(self, idx, feats):
        np.save(os.path.join(self.output_path, self.scene_list[idx]) + "features.npy", feats.cpu().numpy())
        
    def save(self):
        # if not self.glob:
        #     print("Acc %d/%d=%.4f" % (self.correct, self.total, self.correct / self.total))
        for idx, feats in tqdm(self.feats):
            self.save_one_feats(idx, feats)
    
    