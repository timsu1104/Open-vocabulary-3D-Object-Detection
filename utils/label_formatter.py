from typing import List
import torch
import numpy as np
import os

from scipy.stats import mode

import multiprocessing as mp

def box_3d_iou(box_q, box_k, typ='vv', eps=1e-5):
    """
    3d iou between axis aligned boxes
    box_q: 6, 
    box_k: B, 6
    box: xyz xyz
    
    return: iou: B, 
    """
    
    box_q = box_q[None, :]
    
    if typ == 'vv':
        x1q = box_q[:,0]
        y1q = box_q[:,1]
        z1q = box_q[:,2]
        x2q = box_q[:,3]
        y2q = box_q[:,4]
        z2q = box_q[:,5]
        x1k = box_k[:,0]
        y1k = box_k[:,1]
        z1k = box_k[:,2]
        x2k = box_k[:,3]
        y2k = box_k[:,4]
        z2k = box_k[:,5]
    elif typ == 'cs':
        x1q = box_q[:,0] - box_q[:,3] / 2
        y1q = box_q[:,1] - box_q[:,4] / 2
        z1q = box_q[:,2] - box_q[:,5] / 2
        x2q = box_q[:,0] + box_q[:,3] / 2
        y2q = box_q[:,1] + box_q[:,4] / 2
        z2q = box_q[:,2] + box_q[:,5] / 2
        x1k = box_k[:,0] - box_k[:,3] / 2
        y1k = box_k[:,1] - box_k[:,4] / 2
        z1k = box_k[:,2] - box_k[:,5] / 2
        x2k = box_k[:,0] + box_k[:,3] / 2
        y2k = box_k[:,1] + box_k[:,4] / 2
        z2k = box_k[:,2] + box_k[:,5] / 2
        

    box_q_volume = (x2q-x1q) * (y2q-y1q) * (z2q-z1q)
    box_k_volume = (x2k-x1k) * (y2k-y1k) * (z2k-z1k)

    xi = np.maximum(x1q, x1k)
    yi = np.maximum(y1q, y1k)
    zi = np.maximum(z1q, z1k)
    corner_xi = np.minimum(x2q, x2k)
    corner_yi = np.minimum(y2q, y2k)
    corner_zi = np.minimum(z2q, z2k)

    intersection = np.maximum(corner_xi - xi, 0) * np.maximum(corner_yi - yi, 0) * np.maximum(corner_zi - zi, 0)

    iou = intersection / (box_q_volume + box_k_volume - intersection + eps)

    return iou

class LabelFormatter():
    def __init__(self, box_path, output_path, label_path, scene_list) -> None:
        self.boxes = []
        self.pseudo_box_dir = box_path
        self.output_path = output_path
        self.scene_list = scene_list
        self.raw_label_path = os.path.join(label_path, "{}.npy")
        self.IGNORE_LABEL = -100
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }
    
    def step(self, outputs, batch_data_label):
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
        sem_cls_prob = outputs["sem_cls_prob"] # batch, nq, numcls
        obj_prob = outputs["objectness_prob"] # batch, nq, numcls
        B, Q, _ = sem_cls_prob.size()
        center = outputs["center_unnormalized"]
        size = outputs["size_unnormalized"]
        score, label = torch.max(sem_cls_prob, dim=-1)
        boxes = torch.cat([center, size, torch.stack([label, score, obj_prob, torch.repeat_interleave(batch_data_label["scan_idx"][:, None], Q, dim=1)], -1)], dim=-1).view(B * Q, 10).cpu().numpy()
        self.boxes.append(boxes)
        
        # the following is for sanity check of gt boxes
        # center = batch_data_label["gt_box_centers"]
        # size = batch_data_label["gt_box_sizes"]
        # mask = batch_data_label["gt_box_present"].bool()
        # label = batch_data_label["gt_box_sem_cls_label"]
        # score = torch.ones_like(label)
        # boxes = torch.cat([center, size, torch.stack([label, score, torch.repeat_interleave(batch_data_label["scan_idx"][:, None], 64, dim=1)], -1)], dim=-1)[mask].cpu().numpy()
        # self.boxes.append(boxes)
    
    def compute(self, k, th_s, th_o):
        """
        compute top-k proposals for each class
        """
        self.boxes = np.concatenate(self.boxes, 0)
        # assert np.isin(self.boxes[:, 6], list(range(18))).all(), np.unique(self.boxes[:, 6])
        self.pseudo_boxes = []
        for label in range(18):
            boxes = self.boxes[self.boxes[:, 6] == label]
            scores = boxes[:, 7]
            obj_prob = boxes[:, 8]
            # topk_ind = np.argpartition(scores, -k)[-k:]
            # boxes = boxes[topk_ind]
            # scores = scores[topk_ind]
            self.pseudo_boxes.append(boxes[np.logical_and(scores >= th_s, obj_prob >= th_o)])
        self.pseudo_boxes = np.concatenate(self.pseudo_boxes, 0)
    
    def gen_pseudo(self, idx):
        scan_name = self.scene_list[idx]
        raw_pc_data = np.load(self.raw_label_path.format(scan_name))
        point_clouds = raw_pc_data[:, :3]
        semantic_labels = raw_pc_data[:, 3]
        sem_seg_labels = self.project_label(semantic_labels, True)
        # instance_bboxes_orig = np.load(os.path.join(self.pseudo_box_dir, scan_name) + "_bbox.npy")
        # instance_bboxes = instance_bboxes_orig.copy() 
        instance_bboxes = np.zeros((0, 7))
        mask = self.pseudo_boxes[:, -1] == idx
        numBox = 0
        numBox = mask.sum()
        if numBox > 0:
            boxes = self.pseudo_boxes[mask]
            
            # filter out duplicate boxes
            filtered_box = []
            for box in boxes :
                # if instance_bboxes_orig.shape[0] == 0 or \
                #     box_3d_iou(box, instance_bboxes_orig, typ='cs').max() < 0.1:
                    assert box[6] >= 0
                    mask = self.crop_pc(point_clouds, box) * (sem_seg_labels != self.IGNORE_LABEL)
                    if mask.sum() > 0:
                        stats = mode(sem_seg_labels[mask], keepdims=False)
                        if stats.mode == box[6]: 
                            filtered_box.append(box)
            
            if len(filtered_box) > 0:
                filtered_box = np.stack(filtered_box, 0)
                instance_bboxes = np.concatenate([instance_bboxes[:, :7], filtered_box[:, :7]], 0)
            numBox = len(filtered_box)
            
        np.save(os.path.join(self.output_path, scan_name) + "_bbox.npy", instance_bboxes)
        return numBox
        
    def save(self):
        p = mp.Pool(processes=mp.cpu_count())
        l = p.map(self.gen_pseudo, range(len(self.scene_list)))
        p.close()
        p.join()
        return sum(l)
    
    def process(self, k, th_s, th_o):
        self.compute(k, th_s, th_o)
        l = self.save()
        print("Done! Acquired {} boxes.".format(l))
    
    ### Utilities ###
    
    def crop_pc(self, pc, box):
        mask1 = np.prod(pc >= box[0:3] - box[3:6] / 2, axis=-1, keepdims=False)
        mask2 = np.prod(pc <= box[0:3] + box[3:6] / 2, axis=-1, keepdims=False)
        mask: np.ndarray = (mask1 * mask2).astype('bool')
        assert mask.ndim == 1
        return mask
    
    def project_label(self, semantic_labels, PSEUDO_FLAG):
        """
        Input: nyu40 label
        Output: 0-17, -100 label
        """
        if not PSEUDO_FLAG:
            sem_seg_labels = np.ones_like(semantic_labels) * self.IGNORE_LABEL

            for _c in self.nyu40ids:
                sem_seg_labels[
                    semantic_labels == _c
                ] = self.nyu40id2class[_c]
        else:
            sem_seg_labels = semantic_labels
            sem_seg_labels[semantic_labels >= 18] = self.IGNORE_LABEL
        
        return sem_seg_labels