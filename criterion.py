# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from utils.box_util import generalized_box3d_iou
from utils.image_util import SUNRGBD_Calibration_cuda, project_box_3d_cuda
from utils.dist import all_reduce_average
from utils.misc import huber_loss
from utils.ulip_losses import CLIPLoss
from models.model_3detr import load_text_embed
from scipy.optimize import linear_sum_assignment

from time import time

from detectron2.structures import Boxes, Instances

class Matcher(nn.Module):
    def __init__(self, cost_class, cost_objectness, cost_giou, cost_center):
        """
        Parameters:
            cost_class:
        Returns:

        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_objectness = cost_objectness
        self.cost_giou = cost_giou
        self.cost_center = cost_center

    @torch.no_grad()
    def forward(self, outputs, targets):

        batchsize = outputs["sem_cls_prob"].shape[0]
        nqueries = outputs["sem_cls_prob"].shape[1]
        # ngt = targets["gt_box_sem_cls_label"].shape[1]
        nactual_gt = targets["nactual_gt"]

        # classification cost: batch x nqueries x ngt matrix
        # pred_cls_prob = outputs["sem_cls_prob"]
        # gt_box_sem_cls_labels = (
        #     targets["gt_box_sem_cls_label"]
        #     .unsqueeze(1)
        #     .expand(batchsize, nqueries, ngt)
        # )
        # class_mat = -torch.gather(pred_cls_prob, 2, gt_box_sem_cls_labels)
        class_mat = 0

        # objectness cost: batch x nqueries x 1
        objectness_mat = -outputs["objectness_prob"].unsqueeze(-1)

        # center cost: batch x nqueries x ngt
        center_mat = outputs["center_dist"].detach()

        # giou cost: batch x nqueries x ngt
        giou_mat = -outputs["gious"].detach()

        final_cost = (
            self.cost_class * class_mat
            + self.cost_objectness * objectness_mat
            + self.cost_center * center_mat
            + self.cost_giou * giou_mat
        )

        final_cost = final_cost.detach().cpu().numpy()
        assignments = []

        # auxiliary variables useful for batched loss computation
        batch_size, nprop = final_cost.shape[0], final_cost.shape[1]
        per_prop_gt_inds = torch.zeros(
            [batch_size, nprop], dtype=torch.int64, device=objectness_mat.device
        )
        proposal_matched_mask = torch.zeros(
            [batch_size, nprop], dtype=torch.float32, device=objectness_mat.device
        )
        for b in range(batchsize):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=objectness_mat.device)
                    for x in assign
                ]
                per_prop_gt_inds[b, assign[0]] = assign[1]
                proposal_matched_mask[b, assign[0]] = 1
            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_prop_gt_inds": per_prop_gt_inds,
            "proposal_matched_mask": proposal_matched_mask,
        }


class SetCriterion(nn.Module):
    def __init__(self, matcher, dataset_config, loss_weight_dict, text_embed):
        super().__init__()
        self.dataset_config = dataset_config
        self.matcher = matcher
        self.loss_weight_dict = loss_weight_dict

        semcls_percls_weights = torch.ones(dataset_config.num_semcls)
        # semcls_percls_weights = torch.ones(dataset_config.num_semcls + 1)
        # semcls_percls_weights[-1] = loss_weight_dict["loss_no_object_weight"]
        del loss_weight_dict["loss_no_object_weight"]
        self.register_buffer("semcls_percls_weights", semcls_percls_weights)
        
        self.clip_loss = CLIPLoss(text_embed)

        self.loss_functions = {
            "loss_sem_cls": self.loss_sem_cls,
            "loss_obj": self.loss_obj,
            "loss_angle": self.loss_angle,
            "loss_center": self.loss_center,
            "loss_size": self.loss_size,
            "loss_giou": self.loss_giou,
            "loss_2dalignment": self.loss_2dalignment,
            # this isn't used during training and is logged for debugging.
            # thus, this loss does not have a loss_weight associated with it.
            "loss_cardinality": self.loss_cardinality,
        }

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, assignments, assignments_all, assignments_sem):
        # Count the number of predictions that are objects
        # Cardinality is the error between predicted #objects and ground truth objects

        pred_logits = outputs["objectness_logits"]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        pred_objects = (pred_logits.argmax(-1) != 1).sum(1)
        card_err = F.l1_loss(pred_objects.float(), targets["nactual_gt"])
        return {"loss_cardinality": card_err}
    
    def loss_2dalignment(self, outputs, targets, assignments, assignments_all, assignments_sem):
        visual_embeds = outputs['visual_embeds']
        # targets["num_boxes_replica"] == 0 or 
        if 'feature_2d' not in targets.keys():
            return {"loss_2dalignment": visual_embeds.new_zeros(1, requires_grad=True).squeeze()}
        
        # clip_logits = outputs["batch_clip_logits"] # B*Q, C
        clip_logits = torch.gather(
            targets["feature_2d"], 1, torch.broadcast_to(assignments_all["per_prop_gt_inds"].unsqueeze(-1), visual_embeds.size())
        ) # B, Q, C
        mask = (assignments_all["proposal_matched_mask"].int() != 0) * (~(torch.prod(clip_logits == 0, -1).bool()))
        
        assert visual_embeds.size(-1) == clip_logits.size(-1)
        # print("M", mask.sum(), (assignments_all["proposal_matched_mask"].int() != 0).sum(), (~(torch.prod(clip_logits == 0, -1).bool())).sum())
        distill_loss = (1 - F.cosine_similarity(visual_embeds[mask], clip_logits[mask], dim=-1)).mean()
        
        return {"loss_2dalignment": distill_loss}

    def loss_global_alignment(self, global_feats, targets, assignments, assignments_all, assignments_sem):
        if global_feats is None or 'feature_2d' not in targets.keys():
            return {"loss_glob_alignment": torch.tensor(0., device=assignments["proposal_matched_mask"].device)}
        img_feats = targets["feature_global"]
        glob_align_loss = (1 - F.cosine_similarity(global_feats, img_feats, dim=-1)).sum()
        
        return {"loss_glob_alignment": glob_align_loss}

    def loss_sem_cls(self, outputs, targets, assignments, assignments_all, assignments_sem):

        # # Not vectorized version
        # pred_logits = outputs["sem_cls_logits"]
        # assign = assignments["assignments"]

        # sem_cls_targets = torch.ones((pred_logits.shape[0], pred_logits.shape[1]),
        #                         dtype=torch.int64, device=pred_logits.device)

        # # initialize to background/no-object class
        # sem_cls_targets *= (pred_logits.shape[-1] - 1)

        # # use assignments to compute labels for matched boxes
        # for b in range(pred_logits.shape[0]):
        #     if len(assign[b]) > 0:
        #         sem_cls_targets[b, assign[b][0]] = targets["gt_box_sem_cls_label"][b, assign[b][1]]

        # sem_cls_targets = sem_cls_targets.view(-1)
        # pred_logits = pred_logits.reshape(sem_cls_targets.shape[0], -1)
        # loss = F.cross_entropy(pred_logits, sem_cls_targets, self.semcls_percls_weights, reduction="mean")

        pred_logits = outputs["sem_cls_logits"]
        
        if targets["num_boxes_replica"] == 0:
            return {"loss_sem_cls": pred_logits.new_zeros(1).squeeze()}
        
        gt_box_label = torch.gather(
            targets["gt_box_sem_cls_label"], 1, assignments_sem["per_prop_gt_inds"]
        )
        mask = assignments_sem["proposal_matched_mask"].int() != 0
        # gt_box_label[assignments_sem["proposal_matched_mask"].int() == 0] = (
        #     pred_logits.shape[-1] - 1
        # )
        loss = F.cross_entropy(
            pred_logits[mask],
            gt_box_label[mask],
            self.semcls_percls_weights,
            reduction="mean",
        )

        return {"loss_sem_cls": loss}
    
    def loss_obj(self, outputs, targets, assignments, assignments_all, assignments_sem):
        
        objectness_logits = outputs["objectness_logits"]

        if targets["num_boxes_replica"] == 0:
            return {"loss_obj": objectness_logits.new_zeros(1).squeeze()}
        
        objectness_prob = torch.nn.functional.softmax(
            objectness_logits, 
            dim=-1)[..., 0]
        
        loss = F.binary_cross_entropy(
            objectness_prob, 
            assignments["proposal_matched_mask"].float(),
            reduction="mean",
        )

        return {"loss_obj": loss}

    def loss_angle(self, outputs, targets, assignments, assignments_all, assignments_sem):
        angle_logits = outputs["angle_logits"]
        angle_residual = outputs["angle_residual_normalized"]

        if targets["num_boxes_replica"] > 0:
            gt_angle_label = targets["gt_angle_class_label"]
            gt_angle_residual = targets["gt_angle_residual_label"]
            gt_angle_residual_normalized = gt_angle_residual / (
                np.pi / self.dataset_config.num_angle_bin
            )

            # # Non vectorized version
            # assignments = assignments["assignments"]
            # p_angle_logits = []
            # p_angle_resid = []
            # t_angle_labels = []
            # t_angle_resid = []

            # for b in range(angle_logits.shape[0]):
            #     if len(assignments[b]) > 0:
            #         p_angle_logits.append(angle_logits[b, assignments[b][0]])
            #         p_angle_resid.append(angle_residual[b, assignments[b][0], gt_angle_label[b][assignments[b][1]]])
            #         t_angle_labels.append(gt_angle_label[b, assignments[b][1]])
            #         t_angle_resid.append(gt_angle_residual_normalized[b, assignments[b][1]])

            # p_angle_logits = torch.cat(p_angle_logits)
            # p_angle_resid = torch.cat(p_angle_resid)
            # t_angle_labels = torch.cat(t_angle_labels)
            # t_angle_resid = torch.cat(t_angle_resid)

            # angle_cls_loss = F.cross_entropy(p_angle_logits, t_angle_labels, reduction="sum")
            # angle_reg_loss = huber_loss(p_angle_resid.flatten() - t_angle_resid.flatten()).sum()

            gt_angle_label = torch.gather(
                gt_angle_label, 1, assignments["per_prop_gt_inds"]
            )
            angle_cls_loss = F.cross_entropy(
                angle_logits.transpose(2, 1), gt_angle_label, reduction="none"
            )
            angle_cls_loss = (
                angle_cls_loss * assignments["proposal_matched_mask"]
            ).sum()

            gt_angle_residual_normalized = torch.gather(
                gt_angle_residual_normalized, 1, assignments["per_prop_gt_inds"]
            )
            gt_angle_label_one_hot = torch.zeros_like(
                angle_residual, dtype=torch.float32
            )
            gt_angle_label_one_hot.scatter_(2, gt_angle_label.unsqueeze(-1), 1)

            angle_residual_for_gt_class = torch.sum(
                angle_residual * gt_angle_label_one_hot, -1
            )
            angle_reg_loss = huber_loss(
                angle_residual_for_gt_class - gt_angle_residual_normalized, delta=1.0
            )
            angle_reg_loss = (
                angle_reg_loss * assignments["proposal_matched_mask"]
            ).sum()

            angle_cls_loss /= targets["num_boxes"]
            angle_reg_loss /= targets["num_boxes"]
        else:
            angle_cls_loss = torch.zeros(1, device=angle_logits.device).squeeze()
            angle_reg_loss = torch.zeros(1, device=angle_logits.device).squeeze()
        return {"loss_angle_cls": angle_cls_loss, "loss_angle_reg": angle_reg_loss}

    def loss_center(self, outputs, targets, assignments, assignments_all, assignments_sem):
        center_dist = outputs["center_dist"]
        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # assign = assignments["assignments"]
            # center_loss = torch.zeros(1, device=center_dist.device).squeeze()
            # for b in range(center_dist.shape[0]):
            #     if len(assign[b]) > 0:
            #         center_loss += center_dist[b, assign[b][0], assign[b][1]].sum()

            # select appropriate distances by using proposal to gt matching
            center_loss = torch.gather(
                center_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
            ).squeeze(-1)
            # zero-out non-matched proposals
            center_loss = center_loss * assignments["proposal_matched_mask"]
            center_loss = center_loss.sum()

            if targets["num_boxes"] > 0:
                center_loss /= targets["num_boxes"]
        else:
            center_loss = torch.zeros(1, device=center_dist.device).squeeze()

        return {"loss_center": center_loss}

    def loss_giou(self, outputs, targets, assignments, assignments_all, assignments_sem):
        gious_dist = 1 - outputs["gious"]

        # # Non vectorized version
        # giou_loss = torch.zeros(1, device=gious_dist.device).squeeze()
        # assign = assignments["assignments"]

        # for b in range(gious_dist.shape[0]):
        #     if len(assign[b]) > 0:
        #         giou_loss += gious_dist[b, assign[b][0], assign[b][1]].sum()

        # select appropriate gious by using proposal to gt matching
        giou_loss = torch.gather(
            gious_dist, 2, assignments["per_prop_gt_inds"].unsqueeze(-1)
        ).squeeze(-1)
        # zero-out non-matched proposals
        giou_loss = giou_loss * assignments["proposal_matched_mask"]
        giou_loss = giou_loss.sum()

        if targets["num_boxes"] > 0:
            giou_loss /= targets["num_boxes"]

        return {"loss_giou": giou_loss}

    def loss_size(self, outputs, targets, assignments, assignments_all, assignments_sem):
        gt_box_sizes = targets["gt_box_sizes_normalized"]
        pred_box_sizes = outputs["size_normalized"]

        if targets["num_boxes_replica"] > 0:

            # # Non vectorized version
            # p_sizes = []
            # t_sizes = []
            # assign = assignments["assignments"]
            # for b in range(pred_box_sizes.shape[0]):
            #     if len(assign[b]) > 0:
            #         p_sizes.append(pred_box_sizes[b, assign[b][0]])
            #         t_sizes.append(gt_box_sizes[b, assign[b][1]])
            # p_sizes = torch.cat(p_sizes)
            # t_sizes = torch.cat(t_sizes)
            # size_loss = F.l1_loss(p_sizes, t_sizes, reduction="sum")

            # construct gt_box_sizes as [batch x nprop x 3] matrix by using proposal to gt matching
            gt_box_sizes = torch.stack(
                [
                    torch.gather(
                        gt_box_sizes[:, :, x], 1, assignments["per_prop_gt_inds"]
                    )
                    for x in range(gt_box_sizes.shape[-1])
                ],
                dim=-1,
            )
            size_loss = F.l1_loss(pred_box_sizes, gt_box_sizes, reduction="none").sum(
                dim=-1
            )

            # zero-out non-matched proposals
            size_loss *= assignments["proposal_matched_mask"]
            size_loss = size_loss.sum()

            size_loss /= targets["num_boxes"]
        else:
            size_loss = torch.zeros(1, device=pred_box_sizes.device).squeeze()
        return {"loss_size": size_loss}
    
    def crop_pc(self, pc, center, size):
        mask1 = np.prod(pc >= center - size / 2, -1)
        mask2 = np.prod(pc <= center + size / 2, -1)
        mask = (mask1 * mask2).astype('bool')
        assert mask.sum() > 0
        return pc[mask]

    def single_output_forward(self, outputs, targets, clip, glob=None):
        # start = time()
        # torch.cuda.empty_cache()
        # print("empty time", time() - start)
        start = time()
        
        center_dist = torch.cdist(
            outputs["center_normalized"], targets["gt_box_centers_normalized"], p=1
        )
        outputs["center_dist"] = center_dist
        
        # semantics
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt_sem"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )

        outputs["gious"] = gious
        
        nactual_gt = targets["nactual_gt"].clone()
        targets["nactual_gt"] = targets["nactual_gt_sem"]
        assignments_sem = self.matcher(outputs, targets)
        targets["nactual_gt"] = nactual_gt
        
        if 'image_height' not in targets.keys():
            
            gious = generalized_box3d_iou(
                outputs["box_corners"],
                targets["gt_box_corners"],
                targets["nactual_gt_all"],
                rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
                needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
            )

            outputs["gious"] = gious
            
            nactual_gt = targets["nactual_gt"].clone()
            targets["nactual_gt"] = targets["nactual_gt_all"]
            assignments_all = self.matcher(outputs, targets)
            targets["nactual_gt"] = nactual_gt
        
        elif "feature_2d" in targets.keys():
            from utils.image_util import box_2d_iou_tensor
            batch_centers: torch.Tensor = outputs["center_unnormalized"]
            batch_sizes: torch.Tensor = outputs["size_unnormalized"]
            batch_angles: torch.Tensor = outputs["angle_continuous"]
            # feature_2d: torch.Tensor = targets["feature_2d"]
            # boxes_2d = feature_2d[:, -5:-1]
            # box_masks = feature_2d[:, -1]
            # targets["feature_2d"] = feature_2d[:, :-5]
            length = targets["gt_box_all"].sum(1)
            
            hs, ws = targets["image_height"], targets["image_width"]
            
            masks = targets["gt_box_all"]
            batch_gt_centers: torch.Tensor = targets["gt_box_centers"]
            batch_gt_sizes: torch.Tensor = targets["gt_box_sizes"]
            batch_gt_angles: torch.Tensor = targets["gt_box_angles"]
            
            calib_Rtilts: torch.Tensor = targets["calib_Rtilt"]
            calib_Ks: torch.Tensor = targets["calib_K"]
            with torch.no_grad():
                per_prop_gt_inds = []
                proposal_matched_mask = []
                for idx, (h, w, calib_Rtilt, calib_K, centers, sizes, angles) in \
                    enumerate(zip(hs, ws, calib_Rtilts, calib_Ks, batch_centers, batch_sizes, batch_angles)):
                    if masks[idx].sum() == 0:
                        per_prop_gt_inds.append(h.new_zeros(centers.size(0)))
                        proposal_matched_mask.append(h.new_zeros(centers.size(0)))
                        continue
                        
                    gt_centers = batch_gt_centers[idx][masks[idx].bool()]
                    gt_sizes = batch_gt_sizes[idx][masks[idx].bool()]
                    gt_angles = batch_gt_angles[idx][masks[idx].bool()]
                
                    calib = SUNRGBD_Calibration_cuda(calib_Rtilt, calib_K)
                    boxes = project_box_3d_cuda(calib, centers, sizes, angles)
                    box_2d = project_box_3d_cuda(calib, gt_centers, gt_sizes, gt_angles)
                    
                    # clip to image region
                    max_coords = torch.broadcast_to(torch.stack([w, h, w, h]).unsqueeze(0), boxes.size())
                    boxes = torch.clamp_min(boxes, 0)
                    boxes = torch.minimum(boxes, max_coords)
                    areas = torch.prod(boxes[..., 2:] - boxes[..., :2], -1)
                    valid_mask = areas > 100 # Q
                    
                    # box_mask = box_masks[idx].bool()
                    # box_2d = boxes_2d[idx] # boxes_2d[idx, box_mask]
                    iou, iou_max, nmax = box_2d_iou_tensor(boxes, box_2d)
                    
                    iou_mask = iou_max > 0.3
                    ind_mask = nmax < length[idx]
                    # print(valid_mask.sum(), iou_mask.sum(), ind_mask.sum())
                    # print("S", (valid_mask * iou_mask * ind_mask).sum())
                    
                    per_prop_gt_inds.append(nmax)
                    proposal_matched_mask.append(valid_mask * iou_mask * ind_mask)
                    
                assignments_all = {
                    "per_prop_gt_inds": torch.stack(per_prop_gt_inds),
                    "proposal_matched_mask": torch.stack(proposal_matched_mask),
                }
        else:
            assignments_all = None
        
        gious = generalized_box3d_iou(
            outputs["box_corners"],
            targets["gt_box_corners"],
            targets["nactual_gt"],
            rotated_boxes=torch.any(targets["gt_box_angles"] > 0).item(),
            needs_grad=(self.loss_weight_dict["loss_giou_weight"] > 0),
        )

        outputs["gious"] = gious
        assignments = self.matcher(outputs, targets)
        
        # Crop images
        # print("before clip time", time() - start)
        # start = time()
        # assert clip is not None
        # batch_centers: torch.Tensor = outputs["center_unnormalized"]
        # batch_sizes: torch.Tensor = outputs["size_unnormalized"]
        # batch_angles: torch.Tensor = outputs["angle_continuous"]
        
        # images_1d, h, w = targets["image"], targets["image_height"], targets["image_width"]
        # images = [
        #     image_1d[:height*width*3].view(height, width, 3) 
        #         for image_1d, height, width in zip(images_1d, h, w)
        #         ]
        
        # calib_Rtilts: torch.Tensor = targets["calib_Rtilt"]
        # calib_Ks: torch.Tensor = targets["calib_K"]
        # with torch.no_grad():
        #     batch_clip_logits = []
        #     for image, calib_Rtilt, calib_K, centers, sizes, angles in \
        #         zip(images, calib_Rtilts, calib_Ks, batch_centers, batch_sizes, batch_angles):
        #         calib = SUNRGBD_Calibration_cuda(calib_Rtilt, calib_K)
        #         # boxes = torch.vstack([project_box_3d_cuda(calib, center, size, angle) for center, size, angle in zip(centers, sizes, angles)]) # N, 4
        #         boxes = project_box_3d_cuda(calib, centers, sizes, angles)
                
        #         # clip to image region
        #         h, w, _ = image.size()
        #         max_coords = torch.broadcast_to(torch.tensor([[w, h, w, h]], device=image.device), boxes.size())
        #         boxes = torch.clamp_min(boxes, 0)
        #         boxes = torch.minimum(boxes, max_coords)
                
        #         batch_clip_logits.append({
        #             'image': image.permute(2, 0, 1).contiguous(),
        #             'instances': Instances((image.shape[0], image.shape[1]), gt_boxes=Boxes(boxes))
        #         })
        #     batch_clip_logits = clip.inference(batch_clip_logits, do_postprocess=False)
        #     outputs["batch_clip_logits"] = batch_clip_logits
        # print("clip time", time() - start)
        start = time()

        losses = {}

        for k in self.loss_functions:
            loss_wt_key = k + "_weight"
            if (
                loss_wt_key in self.loss_weight_dict
                and self.loss_weight_dict[loss_wt_key] > 0
            ) or loss_wt_key not in self.loss_weight_dict:
                # only compute losses with loss_wt > 0
                # certain losses like cardinality are only logged and have no loss weight
                curr_loss = self.loss_functions[k](outputs, targets, assignments, assignments_all, assignments_sem)
                losses.update(curr_loss)
        
        if self.loss_weight_dict["loss_glob_alignment_weight"] > 0:
            losses.update(self.loss_global_alignment(glob, targets, assignments, assignments_all, assignments_sem))

        final_loss = 0
        for k in self.loss_weight_dict:
            if self.loss_weight_dict[k] > 0:
                losses[k.replace("_weight", "")] *= self.loss_weight_dict[k]
                final_loss += losses[k.replace("_weight", "")]
        # print("after clip time", time() - start)
        return final_loss, losses

    def forward(self, outputs, targets, clip=None):
        nactual_gt = targets["gt_box_present"].sum(axis=1).long()
        num_boxes = torch.clamp(all_reduce_average(nactual_gt.sum()), min=1).item()
        targets["nactual_gt"] = nactual_gt
        targets["num_boxes"] = num_boxes
        targets[
            "num_boxes_replica"
        ] = nactual_gt.sum().item()  # number of boxes on this worker for dist training
        
        targets["nactual_gt_all"] = targets["gt_box_all"].sum(axis=1).long()
        targets["nactual_gt_sem"] = targets["gt_box_num"].long()
        
        loss, loss_dict = self.single_output_forward(outputs["outputs"], targets, clip, glob=outputs["global_feats"])

        if "aux_outputs" in outputs:
            for k in range(len(outputs["aux_outputs"])):
                interm_loss, interm_loss_dict = self.single_output_forward(
                    outputs["aux_outputs"][k], targets, clip
                )

                loss += interm_loss
                for interm_key in interm_loss_dict:
                    loss_dict[f"{interm_key}_{k}"] = interm_loss_dict[interm_key]
        return loss, loss_dict


def build_criterion(args, dataset_config):
    matcher = Matcher(
        cost_class=args.matcher_cls_cost,
        cost_giou=args.matcher_giou_cost,
        cost_center=args.matcher_center_cost,
        cost_objectness=args.matcher_objectness_cost,
    )

    loss_weight_dict = {
        "loss_giou_weight": args.loss_giou_weight,
        "loss_sem_cls_weight": args.loss_sem_cls_weight,
        "loss_obj_weight": args.loss_obj_weight,
        "loss_no_object_weight": args.loss_no_object_weight,
        "loss_angle_cls_weight": args.loss_angle_cls_weight,
        "loss_angle_reg_weight": args.loss_angle_reg_weight,
        "loss_center_weight": args.loss_center_weight,
        "loss_size_weight": args.loss_size_weight,
        "loss_2dalignment_weight": args.loss_2dalignment_weight,
        "loss_glob_alignment_weight": args.loss_glob_alignment_weight,
    }
    text_embed = load_text_embed(args)
    criterion = SetCriterion(matcher, dataset_config, loss_weight_dict, text_embed)
    return criterion
