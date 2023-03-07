# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import numpy as np
import datetime
import logging
import math
import time
import sys, os

from tqdm import tqdm

from torch.distributed.distributed_c10d import reduce
from utils.ap_calculator import APCalculator
from utils.label_formatter import LabelFormatter
from utils.feature_formatter import FeatureFormatter
from utils.misc import SmoothedValue
from utils.dist import (
    all_gather_dict,
    all_reduce_average,
    is_primary,
    reduce_dict,
    barrier,
)

from utils.image_util import SUNRGBD_Calibration_cuda, project_box_3d_cuda, project_box_3d_aabb
from utils.box_util import box_3d_iou_tensor

from detectron2.structures import Boxes, Instances

def compute_learning_rate(args, curr_epoch_normalized):
    assert curr_epoch_normalized <= 1.0 and curr_epoch_normalized >= 0.0
    if (
        curr_epoch_normalized <= (args.warm_lr_epochs / args.max_epoch)
        and args.warm_lr_epochs > 0
    ):
        # Linear Warmup
        curr_lr = args.warm_lr + curr_epoch_normalized * args.max_epoch * (
            (args.base_lr - args.warm_lr) / args.warm_lr_epochs
        )
    else:
        # Cosine Learning Rate Schedule
        curr_lr = args.final_lr + 0.5 * (args.base_lr - args.final_lr) * (
            1 + math.cos(math.pi * curr_epoch_normalized)
        )
    return curr_lr


def adjust_learning_rate(args, optimizer, curr_epoch):
    curr_lr = compute_learning_rate(args, curr_epoch)
    for param_group in optimizer.param_groups:
        param_group["lr"] = curr_lr
    return curr_lr


def train_one_epoch(
    args,
    curr_epoch,
    model,
    regionclip, 
    ema, 
    optimizer,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
):

    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=False,
    )

    curr_iter = curr_epoch * len(dataset_loader)
    max_iters = args.max_epoch * len(dataset_loader)
    net_device = next(model.parameters()).device

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)

    model.train()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        curr_lr = adjust_learning_rate(args, optimizer, curr_iter / max_iters)
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        # Forward pass
        optimizer.zero_grad()
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        s1 = time.time()
        outputs = model(inputs)
        t1 = time.time()
        # print("Model Inference time ", t1-s1)
        if args.use_pseudo_labels: 
            with ema.average_parameters():
                teacher_outputs = model(inputs)

        # Compute loss
        s1 = time.time()
        loss, loss_dict = criterion(outputs, batch_data_label, clip=regionclip)
        t1 = time.time()
        # print("Loss computation time ", t1-s1)

        loss_reduced = all_reduce_average(loss)
        loss_dict_reduced = reduce_dict(loss_dict)

        if not math.isfinite(loss_reduced.item()):
            logging.info(f"Loss in not finite. Training will be stopped.")
            sys.exit(1)

        loss.backward()
        if args.clip_gradient > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_gradient)
        optimizer.step()

        if curr_iter % args.log_metrics_every == 0:
            # This step is slow. AP is computed approximately and locally during training.
            # It will gather outputs and ground truth across all ranks.
            # It is memory intensive as point_cloud ground truth is a large tensor.
            # If GPU memory is not an issue, uncomment the following lines.
            # outputs["outputs"] = all_gather_dict(outputs["outputs"])
            # batch_data_label = all_gather_dict(batch_data_label)
            ap_calculator.step_meter(outputs, batch_data_label)

        time_delta.update(time.time() - curr_time)
        loss_avg.update(loss_reduced.item())
        
        # print("Elapsed", time_delta.avg, "{}/{}".format(batch_idx, len(dataset_loader)))

        # logging
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            eta_seconds = (max_iters - curr_iter) * time_delta.avg
            eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
            print(
                f"Epoch [{curr_epoch}/{args.max_epoch}]; Iter [{curr_iter}/{max_iters}]; Loss {loss_avg.avg:0.2f}; LR {curr_lr:0.2e}; Iter time {time_delta.avg:0.2f}; ETA {eta_str}; Mem {mem_mb:0.2f}MB"
            )
            logger.log_scalars(loss_dict_reduced, curr_iter, prefix="Train_details/")

            train_dict = {}
            train_dict["lr"] = curr_lr
            train_dict["memory"] = mem_mb
            train_dict["loss"] = loss_avg.avg
            train_dict["batch_time"] = time_delta.avg
            logger.log_scalars(train_dict, curr_iter, prefix="Train/")

        curr_iter += 1
        barrier()

    return ap_calculator


@torch.no_grad()
def evaluate(
    args,
    curr_epoch,
    model,
    clip,
    criterion,
    dataset_config,
    dataset_loader,
    logger,
    curr_train_iter,
):

    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25, 0.5],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    loss_avg = SmoothedValue(window_size=10)
    model.eval()
    barrier()
    epoch_str = f"[{curr_epoch}/{args.max_epoch}]" if curr_epoch > 0 else ""

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)

        # Compute loss
        loss_str = ""
        if criterion is not None:
            loss, loss_dict = criterion(outputs, batch_data_label, clip)

            loss_reduced = all_reduce_average(loss)
            loss_dict_reduced = reduce_dict(loss_dict)
            loss_avg.update(loss_reduced.item())
            loss_str = f"Loss {loss_avg.avg:0.2f};"

        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        ap_calculator.step_meter(outputs, batch_data_label)
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Evaluate {epoch_str}; Batch [{curr_iter}/{num_batches}]; {loss_str} Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
            if criterion is not None:
                test_dict["loss"] = loss_avg.avg
        curr_iter += 1
        barrier()
    if is_primary():
        if criterion is not None:
            logger.log_scalars(
                loss_dict_reduced, curr_train_iter, prefix="Test_details/"
            )
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return ap_calculator


@torch.no_grad()
def inference(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset, 
    dataset_loader,
    logger,
    curr_train_iter,
):
    
    # ap calculator is exact for evaluation. This is slower than the ap calculator used during training.
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )
    
    label_formatter = LabelFormatter(args.in_dir, args.out_dir, args.feature_2d_dir, dataset.scan_names)
    
    assert args.out_dir is not None, f"Please specify a path to save pseudo labels using --out_dir."
    os.makedirs(args.out_dir, exist_ok=True)

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    model.eval()
    barrier()

    for batch_idx, batch_data_label in enumerate(dataset_loader):
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)


        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        
        label_formatter.step(outputs["outputs"], batch_data_label)
        # ap_calculator.step_meter(outputs, batch_data_label)
        
        time_delta.update(time.time() - curr_time)
        if is_primary() and curr_iter % args.log_every == 0:
            mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(
                f"Infer; Batch [{curr_iter}/{num_batches}]; Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
            )

            test_dict = {}
            test_dict["memory"] = mem_mb
            test_dict["batch_time"] = time_delta.avg
        curr_iter += 1
        barrier()
    if is_primary():
        logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return label_formatter, ap_calculator


@torch.no_grad()
def inference_gt_feature(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset, 
    dataset_loader,
    logger,
    curr_train_iter,
):
    text_embedding = torch.load(args.clip_embed_path).float()
    
    label_formatter = FeatureFormatter(args.out_dir, dataset.scan_names, text_embedding)
    
    assert args.out_dir is not None, f"Please specify a path to save pseudo labels using --out_dir."
    os.makedirs(args.out_dir, exist_ok=True)

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    model.eval()
    barrier()

    for batch_idx, batch_data_label in enumerate(tqdm(dataset_loader)):
        torch.cuda.empty_cache()
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)

        masks = batch_data_label["gt_box_present"]
        batch_centers: torch.Tensor = batch_data_label["gt_box_centers"]
        batch_sizes: torch.Tensor = batch_data_label["gt_box_sizes"]
        batch_angles: torch.Tensor = batch_data_label["gt_box_angles"]
        
        if "image_height" in batch_data_label: # sunrgbd
            images_1d, h, w = batch_data_label["image"], batch_data_label["image_height"], batch_data_label["image_width"]
            images = [
                image_1d[:height*width*3].view(height, width, 3) 
                    for image_1d, height, width in zip(images_1d, h, w)
                    ] # B, (H, W, 3)
            
            calib_Rtilts: torch.Tensor = batch_data_label["calib_Rtilt"]
            calib_Ks: torch.Tensor = batch_data_label["calib_K"]

            batch_clip_logits = []
            for frame_image, calib_Rtilt, calib_K, centers, sizes, angles, mask in \
                zip(images, calib_Rtilts, calib_Ks, batch_centers, batch_sizes, batch_angles, masks.bool()):
                calib = SUNRGBD_Calibration_cuda(calib_Rtilt, calib_K)
                boxes = project_box_3d_cuda(calib, centers[mask], sizes[mask], angles[mask])
                
                # clip to image region
                h, w, _ = frame_image.size()
                max_coords = torch.broadcast_to(torch.tensor([[w, h, w, h]], device=frame_image.device), boxes.size())
                boxes = torch.clamp_min(boxes, 0)
                boxes = torch.minimum(boxes, max_coords)
                areas = torch.prod(boxes[..., 2:] - boxes[..., :2], -1)
                box_mask = areas > 1000 # N
                
                torch.cuda.empty_cache()
                # print(boxes.shape)
                # print(frame_image.shape)
                pl_holder = boxes.new_zeros(mask.sum(), 640)
                if box_mask.sum() > 0:
                    pl_holder[box_mask] = model.inference([{
                        'image': frame_image.permute(2, 0, 1).contiguous(),
                        'instances': Instances((frame_image.shape[0], frame_image.shape[1]), gt_boxes=Boxes(boxes[box_mask]))
                    }], do_postprocess=False)
                batch_clip_logits.append(torch.cat([pl_holder, boxes, box_mask.unsqueeze(-1)], -1))
            batch_data_label["batch_clip_logits"] = batch_clip_logits
        else:
            images = batch_data_label["images"] # B, F, (H, W, 3)
            poses = batch_data_label["poses"] # B, F, 4, 4
            frame_length = batch_data_label["frame_length"] # B, F, 4, 4
            intrinsics = batch_data_label["intrinsics"] # B, 4, 4
            axis_aligned_matrix = batch_data_label["axis_align_mat"]

            batch_clip_logits = []
            for frame_image, pose, intrinsic, axis_aligned_mat, centers, sizes, len_frame, mask in \
                zip(images, poses, intrinsics, axis_aligned_matrix, batch_centers, batch_sizes, frame_length, masks.bool()):
                
                if mask.sum() == 0:
                    batch_clip_logits.append(mask.new_zeros((0, 640)))
                    continue
                
                projection = [project_box_3d_aabb(centers[mask], sizes[mask], camera2world, intrinsic, axis_aligned_mat) for camera2world in pose[:len_frame]]
                boxes = torch.stack([x[0] for x in projection]) # F, N, 4
                depth_mask = torch.stack([x[1].min(-1)[0] > 0 for x in projection]) # F, N
                # orig_areas = torch.prod(boxes[..., 2:] - boxes[..., :2], -1)
                
                # clip to image region
                _, h, w, _ = frame_image.size()
                max_coords = torch.broadcast_to(torch.tensor([[w, h, w, h]], device=frame_image.device), boxes.size())
                boxes = torch.clamp_min(boxes, 0)
                boxes = torch.minimum(boxes, max_coords)
                areas = torch.prod(boxes[..., 2:] - boxes[..., :2], -1)
                # print(areas)
                # box_mask = areas/orig_areas > 0.1 # F, N
                box_mask = (areas > 1000) * depth_mask # F, N
                
                results = []
                for image, box, bmask in zip(frame_image[:len_frame], boxes, box_mask):
                    raw_res: torch.Tensor = model.inference([{
                        'image': image.permute(2, 0, 1).contiguous(),
                        'instances': Instances((image.shape[0], image.shape[1]), gt_boxes=Boxes(box[bmask]))
                    }], do_postprocess=False)
                    pl_holder = raw_res.new_zeros(mask.sum(), 640)
                    if bmask.sum() > 0:
                        pl_holder[bmask] = raw_res
                    results.append(pl_holder)
                    
                results = torch.stack(results, 1).contiguous() # N, F, C
                valid_mask = box_mask.transpose(0, 1)
                clip_logits = []
                for res, mk in zip(results, valid_mask):
                    if mk.sum() > 0:
                        clip_logits.append(res[mk].mean(0))
                    else:
                        clip_logits.append(results.new_zeros(640))
                clip_logits = torch.stack(clip_logits) # N, C
                assert clip_logits.size(0) == mask.sum(), (clip_logits.size(0), mask.sum())
                batch_clip_logits.append(clip_logits)
                
            batch_data_label["batch_clip_logits"] = batch_clip_logits # B, (N, C)
        
        label_formatter.step(batch_data_label)
        
        time_delta.update(time.time() - curr_time)
        # if is_primary() and curr_iter % args.log_every == 0:
        #     mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        #     print(
        #         f"Infer; Batch [{curr_iter}/{num_batches}]; Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
        #     )

        #     test_dict = {}
        #     test_dict["memory"] = mem_mb
        #     test_dict["batch_time"] = time_delta.avg
        # curr_iter += 1
        barrier()
    # if is_primary():
    #     logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return label_formatter

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[..., [0, 1, 2]] = pc2[..., [0, 2, 1]]  # depth X,Y,Z = cam X,Z,-Y
    pc2[..., 2] *= -1
    return pc2

@torch.no_grad()
def inference_prop_feature(
    args,
    curr_epoch,
    model,
    clip,
    dataset_config,
    dataset, 
    dataset_loader,
    logger,
    curr_train_iter,
):
    text_embedding = torch.load(args.clip_embed_path).cuda().float()
    
    ap_calculator = APCalculator(
        dataset_config=dataset_config,
        ap_iou_thresh=[0.25],
        class2type_map=dataset_config.class2type,
        exact_eval=True,
    )

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    model.eval()
    barrier()

    for batch_idx, batch_data_label in enumerate(tqdm(dataset_loader)):
        torch.cuda.empty_cache()
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
            
        inputs = {
            "point_clouds": batch_data_label["point_clouds"],
            "point_cloud_dims_min": batch_data_label["point_cloud_dims_min"],
            "point_cloud_dims_max": batch_data_label["point_cloud_dims_max"],
        }
        outputs = model(inputs)

        batch_centers: torch.Tensor = outputs["outputs"]["center_unnormalized"]
        batch_sizes: torch.Tensor = outputs["outputs"]["size_unnormalized"]
        batch_angles: torch.Tensor = outputs["outputs"]["angle_continuous"]
        sem_cls_prob: torch.Tensor = outputs["outputs"]["sem_cls_prob"] # B, Q, C
        objectness_prob: torch.Tensor = outputs["outputs"]["objectness_prob"] # B, Q, C
        
        masks = batch_data_label["gt_box_present"]
        batch_gt_centers: torch.Tensor = batch_data_label["gt_box_centers"]
        batch_gt_sizes: torch.Tensor = batch_data_label["gt_box_sizes"]
        batch_gt_angles: torch.Tensor = batch_data_label["gt_box_angles"]
        batch_gt_cls: torch.Tensor = batch_data_label["gt_box_sem_cls_label"]
        
        # if is_primary():
        #     print("obj", objectness_prob)
        #     print("sem", sem_cls_prob)
        
        if "image_height" in batch_data_label: # sunrgbd
            images_1d, h, w = batch_data_label["image"], batch_data_label["image_height"], batch_data_label["image_width"]
            images = [
                image_1d[:height*width*3].view(height, width, 3) 
                    for image_1d, height, width in zip(images_1d, h, w)
                    ] # B, (H, W, 3)
            
            calib_Rtilts: torch.Tensor = batch_data_label["calib_Rtilt"]
            calib_Ks: torch.Tensor = batch_data_label["calib_K"]

            batch_clip_logits = []
            for batch_offset, (image, calib_Rtilt, calib_K, centers, sizes, angles) in \
                enumerate(zip(images, calib_Rtilts, calib_Ks, batch_centers, batch_sizes, batch_angles)):
                
                # centers = batch_gt_centers[batch_offset][masks[batch_offset].bool()]
                # sizes = batch_gt_sizes[batch_offset][masks[batch_offset].bool()]
                # angles = batch_gt_angles[batch_offset][masks[batch_offset].bool()]
                # gt_cls = batch_gt_cls[batch_offset][masks[batch_offset].bool()]
                
                calib = SUNRGBD_Calibration_cuda(calib_Rtilt, calib_K)
                boxes = project_box_3d_cuda(calib, centers, sizes, angles)
                
                # clip to image region
                h, w, _ = image.size()
                max_coords = torch.broadcast_to(torch.tensor([[w, h, w, h]], device=image.device), boxes.size())
                boxes = torch.clamp_min(boxes, 0)
                boxes = torch.minimum(boxes, max_coords)
                areas = torch.prod(boxes[..., 2:] - boxes[..., :2], -1)
                bmask = (areas > 100000) # N
                
                # # visualize
                # from detectron2.utils.visualizer import Visualizer
                # import cv2
                # os.makedirs('visualize', exist_ok=True)
                # print(dataset.scan_names[batch_data_label["scan_idx"][0]])
                # if bmask.sum() > 0:
                #     ret = Instances([h, w])
                #     ret.pred_boxes = Boxes(boxes[bmask].cpu().numpy())
                #     # ret.pred_classes = gt_cls[bmask].cpu().numpy()
                #     vis = Visualizer(image.cpu().numpy(), {"thing_classes": dataset.dataset_config.class2type})
                #     vis_pred = vis.draw_instance_predictions(ret).get_image()
                #     cv2.imwrite(os.path.join('visualize', 'sunrgbd.jpg'), vis_pred[:, :, ::-1])
                # exit(0)
                
                results = image.new_zeros(args.nqueries, 640)
                if bmask.sum() > 0: 
                    raw_res: torch.Tensor = clip.inference([{
                        'image': image.clone().permute(2, 0, 1).contiguous(),
                        'instances': Instances((image.shape[0], image.shape[1]), gt_boxes=Boxes(boxes[bmask]))
                    }], do_postprocess=False)
                    results[bmask] = raw_res
                
                objectness = image.new_zeros(args.nqueries)
                if bmask.sum() > 0: 
                    obj = clip.inference_rpn(image.clone(), boxes[bmask])
                    if obj is None:
                        bmask.fill_(False)
                    else:
                        objectness[bmask] = obj
                
                if bmask.sum() > 0:
                    clip_logits = results # Q, C
                    obj_prob = objectness# Q
                    clip_prob = torch.nn.functional.softmax((clip_logits @ text_embedding.transpose(0, 1)), -1)
                    # if is_primary():
                    #     print(clip_prob)
                    # objectness = torch.max(clip_prob, -1)[0]
                    sem_cls_prob[batch_offset, bmask] = clip_prob[bmask]
                    objectness_prob[batch_offset, bmask] = obj_prob[bmask]
                    print(obj_prob)
                
            batch_data_label["batch_clip_logits"] = batch_clip_logits
        else:
            images = batch_data_label["images"] # B, F, (H, W, 3)
            poses = batch_data_label["poses"] # B, F, 4, 4
            frame_length = batch_data_label["frame_length"] # B, F, 4, 4
            intrinsics = batch_data_label["intrinsics"] # B, 4, 4
            axis_aligned_matrix = batch_data_label["axis_align_mat"]

            batch_clip_logits = []
            for batch_offset, (frame_image, pose, intrinsic, axis_aligned_mat, centers, sizes, len_frame) in \
                enumerate(zip(images, poses, intrinsics, axis_aligned_matrix, batch_centers, batch_sizes, frame_length)):
                    
                # centers = batch_gt_centers[batch_offset][masks[batch_offset].bool()]
                # sizes = batch_gt_sizes[batch_offset][masks[batch_offset].bool()]
                # gt_cls = batch_gt_cls[batch_offset][masks[batch_offset].bool()]
                
                projection = [project_box_3d_aabb(centers, sizes, camera2world, intrinsic, axis_aligned_mat) for camera2world in pose[:len_frame]]
                boxes = torch.stack([x[0] for x in projection]) # F, N, 4
                depth_mask = torch.stack([x[1].min(-1)[0] > 0 for x in projection]) # F, N
                # orig_areas = torch.prod(boxes[..., 2:] - boxes[..., :2], -1)
                
                # clip to image region
                _, h, w, _ = frame_image.size()
                max_coords = torch.broadcast_to(torch.tensor([[w, h, w, h]], device=frame_image.device), boxes.size())
                boxes = torch.clamp_min(boxes, 0)
                boxes = torch.minimum(boxes, max_coords)
                areas = torch.prod(boxes[..., 2:] - boxes[..., :2], -1)
                # print(areas)
                # box_mask = areas/orig_areas > 0.1 # F, N
                box_mask = (areas > 10000) * depth_mask # F, N
                # print(torch.sort(areas[box_mask])[0])                
                
                results = []
                objectness = []
                for image, box, bmask in zip(frame_image[:len_frame], boxes, box_mask):
                    
                    pl_holder = image.new_zeros(args.nqueries, 640)
                    if bmask.sum() > 0: 
                        raw_res: torch.Tensor = clip.inference([{
                            'image': image.clone().permute(2, 0, 1).contiguous(),
                            'instances': Instances((image.shape[0], image.shape[1]), gt_boxes=Boxes(box[bmask]))
                        }], do_postprocess=False)
                        pl_holder[bmask] = raw_res
                    results.append(pl_holder)
                    
                    pl_holder = image.new_zeros(args.nqueries)
                    if bmask.sum() > 0: 
                        obj = clip.inference_rpn(image.clone(), box[bmask])
                        if obj is None:
                            bmask.fill_(False)
                        else:
                            pl_holder[bmask] = obj
                    objectness.append(pl_holder)
                    
                results = torch.stack(results, 1).contiguous() # Q, F, C
                objectness = torch.stack(objectness, 1).contiguous() # Q, F
                
                # # visualize
                # from detectron2.utils.visualizer import Visualizer
                # import cv2
                # os.makedirs('visualize', exist_ok=True)
                # print(dataset.scan_names[batch_data_label["scan_idx"][0]])
                # for idx, (image, box, bmask) in enumerate(zip(frame_image[:len_frame], boxes, box_mask)):
                #     if bmask.sum() > 0:
                #         ret = Instances([h, w])
                #         ret.scores = objectness.t()[idx, bmask].cpu().numpy()
                #         ret.pred_boxes = Boxes(box[bmask].cpu().numpy())
                #         ret.pred_classes = gt_cls[bmask].cpu().numpy()
                #         vis = Visualizer(image.cpu().numpy(), {"thing_classes": dataset.dataset_config.class2type})
                #         vis_pred = vis.draw_instance_predictions(ret).get_image()
                #         cv2.imwrite(os.path.join('visualize', '%d.jpg'%idx), vis_pred[:, :, ::-1])
                # exit(0)
                    
                valid_mask = box_mask.transpose(0, 1).contiguous() # Q, F
                if valid_mask.sum() > 0:
                    clip_logits = []
                    obj_prob = []
                    for res, obj, mk in zip(results, objectness, valid_mask):
                        if mk.sum() > 0:
                            clip_logits.append(res[mk].mean(0))
                            obj_prob.append(obj[mk].max(0)[0])
                    clip_logits = torch.stack(clip_logits) # Q, C
                    obj_prob = torch.stack(obj_prob) # Q
                    clip_prob = torch.nn.functional.softmax((clip_logits @ text_embedding.transpose(0, 1)), -1)
                    # if is_primary():
                    #     print(clip_prob)
                    # objectness = torch.max(clip_prob, -1)[0]
                    clip_mask = valid_mask.sum(-1) > 0
                    print((~clip_mask).sum())
                    sem_cls_prob[batch_offset, clip_mask] = clip_prob
                    # objectness_prob[batch_offset, clip_mask] = obj_prob
                    print(obj_prob)
                
                # mask = masks[batch_offset].bool()
                # if mask.sum() > 0:
                #     gt_centers = batch_gt_centers[batch_offset, mask]
                #     gt_sizes = batch_gt_sizes[batch_offset, mask]
                #     gt_box = torch.cat([gt_centers, gt_sizes], -1)
                #     pred_box = torch.cat([centers, sizes], -1)
                #     ious = torch.stack([box_3d_iou_tensor(pred, gt_box, typ='cs').max(dim=0)[0] for pred in pred_box])
                #     objectness = torch.zeros_like(ious)
                #     objectness[ious >= 0.25] = ious[ious >= 0.25]
                #     assert objectness_prob[batch_offset].size() == objectness.size()
                #     objectness_prob[batch_offset] = objectness
            
            outputs["outputs"]["sem_cls_prob"] = sem_cls_prob
            outputs["outputs"]["objectness_prob"] = objectness_prob
        
        # Memory intensive as it gathers point cloud GT tensor across all ranks
        outputs["outputs"] = all_gather_dict(outputs["outputs"])
        batch_data_label = all_gather_dict(batch_data_label)
        
        ap_calculator.step_meter(outputs["outputs"], batch_data_label)
        
        time_delta.update(time.time() - curr_time)
        # if is_primary() and curr_iter % args.log_every == 0:
        #     mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        #     print(
        #         f"Infer; Batch [{curr_iter}/{num_batches}]; Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
        #     )

        #     test_dict = {}
        #     test_dict["memory"] = mem_mb
        #     test_dict["batch_time"] = time_delta.avg
        # curr_iter += 1
        barrier()
    # if is_primary():
    #     logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return ap_calculator

@torch.no_grad()
def inference_global_feature(
    args,
    curr_epoch,
    model,
    dataset_config,
    dataset, 
    dataset_loader,
    logger,
    curr_train_iter,
):
    
    label_formatter = FeatureFormatter(args.out_dir, dataset.scan_names, glob=True)
    
    assert args.out_dir is not None, f"Please specify a path to save pseudo labels using --out_dir."
    os.makedirs(args.out_dir, exist_ok=True)

    curr_iter = 0
    net_device = next(model.parameters()).device
    num_batches = len(dataset_loader)

    time_delta = SmoothedValue(window_size=10)
    model.eval()
    barrier()

    for batch_idx, batch_data_label in enumerate(tqdm(dataset_loader)):
        torch.cuda.empty_cache()
        curr_time = time.time()
        for key in batch_data_label:
            batch_data_label[key] = batch_data_label[key].to(net_device)
        
        if "image_height" in batch_data_label: # sunrgbd
            images_1d, h, w = batch_data_label["image"], batch_data_label["image_height"], batch_data_label["image_width"]
            images = [
                image_1d[:height*width*3].view(height, width, 3) 
                    for image_1d, height, width in zip(images_1d, h, w)
                    ] # B, (H, W, 3)

            batch_clip_logits = []
            for frame_image, height, width in zip(images, h, w):
                
                batch_clip_logits.append(model.inference([{
                    'image': frame_image.permute(2, 0, 1).contiguous(),
                    'instances': Instances((frame_image.shape[0], frame_image.shape[1]), gt_boxes=Boxes(torch.tensor([[0, 0, width, height]], device=frame_image.device)))
                }], do_postprocess=False).squeeze(0))
            assert batch_clip_logits[0].ndim == 1, batch_clip_logits[0].shape
            batch_data_label["batch_clip_logits"] = batch_clip_logits
        else:
            images = batch_data_label["images"] # B, F, (H, W, 3)
            frame_length = batch_data_label["frame_length"] # B, F, 4, 4

            batch_clip_logits = []
            for frame_image, len_frame in zip(images, frame_length):
                _, h, w, _ = frame_image.size()
                results = []
                for image in frame_image[:len_frame]:
                    raw_res = model.inference([{
                        'image': image.permute(2, 0, 1).contiguous(),
                        'instances': Instances((image.shape[0], image.shape[1]), gt_boxes=Boxes(torch.tensor([[w, h, w, h]], device=frame_image.device)))
                    }], do_postprocess=False)
                    results.append(raw_res)
                results = torch.cat(results, 0).mean(0) # GAP
                assert results.size(0) == 640, results.size()
                    
                batch_clip_logits.append(results)
                
            batch_data_label["batch_clip_logits"] = batch_clip_logits # B, (N, C)
        
        label_formatter.step(batch_data_label)
        
        time_delta.update(time.time() - curr_time)
        # if is_primary() and curr_iter % args.log_every == 0:
        #     mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        #     print(
        #         f"Infer; Batch [{curr_iter}/{num_batches}]; Iter time {time_delta.avg:0.2f}; Mem {mem_mb:0.2f}MB"
        #     )

        #     test_dict = {}
        #     test_dict["memory"] = mem_mb
        #     test_dict["batch_time"] = time_delta.avg
        # curr_iter += 1
        barrier()
    # if is_primary():
    #     logger.log_scalars(test_dict, curr_train_iter, prefix="Test/")

    return label_formatter
