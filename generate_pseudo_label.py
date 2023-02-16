# Copyright (c) Facebook, Inc. and its affiliates.

"""
Example: 
python generate_pseudo_label.py --dataset_name scannet --nqueries 256 --test_ckpt outputs/scannet_ep1080_GSSlseg_woprior/checkpoint.pth --pseudo_label_dir /share/suzhengyuan/data/RegionCLIP_boxes/3DETR_adjusted/3D_LSeg_woprior --in_dir /share/suzhengyuan/data/RegionCLIP_boxes/3D_LSeg_woprior --out_dir /share/suzhengyuan/data/RegionCLIP_boxes/test/3D_LSeg_woprior_phase1 --feature_2d_dir /data1/lseg_data/data/scannet/pseudo_labels --conf_thresh 0.9 --obj_thresh 0.9
"""

import argparse
import os

import numpy as np
import torch
from torch.multiprocessing import set_start_method
from torch.utils.data import DataLoader, DistributedSampler

# 3DETR codebase specific imports
from datasets import build_dataset
from engine import inference
from models import build_model, build_ULIP
from utils.dist import init_distributed, is_distributed, get_rank
from utils.misc import my_worker_init_fn
from utils.logger import Logger

project_name = "OVDet"
group_name = "Test"

def make_args_parser():
    parser = argparse.ArgumentParser("3D Detection Using Transformers", add_help=False)

    ##### Optimizer #####
    parser.add_argument("--base_lr", default=5e-4, type=float)
    parser.add_argument("--warm_lr", default=1e-6, type=float)
    parser.add_argument("--warm_lr_epochs", default=9, type=int)
    parser.add_argument("--final_lr", default=1e-6, type=float)
    parser.add_argument("--lr_scheduler", default="cosine", type=str)
    parser.add_argument("--weight_decay", default=0.1, type=float)
    parser.add_argument("--filter_biases_wd", default=False, action="store_true")
    parser.add_argument(
        "--clip_gradient", default=0.1, type=float, help="Max L2 norm of the gradient"
    )

    ##### Model #####
    parser.add_argument(
        "--model_name",
        default="3detr",
        type=str,
        help="Name of the model",
        choices=["3detr"],
    )
    ### Encoder
    parser.add_argument(
        "--enc_type", default="vanilla", choices=["masked", "maskedv2", "vanilla"]
    )
    # Below options are only valid for vanilla encoder
    parser.add_argument("--enc_nlayers", default=3, type=int)
    parser.add_argument("--enc_dim", default=256, type=int)
    parser.add_argument("--enc_ffn_dim", default=128, type=int)
    parser.add_argument("--enc_dropout", default=0.1, type=float)
    parser.add_argument("--enc_nhead", default=4, type=int)
    parser.add_argument("--enc_pos_embed", default=None, type=str)
    parser.add_argument("--enc_activation", default="relu", type=str)

    ### Decoder
    parser.add_argument("--dec_nlayers", default=8, type=int)
    parser.add_argument("--dec_dim", default=256, type=int)
    parser.add_argument("--dec_ffn_dim", default=256, type=int)
    parser.add_argument("--dec_dropout", default=0.1, type=float)
    parser.add_argument("--dec_nhead", default=4, type=int)

    ### MLP heads for predicting bounding boxes
    parser.add_argument("--mlp_dropout", default=0.3, type=float)
    parser.add_argument(
        "--nsemcls",
        default=-1,
        type=int,
        help="Number of semantic object classes. Can be inferred from dataset",
    )

    ### Other model params
    parser.add_argument("--preenc_npoints", default=2048, type=int)
    parser.add_argument(
        "--pos_embed", default="fourier", type=str, choices=["fourier", "sine"]
    )
    parser.add_argument("--nqueries", default=256, type=int)
    parser.add_argument("--use_color", default=False, action="store_true")

    ##### Dataset #####
    parser.add_argument(
        "--dataset_name", required=True, type=str, choices=["scannet", "sunrgbd"]
    )
    parser.add_argument(
        "--dataset_root_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--meta_data_dir",
        type=str,
        default=None,
        help="Root directory containing the metadata files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--dataset_num_workers", default=4, type=int)
    parser.add_argument("--batchsize_per_gpu", default=8, type=int)
    
    # pseudo label
    parser.add_argument(
        "--pseudo_label_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--in_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--clip_embed_path",
        type=str,
        default="/home/zhengyuan/packages/RegionCLIP/datasets/custom_concepts/concepts_3detr.pth",
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--ulip_ckpt_path",
        type=str,
        default="/home/zhengyuan/code/ULIP/pretraining_ScanObjectNN/checkpoints/pointMLP_ULIP-20230131063533/best_checkpoint.pth",
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument(
        "--feature_2d_dir",
        type=str,
        default=None,
        help="Root directory containing the dataset files. \
              If None, default values from scannet.py/sunrgbd.py are used",
    )
    parser.add_argument("--use_pbox", default=False, action="store_true")
    parser.add_argument("--use_2d_feature", default=False, action="store_true")
    parser.add_argument("--use_image", default=False, action="store_true")

    ##### Training #####
    parser.add_argument("--start_epoch", default=-1, type=int)
    parser.add_argument("--max_epoch", default=720, type=int)
    parser.add_argument("--eval_every_epoch", default=10, type=int)
    parser.add_argument("--seed", default=0, type=int)

    ##### Testing #####
    parser.add_argument("--test_only", default=False, action="store_true")
    parser.add_argument("--test_ckpt", default=None, type=str)
    parser.add_argument("--topk", default=50, type=int)
    parser.add_argument("--conf_thresh", default=0, type=float)
    parser.add_argument("--obj_thresh", default=0, type=float)

    ##### I/O #####
    parser.add_argument("--checkpoint_dir", default=None, type=str)
    parser.add_argument("--log_every", default=10, type=int)
    parser.add_argument("--log_metrics_every", default=20, type=int)
    parser.add_argument("--save_separate_checkpoint_every_epoch", default=100, type=int)
    parser.add_argument("--pseudo_label_saving_path", default=None, type=str)

    ##### Distributed Training #####
    parser.add_argument("--ngpus", default=1, type=int)
    parser.add_argument("--dist_url", default="tcp://localhost:12345", type=str)

    return parser

def test_model(args, model, model_no_ddp, dataset_config, datasets, dataloaders):
    
    assert not (args.test_ckpt is None or not os.path.isfile(args.test_ckpt)), f"Please specify a test checkpoint using --test_ckpt. Found invalid value {args.test_ckpt}"

    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu"))
    model_no_ddp.load_state_dict(sd["model"])
    logger = Logger()
    epoch = -1
    curr_iter = 0
    label_formatter, ap_calculator = inference(
        args,
        epoch,
        model,
        dataset_config,
        datasets["inference"],
        dataloaders["inference"],
        logger,
        curr_iter,
    )
    
    # metrics = ap_calculator.compute_metrics()
    # ap25 = metrics[0.25]["mAP"]
    # metric_str = ap_calculator.metrics_to_str(metrics, per_class=True)
    # if is_primary():
    #     print("==" * 10)
    #     print(f"Evaluate Epoch [{epoch}/{args.max_epoch}]; Metrics {metric_str}")
    #     print("==" * 10)
        
    label_formatter.process(args.topk, args.conf_thresh, args.obj_thresh)


def main(local_rank, args):
    if args.ngpus > 1:
        print(
            "Initializing Distributed Training. This is in BETA mode and hasn't been tested thoroughly. Use at your own risk :)"
        )
        print("To get the maximum speed-up consider reducing evaluations on val set by setting --eval_every_epoch to greater than 50")
        init_distributed(
            local_rank,
            global_rank=local_rank,
            world_size=args.ngpus,
            dist_url=args.dist_url,
            dist_backend="nccl",
        )

    print(f"Called with args: {args}")
    torch.cuda.set_device(local_rank)
    np.random.seed(args.seed + get_rank())
    torch.manual_seed(args.seed + get_rank())
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + get_rank())

    datasets, dataset_config = build_dataset(args)
    model, _ = build_model(args, dataset_config)
    ulip = build_ULIP(args, dataset_config)
    model = model.cuda(local_rank)
    ulip = ulip.cuda(local_rank)
    model_no_ddp = model

    if is_distributed():
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank]
        )

    dataloaders = {}
    dataset_splits = ["inference"]
    for split in dataset_splits:
        shuffle = False
        if is_distributed():
            sampler = DistributedSampler(datasets[split], shuffle=shuffle)
        else:
            sampler = torch.utils.data.SequentialSampler(datasets[split])

        dataloaders[split] = DataLoader(
            datasets[split],
            sampler=sampler,
            batch_size=args.batchsize_per_gpu,
            num_workers=args.dataset_num_workers,
            worker_init_fn=my_worker_init_fn,
        )
        dataloaders[split + "_sampler"] = sampler

    test_model(args, model, model_no_ddp, dataset_config, datasets, dataloaders)


def launch_distributed(args):
    world_size = args.ngpus
    if world_size == 1:
        main(local_rank=0, args=args)
    else:
        torch.multiprocessing.spawn(main, nprocs=world_size, args=(args,))


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    parser = make_args_parser()
    args = parser.parse_args()
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    launch_distributed(args)
