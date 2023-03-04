# Copyright (c) Facebook, Inc. and its affiliates.
from .scannet import ScannetDetectionDataset, ScannetDatasetConfig
from .sunrgbd import SunrgbdDetectionDataset, SunrgbdDatasetConfig


DATASET_FUNCTIONS = {
    "scannet": [ScannetDetectionDataset, ScannetDatasetConfig],
    "sunrgbd": [SunrgbdDetectionDataset, SunrgbdDatasetConfig],
}


def build_dataset(args):
    dataset_builder = DATASET_FUNCTIONS[args.dataset_name][0]
    dataset_config = DATASET_FUNCTIONS[args.dataset_name][1]()
    
    dataset_dict = {
        "train": dataset_builder(
            dataset_config, 
            split_set="train", 
            root_dir=args.dataset_root_dir, 
            pseudo_box_dir=args.pseudo_label_dir, 
            gss_box_dir=args.gss_box_dir, 
            feature_2d_dir=args.feature_2d_dir, 
            gss_feats_dir=args.gss_feats_dir, 
            feature_global_dir=args.feature_global_dir,
            pseudo_feature_dir=args.pseudo_feats_dir,
            meta_data_dir=args.meta_data_dir, 
            use_color=args.use_color,
            use_image=args.use_image,
            augment=True,
            use_pbox=args.use_pbox,
            use_gss=args.use_gss,
            use_2d_feature=args.use_2d_feature
        ),
        "test": dataset_builder(
            dataset_config, 
            split_set="val", 
            root_dir=args.dataset_root_dir, 
            use_color=args.use_color,
            use_image=args.use_image,
            augment=False
        ),
        "inference": dataset_builder(
            dataset_config, 
            split_set="train", 
            close_set=True,
            root_dir=args.dataset_root_dir, 
            pseudo_box_dir=args.pseudo_label_dir, 
            gss_box_dir=args.gss_box_dir, 
            # feature_2d_dir=args.feature_2d_dir, 
            meta_data_dir=args.meta_data_dir, 
            use_color=args.use_color,
            use_image=args.use_image,
            only_pbox=args.use_pbox,
            use_gss=args.use_gss,
            augment=False
        ),
    }
    return dataset_dict, dataset_config
    