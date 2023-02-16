"""
Scripts for box evaluations. 

Usage
python evaluate_box.py --box 3D_LSeg_woprior --test
"""
import os
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

from utils.evaluation.pr_helper import PRCalculator
from utils.io_utils import class2type, nyu40ids, get_scene_list
from utils.constants import DATASET_ROOT_DIR, BOX_ROOT

# SCANNET_3DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/test/3D_GSS_LSeg_multi_th0.9_0.9"
# SCANNET_3DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/3D_GSS_GT_multi_softnms_m3"
# SCANNET_3DBOXS = "/share/suzhengyuan/data/RegionCLIP_boxes/3D_GSS_LSeg_multi_test"

def compute_precision_recall(scan_names, tag, iou_thresh):
    ap_calculator = PRCalculator(iou_thresh, class2type) 
    all_p = []
    for scan_name in tqdm(scan_names):
        prop_i = np.load(os.path.join(os.path.join(BOX_ROOT, tag), scan_name+'_bbox.npy'))
        if prop_i.shape[0] == 0: continue
        class_ind = prop_i[:, 6]
        batch_pred_map_cls = [(class_ind[j], prop_i[j, :6], np.prod(prop_i[j, 3:6])) for j in range(len(class_ind))]
        all_p += [prop_i.shape[0]]
        
        gt_boxes = np.load(os.path.join(DATASET_ROOT_DIR, scan_name+'_bbox.npy'))
        class_ind = [np.where(nyu40ids == x)[0][0] for x in gt_boxes[:, 6]]   
        assert gt_boxes.shape[0] == len(class_ind)
        batch_gt_map_cls = [(class_ind[j], gt_boxes[j, :6]) for j in range(len(class_ind))]

        ap_calculator.step([batch_pred_map_cls], [batch_gt_map_cls])
    print('-'*10, f'prop: iou_thresh: {iou_thresh}', '-'*10)
    print("avg num: %.2f total number: %d" % (np.mean(all_p), sum(all_p)))
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        print('eval %s: %f'%(key, metrics_dict[key]))

if __name__ == '__main__':
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--box", default="test", type=str)
    parser.add_argument("--thresh", default=0.25, type=float)
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()
    
    if args.test: 
        compute_precision_recall(['scene0000_00'], args.box, args.thresh)
    else:
        scan_names = get_scene_list()
        compute_precision_recall(scan_names, args.box, args.thresh)