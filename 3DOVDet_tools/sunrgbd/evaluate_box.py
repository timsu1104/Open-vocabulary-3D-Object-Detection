"""
Scripts for box evaluations. 

Example Usage: 
python sunrgbd/evaluate_box.py --box $box_tag (--test)
"""

import os, sys
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

# HACK: add cwd
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.evaluation.pr_helper import PRCalculator
from utils.constants import const_sunrgbd

def compute_precision_recall(scan_names, args):
    tag = args.box
    iou_thresh = args.thresh
    aabb_f = args.aabb
    
    ap_calculator = PRCalculator(iou_thresh, const_sunrgbd.class2type, obb=True) 
    all_p = []
    for scan_name in tqdm(scan_names):
        prop_i = np.load(os.path.join(os.path.join(const_sunrgbd.box_root, tag), scan_name+'_bbox.npy'))
        if prop_i.shape[0] == 0: continue
        prop_i[:, 3:6] *= 2
        if aabb_f:
            box_len = 6
            class_ind = (prop_i[:, 6]).copy()
            prop_i[:, 6] = np.pi/2
        else:
            box_len = 7
            class_ind = (prop_i[:, 7]).copy()
        batch_pred_map_cls = [(class_ind[j], prop_i[j, :7], prop_i[j, box_len+1]) for j in range(len(class_ind))]
        all_p += [prop_i.shape[0]]
        
        gt_boxes = np.load(os.path.join(const_sunrgbd.train_dir, scan_name+'_bbox.npy'))
        if gt_boxes.shape[0] == 0: continue
        class_ind = gt_boxes[:, 7]
        gt_boxes[:, 3:6] *= 2
        batch_gt_map_cls = [(class_ind[j], gt_boxes[j, :7]) for j in range(len(class_ind))]

        ap_calculator.step([batch_pred_map_cls], [batch_gt_map_cls])
    print('-'*10, f'prop: iou_thresh: {iou_thresh}', '-'*10)
    print("avg num: %.2f total number: %d" % (np.mean(all_p), sum(all_p)))
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        print('eval %s: %f'%(key, metrics_dict[key]))

if __name__ == '__main__':
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--box", default="test", type=str)
    parser.add_argument("--aabb", default=False, action="store_true")
    parser.add_argument("--thresh", default=0.25, type=float)
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()
    
    if args.test: 
        compute_precision_recall(['005051'], args)
    else:
        scan_names = const_sunrgbd.get_scan_names()
        compute_precision_recall(scan_names, args)