"""
Scripts for box evaluations. 

Example Usage: 
python sunrgbd/evaluate_gss_MABO.py --box $box_tag (--test)
"""

import os, sys
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from functools import partial
import multiprocessing as mp
mp.set_start_method('forkserver', force=True)

# HACK: add cwd
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.constants import const_sunrgbd
from utils.box_3d_utils import get_iou_obb

def compute_MABO(scan_name, args):
    aabb_f = args.aabb
    abo = [[] for _ in range(17)]
    
    prop_i = np.load(os.path.join(const_sunrgbd.gss_root, args.box, scan_name + "_prop.npy"))
    if prop_i.shape[0] > 0: 
        prop_i[:, 3:6] *= 2
        
        if aabb_f:
            prop_i[:, 6] = np.pi/2
        
        gt_boxes = np.load(os.path.join(const_sunrgbd.train_dir, scan_name+'_bbox.npy'))
        gt_class_ind = gt_boxes[:, 6] if aabb_f else gt_boxes[:, 7] 
        gt_boxes[:, 3:6] *= 2

        for c in np.unique(gt_class_ind.astype(np.uint8)):
            mask = gt_class_ind == c
            for gt_box in gt_boxes[mask]:
                iou = [get_iou_obb(box, gt_box) for box in prop_i[:, :7]]
                abo[c.item()].append(max(iou))
    return abo

def get_MABO_stats(scan_names, args):
    
    with mp.Pool(processes=mp.cpu_count() // 2) as p:
        results = tqdm(p.map(partial(compute_MABO, args=args), scan_names))
        
    abo = [[] for _ in range(17)]
    for scan_abo in results:
        for c in range(17):
            abo[c].extend(scan_abo[c])
    
    mabo = list(map(lambda x: sum(x) / len(x) if len(x) > 0 else 0, abo))
    return np.array(mabo)

if __name__ == '__main__':
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--box", default="SZ+V-obb-V+F-obb", type=str)
    parser.add_argument("--thresh", default=0.25, type=float)
    parser.add_argument("--aabb", default=False, action="store_true")
    parser.add_argument("--test", default=False, action="store_true")
    args = parser.parse_args()
    
    mp.freeze_support()
    if args.test: 
        mabo = get_MABO_stats(['005051'], args)
    else:
        scan_names = const_sunrgbd.get_scan_names()
        mabo = get_MABO_stats(scan_names, args)

    print('-'*10, f'prop: iou_thresh: {args.thresh}', '-'*10)
    print("Eval: MABO %.4f" % mabo.mean())
    for id, name in const_sunrgbd.class2type.items():
        print("Eval %s: MABO %.4f" % (name, mabo[id]))