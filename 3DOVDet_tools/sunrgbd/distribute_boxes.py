"""
Split RegionCLIP detected box into multiple files. 

Example Usage: 
python sunrgbd/distribute_boxes.py -o 2D -i inference_sunrgbd
"""
import os, sys
import numpy as np
import torch
from time import time
from argparse import ArgumentParser

# HACK: add cwd
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.constants import const_sunrgbd, regionclip_root

if __name__ == '__main__':
    
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument('-i', "--input", default="inference_sunrgbd_ov3detic", type=str)
    parser.add_argument('-o', "--output", default="2D_ov3detic", type=str)
    parser.add_argument("--thresh", default=0.7, type=float)
    args = parser.parse_args()
    
    output_path = os.path.join(const_sunrgbd.box_root, args.output)
    os.makedirs(output_path, exist_ok=True)
    
    start = time()
    
    custom_img_path = const_sunrgbd.image_data_dir
    outfn_list = [os.path.join(output_path, (os.path.splitext(item)[0] + '.npy')) for item in os.listdir(custom_img_path)]
    
    preds = torch.load(os.path.join(regionclip_root, 'output', args.input, 'instances_predictions.pth'))
    assert len(preds) == len(outfn_list), f"{len(preds)} {len(outfn_list)}"
    
    stats = []
    for pred, outfn in zip(preds, outfn_list):
        s = time()
        os.makedirs(os.path.split(outfn)[0], exist_ok=True)
        filtered_box = np.array([
                x['bbox'] + [x['score'], x['category_id']] # (0, 19)
             for x in pred['instances'] if (x['score'] > args.thresh) and (x['category_id'] <= 19) # remove background
        ])
        np.save(outfn, filtered_box)
        print("Saved {}, box number {}. Time elapsed {}s. ".format(outfn, len(filtered_box), time() - s))
        
        stats.append(len(filtered_box))
    print("Done! min box {}, max box {}, avg box {}. Elapsed {}s. ".format(min(stats), max(stats), sum(stats) / len(stats), time() - start))