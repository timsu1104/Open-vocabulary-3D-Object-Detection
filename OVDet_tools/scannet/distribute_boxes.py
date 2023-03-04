"""
Split RegionCLIP detected box into multiple files. 
"""
import os
import numpy as np
import torch
from time import time

# configs
IMAGE_PATH = '/home/zhengyuan/packages/RegionCLIP/datasets/custom_images'
INPUT_PATH = '/home/zhengyuan/packages/RegionCLIP/output/inference_scannet_nyu38_multigpu'
OUTPUT_PATH = '/share/suzhengyuan/data/RegionCLIP_boxes/2D_nyu38_thresh0.7'
os.makedirs(OUTPUT_PATH, exist_ok=True)
THRESHOLD = 0.7

# preprocessing
custom_img_path = IMAGE_PATH
scene_list = [scene for scene in os.listdir(custom_img_path) if os.path.isdir(os.path.join(custom_img_path, scene))]
custom_img_list = [os.path.join(custom_img_path, scene, 'color', item) for scene in scene_list for item in os.listdir(os.path.join(custom_img_path, scene, 'color')) if os.path.splitext(item)[1] == '.jpg']
outfn_list = [os.path.join(OUTPUT_PATH, (os.path.splitext(item)[0] + '.npy')[len(custom_img_path)+1:]) for item in custom_img_list] 

# file list
preds = torch.load(os.path.join(INPUT_PATH, 'instances_predictions.pth'))
assert len(preds) == len(outfn_list), f"{len(preds)} {len(outfn_list)}"


if __name__ == '__main__':
    start = time()
    stats = []
    for pred, outfn in zip(preds, outfn_list):
        s = time()
        os.makedirs(os.path.split(outfn)[0], exist_ok=True)
        filtered_box = np.array([
                x['bbox'] + [x['score'], x['category_id']] # (0, 17)
             for x in pred['instances'] if (x['score'] > THRESHOLD)# and (x['category_id'] <= 17) # remove background
        ])
        np.save(outfn, filtered_box)
        print("Saved {}, box number {}. Time elapsed {}s. ".format(outfn, len(filtered_box), time() - s))
        
        stats.append(len(filtered_box))
    print("Done! min box {}, max box {}, avg box {}. Elapsed {}s. ".format(min(stats), max(stats), sum(stats) / len(stats), time() - start))