"""
Rename pseudo labels. 
"""
import os, pickle
import numpy as np
import multiprocessing as mp
from utils.io_utils import get_scene_list

INPUT_PSEUDO_FEATURE_PATH = '/data1/lseg_data/data/scannet/scannet_data/lseg_feats_maxpool'
INPUT_PSEUDO_LABEL_PATH = ''
OUTPUT_PATH = '/data/suzhengyuan/ScanRefer/LSeg_data'

os.makedirs(OUTPUT_PATH, exist_ok=True)

def move(scan_name: str):
    feat = pickle.load(open(os.path.join(INPUT_PSEUDO_FEATURE_PATH, scan_name + '.pkl'), 'rb')) # N, C=512
    np.save(os.path.join(OUTPUT_PATH, scan_name + '_feats_lseg.npy'), feat)
    print("Done {}".format(scan_name))

scene_list = get_scene_list()
p = mp.Pool(processes=mp.cpu_count() // 4 * 3)
p.map(move, scene_list)
p.close()
p.join()