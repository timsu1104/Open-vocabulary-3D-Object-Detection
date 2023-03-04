import os, sys
import numpy as np
import multiprocessing as mp
from tqdm import tqdm

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.io_utils import load_label, get_scene_list
from utils.projection import ProjectionHelper
from time import time



SCANNET_LABEL_ROOT = "/data1/lseg_data/data/scannet/pseudo_labels_2d" # LSeg segmentation result
PSEUDO_LABEL_PATH = os.path.join(SCANNET_LABEL_ROOT, "{}.npy")
DATASET_ROOT_DIR = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/scannet/scannet_train_detection_data'
SCANNET_LABEL_PATH = os.path.join(DATASET_ROOT_DIR, "{}_sem_label.npy")
SCANNET_FRAMES_ROOT = "/data/suzhengyuan/ScanRefer/scannet_train_images/frames_square"
SCANNET_FRAMES = os.path.join(SCANNET_FRAMES_ROOT, "{}/{}") # scene_id, mode
SCANNET_FRAME_PATH = os.path.join(SCANNET_FRAMES, "{}") # name of the file
SCENE_ID = "scene0000_00"
FRAME_ID = "20"

# scan_name = SCENE_ID
# frame_id = FRAME_ID
PROJECTOR = ProjectionHelper(0.1, 10.0, [240, 320])

def match(scan_name):
    frame_list = list(map(lambda x: x.split(".")[0], sorted(os.listdir(SCANNET_FRAMES.format(scan_name, "color")))))

    semantic_labels = np.stack([load_label(SCANNET_FRAME_PATH.format(scan_name, "label-mapped", "{}.png".format(frame_id))) for frame_id in frame_list])
    x = np.load(PSEUDO_LABEL_PATH.format(scan_name), allow_pickle=True)
    pseudo_labels = np.stack([x.item()[int(frame_id)].astype(np.int64) for frame_id in frame_list])

    semantic_labels = PROJECTOR.project_label(semantic_labels, False).astype(np.int64)
    pseudo_labels = PROJECTOR.project_label(pseudo_labels, True)
    
    # count = 0
    # for plabel, gtlabel in zip(pseudo_labels, semantic_labels):
    #     if np.mean(np.isin(plabel, np.unique(gtlabel))) >= 0.5:
    #         count += 1
    # total = len(frame_list)
    
    count = np.sum(pseudo_labels == semantic_labels)
    total = semantic_labels.size
    
    print(scan_name, count/total)
    return count, total

# print(np.unique(semantic_labels))
# print(np.unique(pseudo_labels))
if __name__ == '__main__':
    scene_list = get_scene_list()
    start = time()
    p = mp.Pool(mp.cpu_count() // 2)
    result = p.map(match, scene_list)
    p.close()
    p.join()
    count = sum(map(lambda x: x[0], result))
    s = sum(map(lambda x: x[1], result))
    print("Done! Elapsed {}s. Correctness {}/{}".format(time() - start, count, s))