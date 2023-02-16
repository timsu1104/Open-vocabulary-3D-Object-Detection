import os
import numpy as np
import multiprocessing as mp
from utils.io_utils import get_scene_list, nyu40ids
from argparse import ArgumentParser
from functools import partial
from utils.constants import BOX_ROOT


def move(scan_name: str, tag):
    box = np.load(os.path.join(BOX_ROOT, tag, scan_name+'_bbox.npy'))[:, 0:7] # remove the score and size column
    assert box.shape[1] == 7, box.shape
    box[:, -1] = nyu40ids[box[:, -1].astype(np.int64)]
    os.makedirs(os.path.join(BOX_ROOT, '3DETR_adjusted', tag), exist_ok=True)
    np.save(os.path.join(BOX_ROOT, '3DETR_adjusted', tag, scan_name+'_bbox.npy'), box)
    return box.shape[0]

if __name__ == '__main__':
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--box", default="test", type=str)
    args = parser.parse_args()

    scene_list = get_scene_list()
    p = mp.Pool(processes=mp.cpu_count() // 4 * 3)
    l = p.map(partial(move, tag=args.box), scene_list)
    p.close()
    p.join()

    print(f"Max number of box {max(l)}")

"""
Available Boxes (/share/suzhengyuan/data/RegionCLIP_boxes/3DETR_adjusted):
3D_GSS_gt: GSS + gt pool, max 1189
3D_GSS_GT_multi: GSS + regionclip + gt 3d, max 510
3D_GSS_GT_single: GSS + regionclip + gt 2d, max 76
3D_GSS_LSeg_single: GSS + regionclip + lseg 2d, max 89
"""