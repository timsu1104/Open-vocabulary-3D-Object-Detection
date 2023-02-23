import os, sys
import numpy as np
from tqdm import tqdm
import shutil
from argparse import ArgumentParser

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from utils.constants import const_sunrgbd

if __name__ == '__main__':
    
    parser = ArgumentParser("3D Detection Using Transformers")
    parser.add_argument("--box", default="3D_test", type=str)
    parser.add_argument("--list", default=False, action="store_true")
    parser.add_argument("--rm", default=False, action="store_true")
    parser.add_argument("--stats", default=False, action="store_true")
    args = parser.parse_args()
    
    if args.list:
        print(os.listdir(const_sunrgbd.box_root))
    if args.rm:
        print("removing %s" % os.path.join(const_sunrgbd.box_root, args.box))
        shutil.rmtree(os.path.join(const_sunrgbd.box_root, args.box))
    if args.stats:
        scene_list = const_sunrgbd.get_scan_names()
        boxes = [np.load(os.path.join(const_sunrgbd.box_root, args.box, scan_name + "_bbox.npy")).shape[0] for scan_name in tqdm(scene_list)]
        order = np.argsort(boxes)
        print("Max box %d in %s" % (max(boxes), scene_list[order[-1]]))
        print("The top 10: ", end="")
        for i in range(1, 11):
            print(scene_list[order[-i]], end=" ")