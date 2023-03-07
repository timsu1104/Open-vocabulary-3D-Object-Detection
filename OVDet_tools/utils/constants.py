import os


class ScannetConstants(object):
    def __init__(self) -> None:
        self.root_dir = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/scannet/scannet_detection_data'
        self.box_root = '/share/suzhengyuan/data/RegionCLIP_boxes'
        return


class SunrgbdConstants(object):
    def __init__(self) -> None:
        self.root_dir = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/sunrgbd/sunrgbd_pc_bbox_50k_v1'
        self.suffix = '_ov3detic'
        self.train_dir = self.root_dir + '_train' + self.suffix
        self.val_dir = self.root_dir + '_val' + self.suffix
        
        self.raw_data_dir = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/sunrgbd/sunrgbd_trainval'
        self.image_data_dir = os.path.join(self.raw_data_dir, 'image')
        self.depth_data_dir = os.path.join(self.raw_data_dir, 'depth_raw')
        self.calib_data_dir = os.path.join(self.raw_data_dir, 'calib')
        self.seglabel_data_dir = os.path.join(self.raw_data_dir, 'seg_label')
        
        self.box_root = '/share/suzhengyuan/data/RegionCLIP_boxes_sunrgbd'
        self.vis_root = "./visualizations_sunrgbd"
        self.gss_root = "/share/suzhengyuan/code/WyPR/gss/computed_proposal_sunrgbd"
        
        # self.type2class = {
        #     'bathtub': 0,
        #     'bed': 1,
        #     'bookshelf': 2,
        #     'box': 3,
        #     'chair': 4,
        #     'counter': 5,
        #     'desk': 6,
        #     'door': 7,
        #     'dresser': 8,
        #     'lamp': 9,
        #     'nightstand': 10,
        #     'pillow': 11,
        #     'sink': 12,
        #     'sofa': 13,
        #     'table': 14,
        #     'tv': 15,
        #     'toilet': 16
        # }
        
        self.type2class = {
            'toilet': 0,
            'bed': 1,
            'chair': 2,
            'bathtub': 3,
            'sofa': 4,
            'dresser': 5,
            'scanner': 6,
            'fridge': 7,
            'lamp': 8,
            'desk': 9,
            'table': 10,
            'stand': 11,
            'cabinet': 12,
            'counter': 13,
            'bin': 14,
            'bookshelf': 15,
            'pillow': 16,
            'microwave': 17,
            'sink': 18,
            'stool': 19
        }
        
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.orient = True
        self.num_sem_cls = len(self.type2class.keys())
        
    def get_scan_names(self):
        data_path = self.root_dir + "_%s" % ("trainval")
        sub_splits = ["train", "val"]
        all_paths = []
        for sub_split in sub_splits:
            data_path = data_path.replace("trainval", sub_split)
            basenames = sorted(
                list(set([os.path.basename(x)[0:6] for x in os.listdir(data_path)]))
            )
            basenames = [os.path.join(data_path, x) for x in basenames]
            all_paths.extend(basenames)
        all_paths.sort()
        scan_names = all_paths
        scan_names = list(set([os.path.split(scan_name)[1] for scan_name in scan_names]))
        return scan_names

class Matterport3DConstants(object):
    def __init__(self) -> None:
        self.root_dir = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/matterport3d/matterport_detection_data'
        self.meta_dir = '/share/suzhengyuan/code/ScanRefer-3DVG/votenet/matterport3d/meta_data'
        
        self.raw_data_dir = '/share/suzhengyuan/data/matterport3d/v1/scans'
        self.image_data_dir = os.path.join(self.raw_data_dir, '{}', 'matterport_color_images')
        self.calib_data_dir = os.path.join(self.raw_data_dir, 'calib')
        self.seglabel_data_dir = os.path.join(self.raw_data_dir, 'seg_label')
        
        self.box_root = '/share/suzhengyuan/data/RegionCLIP_boxes_sunrgbd'
        self.vis_root = "./visualizations_sunrgbd"
        self.gss_root = "/share/suzhengyuan/code/WyPR/gss/computed_proposal_sunrgbd"
        
        self.type2class = {
            'toilet': 0,
            'bed': 1,
            'chair': 2,
            'bathtub': 3,
            'sofa': 4,
            'dresser': 5,
            'scanner': 6,
            'fridge': 7,
            'lamp': 8,
            'desk': 9,
            'table': 10,
            'stand': 11,
            'cabinet': 12,
            'counter': 13,
            'bin': 14,
            'bookshelf': 15,
            'pillow': 16,
            'microwave': 17,
            'sink': 18,
            'stool': 19
        }
        
        self.class2type = {self.type2class[t]: t for t in self.type2class}
        self.orient = True
        self.num_sem_cls = len(self.type2class.keys())
        
    def get_scan_names(self):
        with open(os.path.join(self.meta_dir, 'matterport_train.txt'), "r") as f:
            scan_names = f.read().splitlines()
        return scan_names
        
const_scannet = ScannetConstants()
const_sunrgbd = SunrgbdConstants()
const_matterport = Matterport3DConstants()
regionclip_root = '/home/zhengyuan/packages/RegionCLIP'