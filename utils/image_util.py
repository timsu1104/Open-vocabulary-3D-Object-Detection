"""
Modified from 3DVG Transformer by Zhengyuan Su.
"""

from imageio import imread
from PIL import Image
import math
import torchvision.transforms as transforms
from utils.projection import ProjectionHelper

import torch
from torchvision.transforms import InterpolationMode
import numpy as np

INTRINSICS = [[37.01983, 0, 20, 0],[0, 38.52470, 15.5, 0],[0, 0, 1, 0],[0, 0, 0, 1]]

class image_processor():
    def __init__(self):
        self.PROJECTOR = ProjectionHelper(INTRINSICS, 0.1, 4.0, [41, 32], 0.05)
    
    def to_tensor(self, arr):
        return torch.Tensor(arr).cuda()
    
    def resize_crop_image(self, image, new_image_dims):
        image_dims = [image.shape[1], image.shape[0]]
        if image_dims == new_image_dims:
            return image
        resize_width = int(math.floor(new_image_dims[1] * float(image_dims[0]) / float(image_dims[1])))
        image = transforms.Resize([new_image_dims[1], resize_width], interpolation=InterpolationMode.NEAREST)(Image.fromarray(image))
        image = transforms.CenterCrop([new_image_dims[1], new_image_dims[0]])(image)
        image = np.array(image)
        
        return image
    
    def load_image(self, file, image_dims):
        image = imread(file)
        # preprocess
        image = self.resize_crop_image(image, image_dims)
        if len(image.shape) == 3: # color image
            image =  np.transpose(image, [2, 0, 1])  # move feature to front
            image = transforms.Normalize(mean=[0.496342, 0.466664, 0.440796], std=[0.277856, 0.28623, 0.291129])(torch.Tensor(image.astype(np.float32) / 255.0))
        elif len(image.shape) == 2: # label image
    #         image = np.expand_dims(image, 0)
            pass
        else:
            raise
            
        return image

    def load_pose(self, filename):
        lines = open(filename).read().splitlines()
        assert len(lines) == 4
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]

        return np.asarray(lines).astype(np.float32)

    def load_depth(self, file, image_dims):
        depth_image = imread(file)
        # preprocess
        depth_image = self.resize_crop_image(depth_image, image_dims)
        depth_image = depth_image.astype(np.float32) / 1000.0

        return depth_image
    
    def compute_projection(self, points, depth, camera_to_world):
        """
        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param depth: depth map (size: proj_image)
        :param camera_to_world: camera pose (4, 4)
        
        :return indices_3d (array with point indices that correspond to a pixel),
        :return indices_2d (array with pixel indices that correspond to a point)

        note:
            the first digit of indices represents the number of relevant points
            the rest digits are for the projection mapping
        """
        num_points = points.shape[0]
        num_frames = depth.shape[0]
        indices_3ds = torch.zeros(num_frames, num_points + 1).long().cuda()
        indices_2ds = torch.zeros(num_frames, num_points + 1).long().cuda()

        for i in range(num_frames):
            indices = self.PROJECTOR.compute_projection(self.to_tensor(points), self.to_tensor(depth[i]), self.to_tensor(camera_to_world[i]))
            if indices:
                indices_3ds[i] = indices[0].long()
                indices_2ds[i] = indices[1].long()
                print("found {} mappings in {} points from frame {}".format(indices_3ds[i][0], num_points, i))
            
        return indices_3ds, indices_2ds