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
    
    def load_image(self, file):
        image = imread(file).astype(np.float32)
            
        return image

    def load_pose(self, filename):
        lines = open(filename).read().splitlines()
        assert len(lines) == 4
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]

        return np.asarray(lines).astype(np.float32)
    
    def load_intrinsic(self, filename):
        lines = open(filename).read().splitlines()
        assert len(lines) == 4
        lines = [[x[0],x[1],x[2],x[3]] for x in (x.split(" ") for x in lines)]

        return np.asarray(lines).astype(np.float32)

    def load_depth(self, file):
        depth_image = imread(file)
        depth_image = depth_image.astype(np.float32) / 1000.0

        return depth_image
    
    def read_alignment(self, meta_file):
        lines = open(meta_file).readlines()
        axis_align_matrix = None
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
        return axis_align_matrix
    
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
    
    @staticmethod
    def rotz(t):
        """Rotation about the z-axis."""
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s,  0],
                        [s,  c,  0],
                        [0,  0,  1]])
    
    @staticmethod
    def project_box_3d(calib, center, size, heading_angle=0):
        R = image_processor.rotz(-1*heading_angle)
        l,w,h = size
        x_corners = [-l,l,l,-l,-l,l,l,-l]
        y_corners = [w,w,-w,-w,w,w,-w,-w]
        z_corners = [h,h,h,h,-h,-h,-h,-h]
        corners_3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0,:] += center[0]
        corners_3d[1,:] += center[1]
        corners_3d[2,:] += center[2]
        corners_2d, _ = calib.project_upright_depth_to_image(np.transpose(corners_3d))
        y1, x1 = np.min(corners_2d, 0)
        y2, x2 = np.max(corners_2d, 0)
        return np.array([x1, y1, x2, y2])
    
def project_box_3d_cuda(calib, center, size, heading_angle):
    """
    center: B, Q, 3
    size: B, Q, 3
    heading_angle: B, Q
    """
    R = rotz_cuda(-heading_angle)
    l,w,h = torch.tensor_split(size / 2, 3, dim=-1)
    x_corners = torch.cat([-l,l,l,-l,-l,l,l,-l], -1) # ..., 8
    y_corners = torch.cat([w,w,-w,-w,w,w,-w,-w], -1)
    z_corners = torch.cat([h,h,h,h,-h,-h,-h,-h], -1)
    corners_3d = R @ torch.stack([x_corners, y_corners, z_corners], -2) # ..., 3, 8
    corners_3d += center[..., None]
    corners_2d, _ = calib.project_upright_depth_to_image(corners_3d.transpose(-1, -2))
    # y1, x1 = torch.tensor_split(torch.min(corners_2d, -2)[0], 2, dim=-1) # ..., 2
    # y2, x2 = torch.tensor_split(torch.max(corners_2d, -2)[0], 2, dim=-1) # ..., 2
    x1, y1 = torch.tensor_split(torch.min(corners_2d, -2)[0], 2, dim=-1) # ..., 2
    x2, y2 = torch.tensor_split(torch.max(corners_2d, -2)[0], 2, dim=-1) # ..., 2
    box_2d = torch.stack([x1, y1, x2, y2], -2).squeeze(-1)
    return box_2d

def project_box_3d_aabb(center, size, pose:torch.Tensor, intrinsic, axis_aligned_mat):
    """
    center: B, Q, 3
    size: B, Q, 3
    """
    l,w,h = torch.tensor_split(size / 2, 3, dim=-1)
    x_corners = torch.cat([-l,l,l,-l,-l,l,l,-l], -1) # ..., 8
    y_corners = torch.cat([w,w,-w,-w,w,w,-w,-w], -1)
    z_corners = torch.cat([h,h,h,h,-h,-h,-h,-h], -1)
    corners_3d = torch.stack([x_corners, y_corners, z_corners], -2) # ..., 3, 8
    corners_3d += center[..., None]
    corners_3d = corners_3d.transpose(-1, -2) # ..., 8, 3, axis aligned bbox
    
    rot = axis_aligned_mat[:3, :3]
    bias = axis_aligned_mat[:3, 3]
    corners_3d = (corners_3d - bias) @ torch.inverse(rot.t())
    # world_to_camera = torch.inverse(pose)
    # rot = world_to_camera[:3, :3]
    # bias = world_to_camera[:3, 3]
    # corners_3d = (rot @ corners_3d.unsqueeze(-1)).squeeze(-1) + bias
    rot = pose[:3, :3]
    bias = pose[:3, 3]
    corners_3d = (corners_3d - bias) @ torch.inverse(rot.t())
    
    x = (corners_3d[..., 0] * intrinsic[0][0]) / corners_3d[..., 2] + intrinsic[0][2]
    y = (corners_3d[..., 1] * intrinsic[1][1]) / corners_3d[..., 2] + intrinsic[1][2]
    
    corners_2d = torch.stack([x, y], -1) # ..., 8, 2
    # y1, x1 = torch.tensor_split(torch.min(corners_2d, -2)[0], 2, dim=-1) # ..., 2
    # y2, x2 = torch.tensor_split(torch.max(corners_2d, -2)[0], 2, dim=-1) # ..., 2
    x1, y1 = torch.tensor_split(torch.min(corners_2d, -2)[0], 2, dim=-1) # ..., 2
    x2, y2 = torch.tensor_split(torch.max(corners_2d, -2)[0], 2, dim=-1) # ..., 2
    box_2d = torch.cat([x1, y1, x2, y2], -1)
    return box_2d, corners_3d[..., 2]
    
def rotz_cuda(t: torch.Tensor):
    """Rotation about the z-axis. Support batch ops. """
    c = torch.cos(t)
    s = torch.sin(t)
    zeros = t.new_zeros(t.size())
    ones = t.new_ones(t.size())
    r1 = torch.stack([c, -s, zeros], dim=-1)
    r2 = torch.stack([s, c, zeros], dim=-1)
    r3 = torch.stack([zeros, zeros, ones], dim=-1)
    rotation = torch.stack([r1, r2, r3], dim=-2) # B, Q, 3, 3
    return rotation
    
class SUNRGBD_Calibration(object):
    ''' Calibration matrices and utils
        We define five coordinate system in SUN RGBD dataset

        camera coodinate:
            Z is forward, Y is downward, X is rightward

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord

        upright depth coordinate: tilted depth coordinate by Rtilt such that Z is gravity direction,
            Z is up-axis, Y is forward, X is right-ward

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v) 

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box in upright camera coordinate
    '''

    def __init__(self, Rtilt, K):
        self.Rtilt = Rtilt.cpu().numpy()
        self.K = K.cpu().numpy()
        self.f_u = self.K[0,0]
        self.f_v = self.K[1,1]
        self.c_u = self.K[0,2]
        self.c_v = self.K[1,2]
   
    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (N,3) Output: (N,3)
        '''
        # Project upright depth to depth coordinate
        pc2 = np.dot(np.transpose(self.Rtilt), np.transpose(pc[:,0:3])) # (3,n)
        return flip_axis_to_camera(np.transpose(pc2))

    def project_upright_depth_to_image(self, pc):
        ''' Input: (N,3) Output: (N,2) UV and (N,) depth '''
        pc2 = self.project_upright_depth_to_camera(pc)
        uv = np.dot(pc2, np.transpose(self.K)) # (n,3)
        uv[:,0] /= uv[:,2]
        uv[:,1] /= uv[:,2]
        return uv[:,0:2], pc2[:,2]

    def project_upright_depth_to_upright_camera(self, pc):
        return flip_axis_to_camera(pc)

    def project_upright_camera_to_upright_depth(self, pc):
        return flip_axis_to_depth(pc)

    def project_image_to_camera(self, uv_depth):
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v
        pts_3d_camera = np.zeros((n,3))
        pts_3d_camera[:,0] = x
        pts_3d_camera[:,1] = y
        pts_3d_camera[:,2] = uv_depth[:,2]
        return pts_3d_camera

    def project_image_to_upright_camerea(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        return self.project_upright_depth_to_upright_camera(pts_3d_upright_depth)

def flip_axis_to_camera(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (N,3) array
    '''
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[:,1] *= -1
    return pc2

def flip_axis_to_depth(pc):
    pc2 = np.copy(pc)
    pc2[:,[0,1,2]] = pc2[:,[0,2,1]] # depth X,Y,Z = cam X,Z,-Y
    pc2[:,2] *= -1
    return pc2

def flip_axis_to_camera_cuda(pc):
    ''' Flip X-right,Y-forward,Z-up to X-right,Y-down,Z-forward
        Input and output are both (..., N,3) array
    '''
    pc2 = torch.clone(pc)
    pc2[...,[0,1,2]] = pc2[...,[0,2,1]] # cam X,Y,Z = depth X,-Z,Y
    pc2[...,1] *= -1
    return pc2

class SUNRGBD_Calibration_cuda(object):
    ''' Calibration matrices and utils
        We define five coordinate system in SUN RGBD dataset

        camera coodinate:
            Z is forward, Y is downward, X is rightward

        depth coordinate:
            Just change axis order and flip up-down axis from camera coord

        upright depth coordinate: tilted depth coordinate by Rtilt such that Z is gravity direction,
            Z is up-axis, Y is forward, X is right-ward

        upright camera coordinate:
            Just change axis order and flip up-down axis from upright depth coordinate

        image coordinate:
            ----> x-axis (u)
           |
           v
            y-axis (v) 

        depth points are stored in upright depth coordinate.
        labels for 3d box (basis, centroid, size) are in upright depth coordinate.
        2d boxes are in image coordinate

        We generate frustum point cloud and 3d box in upright camera coordinate
    '''

    def __init__(self, Rtilt, K):
        self.Rtilt = Rtilt.float()
        self.K = K.float()
        self.f_u = self.K[0,0]
        self.f_v = self.K[1,1]
        self.c_u = self.K[0,2]
        self.c_v = self.K[1,2]
   
    def project_upright_depth_to_camera(self, pc):
        ''' project point cloud from depth coord to camera coordinate
            Input: (..., N, 3) Output: (..., N, 3)
        '''
        # Project upright depth to depth coordinate
        pc2 = self.Rtilt.T @ pc.transpose(-1, -2) # (3,n)
        return flip_axis_to_camera_cuda(pc2.transpose(-1, -2))

    def project_upright_depth_to_image(self, pc):
        ''' Input: (..., N,3) Output: (..., N,2) UV and (..., N,) depth '''
        pc2 = self.project_upright_depth_to_camera(pc)
        uv = pc2 @ self.K.T # (n,3)
        uv[...,0] /= uv[...,2]
        uv[...,1] /= uv[...,2]
        return uv[...,:2], pc2[..., 2]

def box_2d_iou(pred_boxes, gt_box):
    """
    calculate the iou multiple pred_boxes and 1 gt_box (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    if pred_boxes.shape[0] > 0:
        ixmin = np.maximum(pred_boxes[:, 0], gt_box[0])
        ixmax = np.minimum(pred_boxes[:, 2], gt_box[2])
        iymin = np.maximum(pred_boxes[:, 1], gt_box[1])
        iymax = np.minimum(pred_boxes[:, 3], gt_box[3])

        iw = np.maximum(ixmax - ixmin + 1., 0.)
        ih = np.maximum(iymax - iymin + 1., 0.)

    # 2.calculate the area of inters
        inters = iw * ih

    # 3.calculate the area of union
        uni = ((pred_boxes[:, 2] - pred_boxes[:, 0] + 1.) * (pred_boxes[:, 3] - pred_boxes[:, 1] + 1.) +
               (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
               inters)

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
        iou = inters / uni
        iou_max = np.max(iou)
        nmax = np.argmax(iou)
        return iou, iou_max, nmax
    
def box_2d_iou_tensor(pred_boxes, gt_box):
    """
    calculate the iou between M pred_boxes and N gt_boxes (the same one)
    pred_boxes: multiple predict  boxes coordinate
    gt_box: ground truth bounding  box coordinate
    return: the max overlaps about pred_boxes and gt_box
    """
    # 1. calculate the inters coordinate
    ixmin = torch.maximum(pred_boxes[:, None, 0], gt_box[None, :, 0])
    iymin = torch.maximum(pred_boxes[:, None, 1], gt_box[None, :, 1])
    ixmax = torch.minimum(pred_boxes[:, None, 2], gt_box[None, :, 2])
    iymax = torch.minimum(pred_boxes[:, None, 3], gt_box[None, :, 3])

    iw = torch.maximum(ixmax - ixmin + 1., torch.zeros_like(ixmax))
    ih = torch.maximum(iymax - iymin + 1., torch.zeros_like(ixmax))

    # 2.calculate the area of inters
    inters = iw * ih # M, N

    # 3.calculate the area of union
    uni = (pred_boxes[:, None, 2] - pred_boxes[:, None, 0] + 1.) * (pred_boxes[:, None, 3] - pred_boxes[:, None, 1] + 1.) + (gt_box[None, :, 2] - gt_box[None, :, 0] + 1.) * (gt_box[None, :, 3] - gt_box[None, :, 1] + 1.) - inters
    uni = torch.maximum(uni, 1e-7 * torch.ones_like(uni))

    # 4.calculate the overlaps and find the max overlap ,the max overlaps index for pred_box
    iou = inters / uni
    iou_max, nmax = torch.max(iou, -1) # M
    return iou, iou_max, nmax

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU