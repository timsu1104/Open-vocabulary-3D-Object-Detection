
import numpy as np
from utils.constants import const_sunrgbd

class ProjectionHelper():
    def __init__(self, depth_min, depth_max, image_dims, cuda=True):
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.image_dims = tuple(image_dims)
        self.cuda = cuda
        self.IGNORE_LABEL = -100
        
        depth_size = (640, 480)  # intrinsic matrix is based on 640x480 depth maps.
        self.resize_scale = (depth_size[0] / image_dims[1], depth_size[1] / image_dims[0])
        assert self.resize_scale == (2, 2)
        
        self.nyu40ids = np.array(
            [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        )
        self.nyu40id2class = {
            nyu40id: i for i, nyu40id in enumerate(list(self.nyu40ids))
        }
    
    def adapt_label_format(self, label, typ: str):
        if typ == 'nyu40':
            return label + 1
        elif typ == 'nyu38':
            return label + 3
        elif typ == 'scannet18':
            return self.nyu40ids[label]
        else:
            raise NotImplementedError(typ + " is not founded as label format")
    
    def project_label(self, semantic_labels, PSEUDO_FLAG):
        """
        Input: nyu40 label
        Output: 0-17, -100 label
        """
        # if not PSEUDO_FLAG:
        #     sem_seg_labels = np.ones_like(semantic_labels) * self.IGNORE_LABEL

        #     for _c in self.nyu40ids:
        #         sem_seg_labels[
        #             semantic_labels == _c
        #         ] = self.nyu40id2class[_c]
        # else:
        #     sem_seg_labels = semantic_labels
        #     sem_seg_labels[semantic_labels >= 18] = self.IGNORE_LABEL
        
        if PSEUDO_FLAG:
            sem_seg_labels = semantic_labels - 3
            sem_seg_labels[sem_seg_labels < 0] = self.IGNORE_LABEL
        else:
            sem_seg_labels = np.ones_like(semantic_labels) * self.IGNORE_LABEL

            for _c in self.nyu40ids:
                sem_seg_labels[
                    semantic_labels == _c
                ] = self.nyu40id2class[_c]
        
        return sem_seg_labels
    
    def project_alignment(self, point_cloud, axis_align_matrix):
        pts = np.ones((point_cloud.shape[0], 4))
        pts[:,0:3] = point_cloud[:,0:3]
        pts = np.dot(pts, axis_align_matrix.transpose()) # Nx4
        aligned_vertices = np.copy(point_cloud)
        aligned_vertices[:,0:3] = pts[:,0:3]
        return aligned_vertices

    def get_edge_mask(self, box):
        if box.shape[0] == 0:
            return box
        mask = \
            (box[:, 0] != 0) * \
            (box[:, 1] != 0) * \
            (box[:, 0] + box[:, 2] != self.image_dims[1]) * \
            (box[:, 1] + box[:, 3] != self.image_dims[0])
        mask = mask.astype('bool')
        return box[mask]

    def depth_to_skeleton(self, ux, uy, depth, intrinsic):
        # 2D to 3D coordinates with depth (used in compute_frustum_bounds)
        x = (ux - intrinsic[0][2]) / self.intrinsic[0][0]
        y = (uy - intrinsic[1][2]) / self.intrinsic[1][1]
        return np.array([depth*x, depth*y, depth])
    
    def depth2xyz(self, u, v, depth, intrinsic):
        """
        u, v, depth: np.array
        """
        # create xyz coordinates from image position
        uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
        xyz = (np.linalg.inv(intrinsic[:3, :3]).dot(uv1_points.T) * depth).T
        return xyz

    def resize_intrinsic(self, intrinsic):
        intrinsic[0] /= self.resize_scale[0]
        intrinsic[1] /= self.resize_scale[1]
        return intrinsic

    def compute_frustum_corners(self, camera_to_world, box, intrinsic):
        """
        Computes the coordinates of the viewing frustum corresponding to one image and given camera parameters

        :param camera_to_world: torch tensor of shape (4, 4)
        :param box: XYWH format, 4
        :return: corner_coords: torch tensor of shape (8, 4)
        """
        # input: camera pose (torch.Size([4, 4]))
        # output: coordinates of the corner points of the viewing frustum of the camera

        corner_points = np.ones((8, 4))
            
        x, y, w, h = box[:4]
        
        u = np.array([x, x+w, x+w, x, x, x+w, x+w, x])
        v = np.array([y, y, y+h, y+h, y, y, y+h, y+h])
        d = np.repeat(np.array([self.depth_min, self.depth_max]), 4)
        corner_points[:, :3] = self.depth2xyz(u, v, d, intrinsic)
        
        # camera to world
        corner_coords = camera_to_world @ corner_points[:, :, None]

        return corner_coords

    def compute_frustum_normals(self, corner_coords: np.ndarray) -> np.ndarray:
        """
        Computes the normal vectors (pointing inwards) to the 6 planes that bound the viewing frustum

        :param corner_coords: torch tensor of shape (8, 4), coordinates of the corner points of the viewing frustum
        :return: normals: torch tensor of shape (6, 3)
        """

        normals = np.zeros((6, 3))

        # compute plane normals
        # front plane
        plane_vec1 = corner_coords[3][:3] - corner_coords[0][:3]
        plane_vec2 = corner_coords[1][:3] - corner_coords[0][:3]
        # plane_vec1 /= np.sum(plane_vec1 ** 2, axis=-1, keepdims=True)
        # plane_vec2 /= np.sum(plane_vec2 ** 2, axis=-1, keepdims=True)
        normals[0] = np.cross(plane_vec1.reshape(-1), plane_vec2.reshape(-1))

        # right side plane
        plane_vec1 = corner_coords[2][:3] - corner_coords[1][:3]
        plane_vec2 = corner_coords[5][:3] - corner_coords[1][:3]
        # plane_vec1 /= np.sum(plane_vec1 ** 2, axis=-1, keepdims=True)
        # plane_vec2 /= np.sum(plane_vec2 ** 2, axis=-1, keepdims=True)
        normals[1] = np.cross(plane_vec1.reshape(-1), plane_vec2.reshape(-1))

        # roof plane
        plane_vec1 = corner_coords[3][:3] - corner_coords[2][:3]
        plane_vec2 = corner_coords[6][:3] - corner_coords[2][:3]
        # plane_vec1 /= np.sum(plane_vec1 ** 2, axis=-1, keepdims=True)
        # plane_vec2 /= np.sum(plane_vec2 ** 2, axis=-1, keepdims=True)
        normals[2] = np.cross(plane_vec1.reshape(-1), plane_vec2.reshape(-1))

        # left side plane
        plane_vec1 = corner_coords[0][:3] - corner_coords[3][:3]
        plane_vec2 = corner_coords[7][:3] - corner_coords[3][:3]
        # plane_vec1 /= np.sum(plane_vec1 ** 2, axis=-1, keepdims=True)
        # plane_vec2 /= np.sum(plane_vec2 ** 2, axis=-1, keepdims=True)
        normals[3] = np.cross(plane_vec1.reshape(-1), plane_vec2.reshape(-1))

        # bottom plane
        plane_vec1 = corner_coords[1][:3] - corner_coords[0][:3]
        plane_vec2 = corner_coords[4][:3] - corner_coords[0][:3]
        # plane_vec1 /= np.sum(plane_vec1 ** 2, axis=-1, keepdims=True)
        # plane_vec2 /= np.sum(plane_vec2 ** 2, axis=-1, keepdims=True)
        normals[4] = np.cross(plane_vec1.reshape(-1), plane_vec2.reshape(-1))

        # back plane
        plane_vec1 = corner_coords[6][:3] - corner_coords[5][:3]
        plane_vec2 = corner_coords[4][:3] - corner_coords[5][:3]
        # plane_vec1 /= np.sum(plane_vec1 ** 2, axis=-1, keepdims=True)
        # plane_vec2 /= np.sum(plane_vec2 ** 2, axis=-1, keepdims=True)
        normals[5] = np.cross(plane_vec1.reshape(-1), plane_vec2.reshape(-1))
        
        normals /= np.sum(normals ** 2, axis=-1, keepdims=True)

        return normals

    def points_in_frustum(self, corner_coords, normals, new_pts, return_mask=False):
        """
        Checks whether new_pts ly in the frustum defined by the coordinates of the corners coner_coords

        :param corner_coords: torch tensor of shape (8, 4), coordinates of the corners of the viewing frustum
        :param normals: torch tensor of shape (6, 3), normal vectors of the 6 planes of the viewing frustum
        :param new_pts: (num_points, 3)
        :param return_mask: if False, returns number of new_points in frustum
        :return: if return_mask=True, returns Boolean mask determining whether point is in frustum or not
        """

        # create vectors from point set to the planes
        point_to_plane1 = (new_pts - corner_coords[2][:3].reshape(-1))
        point_to_plane2 = (new_pts - corner_coords[4][:3].reshape(-1))
        point_to_plane1 /= np.sum(point_to_plane1 ** 2, axis=-1, keepdims=True)
        point_to_plane2 /= np.sum(point_to_plane2 ** 2, axis=-1, keepdims=True)

        # check if the scalar product with the normals is positive
        masks = list()
        # for each normal, create a mask for points that lie on the correct side of the plane
        for k, normal in enumerate(normals):
            if k < 3:
                masks.append((point_to_plane1 @ normal[:, None]) < 0)
            else:
                masks.append((point_to_plane2 @ normal[:, None]) < 0)
        mask = np.ones(point_to_plane1.shape[0]) > 0
        mask = mask

        # create a combined mask, which keeps only the points that lie on the correct side of each plane
        for addMask in masks:
            mask = mask * addMask.squeeze()

        if return_mask:
            return mask
        else:
            return np.sum(mask)
    
    def compute_frustum_box(self, points, depths, camera_to_world, boxes, labels, axis_align_matrix, intrinsic, view):
        """
        Computes correspondances of points to pixels

        :param points: tensor containing all points of the point cloud (num_points, 3)
        :param camera_to_world: camera pose (4, 4)
        :param boxes: 2D proposals (numBox, 4 + 1 + 1)
        :param labels: (num_points, ) or (H, W)
        :param num_points: number of points in one sample point cloud (4096)
        :return: boxes_3d (numBox, 6 + 1 + 1) min-max-score-label
        """
        
        boxes_3d = []

        for box in boxes:
            box_label = int(box[-1])
            
            if view == 'multi':
                # compute viewing frustum
                corner_coords = self.compute_frustum_corners(camera_to_world, box, intrinsic)
                normals = self.compute_frustum_normals(corner_coords)

                # check if points are in viewing frustum and only keep according indices
                mask_frustum_bounds = self.points_in_frustum(corner_coords, normals, points, return_mask=True)
                mask_label_bounds = (labels == box_label)
                
                mask = mask_label_bounds * mask_frustum_bounds
                # print(mask_frustum_bounds.sum(), mask.sum(), box_label, np.unique(labels[mask_frustum_bounds]))
                    
                if mask.sum() == 0: continue
                
                sub_points = points[mask.astype(bool)]
            
            elif view == 'single':
                mask = labels == box_label
                if mask.sum() == 0: continue
                # print(intrinsic)
                v, u = np.indices(self.image_dims)
                sub_points = self.depth2xyz(u.ravel(), v.ravel(), depths.ravel(), intrinsic)
                # print(sub_points.min(axis=0), sub_points.max(axis=0))
                sub_points = self.depth2xyz(u[mask], v[mask], depths[mask], intrinsic)
                # print(sub_points.min(axis=0), sub_points.max(axis=0))
                sub_points = np.matmul(sub_points, camera_to_world[:3, :3].T) + camera_to_world[:3, 3]
            
            sub_cloud = self.project_alignment(sub_points, axis_align_matrix)
            # print(sub_cloud.min(axis=0), sub_cloud.max(axis=0))
            box_3d = np.concatenate([sub_cloud.min(axis=0), sub_cloud.max(axis=0), box[-2:]], -1)
            # print(box_3d)
            # print(box_3d, sub_cloud.shape[0], points.min(0), points.max(0))

            boxes_3d.append(box_3d)

        if len(boxes_3d) == 0:
            return None
        
        boxes_3d = np.stack(boxes_3d, 0)
        return boxes_3d
    
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

    def __init__(self, calib_filepath):
        lines = [line.rstrip() for line in open(calib_filepath)]
        Rtilt = np.array([float(x) for x in lines[0].split(' ')])
        self.Rtilt = np.reshape(Rtilt, (3,3), order='F')
        K = np.array([float(x) for x in lines[1].split(' ')])
        self.K = np.reshape(K, (3,3), order='F')
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
    
    def project_image_to_upright_depth(self, uv_depth):
        pts_3d_camera = self.project_image_to_camera(uv_depth)
        pts_3d_depth = flip_axis_to_depth(pts_3d_camera)
        pts_3d_upright_depth = np.transpose(np.dot(self.Rtilt, np.transpose(pts_3d_depth)))
        return pts_3d_upright_depth

    @staticmethod
    def project_label(semantic_labels, opts=None, IGNORE_LABEL=-100):
        if opts.use_gt or opts.use_lseg:
            sunrgbd37ids = [36, 4, 10, 29, 5, 12, 14, 8, 17, 35, 32, 18, 34, 6, 7, 25, 33]
            # sunrgbd37ids = [33, 4, 5, 36, 6, 17, 0, 24, 35, 14, 7, 32, 3, 12, 0, 10, 18, 0, 34, 0]
            id2class = {
                id: i for i, id in enumerate(sunrgbd37ids)
            }
            sem_seg_labels = np.ones_like(semantic_labels) * IGNORE_LABEL

            for _c in sunrgbd37ids:
                if _c == 0: continue
                sem_seg_labels[
                    semantic_labels == _c
                ] = id2class[_c]
        elif opts.use_plus:
            sem_seg_labels = semantic_labels.copy() - 2
            sem_seg_labels[sem_seg_labels < 0] = IGNORE_LABEL
            # sem_seg_labels[semantic_labels >= const_sunrgbd.num_sem_cls] = IGNORE_LABEL
        else:
            sem_seg_labels = semantic_labels.copy()
        return sem_seg_labels

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

def get_edge_mask(box, image_dims):
        if box.shape[0] == 0:
            return box
        mask = \
            (box[:, 0] != 0) * \
            (box[:, 1] != 0) * \
            (box[:, 0] + box[:, 2] != image_dims[1]) * \
            (box[:, 1] + box[:, 3] != image_dims[0])
        mask = mask.astype('bool')
        return box[mask]
