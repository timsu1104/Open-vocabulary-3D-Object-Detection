import numpy as np
from sklearn.decomposition import PCA

from utils.evaluation.eval_det_obb import get_iou_obb

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

def get_oriented_box_pca(pc):
    """
    generate an oriented box on pc. 
    
    pc (N, 3) in upright depth coords
    """
    center = pc.mean(0)
    pc -= center
    
    topview = pc[:, :2]
    ori = PCA(n_components=1).fit(topview).components_[0] # 2
    ori /= np.linalg.norm(ori, 2)
    heading_angle = -np.arctan2(ori[1], ori[0])
    
    R = rotz(heading_angle) # real2canonical
    R_r = rotz(-heading_angle) # canonical2real
    pc_rot = pc @ R.T
    size = pc_rot.max(0) - pc_rot.min(0)
    center_offset = (pc_rot.max(0) + pc_rot.min(0)) / 2
    center += center_offset @ R_r.T
    
    return np.concatenate([center, size, heading_angle[None]], 0)

def box_3d_iou(box_q, box_k, typ='vv', eps=1e-5):
    """
    3d iou between axis aligned boxes
    box_q: 6, 
    box_k: B, 6
    box: xyz xyz
    
    return: iou: B, 
    """
    
    box_q = box_q[None, :]
    
    if typ == 'vv':
        x1q = box_q[:,0]
        y1q = box_q[:,1]
        z1q = box_q[:,2]
        x2q = box_q[:,3]
        y2q = box_q[:,4]
        z2q = box_q[:,5]
        x1k = box_k[:,0]
        y1k = box_k[:,1]
        z1k = box_k[:,2]
        x2k = box_k[:,3]
        y2k = box_k[:,4]
        z2k = box_k[:,5]
    elif typ == 'cs':
        x1q = box_q[:,0] - box_q[:,3] / 2
        y1q = box_q[:,1] - box_q[:,4] / 2
        z1q = box_q[:,2] - box_q[:,5] / 2
        x2q = box_q[:,0] + box_q[:,3] / 2
        y2q = box_q[:,1] + box_q[:,4] / 2
        z2q = box_q[:,2] + box_q[:,5] / 2
        x1k = box_k[:,0] - box_k[:,3] / 2
        y1k = box_k[:,1] - box_k[:,4] / 2
        z1k = box_k[:,2] - box_k[:,5] / 2
        x2k = box_k[:,0] + box_k[:,3] / 2
        y2k = box_k[:,1] + box_k[:,4] / 2
        z2k = box_k[:,2] + box_k[:,5] / 2
        

    box_q_volume = (x2q-x1q) * (y2q-y1q) * (z2q-z1q)
    box_k_volume = (x2k-x1k) * (y2k-y1k) * (z2k-z1k)

    xi = np.maximum(x1q, x1k)
    yi = np.maximum(y1q, y1k)
    zi = np.maximum(z1q, z1k)
    corner_xi = np.minimum(x2q, x2k)
    corner_yi = np.minimum(y2q, y2k)
    corner_zi = np.minimum(z2q, z2k)

    intersection = np.maximum(corner_xi - xi, 0) * np.maximum(corner_yi - yi, 0) * np.maximum(corner_zi - zi, 0)

    iou = intersection / (box_q_volume + box_k_volume - intersection + eps)

    return iou


def nms_3d_faster(boxes, overlap_threshold, old_type=False, eps=1e-8, use_size=False, use_size_score=False, class_wise=False, size_typ=None, lhs=False):
    """
    GSS
    """
        
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    z1 = boxes[:,2]
    x2 = boxes[:,3]
    y2 = boxes[:,4]
    z2 = boxes[:,5]
    score = boxes[:,6]
    label = boxes[:,7]
    volume = (x2-x1)*(y2-y1)*(z2-z1) + eps
    
    assert size_typ in [None, 'Volume', 'Area']
    if size_typ is not None:
        size = boxes[:, 8] if size_typ == 'Volume' else boxes[:, 9]
        if use_size:
            score = size
        elif use_size_score: # geo, score, label, volume
            score *= size

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[I[:last-1]])
        yy1 = np.maximum(y1[i], y1[I[:last-1]])
        zz1 = np.maximum(z1[i], z1[I[:last-1]])
        xx2 = np.minimum(x2[i], x2[I[:last-1]])
        yy2 = np.minimum(y2[i], y2[I[:last-1]])
        zz2 = np.minimum(z2[i], z2[I[:last-1]])
        if class_wise:
            cls1 = label[i]
            cls2 = label[I[:last-1]]

        l = np.maximum(0, xx2-xx1)
        w = np.maximum(0, yy2-yy1)
        h = np.maximum(0, zz2-zz1)

        if old_type:
            o = (l*w*h)/volume[I[:last-1]]
        else:
            inter = l*w*h
            o = inter / (volume[i] + volume[I[:last-1]] - inter)
        if class_wise:
            o = o * (cls1==cls2)
            
        inds = np.where(o>overlap_threshold)[0]
        if lhs:
            len_inds = len(inds)
            for count in range((len_inds) // 2):
                pick.append(I[inds[len_inds - count - 1]])

        I = np.delete(I, np.concatenate(([last-1], inds)))

    return boxes[np.array(pick)]

def nms_3d_faster_obb(boxes, overlap_threshold, use_size=False, use_size_score=False, class_wise=False, size_typ=None, lhs=False):
    """
    GSS
    boxes: (N, 7)
    """
    
    size = boxes[:, 3:6]
    score = boxes[:,-4]
    label = boxes[:,-3]
    
    assert size_typ in [None, 'Volume', 'Area']
    if size_typ is not None:
        size = boxes[:, -2] if size_typ == 'Volume' else boxes[:, -1]
        if use_size:
            score = size
        elif use_size_score: # geo, score, label, volume
            score *= size

    I = np.argsort(score)
    pick = []
    while (I.size!=0):
        last = I.size
        i = I[-1]
        pick.append(i)
        
        o = np.array([
            get_iou_obb(boxes[i, :7], boxes[I[j], :7]) for j in range(last-1)
        ])# may be slow, check!
            
        if class_wise:
            o = o * (label[i]==label[I[:last-1]])
            
        inds = np.where(o>overlap_threshold)[0]
        if lhs:
            len_inds = len(inds)
            for count in range((len_inds) // 2):
                pick.append(I[inds[len_inds - count - 1]])

        I = np.delete(I, np.concatenate(([last-1], inds)))

    return boxes[np.array(pick)]

def vv2cs(box: np.ndarray):
    """
    switch repr from two vertices to center plus size
    box: (B, c) c>=6
    """
    box[:, 3:6] -= box[:, :3]
    box[:, :3] += box[:, 3:6] / 2
    return box

def cs2vv(box: np.ndarray):
    box[:, :3] -= box[:, 3:6] / 2
    box[:, 3:6] += box[:, :3]
    return box