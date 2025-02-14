import lap
import torch
import math
import numpy as np
import scipy
from cython_bbox import bbox_overlaps as bbox_ious
from scipy.spatial.distance import cdist
from tracking_utils import kalman_filter
from sklearn.metrics.pairwise import cosine_similarity

def merge_matches(m1, m2, shape):
    O,P,Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1*M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious


def bbox_ciou(atlbrs, btlbrs):
    """
    Compute the Complete Intersection over Union (CIoU) of two bounding boxes.

    :param atlbrs: Bounding box A in (x1, y1, x2, y2) format.
    :param btlbrs: Bounding box B in (x1, y1, x2, y2) format.

    :return: CIoU value as a float.
    """
    # 计算交集面积
    intersection = max(0, min(atlbrs[2], btlbrs[2]) - max(atlbrs[0], btlbrs[0])) * \
                   max(0, min(atlbrs[3], btlbrs[3]) - max(atlbrs[1], btlbrs[1]))

    # 计算两个边界框的面积
    area_a = (atlbrs[2] - atlbrs[0]) * (atlbrs[3] - atlbrs[1])
    area_b = (btlbrs[2] - btlbrs[0]) * (btlbrs[3] - btlbrs[1])

    # 计算并集面积
    union = area_a + area_b - intersection

    # 计算IoU
    iou = intersection / union

    # 计算两个边界框的中心点
    center_a = ((atlbrs[0] + atlbrs[2]) / 2, (atlbrs[1] + atlbrs[3]) / 2)
    center_b = ((btlbrs[0] + btlbrs[2]) / 2, (btlbrs[1] + btlbrs[3]) / 2)

    # 计算两个中心点之间的欧氏距离
    distance = np.linalg.norm(np.array(center_a) - np.array(center_b))

    # 计算两个边界框的宽高比
    ar_a = (atlbrs[2] - atlbrs[0]) / (atlbrs[3] - atlbrs[1])
    ar_b = (btlbrs[2] - btlbrs[0]) / (btlbrs[3] - btlbrs[1])

    # 计算宽高比差异
    ar_diff = 4 / (np.pi ** 2) * (ar_a - ar_b) ** 2

    # 计算CIoU
    ciou = iou - distance / max(atlbrs[2] - atlbrs[0], atlbrs[3] - atlbrs[1], btlbrs[2] - btlbrs[0],
                                btlbrs[3] - btlbrs[1]) - ar_diff

    return ciou


def ciou(atlbrs, btlbrs):
    """
    Compute the cost based on CIoU.

    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype ciou: np.ndarray
    """
    ciou = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ciou.size == 0:
        return ciou

        # 将输入列表/数组转换为NumPy数组（如果需要）
    atlbrs_np = np.asarray(atlbrs, dtype=np.float)
    btlbrs_np = np.asarray(btlbrs, dtype=np.float)

    # 计算每对边界框之间的DIoU值
    for i in range(len(atlbrs_np)):
        for j in range(len(btlbrs_np)):
            ciou[i, j] = bbox_diou(atlbrs_np[i], btlbrs_np[j])

    return ciou



def bbox_diou(atlbrs, btlbrs):
    """
    Compute the Distance Intersection over Union (DIoU) of two bounding boxes.

    :param atlbrs: Bounding box A in (x1, y1, x2, y2) format.
    :param btlbrs: Bounding box B in (x1, y1, x2, y2) format.

    :return: DIoU value as a float.
    """
    # 计算交集面积
    intersection = max(0, min(atlbrs[2], btlbrs[2]) - max(atlbrs[0], btlbrs[0])) * \
                   max(0, min(atlbrs[3], btlbrs[3]) - max(atlbrs[1], btlbrs[1]))

    # 计算两个边界框的面积
    area_a = (atlbrs[2] - atlbrs[0]) * (atlbrs[3] - atlbrs[1])
    area_b = (btlbrs[2] - btlbrs[0]) * (btlbrs[3] - btlbrs[1])

    # 计算并集面积
    union = area_a + area_b - intersection

    # 计算IoU
    iou = intersection / union

    # 计算两个边界框的中心点
    center_a = ((atlbrs[0] + atlbrs[2]) / 2, (atlbrs[1] + atlbrs[3]) / 2)
    center_b = ((btlbrs[0] + btlbrs[2]) / 2, (btlbrs[1] + btlbrs[3]) / 2)

    # 计算两个中心点之间的欧氏距离
    distance = np.linalg.norm(np.array(center_a) - np.array(center_b))

    # 计算DIoU
    diou = iou - distance / max(atlbrs[2] - atlbrs[0], atlbrs[3] - atlbrs[1], btlbrs[2] - btlbrs[0], btlbrs[3] - btlbrs[1])

    return diou


def diou(atlbrs, btlbrs):
    """
    Compute the cost based on DIoU.

    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype diou: np.ndarray
    """
    diou = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if diou.size == 0:
        return diou

        # 将输入列表/数组转换为NumPy数组（如果需要）
    atlbrs_np = np.asarray(atlbrs, dtype=np.float)
    btlbrs_np = np.asarray(btlbrs, dtype=np.float)

    # 计算每对边界框之间的DIoU值
    for i in range(len(atlbrs_np)):
        for j in range(len(btlbrs_np)):
            diou[i, j] = bbox_diou(atlbrs_np[i], btlbrs_np[j])

    return diou


def bbox_giou(atlbrs, btlbrs):
    """
    Compute the Generalized Intersection over Union (GIoU) of two bounding boxes.

    :param atlbrs: Bounding box A in (x1, y1, x2, y2) format.
    :param btlbrs: Bounding box B in (x1, y1, x2, y2) format.

    :return: GIoU value as a float.
    """
    # 计算交集面积
    intersection = max(0, min(atlbrs[2], btlbrs[2]) - max(atlbrs[0], btlbrs[0])) * \
                   max(0, min(atlbrs[3], btlbrs[3]) - max(atlbrs[1], btlbrs[1]))

    # 计算两个边界框的面积
    area_a = (atlbrs[2] - atlbrs[0]) * (atlbrs[3] - atlbrs[1])
    area_b = (btlbrs[2] - btlbrs[0]) * (btlbrs[3] - btlbrs[1])

    # 计算并集面积
    union = area_a + area_b - intersection

    # 计算IoU
    iou = intersection / union

    # 计算两个边界框的中心点
    center_a = ((atlbrs[0] + atlbrs[2]) / 2, (atlbrs[1] + atlbrs[3]) / 2)
    center_b = ((btlbrs[0] + btlbrs[2]) / 2, (btlbrs[1] + btlbrs[3]) / 2)

    # 计算两个中心点之间的欧氏距离
    distance = np.linalg.norm(np.array(center_a) - np.array(center_b))

    # 计算封闭边界框的坐标
    x1 = min(atlbrs[0], btlbrs[0])
    y1 = min(atlbrs[1], btlbrs[1])
    x2 = max(atlbrs[2], btlbrs[2])
    y2 = max(atlbrs[3], btlbrs[3])

    # 计算封闭边界框的面积
    enclosing_area = (x2 - x1) * (y2 - y1)

    # 计算GIoU
    giou = iou - (enclosing_area - union) / enclosing_area

    return giou


def giou(atlbrs, btlbrs):
    """
    Compute the cost based on GIoU.

    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype giou: np.ndarray
    """
    giou = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if giou.size == 0:
        return giou

        # 将输入列表/数组转换为NumPy数组（如果需要）
    atlbrs_np = np.asarray(atlbrs, dtype=np.float)
    btlbrs_np = np.asarray(btlbrs, dtype=np.float)

    # 计算每对边界框之间的GIoU值
    for i in range(len(atlbrs_np)):
        for j in range(len(btlbrs_np)):
            giou[i, j] = bbox_giou(atlbrs_np[i], btlbrs_np[j])

    return giou



def bbox_eiou(atlbrs, btlbrs):
    """
    Compute the Extended Intersection over Union (EIOU) of two bounding boxes.

    :param atlbrs: Bounding box A in (x1, y1, x2, y2) format.
    :param btlbrs: Bounding box B in (x1, y1, x2, y2) format.

    :return: EIOU value as a float.
    """
    # 计算交集面积
    intersection = max(0, min(atlbrs[2], btlbrs[2]) - max(atlbrs[0], btlbrs[0])) * \
                   max(0, min(atlbrs[3], btlbrs[3]) - max(atlbrs[1], btlbrs[1]))

    # 计算两个边界框的面积
    area_a = (atlbrs[2] - atlbrs[0]) * (atlbrs[3] - atlbrs[1])
    area_b = (btlbrs[2] - btlbrs[0]) * (btlbrs[3] - btlbrs[1])

    # 计算并集面积
    union = area_a + area_b - intersection

    # 计算IoU
    iou = intersection / union

    # 计算两个边界框的中心点
    center_a = ((atlbrs[0] + atlbrs[2]) / 2, (atlbrs[1] + atlbrs[3]) / 2)
    center_b = ((btlbrs[0] + btlbrs[2]) / 2, (btlbrs[1] + btlbrs[3]) / 2)

    # 计算两个中心点之间的欧氏距离
    distance = np.linalg.norm(np.array(center_a) - np.array(center_b))

    # 计算长宽比
    aspect_ratio_a = (atlbrs[2] - atlbrs[0]) / (atlbrs[3] - atlbrs[1])
    aspect_ratio_b = (btlbrs[2] - btlbrs[0]) / (btlbrs[3] - btlbrs[1])

    # 计算长宽比的差异
    aspect_ratio_diff = abs(aspect_ratio_a - aspect_ratio_b) / (aspect_ratio_a + aspect_ratio_b)

    # 计算EIOU
    eiou = iou - distance / max(atlbrs[2] - atlbrs[0], atlbrs[3] - atlbrs[1], btlbrs[2] - btlbrs[0],
                                btlbrs[3] - btlbrs[1]) - aspect_ratio_diff

    return eiou


def eiou(atlbrs, btlbrs):
    """
    Compute the cost based on EIOU.

    :type atlbrs: list[tlbr] | np.ndarray
    :type btlbrs: list[tlbr] | np.ndarray

    :rtype eiou: np.ndarray
    """
    eiou = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if eiou.size == 0:
        return eiou

        # 将输入列表/数组转换为NumPy数组（如果需要）
    atlbrs_np = np.asarray(atlbrs, dtype=np.float)
    btlbrs_np = np.asarray(btlbrs, dtype=np.float)

    # 计算每对边界框之间的EIOU值
    for i in range(len(atlbrs_np)):
        for j in range(len(btlbrs_np)):
            eiou[i, j] = bbox_eiou(atlbrs_np[i], btlbrs_np[j])

    return eiou

def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = eiou(atlbrs, btlbrs)
    # _ious = diou(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    #for i, track in enumerate(tracks):
        #cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # Nomalized features // 计算新检测目标和tracked_tracker的cosine距离

    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """马氏距离"""
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf

    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, cosine_similarities, only_position=False,  lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        # cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
        similarity_weight = cosine_similarities[row]
        motion_importance = 1 - similarity_weight
        # cost_matrix[row] = (lambda_ - 0.1 * similarity_weight) * cost_matrix[row] + (1 - lambda_ - 0.1 *  motion_importance ) * gating_distance
        cost_matrix[row] = (lambda_ - 0.1 * motion_importance) * cost_matrix[row] + (
                1 - lambda_ - 0.1 * similarity_weight) * gating_distance
    return cost_matrix



def iou_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.7):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance

    return cost_matrix
'''
def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    #fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost
'''

def fuses_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix

        # 计算外观特征相似度 reid_sim
    reid_sim = 1 - cost_matrix

    # 计算IOU相似度 iou_sim
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist

    # 定义权重，可以根据实际情况调整
    reid_weight = 0.1 # reid_sim的权重
    iou_weight = 0.9  # iou_sim的权重

    # 计算加权和形式的融合相似度 fuse_sim
    fuse_sim = reid_weight * reid_sim + iou_weight * iou_sim
    fuse_cost = 1 - fuse_sim

    return fuse_cost

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost

def compute_consecutive_frame_similarity(tracks, detections):

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    cur_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    # for i, track in enumerate(tracks):
    # cost_matrix[i, :] = np.maximum(0.0, cdist(track.smooth_feat.reshape(1,-1), det_features, metric))
    pre_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)
    # cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    cosine_distances = np.maximum(0.0, cdist(pre_features, cur_features, metric='cosine'))
    cosine_similarities = 1.0 - cosine_distances


    return cosine_similarities
