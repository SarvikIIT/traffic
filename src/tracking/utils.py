from __future__ import annotations

from typing import List, Tuple

import numpy as np

def iou(box_a: Tuple, box_b: Tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0.0

def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    n, m = len(boxes_a), len(boxes_b)
    mat = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            mat[i, j] = iou(boxes_a[i], boxes_b[j])
    return mat

def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thresh: float = 0.45,
) -> List[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest_boxes = boxes[order[1:]]
        ious = np.array([iou(boxes[i], b) for b in rest_boxes])
        order = order[1:][ious < iou_thresh]
    return keep

def xyxy_to_xywh(box: Tuple) -> Tuple:
    x1, y1, x2, y2 = box
    return (x1, y1, x2 - x1, y2 - y1)

def xywh_to_xyxy(box: Tuple) -> Tuple:
    x, y, w, h = box
    return (x, y, x + w, y + h)

def center_of_box(box: Tuple) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)
