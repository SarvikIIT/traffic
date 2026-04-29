from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..detection.model import Detection
from .utils import iou_matrix

class KalmanBoxTracker:

    _count = 0

    def __init__(self, bbox: Tuple[float, float, float, float]):
        KalmanBoxTracker._count += 1
        self.id = KalmanBoxTracker._count
        self.age = 0
        self.hits = 0
        self.hit_streak = 0
        self.time_since_update = 0

        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        self.x = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float64)
        self._last_box = bbox

    def predict(self) -> Tuple[float, float, float, float]:
        self.x[:4] += self.x[4:]
        self.age += 1
        self.time_since_update += 1
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

    def update(self, bbox: Tuple[float, float, float, float]) -> None:
        x1, y1, x2, y2 = bbox
        new_cx = (x1 + x2) / 2
        new_cy = (y1 + y2) / 2
        new_w = x2 - x1
        new_h = y2 - y1
        self.x[4:] = 0.4 * self.x[4:] + 0.6 * (
            np.array([new_cx, new_cy, new_w, new_h]) - self.x[:4]
        )
        self.x[:4] = np.array([new_cx, new_cy, new_w, new_h])
        self._last_box = bbox
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def get_state(self) -> Tuple[float, float, float, float]:
        cx, cy, w, h = self.x[:4]
        return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]
    class_id: int
    class_name: str
    confidence: float
    age: int = 0
    hits: int = 0
    trajectory: List[Tuple[float, float]] = field(default_factory=list)

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

class MultiObjectTracker:

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        high_thresh: float = 0.6,
        low_thresh: float = 0.1,
        match_thresh: float = 0.8,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.match_thresh = match_thresh
        self._trackers: List[KalmanBoxTracker] = []
        self._class_map: Dict[int, Tuple[int, str]] = {}
        self._conf_map: Dict[int, float] = {}
        self._trajectory_map: Dict[int, List[Tuple[float, float]]] = {}

    def reset(self) -> None:
        self._trackers.clear()
        self._class_map.clear()
        self._conf_map.clear()
        self._trajectory_map.clear()
        KalmanBoxTracker._count = 0

    def update(self, detections: List[Detection]) -> List[Track]:
        predicted_boxes = []
        to_del = []
        for i, t in enumerate(self._trackers):
            pred = t.predict()
            predicted_boxes.append(pred)
            if np.any(np.isnan(pred)):
                to_del.append(i)
        for i in reversed(to_del):
            self._trackers.pop(i)
            predicted_boxes.pop(i)

        high_dets = [d for d in detections if d.confidence >= self.high_thresh]
        low_dets = [
            d for d in detections
            if self.low_thresh <= d.confidence < self.high_thresh
        ]

        matched_t, matched_d = set(), set()
        if self._trackers and high_dets:
            det_boxes = np.array([d.bbox for d in high_dets])
            tr_boxes = np.array(predicted_boxes)
            iou_mat = iou_matrix(tr_boxes, det_boxes)
            for t_idx, d_idx in self._greedy_match(iou_mat, self.match_thresh):
                self._trackers[t_idx].update(high_dets[d_idx].bbox)
                self._conf_map[self._trackers[t_idx].id] = high_dets[d_idx].confidence
                matched_t.add(t_idx)
                matched_d.add(d_idx)

        unmatched_t = [i for i in range(len(self._trackers)) if i not in matched_t]
        if unmatched_t and low_dets:
            tr_boxes = np.array([predicted_boxes[i] for i in unmatched_t])
            det_boxes = np.array([d.bbox for d in low_dets])
            iou_mat = iou_matrix(tr_boxes, det_boxes)
            for local_ti, d_idx in self._greedy_match(iou_mat, self.iou_threshold):
                t_idx = unmatched_t[local_ti]
                self._trackers[t_idx].update(low_dets[d_idx].bbox)
                matched_t.add(t_idx)

        for d_idx, det in enumerate(high_dets):
            if d_idx not in matched_d:
                kt = KalmanBoxTracker(det.bbox)
                self._trackers.append(kt)
                self._class_map[kt.id] = (det.class_id, det.class_name)
                self._conf_map[kt.id] = det.confidence

        active_trackers = []
        for t in self._trackers:
            if t.time_since_update > self.max_age:
                continue
            if t.time_since_update > 0:
                t.hit_streak = 0
            active_trackers.append(t)
        self._trackers = active_trackers

        active_ids = {t.id for t in self._trackers}
        stale_ids = [tid for tid in self._trajectory_map if tid not in active_ids]
        for tid in stale_ids:
            del self._trajectory_map[tid]

        output: List[Track] = []
        for t in self._trackers:
            if t.hits < self.min_hits and t.time_since_update > 0:
                continue
            cls_id, cls_name = self._class_map.get(t.id, (2, "car"))
            bbox = t.get_state()
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if t.id not in self._trajectory_map:
                self._trajectory_map[t.id] = []
            self._trajectory_map[t.id].append(center)
            tr = Track(
                track_id=t.id,
                bbox=bbox,
                class_id=cls_id,
                class_name=cls_name,
                confidence=self._conf_map.get(t.id, 0.0),
                age=t.age,
                hits=t.hits,
                trajectory=list(self._trajectory_map[t.id]),
            )
            output.append(tr)
        return output

    @staticmethod
    def _greedy_match(
        iou_mat: np.ndarray, threshold: float
    ) -> List[Tuple[int, int]]:
        matches = []
        used_rows, used_cols = set(), set()
        n_rows, n_cols = iou_mat.shape
        pairs = sorted(
            [(iou_mat[r, c], r, c) for r in range(n_rows) for c in range(n_cols)],
            reverse=True,
        )
        for score, r, c in pairs:
            if score < threshold:
                break
            if r not in used_rows and c not in used_cols:
                matches.append((r, c))
                used_rows.add(r)
                used_cols.add(c)
        return matches

    def count_crossings(
        self,
        tracks: List[Track],
        line_y: float,
        direction: str = "down",
    ) -> int:
        count = 0
        for t in tracks:
            traj = t.trajectory
            if len(traj) < 2:
                continue
            prev_y = traj[-2][1]
            curr_y = traj[-1][1]
            if direction == "down" and prev_y < line_y <= curr_y:
                count += 1
            elif direction == "up" and prev_y > line_y >= curr_y:
                count += 1
        return count
