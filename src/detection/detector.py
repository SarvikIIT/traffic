from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .model import Detection, DetectionModel

@dataclass
class FrameAnalysis:
    frame_id: int
    detections: List[Detection]
    vehicle_count: int = 0
    density: float = 0.0
    lane_counts: Dict[int, int] = field(default_factory=dict)
    heatmap: Optional[np.ndarray] = None

class VehicleDetector:

    def __init__(
        self,
        model: Optional[DetectionModel] = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        num_lanes: int = 4,
        roi: Optional[Tuple[int, int, int, int]] = None,
    ):
        self.model = model or DetectionModel()
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.num_lanes = num_lanes
        self.roi = roi or (0, 0, frame_width, frame_height)

    def process_frame(
        self, frame: np.ndarray, frame_id: int = 0
    ) -> FrameAnalysis:
        detections = self.model.predict(frame)
        detections = self._apply_roi_filter(detections)

        h, w = frame.shape[:2]
        heatmap = self._build_heatmap(detections, w, h)
        lane_counts = self._count_lanes(detections, w)
        density = self._estimate_density(detections, w, h)

        return FrameAnalysis(
            frame_id=frame_id,
            detections=detections,
            vehicle_count=len(detections),
            density=density,
            lane_counts=lane_counts,
            heatmap=heatmap,
        )

    def compute_queue_length(
        self,
        detections: List[Detection],
        stop_line_y: Optional[float] = None,
        frame_height: Optional[int] = None,
    ) -> float:
        fh = frame_height or self.frame_height
        sl_y = stop_line_y if stop_line_y is not None else fh * 0.8
        vehicles_in_queue = [
            d for d in detections if d.center[1] <= sl_y
        ]
        if not vehicles_in_queue:
            return 0.0
        min_y = min(d.center[1] for d in vehicles_in_queue)
        return float(sl_y - min_y)

    def _apply_roi_filter(self, detections: List[Detection]) -> List[Detection]:
        x1r, y1r, x2r, y2r = self.roi
        return [
            d for d in detections
            if x1r <= d.center[0] <= x2r and y1r <= d.center[1] <= y2r
        ]

    def _build_heatmap(
        self, detections: List[Detection], width: int, height: int, sigma: int = 30
    ) -> np.ndarray:
        heatmap = np.zeros((height, width), dtype=np.float32)
        for d in detections:
            cx, cy = int(d.center[0]), int(d.center[1])
            x1 = max(0, cx - sigma)
            x2 = min(width, cx + sigma)
            y1 = max(0, cy - sigma)
            y2 = min(height, cy + sigma)
            heatmap[y1:y2, x1:x2] += d.confidence
        return heatmap

    def _count_lanes(
        self, detections: List[Detection], frame_width: int
    ) -> Dict[int, int]:
        lane_width = frame_width / self.num_lanes
        counts: Dict[int, int] = {i: 0 for i in range(self.num_lanes)}
        for d in detections:
            lane_idx = min(int(d.center[0] / lane_width), self.num_lanes - 1)
            counts[lane_idx] += 1
        return counts

    @staticmethod
    def _estimate_density(
        detections: List[Detection], width: int, height: int
    ) -> float:
        frame_area = width * height
        if frame_area == 0:
            return 0.0
        return len(detections) / (frame_area / 10_000)
