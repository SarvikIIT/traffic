from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..detection.model import Detection
from ..tracking.tracker import Track

CLASS_COLOURS = {
    "car":        (0, 200, 0),
    "truck":      (0, 0, 220),
    "bus":        (220, 100, 0),
    "motorcycle": (220, 0, 220),
}
DEFAULT_COLOUR = (200, 200, 0)
TRACK_COLOUR   = (0, 255, 255)

def draw_detections(
    frame: np.ndarray,
    detections: List[Detection],
    show_label: bool = True,
    show_confidence: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    out = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        colour = CLASS_COLOURS.get(det.class_name, DEFAULT_COLOUR)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)
        if show_label:
            label = det.class_name
            if show_confidence:
                label += f" {det.confidence:.2f}"
            cv2.putText(
                out, label, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )
    return out

def draw_tracks(
    frame: np.ndarray,
    tracks: List[Track],
    show_id: bool = True,
    draw_trajectory: bool = True,
    thickness: int = 2,
) -> np.ndarray:
    out = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = (int(v) for v in track.bbox)
        colour = CLASS_COLOURS.get(track.class_name, TRACK_COLOUR)
        cv2.rectangle(out, (x1, y1), (x2, y2), colour, thickness)

        if show_id:
            cv2.putText(
                out, f"ID:{track.track_id}",
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA,
            )

        if draw_trajectory and len(track.trajectory) > 1:
            pts = [(int(p[0]), int(p[1])) for p in track.trajectory[-20:]]
            for i in range(1, len(pts)):
                alpha = i / len(pts)
                c = tuple(int(v * alpha) for v in colour)
                cv2.line(out, pts[i - 1], pts[i], c, 1)

    return out

def draw_heatmap(
    frame: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    hm_norm = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    hm_colour = cv2.applyColorMap(hm_norm, colormap)
    if hm_colour.shape[:2] != frame.shape[:2]:
        hm_colour = cv2.resize(hm_colour, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 1 - alpha, hm_colour, alpha, 0)

def draw_stats_overlay(
    frame: np.ndarray,
    vehicle_count: int,
    density: float,
    fps: Optional[float] = None,
    colour: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    out = frame.copy()
    h = frame.shape[0]
    lines = [
        f"Vehicles: {vehicle_count}",
        f"Density: {density:.3f}",
    ]
    if fps is not None:
        lines.append(f"FPS: {fps:.1f}")
    for i, text in enumerate(lines):
        cv2.putText(
            out, text, (10, h - 20 - i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA,
        )
    return out
