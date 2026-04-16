from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

@dataclass
class Detection:
    bbox: Tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]

VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

class DetectionModel:

    def __init__(
        self,
        weights: str = "yolov8n",
        device: str = "auto",
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        vehicle_classes: Optional[List[int]] = None,
        imgsz: int = 640,
    ):
        self.weights = weights
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.vehicle_classes = vehicle_classes or list(VEHICLE_CLASS_IDS.keys())
        self.imgsz = imgsz
        self._model = None
        self._device = device
        self._load()

    def _load(self) -> None:
        try:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
            if self._device != "auto":
                self._model.to(self._device)
        except ImportError:
            raise ImportError(
                "ultralytics is required for detection. "
                "Install with: pip install ultralytics"
            )

    def predict(self, frame: np.ndarray) -> List[Detection]:
        results = self._model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.vehicle_classes,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: List[Detection] = []
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASS_IDS:
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0]),
                        class_id=cls_id,
                        class_name=VEHICLE_CLASS_IDS.get(cls_id, "vehicle"),
                    )
                )
        return detections

    def predict_batch(self, frames: List[np.ndarray]) -> List[List[Detection]]:
        return [self.predict(f) for f in frames]
