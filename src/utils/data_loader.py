from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple

import cv2
import numpy as np

class VideoLoader:

    def __init__(
        self,
        source: str,
        target_fps: Optional[float] = None,
        resize: Optional[Tuple[int, int]] = None,
        max_frames: Optional[int] = None,
    ):
        self.source = source
        self.target_fps = target_fps
        self.resize = resize
        self.max_frames = max_frames

        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise IOError(f"Cannot open video source: {source}")

        self.src_fps: float = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self._frame_skip = max(1, int(self.src_fps / target_fps)) if target_fps else 1

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        frame_idx = 0
        read_count = 0
        while True:
            ret, frame = self._cap.read()
            if not ret:
                break
            read_count += 1
            if read_count % self._frame_skip != 0:
                continue
            if self.resize:
                frame = cv2.resize(frame, self.resize)
            yield frame_idx, frame
            frame_idx += 1
            if self.max_frames and frame_idx >= self.max_frames:
                break
        self._cap.release()

    def __len__(self) -> int:
        if self.max_frames:
            return min(self.max_frames, self.total_frames // self._frame_skip)
        return self.total_frames // self._frame_skip

    def release(self) -> None:
        self._cap.release()

class DatasetScanner:

    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".m4v"}

    def __init__(self, root: str):
        self.root = Path(root)

    def scan(self) -> List[Path]:
        videos = []
        for ext in self.VIDEO_EXTS:
            videos.extend(self.root.rglob(f"*{ext}"))
        return sorted(videos)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.scan())

class AnnotationLoader:

    def __init__(self, annotation_path: str):
        with open(annotation_path, "r") as f:
            self._data = json.load(f)
        self._by_image = {}
        for ann in self._data.get("annotations", []):
            img_id = ann["image_id"]
            self._by_image.setdefault(img_id, []).append(ann)

    def get_annotations(self, image_id: int) -> List[dict]:
        return self._by_image.get(image_id, [])

    @property
    def categories(self) -> List[dict]:
        return self._data.get("categories", [])

    @property
    def images(self) -> List[dict]:
        return self._data.get("images", [])

def load_processed_features(path: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npz":
        data = np.load(p)
        return data[list(data.files)[0]]
    return np.load(p)

def load_graph_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_graph_json(graph: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(graph, f, indent=2)
