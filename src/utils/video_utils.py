from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    ret, frame = cap.read()
    return ret, frame if ret else None

def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def normalize_frame(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 255.0

def get_video_properties(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    props = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "duration_sec": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        "codec": int(cap.get(cv2.CAP_PROP_FOURCC)),
    }
    cap.release()
    return props

class VideoWriter:

    def __init__(
        self,
        output_path: str,
        fps: float = 30.0,
        width: int = 1920,
        height: int = 1080,
        codec: str = "mp4v",
    ):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        self._writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()

def extract_frames(
    video_path: str,
    output_dir: str,
    every_n: int = 1,
    max_frames: Optional[int] = None,
    resize: Optional[Tuple[int, int]] = None,
) -> List[str]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    saved = []
    idx = count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n == 0:
            if resize:
                frame = cv2.resize(frame, resize)
            out = str(Path(output_dir) / f"frame_{idx:06d}.jpg")
            cv2.imwrite(out, frame)
            saved.append(out)
            idx += 1
            if max_frames and idx >= max_frames:
                break
        count += 1
    cap.release()
    return saved

def compute_optical_flow(
    prev_gray: np.ndarray, curr_gray: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return mag, ang

def draw_counting_line(
    frame: np.ndarray,
    ratio: float = 0.6,
    color: Tuple[int, int, int] = (0, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    h, w = frame.shape[:2]
    y = int(h * ratio)
    cv2.line(frame, (0, y), (w, y), color, thickness)
    return frame
