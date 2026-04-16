import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.model import DetectionModel
from src.detection.detector import VehicleDetector
from src.tracking.tracker import MultiObjectTracker
from src.visualization.detection_viz import (
    draw_detections, draw_tracks, draw_heatmap, draw_stats_overlay
)
from src.utils.video_utils import VideoWriter, get_video_properties
from src.utils.config import load_config
from src.utils.logger import setup_logging, get_logger

def parse_args():
    parser = argparse.ArgumentParser(description="Process traffic video")
    parser.add_argument("--input", type=str, required=True,
                        help="Input video path or camera index")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="yolov8n",
                        help="Detection model (yolov8n/s/m/l or path to .pt)")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show live preview window")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--save-json", action="store_true", default=True)
    parser.add_argument("--heatmap", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log = get_logger("process_video")

    Path(args.output).mkdir(parents=True, exist_ok=True)

    src = int(args.input) if args.input.isdigit() else args.input
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        log.error(f"Cannot open: {args.input}")
        sys.exit(1)

    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log.info(f"Video: {width}×{height} @ {fps:.1f} FPS | {total} frames")

    log.info(f"Loading detector: {args.model}")
    det_model = DetectionModel(
        weights=args.model, device=args.device, confidence=args.conf
    )
    detector = VehicleDetector(model=det_model, frame_width=width, frame_height=height)
    tracker  = MultiObjectTracker(
        max_age=cfg.get("tracking.max_age", 30),
        min_hits=cfg.get("tracking.min_hits", 3),
        high_thresh=args.conf,
    )

    out_stem = Path(args.input).stem if not args.input.isdigit() else "webcam"
    video_out_path = str(Path(args.output) / f"{out_stem}_annotated.mp4")
    writer = VideoWriter(video_out_path, fps=fps, width=width, height=height) \
             if args.save_video else None

    frame_results = []
    total_count_crossing = 0
    prev_time = time.time()
    frame_idx = 0

    log.info("Starting processing...")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if args.max_frames and frame_idx >= args.max_frames:
                break

            analysis = detector.process_frame(frame, frame_id=frame_idx)
            tracks   = tracker.update(analysis.detections)

            curr_time = time.time()
            proc_fps  = 1.0 / max(curr_time - prev_time, 1e-6)
            prev_time = curr_time

            vis = draw_tracks(frame, tracks)
            if args.heatmap and analysis.heatmap is not None:
                vis = draw_heatmap(vis, analysis.heatmap, alpha=0.3)
            vis = draw_stats_overlay(vis, analysis.vehicle_count, analysis.density, proc_fps)

            if writer:
                writer.write(vis)
            if args.show:
                cv2.imshow("Traffic Analysis", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_results.append({
                "frame_id": frame_idx,
                "timestamp_sec": frame_idx / fps,
                "vehicle_count": analysis.vehicle_count,
                "density": round(analysis.density, 4),
                "lane_counts": analysis.lane_counts,
                "track_ids": [t.track_id for t in tracks],
            })

            if frame_idx % 100 == 0:
                log.info(f"  Frame {frame_idx}/{total} | "
                         f"Vehicles: {analysis.vehicle_count} | "
                         f"FPS: {proc_fps:.1f}")
            frame_idx += 1

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    avg_count   = np.mean([r["vehicle_count"] for r in frame_results]) if frame_results else 0
    avg_density = np.mean([r["density"] for r in frame_results]) if frame_results else 0
    log.info(f"Processed {frame_idx} frames | "
             f"Avg vehicles: {avg_count:.1f} | "
             f"Avg density: {avg_density:.4f}")

    if args.save_json:
        summary = {
            "input": args.input,
            "total_frames": frame_idx,
            "fps": fps,
            "avg_vehicle_count": float(avg_count),
            "avg_density": float(avg_density),
            "frames": frame_results,
        }
        json_path = str(Path(args.output) / f"{out_stem}_results.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"Results saved to {json_path}")

    if args.save_video and writer:
        log.info(f"Annotated video saved to {video_out_path}")

    return frame_results

if __name__ == "__main__":
    main()
