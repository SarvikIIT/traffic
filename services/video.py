import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection.model import DetectionModel
from src.detection.detector import VehicleDetector
from src.tracking.tracker import MultiObjectTracker
from src.utils.config import load_config
from src.utils.db import DatabaseManager, TrafficReading, VideoJob, get_db
from src.utils.logger import setup_logging, get_logger
from src.utils.video_utils import VideoWriter

import cv2
import numpy as np
from datetime import datetime, timezone

def process_job(job: VideoJob, detector: VehicleDetector,
                tracker: MultiObjectTracker, db: DatabaseManager,
                cfg, log) -> None:
    cap = cv2.VideoCapture(job.input_path)
    if not cap.isOpened():
        with db.session() as s:
            s.query(VideoJob).filter(VideoJob.job_id == job.job_id).update(
                {"status": "failed", "error_message": "Cannot open video",
                 "completed_at": datetime.now(timezone.utc)}
            )
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with db.session() as s:
        s.query(VideoJob).filter(VideoJob.job_id == job.job_id).update(
            {"status": "running", "total_frames": total}
        )

    tracker.reset()
    frame_idx = 0
    batch_readings_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        analysis = detector.process_frame(frame, frame_idx)
        tracks   = tracker.update(analysis.detections)

        if frame_idx % int(fps) == 0:
            density = analysis.density
            congestion = min(1.0, density / 0.08)
            batch_readings_data.append({
                "intersection_id": job.intersection_id or "unknown",
                "camera_id": f"cam_{job.job_id[:8]}",
                "vehicle_count": analysis.vehicle_count,
                "density": density,
                "avg_speed": max(5.0, 60.0 - density * 400.0),
                "queue_length": detector.compute_queue_length(analysis.detections),
                "flow_rate": density * 20.0,
                "congestion_level": congestion,
            })

        frame_idx += 1
        if frame_idx % 300 == 0:
            with db.session() as s:
                for rd in batch_readings_data:
                    s.add(TrafficReading(**rd))
                s.query(VideoJob).filter(VideoJob.job_id == job.job_id).update(
                    {"processed_frames": frame_idx}
                )
            batch_readings_data = []
            log.info(f"Job {job.job_id[:8]} – frame {frame_idx}/{total}")

    cap.release()
    with db.session() as s:
        for rd in batch_readings_data:
            s.add(TrafficReading(**rd))
        s.query(VideoJob).filter(VideoJob.job_id == job.job_id).update(
            {"status": "done", "processed_frames": frame_idx,
             "completed_at": datetime.now(timezone.utc)}
        )
    log.info(f"Job {job.job_id[:8]} complete – {frame_idx} frames processed.")

def run_service():
    cfg = load_config()
    setup_logging(cfg.get("system.log_level", "INFO"), cfg.get("system.log_dir", "logs"))
    log = get_logger("video_processor")
    db  = get_db(f"sqlite:///{cfg.get('database.sqlite_path', 'data/traffic.db')}")
    db.create_tables()

    log.info("Video processor service started. Polling for jobs...")
    det_model = DetectionModel(
        weights=cfg.get("detection.model", "yolov8n"),
        confidence=cfg.get("detection.confidence", 0.5),
        device=cfg.get("detection.device", "auto"),
    )
    detector = VehicleDetector(model=det_model)
    tracker  = MultiObjectTracker()

    while True:
        job = None
        with db.session() as s:
            job = (
                s.query(VideoJob)
                .filter(VideoJob.status == "pending")
                .order_by(VideoJob.created_at.asc())
                .first()
            )
            if job:
                s.expunge(job)

        if job:
            log.info(f"Processing job {job.job_id}: {job.input_path}")
            try:
                process_job(job, detector, tracker, db, cfg, log)
            except Exception as e:
                log.error(f"Job {job.job_id} failed: {e}")
                with db.session() as s:
                    s.query(VideoJob).filter(VideoJob.job_id == job.job_id).update(
                        {"status": "failed", "error_message": str(e),
                         "completed_at": datetime.now(timezone.utc)}
                    )
        else:
            time.sleep(5)

if __name__ == "__main__":
    run_service()
