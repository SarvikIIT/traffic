from __future__ import annotations

import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.utils.config import load_config
from src.utils.db import DatabaseManager, TrafficReading, SignalState, TrafficPrediction, VideoJob
from src.utils.logger import setup_logging, get_logger

_cfg = None
_db: Optional[DatabaseManager] = None
_log = None

@asynccontextmanager
async def lifespan(application: FastAPI):
    global _cfg, _db, _log
    if _db is None:
        _cfg = load_config()
        setup_logging(_cfg.get("system.log_level", "INFO"), _cfg.get("system.log_dir", "logs"))
        _log = get_logger("api")
        db_url = _cfg.get("database.sqlite_path", "data/traffic.db")
        _db = DatabaseManager(f"sqlite:///{db_url}")
        _db.create_tables()
        _log.info("API started – database ready.")
    yield

app = FastAPI(
    title="Traffic Digital Twin API",
    description="City-Scale Traffic Management System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrafficReadingIn(BaseModel):
    intersection_id: str
    camera_id: str = "cam_default"
    vehicle_count: int = 0
    density: float = 0.0
    avg_speed: float = 50.0
    queue_length: float = 0.0
    flow_rate: float = 0.0
    congestion_level: float = 0.0

class SignalOptimizeRequest(BaseModel):
    intersection_ids: List[str]
    optimization_mode: str = Field(default="global", pattern="^(global|local)$")

class VideoJobRequest(BaseModel):
    input_path: str
    intersection_id: Optional[str] = None

class TrafficReadingOut(BaseModel):
    intersection_id: str
    timestamp: str
    vehicle_count: int
    density: float
    avg_speed: float
    queue_length: float
    flow_rate: float
    congestion_level: float

class PredictionOut(BaseModel):
    intersection_id: str
    horizon_minutes: int
    predicted_density: Optional[float]
    predicted_flow: Optional[float]
    predicted_queue: Optional[float]
    confidence: Optional[float]
    created_at: str

@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": "1.0.0",
    }

@app.get("/api/v1/traffic/density")
async def get_traffic_density(
    intersection_id: str = Query(..., description="Intersection identifier"),
    limit: int = Query(default=1, ge=1, le=100),
) -> List[TrafficReadingOut]:
    with _db.session() as s:
        rows = (
            s.query(TrafficReading)
            .filter(TrafficReading.intersection_id == intersection_id)
            .order_by(TrafficReading.timestamp.desc())
            .limit(limit)
            .all()
        )
    if not rows:
        raise HTTPException(status_code=404, detail=f"No data for {intersection_id}")
    return [
        TrafficReadingOut(
            intersection_id=r.intersection_id,
            timestamp=r.timestamp.isoformat(),
            vehicle_count=r.vehicle_count,
            density=r.density,
            avg_speed=r.avg_speed,
            queue_length=r.queue_length,
            flow_rate=r.flow_rate,
            congestion_level=r.congestion_level,
        )
        for r in rows
    ]

@app.post("/api/v1/traffic/update", status_code=201)
async def update_traffic_reading(reading: TrafficReadingIn) -> Dict[str, str]:
    with _db.session() as s:
        row = TrafficReading(
            intersection_id=reading.intersection_id,
            camera_id=reading.camera_id,
            vehicle_count=reading.vehicle_count,
            density=reading.density,
            avg_speed=reading.avg_speed,
            queue_length=reading.queue_length,
            flow_rate=reading.flow_rate,
            congestion_level=reading.congestion_level,
        )
        s.add(row)
    return {"status": "created"}

@app.get("/api/v1/traffic/predict")
async def predict_traffic(
    intersection_id: str = Query(...),
    horizon: int = Query(default=30, ge=5, le=120),
) -> PredictionOut:
    with _db.session() as s:
        pred = (
            s.query(TrafficPrediction)
            .filter(
                TrafficPrediction.intersection_id == intersection_id,
                TrafficPrediction.horizon_minutes == horizon,
            )
            .order_by(TrafficPrediction.created_at.desc())
            .first()
        )
    if pred is None:
        return PredictionOut(
            intersection_id=intersection_id,
            horizon_minutes=horizon,
            predicted_density=None,
            predicted_flow=None,
            predicted_queue=None,
            confidence=None,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
    return PredictionOut(
        intersection_id=pred.intersection_id,
        horizon_minutes=pred.horizon_minutes,
        predicted_density=pred.predicted_density,
        predicted_flow=pred.predicted_flow,
        predicted_queue=pred.predicted_queue,
        confidence=pred.confidence,
        created_at=pred.created_at.isoformat(),
    )

@app.get("/api/v1/traffic/network")
async def get_network_status() -> Dict[str, Any]:
    with _db.session() as s:
        from sqlalchemy import func
        latest = (
            s.query(
                TrafficReading.intersection_id,
                func.max(TrafficReading.timestamp).label("max_ts"),
            )
            .group_by(TrafficReading.intersection_id)
            .subquery()
        )
        rows = (
            s.query(TrafficReading)
            .join(
                latest,
                (TrafficReading.intersection_id == latest.c.intersection_id)
                & (TrafficReading.timestamp == latest.c.max_ts),
            )
            .all()
        )
    return {
        "total_intersections": len(rows),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intersections": [
            {
                "id": r.intersection_id,
                "density": r.density,
                "congestion": r.congestion_level,
                "queue": r.queue_length,
            }
            for r in rows
        ],
    }

@app.post("/api/v1/signals/optimize")
async def optimize_signals(req: SignalOptimizeRequest) -> Dict[str, Any]:
    with _db.session() as s:
        recommendations = []
        for iid in req.intersection_ids:
            row = (
                s.query(TrafficReading)
                .filter(TrafficReading.intersection_id == iid)
                .order_by(TrafficReading.timestamp.desc())
                .first()
            )
            if row is None:
                continue
            green_ns = max(15, min(60, 30 + int(row.density * 10)))
            green_ew = max(15, 60 - green_ns)
            recommendations.append({
                "intersection_id": iid,
                "phase": "NS_GREEN",
                "green_duration": green_ns,
                "cycle_length": green_ns + green_ew + 6,
                "optimized": True,
            })
            sig = SignalState(
                intersection_id=iid,
                phase="NS_GREEN",
                green_duration=green_ns,
                cycle_length=green_ns + green_ew + 6,
                optimized=1,
            )
            s.add(sig)
    return {
        "mode": req.optimization_mode,
        "recommendations": recommendations,
        "optimized_count": len(recommendations),
    }

@app.post("/api/v1/jobs/video", status_code=201)
async def submit_video_job(
    request: VideoJobRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, str]:
    job_id = str(uuid.uuid4())
    with _db.session() as s:
        job = VideoJob(
            job_id=job_id,
            input_path=request.input_path,
            intersection_id=request.intersection_id,
            status="pending",
        )
        s.add(job)
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str) -> Dict[str, Any]:
    with _db.session() as s:
        job = s.query(VideoJob).filter(VideoJob.job_id == job_id).first()
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "job_id": job.job_id,
        "status": job.status,
        "input_path": job.input_path,
        "intersection_id": job.intersection_id,
        "created_at": job.created_at.isoformat(),
        "processed_frames": job.processed_frames,
        "total_frames": job.total_frames,
    }
