"""Integration tests for the FastAPI REST API."""

import pytest
from fastapi.testclient import TestClient

from services.api import app
from src.utils.db import DatabaseManager, TrafficReading, Base


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("db")
    db_url = f"sqlite:///{tmp_path}/test.db"

    # Override startup to use test DB
    import services.api as api_module
    test_db = DatabaseManager(db_url)
    test_db.create_tables()
    api_module._db = test_db

    with TestClient(app) as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────

def test_health_check(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


# ─── Traffic density ─────────────────────────────────────────────────────────

def test_update_reading(client):
    payload = {
        "intersection_id": "INT_001",
        "camera_id": "cam_01",
        "vehicle_count": 12,
        "density": 0.045,
        "avg_speed": 45.0,
        "queue_length": 25.5,
        "flow_rate": 0.9,
        "congestion_level": 0.4,
    }
    r = client.post("/api/v1/traffic/update", json=payload)
    assert r.status_code == 201
    assert r.json()["status"] == "created"


def test_get_density_exists(client):
    # First insert a reading
    client.post("/api/v1/traffic/update", json={
        "intersection_id": "INT_002",
        "vehicle_count": 5,
        "density": 0.02,
    })
    r = client.get("/api/v1/traffic/density?intersection_id=INT_002")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[0]["intersection_id"] == "INT_002"


def test_get_density_missing(client):
    r = client.get("/api/v1/traffic/density?intersection_id=NONEXISTENT")
    assert r.status_code == 404


# ─── Predictions ─────────────────────────────────────────────────────────────

def test_predict_no_data(client):
    # Should return placeholder, not 500
    r = client.get("/api/v1/traffic/predict?intersection_id=INT_999&horizon=30")
    assert r.status_code == 200
    data = r.json()
    assert data["horizon_minutes"] == 30


# ─── Network status ──────────────────────────────────────────────────────────

def test_network_status(client):
    r = client.get("/api/v1/traffic/network")
    assert r.status_code == 200
    data = r.json()
    assert "total_intersections" in data
    assert "intersections" in data


# ─── Signal optimization ─────────────────────────────────────────────────────

def test_optimize_signals_empty_intersections(client):
    r = client.post("/api/v1/signals/optimize", json={
        "intersection_ids": ["INT_999_EMPTY"],
        "optimization_mode": "global",
    })
    assert r.status_code == 200
    assert r.json()["optimized_count"] == 0


def test_optimize_signals_with_data(client):
    # Insert data for intersection
    client.post("/api/v1/traffic/update", json={
        "intersection_id": "INT_OPT",
        "vehicle_count": 20,
        "density": 0.06,
        "congestion_level": 0.5,
    })
    r = client.post("/api/v1/signals/optimize", json={
        "intersection_ids": ["INT_OPT"],
        "optimization_mode": "local",
    })
    assert r.status_code == 200
    data = r.json()
    assert data["optimized_count"] == 1
    rec = data["recommendations"][0]
    assert rec["intersection_id"] == "INT_OPT"
    assert rec["green_duration"] >= 15


# ─── Video jobs ───────────────────────────────────────────────────────────────

def test_submit_video_job(client):
    r = client.post("/api/v1/jobs/video", json={
        "input_path": "/tmp/test_video.mp4",
        "intersection_id": "INT_001",
    })
    assert r.status_code == 201
    data = r.json()
    assert "job_id" in data
    assert data["status"] == "pending"
    return data["job_id"]


def test_get_job_status(client):
    # Submit job first
    r = client.post("/api/v1/jobs/video", json={"input_path": "/tmp/x.mp4"})
    job_id = r.json()["job_id"]
    r2 = client.get(f"/api/v1/jobs/{job_id}")
    assert r2.status_code == 200
    assert r2.json()["job_id"] == job_id


def test_get_job_not_found(client):
    r = client.get("/api/v1/jobs/nonexistent-uuid")
    assert r.status_code == 404
