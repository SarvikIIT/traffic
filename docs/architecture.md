# System Architecture

## Overview

The Traffic Digital Twin processes live or recorded video feeds through a four-stage ML pipeline.

```
Video Feeds → Detection → Graph Construction → GNN Prediction → RL Optimization → Dashboard
```

## Components

### 1. Vehicle Detection (`src/detection/`)
- **DetectionModel** – wraps YOLOv8 (ultralytics); filters COCO classes 2,3,5,7
- **VehicleDetector** – per-frame pipeline: ROI filtering, heatmap, lane counts, density

### 2. Multi-Object Tracking (`src/tracking/`)
- **MultiObjectTracker** – ByteTrack-style IoU matching + Kalman motion model
- Two-round association (high-conf first, then low-conf vs unmatched tracks)

### 3. Traffic Graph (`src/graph/`)
- **TrafficGraphBuilder** – NetworkX DiGraph; nodes=intersections, edges=roads
- **TrafficGraphDataset** – sliding-window PyTorch dataset (T_in, N, F)
- **STGCN** – Spatial-Temporal GCN with GLU temporal convolutions

### 4. Reinforcement Learning (`src/rl/`)
- **TrafficSignalEnv** – Gymnasium env; simulates Poisson vehicle arrivals
- **TrafficSignalAgent** – Stable-Baselines3 PPO/A2C/DQN wrapper
- **Reward** – composite: -Δwait −Δqueue +throughput −switch_penalty

### 5. Services
- **api.py** – FastAPI REST API (uvicorn)
- **dashboard.py** – Streamlit interactive dashboard
- **video_processor.py** – job-queue worker for video analysis
- **traffic_predictor.py** – periodic STGCN inference → DB
- **signal_optimizer.py** – periodic RL-based signal recommendations → DB

### 6. Persistence (`src/utils/db.py`)
- SQLAlchemy ORM; SQLite by default, PostgreSQL for production
- Tables: `traffic_readings`, `signal_states`, `traffic_predictions`, `video_jobs`

## Data Flow

```
VideoCapture → VehicleDetector → MultiObjectTracker
                     ↓
             TrafficReading (DB)
                     ↓
             TrafficGraphBuilder.update_intersection()
                     ↓
             STGCN.predict()  →  TrafficPrediction (DB)
                     ↓
             TrafficSignalAgent.predict()  →  SignalState (DB)
                     ↓
             FastAPI / Streamlit Dashboard
```
