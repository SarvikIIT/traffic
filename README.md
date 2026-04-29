# City-Scale Traffic Digital Twin Using Machine Learning

A traffic management system for urban intersections that uses computer vision, graph neural networks, and reinforcement learning to build a digital twin of city traffic infrastructure. The system processes traffic video feeds, builds dynamic traffic flow models, and optimizes signal timings to reduce congestion.

Built as a Exploratory project focused on the Varanasi Lanka area (Lanka Chowk → BHU Gate → Assi → Sigra corridor).

## Overview

The pipeline has four main stages:
1. **Vehicle Detection** — YOLOv8 processes traffic camera feeds to detect vehicles and measure density
2. **Traffic Graph** — Intersections and roads are modeled as a directed graph using NetworkX
3. **Congestion Prediction** — A Spatial-Temporal GCN (STGCN) predicts traffic flow across intersections
4. **Signal Optimization** — A PPO reinforcement learning agent recommends signal timing adjustments

All results are exposed via a Streamlit dashboard.

## System Architecture

```
Traffic Cameras → YOLOv8 Detection → Kalman Tracker
                         ↓
                 TrafficReading (SQLite DB)
                         ↓
                 TrafficGraphBuilder
                         ↓
            STGCN Prediction  →  TrafficPrediction (DB)
                         ↓
            PPO RL Agent      →  SignalState (DB)
                         ↓
                Streamlit Dashboard
```

## Tech Stack

| Area | Libraries |
|------|-----------|
| Detection | YOLOv8 (ultralytics), OpenCV |
| Tracking | Kalman filter + ByteTrack-style IoU matching |
| Graph | NetworkX, PyTorch Geometric |
| Prediction | PyTorch — custom STGCN |
| RL | Stable-Baselines3 (PPO/A2C/DQN), Gymnasium |
| Backend | SQLAlchemy, SQLite |
| Dashboard | Streamlit, Plotly |

## Installation

```bash
git clone https://github.com/SarvikIIT/traffic.git
cd traffic

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

pip install -r requirements.txt
```

Copy the config template and edit settings if needed:
```bash
cp config/config.example.yaml config/config.yaml
```

## Quick Start (Demo Mode)

The fastest way to see everything running:

```bash
# Initialise DB + generate a 3×3 grid graph + start API and dashboard
python app.py --demo
```
## Usage

### Running the full pipeline

```bash
python main.py --init-db --generate-graph
```

Start the dashboard only:
```bash
python main.py --services dashboard
```

### Process a video file

```bash
python inference/process.py --input data/video/sample.mp4 --output results/
```

### Run traffic prediction

```bash
python inference/predict.py --demo
python inference/predict.py --graph data/graphs/city_graph.json --model models/prediction/stgcn_best.pt --horizon 30
```

### Database setup

```bash
python scripts/setup_db.py --seed --nodes 9
```

### Generate city graph

```bash
# Grid graph
python scripts/make_graph.py --rows 5 --cols 5

# Varanasi Lanka-area graph (real intersections)
python scripts/varanasi_graph.py
```

### Delhi dataset scripts

```bash
# Convert Delhi CSV data → database readings
python scripts/convert_delhi.py

# Generate Delhi intersection graph
python scripts/delhi_graph.py
```

## Training Models

**Train the GNN prediction model:**
```bash
python train/gnn.py --synthetic --nodes 25 --epochs 200
# or with real features
python train/gnn.py --graph data/graphs/city_graph.json --epochs 200
```

**Train the RL signal agent:**
```bash
python train/rl.py --timesteps 100000 --eval
```

**Train vehicle detector:**
```bash
python train/detector.py --epochs 100
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=html
```

25 tests across graph, detection, and RL environment.

## Project Structure

```
traffic/
├── config/                  # YAML configuration
├── data/
│   ├── graphs/              # City graph JSON files (varanasi + delhi)
│   └── DelhiTrafficDensityDataset/  # (not in repo, shared separately)
├── inference/
│   ├── predict.py           # Run GNN traffic prediction
│   └── process.py           # Process a video file
├── models/                  # Trained model weights (not in repo)
├── scripts/
│   ├── setup_db.py          # Init/seed database
│   ├── make_graph.py        # Generate grid city graph
│   ├── varanasi_graph.py    # Varanasi Lanka-area graph
│   ├── delhi_graph.py       # Delhi intersection graph
│   ├── convert_delhi.py     # Convert Delhi CSV data to DB
│   ├── features.py          # Extract features from video results
│   └── benchmark.py         # Benchmark all three models
├── services/
│   ├── dashboard.py         # Streamlit dashboard
│   ├── video.py             # Video processing worker
│   ├── predictor.py         # STGCN inference service
│   └── optimizer.py         # RL signal optimizer service
├── src/
│   ├── detection/           # YOLOv8 wrapper + detector
│   ├── graph/               # STGCN (stgcn.py), dataset, graph builder
│   ├── rl/                  # PPO env (env.py), agent, reward
│   ├── tracking/            # Kalman tracker
│   ├── utils/               # DB, config, logger, video utils
│   └── visualization/       # viz.py, mapview.py
├── tests/                   # pytest test suite (24 tests)
├── train/
│   ├── gnn.py               # Train STGCN
│   ├── rl.py                # Train PPO agent
│   └── detector.py          # Train YOLOv8
├── app.py                   # Quick-start (demo mode)
└── main.py                  # Full pipeline orchestrator
```
## Model Results

| Model | Metric | Value |
|-------|--------|-------|
| STGCN (15-min) | MAE | 0.33 |
| STGCN (30-min) | RMSE | 0.55 |
| PPO RL Agent | Wait time reduction | 33.9% |
| YOLOv8n | Avg FPS (CPU) | ~12 |

## License

MIT License — see [LICENSE](LICENSE) for details.

## Contact

- **GitHub**: [SarvikIIT](https://github.com/SarvikIIT)
- **Email**: sarvik1807@gmail.com
- **Issues**: https://github.com/SarvikIIT/traffic/issues
