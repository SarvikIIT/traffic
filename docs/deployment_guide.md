# Deployment Guide

## Local Development

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Initialise database with demo data
python scripts/init_database.py --seed --nodes 9

# 4. Generate city graph
python scripts/generate_graph.py --rows 3 --cols 3

# 5. Start all services
python main.py --init-db

# Or quick demo mode:
python app.py --demo
```

## Training Models

```bash
# Train vehicle detector (uses COCO128 demo by default)
python train/train_detector.py --epochs 100

# Train GNN (synthetic data)
python train/train_gnn.py --synthetic --nodes 25 --epochs 200

# Train RL agent
python train/train_rl_agent.py --timesteps 100000 --eval
```

## Docker Deployment

```bash
cd docker
docker-compose up -d

# Scale video processor workers
docker-compose up -d --scale video_processor=3
```

## Running Tests

```bash
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=html
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TRAFFIC_CONFIG` | Path to config.yaml | `config/config.yaml` |
| `DATABASE_PASSWORD` | PostgreSQL password | (empty) |

## URLs

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Dashboard | http://localhost:8501 |
