# API Reference

Base URL: `http://localhost:8000/api/v1`
Interactive docs: `http://localhost:8000/docs`

## Endpoints

### Health
`GET /health`
Returns system status and version.

### Traffic Density
`GET /traffic/density?intersection_id=INT_001&limit=1`
Returns latest N traffic readings for an intersection.

`POST /traffic/update`
```json
{
  "intersection_id": "INT_001",
  "vehicle_count": 12,
  "density": 0.045,
  "avg_speed": 45.0,
  "queue_length": 25.5,
  "flow_rate": 0.9,
  "congestion_level": 0.4
}
```

### Predictions
`GET /traffic/predict?intersection_id=INT_001&horizon=30`
Returns GNN-based congestion forecast.

### Network
`GET /traffic/network`
Returns summary of all intersections (latest readings).

### Signal Optimization
`POST /signals/optimize`
```json
{
  "intersection_ids": ["INT_001", "INT_002"],
  "optimization_mode": "global"
}
```

### Video Jobs
`POST /jobs/video` – submit a video for processing
`GET /jobs/{job_id}` – check job status
