import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.builder import TrafficGraphBuilder, Intersection, Road
from src.utils.logger import setup_logging, get_logger


CAMERAS = [
    ("DELHI_CAM_1", "Intersection-1 Front", 28.6328, 77.2197),
    ("DELHI_CAM_2", "Intersection-1 Back",  28.6325, 77.2200),
    ("DELHI_CAM_3", "Intersection-2 Front", 28.6340, 77.2215),
    ("DELHI_CAM_4", "Intersection-2 Back",  28.6337, 77.2218),
    ("DELHI_CAM_5", "Intersection-3 Front", 28.6352, 77.2230),
    ("DELHI_CAM_6", "Intersection-3 Back",  28.6349, 77.2233),
]

ROADS = [
    ("DELHI_CAM_1", "DELHI_CAM_2"),
    ("DELHI_CAM_2", "DELHI_CAM_1"),
    ("DELHI_CAM_3", "DELHI_CAM_4"),
    ("DELHI_CAM_4", "DELHI_CAM_3"),
    ("DELHI_CAM_5", "DELHI_CAM_6"),
    ("DELHI_CAM_6", "DELHI_CAM_5"),
    ("DELHI_CAM_2", "DELHI_CAM_3"),
    ("DELHI_CAM_3", "DELHI_CAM_2"),
    ("DELHI_CAM_4", "DELHI_CAM_5"),
    ("DELHI_CAM_5", "DELHI_CAM_4"),
    ("DELHI_CAM_1", "DELHI_CAM_3"),
    ("DELHI_CAM_3", "DELHI_CAM_1"),
]


def main():
    setup_logging()
    log = get_logger("generate_delhi_graph")

    builder = TrafficGraphBuilder()

    for cam_id, name, lat, lon in CAMERAS:
        builder.add_intersection(Intersection(
            intersection_id=cam_id,
            name=name,
            lat=lat,
            lon=lon,
            num_lanes=4,
            capacity=120.0,
            camera_id=cam_id,
        ))

    for src, dst in ROADS:
        builder.add_road(Road(from_id=src, to_id=dst, distance_m=300.0))

    out = "data/graphs/city_graph.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    builder.save(out)
    log.info(f"Delhi graph saved → {out}")
    log.info(f"Nodes: {builder.num_nodes} | Edges: {builder.num_edges}")
    for nid in builder.intersections:
        log.info(f"  {nid}")


if __name__ == "__main__":
    main()
