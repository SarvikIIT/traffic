import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.builder import TrafficGraphBuilder
from src.utils.logger import setup_logging, get_logger

def parse_args():
    p = argparse.ArgumentParser(description="Generate traffic network graph")
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--spacing", type=float, default=500.0,
                   help="Block spacing in metres")
    p.add_argument("--lat", type=float, default=40.7128,
                   help="Base latitude (default: New York City)")
    p.add_argument("--lon", type=float, default=-74.0060,
                   help="Base longitude")
    p.add_argument("--output", type=str, default="data/graphs/city_graph.json")
    p.add_argument("--visualize", action="store_true",
                   help="Show matplotlib graph after generation")
    return p.parse_args()

def main():
    args = parse_args()
    setup_logging()
    log = get_logger("generate_graph")

    log.info(f"Generating {args.rows}×{args.cols} grid city "
             f"({args.rows * args.cols} intersections)")
    builder = TrafficGraphBuilder.create_grid_city(
        rows=args.rows,
        cols=args.cols,
        spacing_m=args.spacing,
        base_lat=args.lat,
        base_lon=args.lon,
    )

    log.info(f"Nodes: {builder.num_nodes} | Edges: {builder.num_edges}")
    builder.save(args.output)
    log.info(f"Graph saved to {args.output}")

    if args.visualize:
        try:
            from src.visualization.mapview import TrafficNetworkVisualizer
            viz = TrafficNetworkVisualizer(builder)
            viz.matplotlib_map()
        except Exception as e:
            log.warning(f"Visualisation failed: {e}")

if __name__ == "__main__":
    main()
