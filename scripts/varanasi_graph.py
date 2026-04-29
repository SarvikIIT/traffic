import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.graph.builder import TrafficGraphBuilder, Intersection, Road
from src.utils.logger import setup_logging, get_logger

def build_varanasi_lanka_graph() -> TrafficGraphBuilder:
    builder = TrafficGraphBuilder()

    intersections = [
        ("LANKA_CHOWK",    "Lanka Chowk",               25.2671,  82.9903, 6,     200),
        ("BHU_GATE",       "BHU Main Gate",              25.2677,  82.9996, 4,     150),
        ("ASSI_CHOWK",     "Assi Chowk",                25.2812,  83.0055, 4,     180),
        ("CHETGANJ",       "Chetganj Crossing",          25.2956,  82.9897, 6,     220),
        ("SUNDERPUR",      "Sunderpur Crossing",         25.2743,  82.9781, 4,     160),
        ("SHIVPUR",        "Shivpur More",               25.2603,  82.9712, 4,     140),
        ("PANDEYPUR",      "Pandeypur Crossing",         25.3120,  82.9762, 4,     170),
        ("SIGRA",          "Sigra Chowk",                25.3028,  82.9841, 6,     250),
        ("MALDAHIYA",      "Maldahiya Crossing",         25.3180,  82.9989, 4,     160),
        ("LAHURABIR",      "Lahurabir Crossing",         25.3102,  83.0067, 6,     240),
        ("GODOWLIA",       "Godowlia Chowk",             25.3095,  83.0118, 4,     180),
        ("DASASWAMEDH",    "Dasaswamedh Ghat Road",      25.3063,  83.0124, 2,     80),
        ("MAHMOORGANJ",    "Mahmoorganj Crossing",       25.2888,  82.9801, 4,     150),
        ("RATHYATRA",      "Rathyatra Crossing",         25.3201,  82.9912, 6,     200),
        ("KABIRCHAURA",    "Kabirchaura Crossing",       25.3175,  82.9907, 4,     180),
    ]

    for iid, name, lat, lon, lanes, cap in intersections:
        builder.add_intersection(Intersection(
            intersection_id=iid,
            name=name,
            lat=lat,
            lon=lon,
            num_lanes=lanes,
            capacity=float(cap),
            camera_id=f"cam_{iid.lower()}",
        ))

    roads = [
        ("LANKA_CHOWK",  "BHU_GATE",     900,  1, 4, 40),
        ("LANKA_CHOWK",  "SUNDERPUR",    1200, 1, 4, 40),
        ("LANKA_CHOWK",  "ASSI_CHOWK",  1800, 1, 4, 30),
        ("BHU_GATE",     "ASSI_CHOWK",  1500, 1, 4, 40),
        ("BHU_GATE",     "MAHMOORGANJ", 1100, 2, 2, 30),
        ("ASSI_CHOWK",   "GODOWLIA",    2800, 0, 4, 30),
        ("SUNDERPUR",    "SHIVPUR",      900,  2, 2, 40),
        ("SUNDERPUR",    "MAHMOORGANJ", 1000, 2, 2, 30),
        ("SUNDERPUR",    "CHETGANJ",    2200, 1, 4, 40),
        ("CHETGANJ",     "SIGRA",        900,  1, 6, 50),
        ("CHETGANJ",     "PANDEYPUR",   1800, 1, 4, 40),
        ("SIGRA",        "MALDAHIYA",    800,  1, 4, 40),
        ("SIGRA",        "RATHYATRA",    600,  1, 4, 40),
        ("SIGRA",        "LAHURABIR",   1100, 1, 4, 40),
        ("MALDAHIYA",    "LAHURABIR",    700,  1, 4, 40),
        ("MALDAHIYA",    "RATHYATRA",    500,  2, 4, 30),
        ("LAHURABIR",    "GODOWLIA",     800,  1, 4, 30),
        ("LAHURABIR",    "KABIRCHAURA",  600,  2, 4, 30),
        ("KABIRCHAURA",  "RATHYATRA",    500,  2, 4, 30),
        ("GODOWLIA",     "DASASWAMEDH",  600,  2, 2, 20),
        ("PANDEYPUR",    "RATHYATRA",    900,  1, 4, 40),
        ("MAHMOORGANJ",  "SHIVPUR",      700,  2, 2, 30),
    ]

    for src, dst, dist, rtype, lanes, speed in roads:
        builder.add_road(Road(src, dst, dist, rtype, lanes, float(speed)))
        builder.add_road(Road(dst, src, dist, rtype, lanes, float(speed)))

    return builder

def main():
    setup_logging()
    log = get_logger("generate_varanasi_graph")

    log.info("Building Varanasi Lanka-area intersection graph...")
    builder = build_varanasi_lanka_graph()
    log.info(f"Nodes: {builder.num_nodes} intersections")
    log.info(f"Edges: {builder.num_edges} road connections")

    for nid, attrs in builder.graph.nodes(data=True):
        log.info(f"  {nid:20s} | {attrs['name']:30s} | {attrs['lat']:.4f}N {attrs['lon']:.4f}E")

    out = "data/graphs/varanasi_lanka_graph.json"
    builder.save(out)
    log.info(f"Saved → {out}")

    builder.save("data/graphs/city_graph.json")
    log.info("Also saved as data/graphs/city_graph.json (default)")
    return builder

if __name__ == "__main__":
    main()
