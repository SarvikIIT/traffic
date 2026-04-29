from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

@dataclass
class Intersection:
    intersection_id: str
    name: str
    lat: float = 0.0
    lon: float = 0.0
    num_lanes: int = 4
    capacity: float = 100.0
    camera_id: Optional[str] = None

    density: float = 0.0
    flow_rate: float = 0.0
    queue_length: float = 0.0
    avg_speed: float = 50.0
    congestion_level: float = 0.0

@dataclass
class Road:
    from_id: str
    to_id: str
    distance_m: float = 500.0
    road_type: int = 1
    num_lanes: int = 2
    speed_limit_kmh: float = 50.0
    free_flow_speed: float = 50.0
    capacity: float = 1000.0

class TrafficGraphBuilder:

    def __init__(self):
        self._G = nx.DiGraph()

    def add_intersection(self, intersection: Intersection) -> None:
        self._G.add_node(intersection.intersection_id, **asdict(intersection))

    def add_road(self, road: Road) -> None:
        self._G.add_edge(road.from_id, road.to_id, **asdict(road))

    def remove_intersection(self, iid: str) -> None:
        self._G.remove_node(iid)

    def update_intersection(self, iid: str, metrics: Dict) -> None:
        if iid in self._G.nodes:
            self._G.nodes[iid].update(metrics)

    @property
    def graph(self) -> nx.DiGraph:
        return self._G

    @property
    def intersections(self) -> List[str]:
        return list(self._G.nodes)

    @property
    def num_nodes(self) -> int:
        return self._G.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._G.number_of_edges()

    def get_node_features(self) -> np.ndarray:
        nodes = list(self._G.nodes(data=True))
        features = []
        for _, attrs in nodes:
            features.append([
                attrs.get("density", 0.0),
                attrs.get("flow_rate", 0.0),
                attrs.get("queue_length", 0.0),
                attrs.get("avg_speed", 50.0),
                attrs.get("congestion_level", 0.0),
                attrs.get("num_lanes", 4) / 10.0,
                attrs.get("capacity", 100.0) / 200.0,
                1.0,
            ])
        return np.array(features, dtype=np.float32)

    def get_adjacency_matrix(self) -> np.ndarray:
        return nx.to_numpy_array(self._G, nodelist=self.intersections)

    def get_edge_index(self) -> np.ndarray:
        node_list = self.intersections
        idx_map = {nid: i for i, nid in enumerate(node_list)}
        rows, cols = [], []
        for u, v in self._G.edges():
            rows.append(idx_map[u])
            cols.append(idx_map[v])
        return np.array([rows, cols], dtype=np.int64)

    def get_congested_nodes(self, threshold: float = 0.7) -> List[str]:
        return [
            nid for nid, attrs in self._G.nodes(data=True)
            if attrs.get("congestion_level", 0.0) >= threshold
        ]

    def shortest_path(self, src: str, dst: str, weight: str = "distance_m"):
        try:
            return nx.shortest_path(self._G, src, dst, weight=weight)
        except nx.NetworkXNoPath:
            return []

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = nx.node_link_data(self._G, edges="links")
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TrafficGraphBuilder":
        with open(path, "r") as f:
            data = json.load(f)
        builder = cls()
        builder._G = nx.node_link_graph(data, directed=True, multigraph=False, edges="links")
        return builder

    @classmethod
    def create_grid_city(
        cls,
        rows: int = 5,
        cols: int = 5,
        spacing_m: float = 500.0,
        base_lat: float = 40.71,
        base_lon: float = -74.00,
    ) -> "TrafficGraphBuilder":
        builder = cls()
        deg_per_m_lat = 1 / 111_320
        deg_per_m_lon = 1 / (111_320 * math.cos(math.radians(base_lat)))

        id_grid: Dict[Tuple[int, int], str] = {}
        for r in range(rows):
            for c in range(cols):
                iid = f"INT_{r:02d}_{c:02d}"
                lat = base_lat + r * spacing_m * deg_per_m_lat
                lon = base_lon + c * spacing_m * deg_per_m_lon
                inter = Intersection(
                    intersection_id=iid,
                    name=f"Row{r}_Col{c}",
                    lat=lat,
                    lon=lon,
                    num_lanes=4,
                    capacity=120.0,
                )
                builder.add_intersection(inter)
                id_grid[(r, c)] = iid

        for r in range(rows):
            for c in range(cols):
                if c + 1 < cols:
                    builder.add_road(Road(id_grid[(r, c)], id_grid[(r, c + 1)], spacing_m))
                    builder.add_road(Road(id_grid[(r, c + 1)], id_grid[(r, c)], spacing_m))
                if r + 1 < rows:
                    builder.add_road(Road(id_grid[(r, c)], id_grid[(r + 1, c)], spacing_m))
                    builder.add_road(Road(id_grid[(r + 1, c)], id_grid[(r, c)], spacing_m))
        return builder
