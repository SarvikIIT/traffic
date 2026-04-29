import numpy as np
import pytest
import torch

from src.graph.builder import TrafficGraphBuilder, Intersection, Road
from src.graph.dataset import TrafficGraphDataset
from src.graph.stgcn import STGCN, GraphConvLayer


def test_grid_city_nodes(grid_builder):
    assert grid_builder.num_nodes == 9


def test_grid_city_edges(grid_builder):
    assert grid_builder.num_edges == 24


def test_add_intersection():
    b = TrafficGraphBuilder()
    b.add_intersection(Intersection("A", "Alpha", lat=40.0, lon=-74.0))
    assert "A" in b.intersections


def test_add_road():
    b = TrafficGraphBuilder()
    b.add_intersection(Intersection("A", "A"))
    b.add_intersection(Intersection("B", "B"))
    b.add_road(Road("A", "B", distance_m=300))
    assert b.num_edges == 1


def test_adjacency_matrix_shape(grid_builder):
    adj = grid_builder.get_adjacency_matrix()
    assert adj.shape == (9, 9)


def test_node_features_shape(grid_builder):
    feat = grid_builder.get_node_features()
    assert feat.shape == (9, 8)


def test_edge_index_shape(grid_builder):
    ei = grid_builder.get_edge_index()
    assert ei.shape[0] == 2
    assert ei.shape[1] > 0


def test_save_load(grid_builder, tmp_path):
    path = str(tmp_path / "graph.json")
    grid_builder.save(path)
    loaded = TrafficGraphBuilder.load(path)
    assert loaded.num_nodes == grid_builder.num_nodes
    assert loaded.num_edges == grid_builder.num_edges


def test_update_intersection(grid_builder):
    iid = grid_builder.intersections[0]
    grid_builder.update_intersection(iid, {"density": 0.75, "congestion_level": 0.8})
    attrs = grid_builder.graph.nodes[iid]
    assert attrs["density"] == 0.75


def test_congested_nodes(grid_builder):
    for i, nid in enumerate(grid_builder.intersections):
        grid_builder.update_intersection(nid, {"congestion_level": 0.9 if i < 3 else 0.1})
    congested = grid_builder.get_congested_nodes(threshold=0.7)
    assert len(congested) == 3


def test_synthetic_dataset_sizes():
    train, val, test = TrafficGraphDataset.generate_synthetic(
        num_nodes=9, timesteps=200, seq_len=6, pred_len=2
    )
    total = len(train) + len(val) + len(test)
    assert total > 0
    assert len(train) > len(val)


def test_dataset_getitem_shapes():
    train, _, _ = TrafficGraphDataset.generate_synthetic(
        num_nodes=9, timesteps=200, seq_len=6, pred_len=2
    )
    x, y, ei = train[0]
    assert x.shape == (6, 9, 8)
    assert y.shape == (2, 9)
    assert ei.shape[0] == 2


def test_graph_conv_forward():
    layer = GraphConvLayer(8, 16)
    x = torch.randn(2, 5, 8)
    adj = torch.eye(5)
    out = layer(x, adj)
    assert out.shape == (2, 5, 16)


def test_stgcn_forward():
    N, T_in, F = 9, 12, 8
    pred_len = 3
    model = STGCN(num_nodes=N, in_channels=F, hidden_channels=32,
                  num_layers=2, pred_len=pred_len, seq_len=T_in)
    x   = torch.randn(2, T_in, N, F)
    adj = torch.eye(N)
    out = model(x, adj)
    assert out.shape == (2, pred_len, N)


def test_stgcn_no_nans():
    N, T_in, F = 9, 12, 8
    model = STGCN(num_nodes=N, in_channels=F, hidden_channels=32,
                  num_layers=2, pred_len=3, seq_len=T_in)
    x   = torch.randn(1, T_in, N, F)
    adj = torch.eye(N)
    out = model(x, adj)
    assert not torch.isnan(out).any()
