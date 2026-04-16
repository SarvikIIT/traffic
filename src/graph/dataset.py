from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class TrafficGraphDataset(Dataset):

    def __init__(
        self,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        seq_len: int = 12,
        pred_len: int = 3,
        target_feat: int = 0,
        normalize: bool = True,
        train_mean: Optional[np.ndarray] = None,
        train_std: Optional[np.ndarray] = None,
    ):
        assert node_features.ndim == 3, "node_features must be (T, N, F)"
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_feat = target_feat

        self.edge_index = torch.tensor(edge_index, dtype=torch.long)

        if normalize:
            if train_mean is None:
                train_mean = node_features.mean(axis=(0, 1), keepdims=True)
            if train_std is None:
                train_std = node_features.std(axis=(0, 1), keepdims=True) + 1e-8
            self.mean = train_mean
            self.std = train_std
            node_features = (node_features - train_mean) / train_std
        else:
            self.mean = np.zeros((1, 1, node_features.shape[2]))
            self.std = np.ones((1, 1, node_features.shape[2]))

        self.data = torch.tensor(node_features, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, :, self.target_feat]
        return x, y, self.edge_index

    @classmethod
    def from_numpy(
        cls,
        node_features: np.ndarray,
        edge_index: np.ndarray,
        **kwargs,
    ) -> "TrafficGraphDataset":
        return cls(node_features, edge_index, **kwargs)

    @classmethod
    def generate_synthetic(
        cls,
        num_nodes: int = 25,
        timesteps: int = 2016,
        seq_len: int = 12,
        pred_len: int = 3,
        seed: int = 42,
    ) -> Tuple["TrafficGraphDataset", "TrafficGraphDataset", "TrafficGraphDataset"]:
        rng = np.random.default_rng(seed)
        t = np.linspace(0, 4 * np.pi * (timesteps / 288), timesteps)
        base = 20 + 30 * (np.sin(t) + 1) / 2
        noise = rng.normal(0, 5, (timesteps, num_nodes, 8))
        features = noise.copy()
        features[:, :, 0] += base[:, None]
        features[:, :, 1] += base[:, None] * 0.8
        features[:, :, 2] += base[:, None] * 0.3
        features = np.clip(features, 0, None)

        rows = cols = int(np.sqrt(num_nodes))
        src, dst = [], []
        for r in range(rows):
            for c in range(cols):
                n = r * cols + c
                if c + 1 < cols:
                    src += [n, n + 1]
                    dst += [n + 1, n]
                if r + 1 < rows:
                    src += [n, n + cols]
                    dst += [n + cols, n]
        edge_index = np.array([src, dst], dtype=np.int64)

        T = timesteps
        t_end = int(T * 0.7)
        v_end = int(T * 0.8)

        train_mean = features[:t_end].mean(axis=(0, 1), keepdims=True)
        train_std = features[:t_end].std(axis=(0, 1), keepdims=True) + 1e-8

        train_ds = cls(features[:t_end], edge_index, seq_len, pred_len,
                       train_mean=train_mean, train_std=train_std)
        val_ds = cls(features[t_end:v_end], edge_index, seq_len, pred_len,
                     train_mean=train_mean, train_std=train_std)
        test_ds = cls(features[v_end:], edge_index, seq_len, pred_len,
                      train_mean=train_mean, train_std=train_std)
        return train_ds, val_ds, test_ds
