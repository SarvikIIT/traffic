from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        support = torch.matmul(x, self.weight)
        out = torch.bmm(adj.unsqueeze(0).expand(x.size(0), -1, -1), support)
        if self.bias is not None:
            out = out + self.bias
        return out

class STConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        spatial_channels: int,
        out_channels: int,
        num_nodes: int,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes

        self.t_conv1 = nn.Conv2d(
            in_channels, spatial_channels * 2,
            kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)
        )
        self.graph_conv = GraphConvLayer(spatial_channels, spatial_channels)
        self.t_conv2 = nn.Conv2d(
            spatial_channels, out_channels * 2,
            kernel_size=(1, kernel_size), padding=(0, kernel_size // 2)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)

        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
            if in_channels != out_channels else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor
    ) -> torch.Tensor:
        B, C, N, T = x.shape
        residual = self.residual(x)

        h = self.t_conv1(x)
        h = F.glu(h, dim=1)

        h = h.permute(0, 3, 2, 1).contiguous()
        h = h.view(B * T, N, -1)
        h = self.graph_conv(h, adj)
        h = F.relu(h)
        h = h.view(B, T, N, -1).permute(0, 3, 2, 1)

        h = self.t_conv2(h)
        h = F.glu(h, dim=1)

        h = self.bn(h)
        h = self.dropout(h)
        return F.relu(h + residual)

class STGCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 3,
        pred_len: int = 3,
        seq_len: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len
        self.seq_len = seq_len

        channels = [in_channels] + [hidden_channels] * num_layers
        self.st_blocks = nn.ModuleList([
            STConvBlock(
                in_channels=channels[i],
                spatial_channels=channels[i + 1],
                out_channels=channels[i + 1],
                num_nodes=num_nodes,
                dropout=dropout,
            )
            for i in range(num_layers)
        ])

        self.temporal_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, (1, 1)),
            nn.ReLU(),
            nn.Conv2d(hidden_channels // 2, pred_len * out_channels, (1, 1)),
        )

    def _normalize_adj(self, adj: torch.Tensor) -> torch.Tensor:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        deg = adj.sum(dim=1).clamp(min=1e-6)
        d_inv_sqrt = deg.pow(-0.5).diag()
        return d_inv_sqrt @ adj @ d_inv_sqrt

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        B, T, N, F = x.shape
        adj_norm = self._normalize_adj(adj)

        h = x.permute(0, 3, 2, 1)

        for block in self.st_blocks:
            h = block(h, adj_norm)

        h = self.temporal_pool(h)
        out = self.output_proj(h)
        out = out.squeeze(-1)
        return out

class TrafficPredictor:
    def __init__(self, model: STGCN, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def predict(
        self,
        node_features: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            x = node_features.unsqueeze(0).to(self.device)
            a = adj.to(self.device)
            out = self.model(x, a)
        return out.squeeze(0).cpu()

    def save(self, path: str) -> None:
        from pathlib import Path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.model.state_dict()}, path)

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        num_nodes: int = 25,
        in_channels: int = 8,
        hidden_channels: int = 64,
        pred_len: int = 3,
        seq_len: int = 12,
        device: str = "cpu",
    ) -> "TrafficPredictor":
        model = STGCN(
            num_nodes=num_nodes,
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            pred_len=pred_len,
            seq_len=seq_len,
        )
        ckpt = torch.load(path, map_location=device, weights_only=False)
        saved_args = ckpt.get("args", {})
        saved_pred = saved_args.get("pred_len", pred_len)
        saved_seq = saved_args.get("seq_len", seq_len)
        if saved_pred != pred_len or saved_seq != seq_len:
            model = STGCN(
                num_nodes=num_nodes,
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                pred_len=saved_pred,
                seq_len=saved_seq,
            )
        model.load_state_dict(ckpt["state_dict"])
        return cls(model, device)
