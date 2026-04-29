from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

CONGESTION_COLORSCALE = [
    [0.0, "#00cc44"],
    [0.5, "#ffcc00"],
    [0.7, "#ff6600"],
    [1.0, "#cc0000"],
]

class TrafficNetworkVisualizer:

    def __init__(self, builder):
        self.builder = builder
        self._G = builder.graph

    def plotly_map(
        self,
        title: str = "Traffic Network",
        width: int = 900,
        height: int = 700,
    ) -> "go.Figure":
        if not HAS_PLOTLY:
            raise ImportError("plotly required: pip install plotly")

        nodes = list(self._G.nodes(data=True))
        node_lats = [d.get("lat", 0) for _, d in nodes]
        node_lons = [d.get("lon", 0) for _, d in nodes]
        node_ids  = [n for n, _ in nodes]
        cong      = [d.get("congestion_level", 0.0) for _, d in nodes]
        queues    = [d.get("queue_length", 0.0) for _, d in nodes]
        texts     = [
            f"{n}<br>Density: {d.get('density',0):.2f}"
            f"<br>Flow: {d.get('flow_rate',0):.1f}"
            f"<br>Queue: {d.get('queue_length',0):.0f}m"
            for n, d in nodes
        ]

        edge_lats, edge_lons = [], []
        for u, v in self._G.edges():
            ul = self._G.nodes[u].get("lat", 0)
            uo = self._G.nodes[u].get("lon", 0)
            vl = self._G.nodes[v].get("lat", 0)
            vo = self._G.nodes[v].get("lon", 0)
            edge_lats += [ul, vl, None]
            edge_lons += [uo, vo, None]

        fig = go.Figure()
        fig.add_trace(go.Scattermapbox(
            lat=edge_lats, lon=edge_lons,
            mode="lines",
            line=dict(width=1.5, color="rgba(128,128,128,0.5)"),
            hoverinfo="none",
            name="Roads",
        ))
        fig.add_trace(go.Scattermapbox(
            lat=node_lats, lon=node_lons,
            mode="markers+text",
            marker=dict(
                size=14,
                color=cong,
                colorscale=CONGESTION_COLORSCALE,
                cmin=0, cmax=1,
                colorbar=dict(title="Congestion"),
            ),
            text=node_ids,
            textposition="top right",
            hovertext=texts,
            hoverinfo="text",
            name="Intersections",
        ))

        center_lat = float(np.mean(node_lats)) if node_lats else 40.71
        center_lon = float(np.mean(node_lons)) if node_lons else -74.00
        fig.update_layout(
            title=title,
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=13),
            width=width, height=height,
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
        )
        return fig

    def plotly_time_series(
        self,
        intersection_ids: List[str],
        metric: str = "density",
        history: Dict = None,
    ) -> "go.Figure":
        if not HAS_PLOTLY:
            raise ImportError("plotly required")
        fig = go.Figure()
        if history:
            for iid in intersection_ids:
                if iid in history:
                    ts = history[iid]
                    fig.add_trace(go.Scatter(
                        x=list(range(len(ts))), y=ts,
                        mode="lines+markers", name=iid,
                    ))
        fig.update_layout(
            title=f"Traffic {metric.capitalize()} Over Time",
            xaxis_title="Time Step",
            yaxis_title=metric.capitalize(),
            template="plotly_dark",
        )
        return fig

    def matplotlib_map(
        self, figsize: Tuple[int, int] = (12, 8), show: bool = True
    ) -> Optional["plt.Figure"]:
        if not HAS_MPL:
            raise ImportError("matplotlib required")

        fig, ax = plt.subplots(figsize=figsize)
        nodes = list(self._G.nodes(data=True))
        node_pos = {n: (d.get("lon", 0), d.get("lat", 0)) for n, d in nodes}
        congestion = {n: d.get("congestion_level", 0.0) for n, d in nodes}

        for u, v in self._G.edges():
            xu, yu = node_pos[u]
            xv, yv = node_pos[v]
            ax.plot([xu, xv], [yu, yv], "gray", linewidth=0.8, alpha=0.6)

        xs = [node_pos[n][0] for n, _ in nodes]
        ys = [node_pos[n][1] for n, _ in nodes]
        cs = [congestion[n] for n, _ in nodes]
        sc = ax.scatter(xs, ys, c=cs, cmap="RdYlGn_r", s=100, zorder=3,
                        vmin=0, vmax=1, edgecolors="k", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="Congestion Level")
        for n, _ in nodes:
            ax.annotate(n, node_pos[n], fontsize=6, ha="center", va="bottom")

        ax.set_title("Traffic Network – Congestion Map")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        plt.tight_layout()
        if show:
            plt.show()
        return fig
