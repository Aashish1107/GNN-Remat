"""
models.py — Reference GNN models for benchmarking.

All models share the same interface:
    model(x, edge_index) -> Tensor [num_nodes, num_classes]

Available: GCN, GraphSAGE, GAT, build(name, ...)
"""
from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, TransformerConv


class GCN(nn.Module):
    """Multi-layer Graph Convolutional Network."""
    def __init__(self, in_channels, hidden, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i+1]) for i in range(num_layers))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x


class GraphSAGE(nn.Module):
    """Multi-layer GraphSAGE."""
    def __init__(self, in_channels, hidden, out_channels, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        dims = [in_channels] + [hidden] * (num_layers - 1) + [out_channels]
        self.convs = nn.ModuleList(SAGEConv(dims[i], dims[i+1]) for i in range(num_layers))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(F.dropout(x, p=self.dropout, training=self.training))
        return x


class GAT(nn.Module):
    """Multi-layer Graph Attention Network."""
    def __init__(self, in_channels, hidden, out_channels, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(GATConv(in_channels, hidden, heads=heads, dropout=dropout))
            elif i == num_layers - 1:
                self.convs.append(GATConv(hidden * heads, out_channels, heads=1, concat=False, dropout=dropout))
            else:
                self.convs.append(GATConv(hidden * heads, hidden, heads=heads, dropout=dropout))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(F.dropout(x, p=self.dropout, training=self.training))
        return x


class GraphTransformer(nn.Module):
    """
    Multi-layer Graph Transformer (TransformerConv).

    TransformerConv computes full Q/K/V projections and multi-head softmax
    attention inside propagate(), producing larger per-edge intermediates than
    GATConv.  This makes it the strongest showcase for propagate-level remat.
    """
    def __init__(self, in_channels, hidden, out_channels, num_layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.convs.append(
                    TransformerConv(in_channels, hidden, heads=heads,
                                    dropout=dropout, concat=True))
            elif i == num_layers - 1:
                self.convs.append(
                    TransformerConv(hidden * heads, out_channels, heads=1,
                                    dropout=dropout, concat=False))
            else:
                self.convs.append(
                    TransformerConv(hidden * heads, hidden, heads=heads,
                                    dropout=dropout, concat=True))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.elu(F.dropout(x, p=self.dropout, training=self.training))
        return x


_REGISTRY = {
    "gcn":         GCN,
    "graphsage":   GraphSAGE,
    "sage":        GraphSAGE,
    "gat":         GAT,
    "transformer": GraphTransformer,
}

def build(name, in_channels=128, hidden=256, out_channels=40, num_layers=3, **kwargs):
    """Instantiate a benchmark model by name: 'gcn', 'graphsage', 'gat', 'transformer'."""
    key = name.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown model {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[key](in_channels=in_channels, hidden=hidden,
                          out_channels=out_channels, num_layers=num_layers, **kwargs)
