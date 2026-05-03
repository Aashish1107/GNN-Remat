"""Tests for core/heuristic.py"""
import torch, torch.nn as nn
from torch_geometric.nn import GCNConv
from gnn_remat.core.detector  import detect
from gnn_remat.core.heuristic import score_layers, select


class ThreeLayerGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 4)
    def forward(self, x, ei):
        return self.conv3(self.conv2(self.conv1(x, ei).relu(), ei).relu(), ei)


def _graph(n=100, f=8, e=400):
    torch.manual_seed(0)
    x  = torch.randn(n, f)
    ei = torch.stack([torch.randint(0, n, (e,)), torch.randint(0, n, (e,))])
    return x, ei


def test_score_layers_returns_one_per_layer():
    assert len(score_layers(detect(ThreeLayerGCN()), *_graph())) == 3

def test_scores_are_positive():
    assert all(s.score > 0 for s in score_layers(detect(ThreeLayerGCN()), *_graph()))

def test_scores_sorted_descending():
    vals = [s.score for s in score_layers(detect(ThreeLayerGCN()), *_graph())]
    assert vals == sorted(vals, reverse=True)

def test_select_threshold_zero_keeps_all():
    assert len(select(detect(ThreeLayerGCN()), *_graph(), threshold=0.0)) == 3

def test_select_huge_threshold_returns_empty():
    assert select(detect(ThreeLayerGCN()), *_graph(), threshold=1e18) == []

def test_select_top_k_limits_count():
    assert len(select(detect(ThreeLayerGCN()), *_graph(), threshold=0.0, top_k=1)) == 1
