"""Tests for core/heuristic.py

The CPU proxy models SAR's Case-1 / Case-2 distinction as a NET savings
estimate (freed_bytes - added_bytes):

  * Non-attention (GCN, SAGE) — Case 1: scatter backward stores no messages,
    so propagate-checkpoint only ADDS the input cost → negative score → skipped.
  * Attention (GAT, Transformer) — Case 2: baseline retains x_j + alpha, which
    the checkpoint frees → positive score on a dense enough graph → selected.
"""
import torch, torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv
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


class ThreeLayerGAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GATConv(8, 64, heads=4)
        self.conv2 = GATConv(256, 64, heads=4)
        self.conv3 = GATConv(256, 4, heads=1, concat=False)
    def forward(self, x, ei):
        return self.conv3(self.conv2(self.conv1(x, ei).relu(), ei).relu(), ei)


def _graph(n=100, f=8, e=400):
    torch.manual_seed(0)
    x  = torch.randn(n, f)
    ei = torch.stack([torch.randint(0, n, (e,)), torch.randint(0, n, (e,))])
    return x, ei


# ── score_layers() ─────────────────────────────────────────────────────────────

def test_score_layers_returns_one_per_layer():
    assert len(score_layers(detect(ThreeLayerGCN()), *_graph())) == 3

def test_non_attention_scores_are_negative():
    # SAR Case 1: scatter backward stores no messages, so propagate-checkpoint
    # only adds memory → net score is negative and the layer should be skipped.
    scored = score_layers(detect(ThreeLayerGCN()), *_graph())
    assert all(s.score < 0 for s in scored)

def test_attention_scores_are_positive_when_dense():
    # SAR Case 2: dense attention layers free x_j + alpha → positive net score.
    scored = score_layers(detect(ThreeLayerGAT()), *_graph())
    assert all(s.score > 0 for s in scored)

def test_attention_scores_higher_than_non_attention():
    gcn = {s.info.name: s.score for s in score_layers(detect(ThreeLayerGCN()), *_graph())}
    gat = {s.info.name: s.score for s in score_layers(detect(ThreeLayerGAT()), *_graph())}
    assert min(gat.values()) > max(gcn.values())

def test_scores_sorted_descending():
    vals = [s.score for s in score_layers(detect(ThreeLayerGCN()), *_graph())]
    assert vals == sorted(vals, reverse=True)


# ── select() ───────────────────────────────────────────────────────────────────

def test_select_skips_non_attention_by_default():
    # GCN layers have negative scores → none selected at the default threshold.
    assert select(detect(ThreeLayerGCN()), *_graph()) == []

def test_select_keeps_dense_attention():
    assert len(select(detect(ThreeLayerGAT()), *_graph())) == 3

def test_select_huge_threshold_returns_empty():
    assert select(detect(ThreeLayerGAT()), *_graph(), threshold=1e18) == []

def test_select_top_k_limits_count():
    assert len(select(detect(ThreeLayerGAT()), *_graph(), top_k=1)) == 1
