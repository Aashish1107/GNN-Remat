"""Smoke test for the benchmark profiler's AMP + 4-condition path (CPU)."""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

from gnn_remat.benchmark.profiler import compare


class _GAT(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = GATConv(8, 4, heads=2)
    def forward(self, x, ei):
        return self.c(x, ei)


def _graph():
    torch.manual_seed(0)
    x  = torch.randn(30, 8)
    ei = torch.stack([torch.randint(0, 30, (80,)), torch.randint(0, 30, (80,))])
    return x, ei


def test_compare_amp_and_chunk_runs():
    x, ei = _graph()
    r = compare(_GAT(), x, ei, num_epochs=2, device=torch.device("cpu"),
                chunk_nodes=5, amp=True)
    for cond in (r.baseline, r.module_ckpt, r.gnn_remat, r.chunked_remat):
        assert cond is not None and cond.ok


if __name__ == "__main__":  # ponytail self-check
    test_compare_amp_and_chunk_runs()
    print("ok")
