"""Tests for the explicit checkpointing plan (detector.plan_of / return_plan)."""
import torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv

from gnn_remat import gnn_remat, plan_of, CheckpointPlan


class _Mixed(nn.Module):
    def __init__(self):
        super().__init__()
        self.gcn  = GCNConv(8, 16)
        self.sage = SAGEConv(16, 16)
        self.gat  = GATConv(16, 4, heads=2)
    def forward(self, x, ei):
        return self.gat(self.sage(self.gcn(x, ei).relu(), ei).relu(), ei)


def test_return_plan_marks_all_layers_aggr():
    model, plan = gnn_remat(_Mixed(), return_plan=True)
    assert isinstance(plan, CheckpointPlan)
    by = {p.name: p for p in plan.layers}
    assert {"gcn", "sage", "gat"} <= set(by)
    assert all(by[n].checkpoint and by[n].granularity == "aggr"
               for n in ("gcn", "sage", "gat"))


def test_plan_reflects_selective_and_chunk():
    model = gnn_remat(_Mixed(), mode="types", layer_types=[GATConv], chunk_nodes=99)
    by = {p.name: p for p in plan_of(model).layers}
    assert by["gat"].checkpoint and by["gat"].chunk_nodes == 99
    assert not by["gcn"].checkpoint and not by["sage"].checkpoint


def test_plan_reflects_module_granularity():
    plan = plan_of(gnn_remat(_Mixed(), granularity="module"))
    assert all(p.granularity == "module" for p in plan.layers if p.checkpoint)


def test_plan_str_renders():
    assert "layer" in str(plan_of(gnn_remat(_Mixed())))


if __name__ == "__main__":  # ponytail self-check
    for fn in (test_return_plan_marks_all_layers_aggr,
               test_plan_reflects_selective_and_chunk,
               test_plan_reflects_module_granularity,
               test_plan_str_renders):
        fn()
    print(plan_of(gnn_remat(_Mixed())))
    print("ok")
