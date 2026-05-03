"""Tests for core/transform.py"""
import copy, torch, torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv
from gnn_remat.core.detector  import detect
from gnn_remat.core.wrapper   import _RematConv
from gnn_remat.core.transform import apply, remove


class TwoLayerGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 16)
        self.conv2 = GCNConv(16, 4)
    def forward(self, x, ei): return self.conv2(self.conv1(x, ei).relu(), ei)


def _graph(n=30, f=8, e=80):
    torch.manual_seed(0)
    x  = torch.randn(n, f)
    ei = torch.stack([torch.randint(0, n, (e,)), torch.randint(0, n, (e,))])
    return x, ei


# apply()
def test_apply_wraps_all_layers():
    remat = apply(TwoLayerGCN(), detect(TwoLayerGCN()))
    assert isinstance(remat.conv1, _RematConv)
    assert isinstance(remat.conv2, _RematConv)

def test_apply_does_not_mutate_original():
    model = TwoLayerGCN();  apply(model, detect(model))
    assert isinstance(model.conv1, GCNConv)   # original untouched

def test_apply_partial_wrapping():
    model = TwoLayerGCN()
    remat = apply(model, detect(model)[:1])   # only conv1
    assert isinstance(remat.conv1, _RematConv)
    assert isinstance(remat.conv2, GCNConv)

def test_apply_output_matches():
    torch.manual_seed(0)
    model = TwoLayerGCN();  remat = apply(model, detect(model))
    x, ei = _graph()
    with torch.no_grad():
        assert torch.allclose(model(x, ei), remat(x, ei), atol=1e-5)

def test_apply_gradients_match():
    torch.manual_seed(0)
    model = TwoLayerGCN();  remat = apply(model, detect(model))
    x, ei = _graph()
    x1 = x.clone().requires_grad_(True);  model(x1, ei).sum().backward()
    x2 = x.clone().requires_grad_(True);  remat(x2, ei).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

# remove()
def test_remove_strips_all_wrappers():
    model = TwoLayerGCN()
    plain = remove(apply(model, detect(model)))
    assert isinstance(plain.conv1, GCNConv)
    assert isinstance(plain.conv2, GCNConv)

def test_remove_preserves_output():
    torch.manual_seed(0)
    model = TwoLayerGCN()
    plain = remove(apply(model, detect(model)))
    x, ei = _graph()
    with torch.no_grad():
        assert torch.allclose(model(x, ei), plain(x, ei), atol=1e-5)
