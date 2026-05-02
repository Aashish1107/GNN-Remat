"""Tests for core/wrapper.py"""
import torch
from torch_geometric.nn import GCNConv, SAGEConv
from gnn_remat.core.wrapper import wrap, _RematConv


def _graph(n=20, f=8, e=60):
    torch.manual_seed(0)
    x  = torch.randn(n, f)
    ei = torch.stack([torch.randint(0, n, (e,)), torch.randint(0, n, (e,))])
    return x, ei


def test_wrap_returns_remat_conv():    
    assert isinstance(wrap(GCNConv(8, 8)), _RematConv)
def test_wrap_stores_conv_ref():
    conv = GCNConv(8, 8);  assert wrap(conv).conv is conv

def test_output_matches_original():
    import copy
    torch.manual_seed(1)
    conv = GCNConv(8, 16);  wrapped = wrap(copy.deepcopy(conv))
    x, ei = _graph()
    with torch.no_grad():
        assert torch.allclose(conv(x, ei), wrapped(x, ei), atol=1e-5)

def test_gradients_match_gcnconv():
    import copy
    torch.manual_seed(2)
    conv = GCNConv(8, 16);  wrapped = wrap(copy.deepcopy(conv))
    x, ei = _graph()
    x1 = x.clone().requires_grad_(True)
    conv(x1, ei).sum().backward()
    x2 = x.clone().requires_grad_(True)
    wrapped(x2, ei).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)

def test_gradients_match_sageconv():
    import copy
    torch.manual_seed(3)
    conv = SAGEConv(8, 16);  wrapped = wrap(copy.deepcopy(conv))
    x, ei = _graph()
    x1 = x.clone().requires_grad_(True);  conv(x1, ei).sum().backward()
    x2 = x.clone().requires_grad_(True);  wrapped(x2, ei).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5)
