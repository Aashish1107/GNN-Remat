"""Tests for the public gnn_remat() API surface."""
import copy, pytest, torch, torch.nn as nn
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from gnn_remat import gnn_remat, remove_remat, detect
from gnn_remat.core.wrapper import _RematConv


class ThreeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(8, 32)
        self.conv2 = SAGEConv(32, 32)
        self.conv3 = GATConv(32, 4, heads=1)
    def forward(self, x, ei):
        return self.conv3(self.conv2(self.conv1(x, ei).relu(), ei).relu(), ei)


def _graph(n=40, f=8, e=120):
    torch.manual_seed(0)
    x  = torch.randn(n, f)
    ei = torch.stack([torch.randint(0, n, (e,)), torch.randint(0, n, (e,))])
    return x, ei


def _count_remat(m): return sum(1 for _, mod in m.named_modules() if isinstance(mod, _RematConv))


# Mode: all
def test_mode_all_wraps_every_layer():
    assert _count_remat(gnn_remat(ThreeLayer())) == 3

def test_mode_all_does_not_mutate_original():
    model = ThreeLayer();  gnn_remat(model)
    assert isinstance(model.conv1, GCNConv)

# Mode: names
def test_mode_names_wraps_only_specified():
    remat = gnn_remat(ThreeLayer(), mode="names", layers=["conv1"])
    assert isinstance(remat.conv1, _RematConv)
    assert isinstance(remat.conv2, SAGEConv)

def test_mode_names_missing_layers_raises():
    with pytest.raises(ValueError):
        gnn_remat(ThreeLayer(), mode="names")

# Mode: types
def test_mode_types_wraps_only_matching():
    remat = gnn_remat(ThreeLayer(), mode="types", layer_types=[GCNConv])
    assert isinstance(remat.conv1, _RematConv)
    assert isinstance(remat.conv2, SAGEConv)

def test_mode_types_missing_types_raises():
    with pytest.raises(ValueError):
        gnn_remat(ThreeLayer(), mode="types")

# Mode: auto
def test_mode_auto_runs_without_error():
    x, ei = _graph()
    remat  = gnn_remat(ThreeLayer(), mode="auto", x=x, edge_index=ei)
    assert remat is not None

def test_mode_auto_missing_x_raises():
    with pytest.raises(ValueError):
        gnn_remat(ThreeLayer(), mode="auto")

# Bad mode
def test_bad_mode_raises_value_error():
    with pytest.raises(ValueError):
        gnn_remat(ThreeLayer(), mode="unknown")

# remove_remat()
def test_remove_remat_strips_all_wrappers():
    plain = remove_remat(gnn_remat(ThreeLayer()))
    assert isinstance(plain.conv1, GCNConv)
    assert isinstance(plain.conv2, SAGEConv)

# detect()
def test_detect_returns_layer_infos():
    infos = detect(ThreeLayer())
    assert len(infos) == 3
    assert infos[0].name == "conv1"

# Gradient parity across all model types
def _grad_parity(ModelCls, in_f=8):
    torch.manual_seed(42)
    from gnn_remat.benchmark.models import build
    model = ModelCls(in_channels=in_f, hidden=16, out_channels=4, num_layers=2)
    remat = gnn_remat(copy.deepcopy(model))
    # eval() disables dropout so both models are deterministic and comparable
    model.eval(); remat.eval()
    x, ei = _graph(f=in_f)
    x1 = x.clone().requires_grad_(True);  model(x1, ei).sum().backward()
    x2 = x.clone().requires_grad_(True);  remat(x2, ei).sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4), \
        f"Grad mismatch: max diff {(x1.grad-x2.grad).abs().max():.2e}"

def test_gradient_parity_gcn():
    from gnn_remat.benchmark.models import GCN
    _grad_parity(GCN)

def test_gradient_parity_sage():
    from gnn_remat.benchmark.models import GraphSAGE
    _grad_parity(GraphSAGE)

def test_gradient_parity_gat():
    from gnn_remat.benchmark.models import GAT
    _grad_parity(GAT)
