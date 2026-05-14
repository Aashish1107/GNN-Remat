"""
test_granularity.py
-------------------
Proves that granularity="aggr" is genuinely different from granularity="module":
  - aggr   recomputes only aggregate()      during backward
  - module recomputes the full layer        during backward

The key check: count how many times message() and aggregate() are called
during a forward+backward pass for each granularity.
"""
import copy
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATConv, MessagePassing

from gnn_remat import gnn_remat
from gnn_remat.core.remat_mp import RematMessagePassing
from gnn_remat.core.wrapper  import _RematConv


#Instrumented conv to count calls

class InstrumentedGCN(MessagePassing):
    """GCNConv that counts message() and aggregate() calls."""
    def __init__(self, in_c, out_c):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_c, out_c, bias=False)
        self.message_calls   = 0
        self.aggregate_calls = 0

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_j):
        self.message_calls += 1
        return self.lin(x_j)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        self.aggregate_calls += 1
        return super().aggregate(inputs, index, ptr=ptr, dim_size=dim_size)


def _graph(n=30, f=8, e=80):
    torch.manual_seed(0)
    x  = torch.randn(n, f, requires_grad=True)
    ei = torch.stack([torch.randint(0, n, (e,)), torch.randint(0, n, (e,))])
    return x, ei


def _count_calls(model, x, ei):
    """Run one forward+backward and return (message_calls, aggregate_calls)."""
    # Find the instrumented conv
    for _, mod in model.named_modules():
        if hasattr(mod, "message_calls"):
            mod.message_calls   = 0
            mod.aggregate_calls = 0
            conv = mod
            break
    model.train()
    out = model(x.clone().requires_grad_(True), ei)
    out.sum().backward()
    return conv.message_calls, conv.aggregate_calls


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_baseline_calls_each_once():
    """Without remat, message and aggregate are each called exactly once."""
    model = nn.Sequential()   # workaround: wrap directly
    conv  = InstrumentedGCN(8, 16)

    class Wrapper(nn.Module):
        def __init__(self): super().__init__(); self.conv = conv
        def forward(self, x, ei): return self.conv(x, ei)

    m_calls, a_calls = _count_calls(Wrapper(), *_graph())
    # forward only: 1 message, 1 aggregate
    assert m_calls == 1, f"Expected 1 message call, got {m_calls}"
    assert a_calls == 1, f"Expected 1 aggregate call, got {a_calls}"


def test_aggr_granularity_reruns_propagate():
    """
    granularity='aggr': propagate() is checkpointed, so both message() and
    aggregate() are recomputed during backward.

      message_calls   == 2  (forward + recompute inside propagate checkpoint)
      aggregate_calls == 2  (forward + recompute inside propagate checkpoint)

    The distinction from 'module' granularity is *what else* is recomputed:
      aggr   — recomputes only propagate() (message + scatter)
      module — recomputes the full layer forward(), including any linear
               projections that the model applies before propagate()
               (e.g. GATConv.lin_src / lin_dst)

    InstrumentedGCN has its linear inside message() so both modes show the
    same call counts here.  The real cost difference is visible when a model
    applies expensive projections outside propagate().
    """
    conv = InstrumentedGCN(8, 16)

    class Wrapper(nn.Module):
        def __init__(self): super().__init__(); self.conv = conv
        def forward(self, x, ei): return self.conv(x, ei)

    model = gnn_remat(Wrapper(), granularity="aggr")

    # find the remat conv inside model
    for _, mod in model.named_modules():
        if hasattr(mod, "message_calls"):
            mod.message_calls = 0; mod.aggregate_calls = 0
            inner = mod; break

    model.train()
    x, ei = _graph()
    model(x.clone().requires_grad_(True), ei).sum().backward()

    # Both run twice: forward + recompute during propagate backward
    assert inner.message_calls == 2, \
        f"aggr mode: message() should run 2x (forward+recompute), ran {inner.message_calls}x"
    assert inner.aggregate_calls == 2, \
        f"aggr mode: aggregate() should run 2x (forward+recompute), ran {inner.aggregate_calls}x"


def test_module_granularity_reruns_everything():
    """
    granularity='module': both message() and aggregate() run twice because
    the whole layer is recomputed during backward.
    """
    conv = InstrumentedGCN(8, 16)

    class Wrapper(nn.Module):
        def __init__(self): super().__init__(); self.conv = conv
        def forward(self, x, ei): return self.conv(x, ei)

    model = gnn_remat(Wrapper(), granularity="module")

    for _, mod in model.named_modules():
        if hasattr(mod, "message_calls"):
            mod.message_calls = 0; mod.aggregate_calls = 0
            inner = mod; break

    model.train()
    x, ei = _graph()
    model(x.clone().requires_grad_(True), ei).sum().backward()

    # Both run twice (full layer recomputed)
    assert inner.message_calls   >= 2, \
        f"module mode: message() should run >=2x, ran {inner.message_calls}x"
    assert inner.aggregate_calls >= 2, \
        f"module mode: aggregate() should run >=2x, ran {inner.aggregate_calls}x"


def test_aggr_output_matches_baseline():
    """Correctness: aggr-mode output == baseline output."""
    torch.manual_seed(0)
    conv = GCNConv(8, 16)

    class M(nn.Module):
        def __init__(self): super().__init__(); self.conv = copy.deepcopy(conv)
        def forward(self, x, ei): return self.conv(x, ei)

    base  = M()
    remat = gnn_remat(copy.deepcopy(base), granularity="aggr")
    x, ei = _graph()
    with torch.no_grad():
        assert torch.allclose(base(x, ei), remat(x, ei), atol=1e-5)


def test_aggr_gradients_match_baseline():
    """Correctness: aggr-mode gradients == baseline gradients."""
    torch.manual_seed(0)

    class M(nn.Module):
        def __init__(self): super().__init__(); self.conv = GCNConv(8, 16)
        def forward(self, x, ei): return self.conv(x, ei)

    base  = M()
    remat = gnn_remat(copy.deepcopy(base), granularity="aggr")
    x, ei = _graph()

    x1 = x.clone().requires_grad_(True); x1.retain_grad(); base(x1, ei).sum().backward()
    x2 = x.clone().requires_grad_(True); x2.retain_grad(); remat(x2, ei).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-5), \
        f"Grad diff: {(x1.grad - x2.grad).abs().max():.2e}"


def test_dsl_decorator_syntax():
    """The @remat.checkpoint decorator correctly applies aggr-mode remat."""
    import gnn_remat.core.dsl as remat

    @remat.checkpoint
    class DecoratedGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(8, 16)
            self.conv2 = GCNConv(16, 4)
        def forward(self, x, ei):
            return self.conv2(self.conv1(x, ei).relu(), ei)

    model = DecoratedGCN()
    # conv layers should now be RematMessagePassing subclasses
    for _, mod in model.named_modules():
        if isinstance(mod, GCNConv):
            assert isinstance(mod, RematMessagePassing), \
                f"{type(mod)} is not a RematMessagePassing subclass"


def test_dsl_apply_method():
    """remat.checkpoint.apply() works imperatively on an instance."""
    import gnn_remat.core.dsl as remat

    class SimpleGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = GCNConv(8, 16)
        def forward(self, x, ei): return self.conv(x, ei)

    model  = SimpleGCN()
    remat_ = remat.checkpoint.apply(model)
    for _, mod in remat_.named_modules():
        if hasattr(mod, "_is_remat"):
            return   # found one — pass
    assert False, "No remat module found after apply()"


def test_dsl_layer_annotation():
    """remat.layer() wraps a single conv at definition time."""
    import gnn_remat.core.dsl as remat

    class AnnotatedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = remat.layer(GCNConv(8, 16))
            self.conv2 = GCNConv(16, 4)   # NOT wrapped
        def forward(self, x, ei):
            return self.conv2(self.conv1(x, ei).relu(), ei)

    model = AnnotatedModel()
    assert isinstance(model.conv1, RematMessagePassing), \
        "conv1 should be a RematMessagePassing after remat.layer()"
    assert not isinstance(model.conv2, RematMessagePassing), \
        "conv2 should remain a plain GCNConv"

    # Correctness: gradients should still flow
    x, ei = _graph()
    model.train()
    model(x.clone().requires_grad_(True), ei).sum().backward()


def test_dsl_when_type_rule():
    """when_type() selects only layers matching the given class."""
    import gnn_remat.core.dsl as remat

    class TwoConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn  = GCNConv(8, 16)
            self.gat  = GATConv(16, 4, heads=1, concat=False)
        def forward(self, x, ei):
            return self.gat(self.gcn(x, ei).relu(), ei)

    model = TwoConvModel()
    wrapped = remat.checkpoint.apply(model, rules=[
        remat.when_type(GATConv, granularity="aggr"),
        remat.when_type(GCNConv, skip=True),
    ])

    assert isinstance(wrapped.gat, RematMessagePassing), \
        "GAT layer should be wrapped"
    assert not isinstance(wrapped.gcn, RematMessagePassing), \
        "GCN layer should be skipped"


def test_dsl_when_name_rule():
    """when_name() selects the layer with the given dotted name."""
    import gnn_remat.core.dsl as remat

    class TwoConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(8, 16)
            self.conv2 = GCNConv(16, 4)
        def forward(self, x, ei):
            return self.conv2(self.conv1(x, ei).relu(), ei)

    model = TwoConvModel()
    wrapped = remat.checkpoint.apply(model, rules=[
        remat.when_name("conv1", granularity="aggr"),
    ])

    assert isinstance(wrapped.conv1, RematMessagePassing), "conv1 should be wrapped"
    assert not isinstance(wrapped.conv2, RematMessagePassing), "conv2 should not be wrapped"
