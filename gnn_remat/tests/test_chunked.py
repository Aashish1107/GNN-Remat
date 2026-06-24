"""
test_chunked.py
---------------
Tests for SAR-inspired destination-node chunked propagation and the
attention-aware heuristic improvements.

Coverage:
  1. Chunked propagation output correctness  (GCN, SAGE, GAT)
  2. Chunked propagation gradient correctness
  3. Chunk count drives chunk size (tiny chunk → many chunks)
  4. Eval mode: chunking gate deactivated (no overhead)
  5. Empty edge ranges handled gracefully (isolated nodes)
  6. auto_chunk_size() returns sensible values
  7. _is_attention_based() detects attention layers correctly
  8. gnn_remat(chunk_nodes=...) integration test (output + gradient)
  9. DSL layer() and checkpoint.apply() thread chunk_nodes correctly
 10. Score ordering: attention layers score higher than non-attention (CPU proxy)
"""
import copy

import pytest
import torch
import torch.nn as nn
from torch_geometric.nn import (
    GCNConv, GATConv, SAGEConv, TransformerConv, MessagePassing,
)

from gnn_remat import gnn_remat, auto_chunk_size, detect
from gnn_remat.core.remat_mp import RematMessagePassing, make_remat_conv
from gnn_remat.core.heuristic import _is_attention_based, score_layers
from gnn_remat.core.detector import detect as _detect
import gnn_remat.core.dsl as remat


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_graph(num_nodes=60, num_feats=16, num_edges=200, seed=0):
    """Return (x, edge_index) for a random homogeneous graph."""
    torch.manual_seed(seed)
    x  = torch.randn(num_nodes, num_feats, requires_grad=True)
    ei = torch.stack([
        torch.randint(0, num_nodes, (num_edges,)),
        torch.randint(0, num_nodes, (num_edges,)),
    ])
    return x, ei


class _SimpleGCN(nn.Module):
    def __init__(self, in_c=16, out_c=8):
        super().__init__()
        self.conv = GCNConv(in_c, out_c)
    def forward(self, x, ei):
        return self.conv(x, ei)


class _SimpleGAT(nn.Module):
    def __init__(self, in_c=16, out_c=8, heads=2):
        super().__init__()
        self.conv = GATConv(in_c, out_c, heads=heads, concat=False)
    def forward(self, x, ei):
        return self.conv(x, ei)


class _SimpleSAGE(nn.Module):
    def __init__(self, in_c=16, out_c=8):
        super().__init__()
        self.conv = SAGEConv(in_c, out_c)
    def forward(self, x, ei):
        return self.conv(x, ei)


class _SimpleTransformer(nn.Module):
    """TransformerConv exercises the full-size FALLBACK chunk path: it passes
    per-dst `query` and per-src `key`/`value` (no `x`), which cannot be safely
    local-remapped by shape on a homogeneous graph."""
    def __init__(self, in_c=16, out_c=8, heads=2):
        super().__init__()
        self.conv = TransformerConv(in_c, out_c, heads=heads, concat=False)
    def forward(self, x, ei):
        return self.conv(x, ei)


def _baseline_output(model_cls, x, ei, **model_kwargs):
    """Forward output of a fresh baseline model (no remat)."""
    m = model_cls(**model_kwargs)
    m.eval()
    with torch.no_grad():
        return m(x, ei)


# ── 1. Output correctness ─────────────────────────────────────────────────────

@pytest.mark.parametrize("model_cls,kwargs", [
    (_SimpleGCN, {}),
    (_SimpleSAGE, {}),
    (_SimpleGAT, {"heads": 2}),
    (_SimpleTransformer, {"heads": 2}),  # fallback (full-size) chunk path
])
def test_chunked_output_matches_baseline(model_cls, kwargs):
    """
    With chunk_nodes set, output must be numerically identical to no-chunk remat.
    Tests GCN (add aggr), SAGE (mean aggr), and GAT (attention softmax).
    """
    x, ei = _random_graph(num_nodes=50, num_edges=150)

    base_model = model_cls(**kwargs)
    base_model.eval()
    with torch.no_grad():
        expected = base_model(x, ei)

    # Use a very small chunk_nodes (5) to force many chunks and stress the logic
    chunked_model = gnn_remat(copy.deepcopy(base_model), chunk_nodes=5)
    chunked_model.eval()
    with torch.no_grad():
        actual = chunked_model(x, ei)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"{model_cls.__name__}: max diff = {(expected - actual).abs().max():.2e}"
    )


def test_chunked_output_single_chunk_equals_no_chunk():
    """
    When chunk_nodes >= num_nodes, the chunked path processes the whole graph
    in one pass — should be identical to plain checkpointing.
    """
    x, ei = _random_graph(num_nodes=30, num_edges=90)

    base  = _SimpleGCN()
    base.eval()
    with torch.no_grad():
        expected = base(x, ei)

    # chunk_nodes larger than num_nodes → single chunk → same as no chunking
    chunked = gnn_remat(copy.deepcopy(base), chunk_nodes=1000)
    chunked.eval()
    with torch.no_grad():
        actual = chunked(x, ei)

    assert torch.allclose(expected, actual, atol=1e-5)


# ── 2. Gradient correctness ───────────────────────────────────────────────────

@pytest.mark.parametrize("model_cls,kwargs", [
    (_SimpleGCN, {}),
    (_SimpleSAGE, {}),
    (_SimpleGAT, {"heads": 2}),
    (_SimpleTransformer, {"heads": 2}),  # fallback (full-size) chunk path
])
def test_chunked_gradients_match_baseline(model_cls, kwargs):
    """
    Input gradients from chunked forward+backward must match the baseline.
    This verifies that torch.cat's CatBackward correctly fans out upstream
    gradients through each chunk's independent checkpoint.
    """
    x, ei = _random_graph(num_nodes=50, num_edges=150)

    base = model_cls(**kwargs)

    # Baseline gradient
    x1 = x.clone().detach().requires_grad_(True)
    base.train()
    base(x1, ei).sum().backward()
    grad_base = x1.grad.clone()

    # Chunked gradient (small chunks to exercise all code paths)
    chunked = gnn_remat(copy.deepcopy(base), chunk_nodes=8)
    x2 = x.clone().detach().requires_grad_(True)
    chunked.train()
    chunked(x2, ei).sum().backward()
    grad_chunked = x2.grad.clone()

    assert torch.allclose(grad_base, grad_chunked, atol=1e-4), (
        f"{model_cls.__name__}: max grad diff = "
        f"{(grad_base - grad_chunked).abs().max():.2e}"
    )


# ── 3. Chunk count scales with chunk_nodes ────────────────────────────────────

def test_chunk_count_is_inversely_proportional():
    """
    Smaller chunk_nodes → more chunks.  We verify this by counting how many
    times _run_chunk is called (monkey-patched counter).
    """
    x, ei = _random_graph(num_nodes=40, num_edges=120)

    base   = _SimpleGCN()
    model  = gnn_remat(copy.deepcopy(base), chunk_nodes=5)  # ceil(40/5) = 8 chunks max
    model2 = gnn_remat(copy.deepcopy(base), chunk_nodes=20) # ceil(40/20) = 2 chunks max

    call_counts = [0, 0]

    # Patch _run_chunk on the inner conv to count calls
    for idx, m in enumerate([model, model2]):
        for _, mod in m.named_modules():
            if isinstance(mod, RematMessagePassing):
                original = mod._run_chunk
                _idx = idx  # capture
                def _counting(cei, fs, fa, sp, _orig=original, _i=_idx):
                    call_counts[_i] += 1
                    return _orig(cei, fs, fa, sp)
                mod._run_chunk = _counting
                break

    model.train();  model(x.clone().requires_grad_(True), ei).sum().backward()
    model2.train(); model2(x.clone().requires_grad_(True), ei).sum().backward()

    # model (chunk_nodes=5) should call _run_chunk more times than model2 (chunk_nodes=20)
    assert call_counts[0] > call_counts[1], (
        f"Smaller chunk_nodes should produce more chunks: "
        f"chunk5={call_counts[0]}, chunk20={call_counts[1]}"
    )


# ── 4. Eval mode gate ─────────────────────────────────────────────────────────

def test_eval_mode_skips_chunking():
    """
    In eval mode, _chunk_nodes should have no effect — output is identical
    to baseline eval and no checkpoint overhead is incurred.
    """
    x, ei = _random_graph(num_nodes=30, num_edges=90)
    base  = _SimpleGAT(heads=2)

    base.eval()
    with torch.no_grad():
        expected = base(x, ei)

    chunked = gnn_remat(copy.deepcopy(base), chunk_nodes=5)
    chunked.eval()
    with torch.no_grad():
        actual = chunked(x, ei)

    assert torch.allclose(expected, actual, atol=1e-5)


# ── 5. Isolated nodes (empty edge chunks) ────────────────────────────────────

def test_isolated_nodes_produce_zeros():
    """
    If some destination-node ranges have no incoming edges (isolated nodes),
    _propagate_chunked must fill those slots with zeros, not crash.
    """
    torch.manual_seed(42)
    num_nodes = 20
    x  = torch.randn(num_nodes, 8)

    # Only nodes 0..9 appear as destinations; nodes 10..19 are isolated
    src = torch.randint(0, num_nodes, (40,))
    dst = torch.randint(0, 10, (40,))        # destinations only in [0, 10)
    ei  = torch.stack([src, dst])

    base = _SimpleGCN(in_c=8, out_c=4)
    base.eval()
    with torch.no_grad():
        expected = base(x, ei)

    # chunk_nodes=5 means chunks: [0-5), [5-10), [10-15), [15-20)
    # last two chunks have no edges → must produce zero rows
    chunked = gnn_remat(copy.deepcopy(base), chunk_nodes=5)
    chunked.eval()
    with torch.no_grad():
        actual = chunked(x, ei)

    assert torch.allclose(expected, actual, atol=1e-5), (
        f"Isolated-node output mismatch: max diff = "
        f"{(expected - actual).abs().max():.2e}"
    )
    # Isolated nodes should have zero output from aggregation
    # (their output comes from the linear on zero aggregate, which varies by model)
    # — just check the shapes match
    assert actual.shape == expected.shape


# ── 6. auto_chunk_size() ─────────────────────────────────────────────────────

def test_auto_chunk_size_returns_positive():
    chunk = auto_chunk_size(num_nodes=50_000, out_channels=256)
    assert chunk > 0, "auto_chunk_size must return a positive integer"


def test_auto_chunk_size_capped_at_num_nodes():
    chunk = auto_chunk_size(num_nodes=100, out_channels=256)
    assert chunk <= 100, "auto_chunk_size must not exceed num_nodes"


def test_auto_chunk_size_decreases_with_more_heads():
    """More attention heads → larger per-edge cost → smaller chunk needed."""
    chunk_1h = auto_chunk_size(50_000, out_channels=256, num_heads=1)
    chunk_8h = auto_chunk_size(50_000, out_channels=256, num_heads=8)
    assert chunk_8h <= chunk_1h, (
        "More heads should require a smaller chunk_nodes "
        f"(1h={chunk_1h}, 8h={chunk_8h})"
    )


def test_auto_chunk_size_minimum_floor():
    """Result is always at least 1000 (floor defined in heuristic.py)."""
    chunk = auto_chunk_size(num_nodes=5_000_000, out_channels=4096, num_heads=64)
    assert chunk >= 1_000


# ── 7. _is_attention_based() ─────────────────────────────────────────────────

def test_gat_is_attention_based():
    conv = GATConv(16, 8, heads=2)
    assert _is_attention_based(conv), "GATConv should be detected as attention-based"


def test_gcn_is_not_attention_based():
    conv = GCNConv(16, 8)
    assert not _is_attention_based(conv), "GCNConv should NOT be attention-based"


def test_sage_is_not_attention_based():
    conv = SAGEConv(16, 8)
    assert not _is_attention_based(conv), "SAGEConv should NOT be attention-based"


# ── 8. Integration: gnn_remat(chunk_nodes=...) ───────────────────────────────

def test_gnn_remat_chunk_nodes_sets_attribute():
    """gnn_remat(chunk_nodes=K) should set _chunk_nodes=K on each wrapped conv."""
    model   = _SimpleGCN()
    wrapped = gnn_remat(model, chunk_nodes=42)
    for _, mod in wrapped.named_modules():
        if isinstance(mod, RematMessagePassing):
            assert mod._chunk_nodes == 42, (
                f"Expected _chunk_nodes=42, got {mod._chunk_nodes}"
            )
            return
    pytest.fail("No RematMessagePassing layer found in wrapped model")


def test_gnn_remat_no_chunk_nodes_is_none():
    """gnn_remat() without chunk_nodes leaves _chunk_nodes=None (standard ckpt only)."""
    model   = _SimpleGCN()
    wrapped = gnn_remat(model)
    for _, mod in wrapped.named_modules():
        if isinstance(mod, RematMessagePassing):
            assert mod._chunk_nodes is None
            return
    pytest.fail("No RematMessagePassing layer found")


@pytest.mark.parametrize("chunk", [3, 7, 15, 50])
def test_gnn_remat_chunked_end_to_end(chunk):
    """Full forward+backward with various chunk sizes produces correct gradients."""
    x, ei = _random_graph(num_nodes=40, num_edges=120)

    base    = _SimpleGAT(heads=2)
    wrapped = gnn_remat(copy.deepcopy(base), chunk_nodes=chunk)

    x1 = x.clone().detach().requires_grad_(True); base.train()
    base(x1, ei).sum().backward()

    x2 = x.clone().detach().requires_grad_(True); wrapped.train()
    wrapped(x2, ei).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-4), (
        f"chunk={chunk}: max grad diff = {(x1.grad - x2.grad).abs().max():.2e}"
    )


# ── 9. DSL threads chunk_nodes correctly ─────────────────────────────────────

def test_dsl_layer_chunk_nodes():
    """remat.layer(conv, chunk_nodes=K) sets _chunk_nodes on the returned conv."""
    conv    = GCNConv(16, 8)
    wrapped = remat.layer(conv, chunk_nodes=99)
    assert isinstance(wrapped, RematMessagePassing)
    assert wrapped._chunk_nodes == 99


def test_dsl_apply_chunk_nodes():
    """remat.checkpoint.apply(model, chunk_nodes=K) propagates chunk_nodes."""
    model   = _SimpleGCN()
    wrapped = remat.checkpoint.apply(model, chunk_nodes=77)
    for _, mod in wrapped.named_modules():
        if isinstance(mod, RematMessagePassing):
            assert mod._chunk_nodes == 77
            return
    pytest.fail("No RematMessagePassing found after checkpoint.apply()")


def test_dsl_decorator_chunk_nodes():
    """@remat.checkpoint(chunk_nodes=K) propagates chunk_nodes via decorator."""

    @remat.checkpoint(chunk_nodes=33)
    class MyGCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = GCNConv(16, 8)
        def forward(self, x, ei):
            return self.conv(x, ei)

    model = MyGCN()
    for _, mod in model.named_modules():
        if isinstance(mod, RematMessagePassing):
            assert mod._chunk_nodes == 33
            return
    pytest.fail("No RematMessagePassing found after @checkpoint(chunk_nodes=33)")


# ── 10. SAR-aware scoring: attention > non-attention ─────────────────────────

def test_attention_layers_score_higher_than_non_attention():
    """
    On CPU, the SAR-aware proxy must score GAT higher than GCN
    so mode='auto' prefers checkpointing attention layers first.
    """
    class MixedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.gcn = GCNConv(16, 8)
            self.gat = GATConv(8, 4, heads=1, concat=False)
        def forward(self, x, ei):
            return self.gat(self.gcn(x, ei).relu(), ei)

    model = MixedModel()
    x, ei = _random_graph(num_nodes=50, num_edges=150)

    infos  = _detect(model)
    scored = score_layers(infos, x, ei)

    # Find GAT and GCN scores
    scores = {s.info.name: s.score for s in scored}
    assert "gat" in scores, "GAT layer not found in scored layers"
    assert "gcn" in scores, "GCN layer not found in scored layers"
    assert scores["gat"] > scores["gcn"], (
        f"GAT should score higher than GCN: GAT={scores['gat']:.0f}, GCN={scores['gcn']:.0f}"
    )


# ── 11. Chunk path selection: local remap vs full-size fallback ──────────────

def test_can_local_remap_x_based_convs():
    """GCN/SAGE/GAT pass only the recognised `x` node kwarg → local remap (the
    path that removes the O(num_dst x F) scatter-output floor)."""
    rmp = make_remat_conv(GCNConv(16, 8))
    N, E = 50, 200
    x = torch.randn(N, 16)
    # single-tensor x (homogeneous)
    assert rmp._can_local_remap({"x": x}, N, N, E) is True
    # src/dst tuple x (bipartite-style)
    assert rmp._can_local_remap({"x": (x, x)}, N, N, E) is True
    # x + a per-edge kwarg (e.g. GAT alpha [E, H]) is still local-remappable
    alpha = torch.randn(E, 2)
    assert rmp._can_local_remap({"x": x, "alpha": alpha}, N, N, E) is True


def test_can_local_remap_falls_back_on_unhandled_dst_kwarg():
    """An unrecognised destination-indexed node kwarg (size == num_dst, not `x`)
    cannot be classified by shape on a homogeneous graph → safe full-size fallback.
    This is the TransformerConv `query` situation."""
    rmp = make_remat_conv(GCNConv(16, 8))
    N, E = 50, 200
    query = torch.randn(N, 2, 8)   # per-destination-node tensor, not named `x`
    key   = torch.randn(N, 2, 8)   # per-source-node tensor
    assert rmp._can_local_remap(
        {"query": query, "key": key}, N, N, E
    ) is False


def test_transformer_runtime_uses_fallback_and_is_correct():
    """End-to-end: TransformerConv chunked output matches baseline even though
    it takes the full-size fallback path (covered by the parametrized tests too,
    asserted here explicitly for the fallback contract)."""
    x, ei = _random_graph(num_nodes=40, num_edges=140)
    base = _SimpleTransformer(heads=2)

    x1 = x.clone().detach().requires_grad_(True); base.train()
    base(x1, ei).sum().backward()

    chunked = gnn_remat(copy.deepcopy(base), chunk_nodes=6)
    x2 = x.clone().detach().requires_grad_(True); chunked.train()
    chunked(x2, ei).sum().backward()

    assert torch.allclose(x1.grad, x2.grad, atol=1e-4), (
        f"Transformer fallback grad diff = {(x1.grad - x2.grad).abs().max():.2e}"
    )


# ── 12. Adaptability: flow="target_to_source" and SparseTensor input ─────────

class _RevConv(MessagePassing):
    """Minimal sum-aggregation conv that aggregates at edge_index[0]
    (flow='target_to_source') — exercises the flow-generic chunk path."""
    def __init__(self, in_c, out_c):
        super().__init__(aggr="add", flow="target_to_source")
        self.lin = nn.Linear(in_c, out_c, bias=False)
    def forward(self, x, ei):
        return self.propagate(ei, x=self.lin(x))
    def message(self, x_j):
        return x_j


def test_chunked_target_to_source_matches_baseline():
    """Chunking must group by the correct central row for flow='target_to_source'."""
    x, ei = _random_graph(num_nodes=50, num_edges=180)

    class M(nn.Module):
        def __init__(self): super().__init__(); self.c = _RevConv(16, 8)
        def forward(self, x, ei): return self.c(x, ei)

    base = M()
    x1 = x.clone().detach().requires_grad_(True); base.train()
    base(x1, ei).sum().backward()

    chunked = gnn_remat(copy.deepcopy(base), chunk_nodes=7)
    x2 = x.clone().detach().requires_grad_(True); chunked.train()
    chunked(x2, ei).sum().backward()

    base.eval(); chunked.eval()
    with torch.no_grad():
        assert torch.allclose(base(x, ei), chunked(x, ei), atol=1e-5)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-4), (
        f"target_to_source grad diff = {(x1.grad - x2.grad).abs().max():.2e}"
    )


def test_sparse_tensor_input_does_not_crash_with_chunking():
    """SparseTensor adj_t input must fall back to the standard path, not crash."""
    SparseTensor = pytest.importorskip("torch_sparse").SparseTensor
    x, ei = _random_graph(num_nodes=40, num_edges=120)
    adj = SparseTensor(row=ei[0], col=ei[1], sparse_sizes=(40, 40)).t()

    base = _SimpleGCN()
    base.eval()
    with torch.no_grad():
        expected = base(x, adj)

    chunked = gnn_remat(copy.deepcopy(base), chunk_nodes=5)  # would chunk a Tensor
    chunked.eval()
    with torch.no_grad():
        actual = chunked(x, adj)                              # SparseTensor → no chunk
    assert torch.allclose(expected, actual, atol=1e-5)
