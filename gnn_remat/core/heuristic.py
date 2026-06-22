"""
heuristic.py
------------
Auto-selects which MessagePassing layers are worth checkpointing,
and provides auto_chunk_size() for SAR-inspired chunked propagation.

Scoring
-------
A layer's *score* estimates the net bytes freed by checkpointing it during a
forward+backward pass.

  score = freed_bytes - added_bytes

Where:
  freed_bytes — per-edge intermediates released from the autograd graph when
                propagate() is checkpointed (e.g. attention coefficients,
                weighted messages in GAT).
  added_bytes — propagate *inputs* saved into ctx.saved_tensors by the
                checkpoint (typically node-feature tensors, num_nodes × out).

On CUDA the score is measured directly: run forward+backward once without the
checkpoint, once with, and return (baseline_peak - remat_peak).  This is exact
but adds overhead proportional to the number of candidate layers.

On CPU (no CUDA memory counters) a SAR-aware proxy is used:

  Attention layers (GAT, Transformer) — SAR Case 2:
    Gradients need x_j AND alpha during backward → large per-edge footprint.
    Proxy multiplier: 3.5× (alpha ~ 1 head-scalar per edge on top of x_j).

  Non-attention layers (GCN, SAGE) — SAR Case 1:
    scatter_add backward only needs d_agg and edge_index; baseline PyTorch
    does NOT save the messages tensor.  Checkpointing ADDS the checkpoint
    input cost.  Proxy multiplier: 0.6× (scores are smaller, selected only
    when graph density is high).

A layer is selected when score ≥ threshold (default: 1.0 byte, i.e. any
positive estimate qualifies).  Pass a larger threshold (e.g. 1e6 for 1 MB)
to be more conservative.

auto_chunk_size()
-----------------
Computes a safe chunk_nodes value for SAR-inspired chunked propagation
(remat_mp.RematMessagePassing._propagate_chunked).  Targets keeping peak
per-chunk edge-tensor memory within safety_factor × free GPU memory.

Public API
----------
    _is_attention_based(module)            -> bool
    score_layers(infos, x, edge_index)     -> list[ScoredLayer]
    select(infos, x, edge_index, ...)      -> list[LayerInfo]
    auto_chunk_size(num_nodes, out_channels, ...)  -> int
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import torch

from .detector import LayerInfo


# ── Attention detection (SAR Case 1 / Case 2) ─────────────────────────────────

# Attributes that are present only in attention-based layers (GAT, Transformer).
# SAR classifies these as "Case 2" — gradients require stored input values
# (x_j, alpha), so they both store MORE per-edge data in the baseline AND
# benefit more from rematerialisation.
_ATTENTION_ATTRS = frozenset({
    "att",          # GATConv v1 — attention parameter vector
    "att_src",      # GATConv v2 — split attention (source side)
    "att_dst",      # GATConv v2 — split attention (destination side)
    "lin_query",    # TransformerConv — query projection
    "lin_key",      # TransformerConv — key projection
    "_alpha",       # GATConv internal cache for return_attention_weights
})


def _is_attention_based(module) -> bool:
    """
    SAR Case-2 detector: does this layer use per-edge learnable attention?

    SAR distinguishes:
      Case 1 (GCN, SAGE): backward gradient of scatter-based aggregation
              does NOT need the input message values → near-free backward.
      Case 2 (GAT, Transformer): backward of attention-weighted aggregation
              DOES need x_j and the attention logits → expensive backward;
              also stores large per-edge tensors in the forward pass.

    Case-2 layers are the primary beneficiaries of propagate-level checkpointing
    and chunked propagation.

    Parameters
    ----------
    module : nn.Module
        A MessagePassing layer to inspect.

    Returns
    -------
    bool
        True if the layer stores learnable per-edge attention weights.
    """
    return any(hasattr(module, attr) for attr in _ATTENTION_ATTRS)


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class ScoredLayer:
    """A LayerInfo augmented with an estimated memory-savings score."""

    info: LayerInfo

    score: float
    """
    Estimated net bytes freed by checkpointing this layer.
    Positive  → checkpointing saves memory (good candidate).
    Near zero → break-even; marginal benefit.
    Negative  → checkpointing adds memory (skip this layer).
    """

    selected: bool = False


# ── CUDA measurement ──────────────────────────────────────────────────────────

def _cuda_savings(module, probe_x: torch.Tensor, edge_index: torch.Tensor) -> float:
    """
    Measure peak-memory delta by running forward+backward with and without
    the propagate-level checkpoint.  Returns bytes saved (positive = good).
    """
    from .remat_mp import make_remat_conv  # local import to avoid circularity

    device = probe_x.device

    def _peak(mod) -> int:
        mod.train()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        x = probe_x.clone().requires_grad_(True)
        out = mod(x, edge_index)
        out.sum().backward()
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device)

    # Two runs each to warm up CUDA allocator caches before measuring.
    _peak(module); baseline = _peak(module)
    remat = make_remat_conv(copy.deepcopy(module))
    _peak(remat); with_remat = _peak(remat)

    return float(baseline - with_remat)


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_layers(
    infos: List[LayerInfo],
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> List[ScoredLayer]:
    """
    Score each candidate layer by estimated bytes saved when checkpointed.

    Parameters
    ----------
    infos : list[LayerInfo]
        Candidates from detect().
    x : torch.Tensor
        Node feature matrix  [num_nodes, in_features].
    edge_index : torch.Tensor
        Graph connectivity   [2, num_edges].

    Returns
    -------
    list[ScoredLayer]
        One entry per candidate, sorted by score descending.
    """
    device = x.device
    cuda = device.type == "cuda"
    num_nodes = x.size(0)
    num_edges = edge_index.size(1)

    scored: List[ScoredLayer] = []

    for info in infos:
        module = info.module
        was_training = module.training
        module.eval()

        in_ch = getattr(module, "in_channels", None) or x.size(-1)
        probe_x = torch.zeros(num_nodes, in_ch, device=device)

        with torch.no_grad():
            try:
                out = module(probe_x, edge_index)
            except Exception:
                module.train(was_training)
                continue
        module.train(was_training)

        if cuda:
            # Exact measurement: run the layer with and without checkpoint.
            score = _cuda_savings(module, probe_x, edge_index)
        else:
            # ── SAR-aware CPU proxy ───────────────────────────────────────────
            # Baseline formula (original):
            #   score ≈ (num_edges / num_nodes) × out_channels × dtype_size
            # This is always positive for degree ≥ 1, which caused mode="auto"
            # to always select all layers — including GCN/SAGE where savings are
            # negative at small scale.
            #
            # SAR improvement: apply a Case-1 / Case-2 multiplier.
            #   Case 2 (attention): large per-edge footprint (alpha + x_j × heads)
            #     → multiplier 3.5×  (clearly worth checkpointing at any density)
            #   Case 1 (no attention): only x_j, scatter backward is free in
            #     baseline PyTorch → checkpointing may ADD overhead at low density
            #     → multiplier 0.6×  (selected only when density is high enough
            #        to overcome the checkpoint-input cost)
            out_ch   = out.size(-1)
            elt_size = out.element_size()
            base     = (num_edges / max(num_nodes, 1)) * out_ch * elt_size

            attn_mult = 3.5 if _is_attention_based(module) else 0.6
            score = base * attn_mult

        scored.append(ScoredLayer(info=info, score=score))

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


# ── Selection ─────────────────────────────────────────────────────────────────

def select(
    infos: List[LayerInfo],
    x: torch.Tensor,
    edge_index: torch.Tensor,
    threshold: float = 1.0,
    top_k: Optional[int] = None,
) -> List[LayerInfo]:
    """
    Return the LayerInfos that are worth checkpointing.

    Parameters
    ----------
    infos : list[LayerInfo]
    x : torch.Tensor
    edge_index : torch.Tensor
    threshold : float
        Minimum score (estimated bytes saved) to include a layer.
        Default 1.0 selects any layer with a positive estimate.
        Use a larger value (e.g. 1e6 for 1 MB) to be conservative.
    top_k : int or None
        If set, return at most the top-k highest-scoring layers.

    Returns
    -------
    list[LayerInfo]
    """
    scored = score_layers(infos, x, edge_index)
    candidates = [s for s in scored if s.score >= threshold]
    if top_k is not None:
        candidates = candidates[:top_k]
    for s in candidates:
        s.selected = True
    return [s.info for s in candidates]


# ── Adaptive chunk size (SAR-inspired) ────────────────────────────────────────

def auto_chunk_size(
    num_nodes: int,
    out_channels: int,
    num_heads: int = 1,
    avg_degree: float = 10.0,
    dtype: torch.dtype = torch.float32,
    safety_factor: float = 0.25,
) -> int:
    """
    Compute a safe chunk_nodes value for SAR-inspired chunked propagation.

    Estimates the largest destination-node chunk that keeps peak per-chunk
    edge-tensor memory (x_j + alpha) within *safety_factor* × free GPU memory.

    The SAR paper targets keeping at most 2 graph partitions in memory at once
    per worker (achieving O(2/N) memory scaling with N workers).  This function
    does the analogous calculation for a single GPU: given available memory and
    per-edge costs, how many destination nodes can one chunk safely process?

    Parameters
    ----------
    num_nodes : int
        Total number of nodes in the graph.  The returned value is capped here.
    out_channels : int
        Output feature dimension of the GNN layer.  Determines the size of x_j.
    num_heads : int
        Number of attention heads (1 for non-attention layers like GCN/SAGE).
        Determines the size of alpha per edge.
    avg_degree : float
        Average number of incoming edges per node.  Used to convert an edge
        memory budget to a node chunk budget.  Default 10 is conservative for
        most real graphs.
    dtype : torch.dtype
        Feature tensor dtype.  Affects per-element byte cost.
    safety_factor : float
        Fraction of free GPU memory (or 1 GB on CPU) to dedicate to per-chunk
        edge tensors.  Lower values leave more headroom for activations and
        model parameters.  Default 0.25 (25 % of free memory).

    Returns
    -------
    int
        Recommended chunk_nodes.  Pass this to gnn_remat(chunk_nodes=...) or
        make_remat_conv(conv, chunk_nodes=...).

    Examples
    --------
    >>> from gnn_remat import gnn_remat, auto_chunk_size
    >>> chunk = auto_chunk_size(50_000, out_channels=256, num_heads=4)
    >>> model  = gnn_remat(model, chunk_nodes=chunk)

    >>> # For non-attention layers (num_heads=1):
    >>> chunk = auto_chunk_size(200_000, out_channels=256)
    >>> model  = gnn_remat(model, chunk_nodes=chunk)
    """
    # ── Memory budget ─────────────────────────────────────────────────────────
    # torch.cuda.is_available() can return True even when CUDA initialisation
    # fails lazily (e.g. broken triton install → DeferredCudaCallError).
    # We therefore guard the actual mem_get_info() call in a try/except so
    # auto_chunk_size() always returns a sensible value regardless of the
    # CUDA environment.
    budget = 1 * 1024 ** 3  # 1 GB safe default (CPU / CUDA-unavailable)
    if torch.cuda.is_available():
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            budget = int(free_bytes * safety_factor)
        except Exception:
            # CUDA detected but unusable (driver mismatch, triton error, etc.)
            # Fall back to the 1 GB CPU budget already set above.
            pass

    # ── Per-edge cost ─────────────────────────────────────────────────────────
    # x_j  : [out_channels] floats per edge   (gathered source features)
    # alpha : [num_heads]   floats per edge    (attention coefficients)
    elt_size: dict[torch.dtype, int] = {
        torch.float64:  8,
        torch.float32:  4,
        torch.float16:  2,
        torch.bfloat16: 2,
    }
    bytes_per_elt  = elt_size.get(dtype, 4)
    bytes_per_edge = (out_channels + num_heads) * bytes_per_elt

    # ── Convert edge budget → node budget ─────────────────────────────────────
    edge_budget = max(budget // bytes_per_edge, 1)
    chunk_nodes = max(1_000, int(edge_budget / max(avg_degree, 1.0)))

    # Cap at total node count (one chunk = whole graph → falls back to regular ckpt)
    return min(chunk_nodes, num_nodes)
