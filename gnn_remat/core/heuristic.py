"""
heuristic.py
------------
Auto-selects which MessagePassing layers are worth checkpointing.

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

On CPU (no CUDA memory counters) a proxy is used:
  score ≈ (num_edges / num_nodes) × out_channels × element_size_bytes
This captures the key variable: graph density.  High-degree graphs have more
per-edge tensors; the node-level checkpoint cost is constant per layer.

A layer is selected when score ≥ threshold (default: 1.0 byte, i.e. any
positive estimate qualifies).  Pass a larger threshold (e.g. 1e6 for 1 MB)
to be more conservative.

Public API
----------
    score_layers(infos, x, edge_index)   -> list[ScoredLayer]
    select(infos, x, edge_index, ...)    -> list[LayerInfo]
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Optional

import torch

from .detector import LayerInfo


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
            # CPU proxy: estimated net bytes = edge tensors freed - node tensors added.
            # edge_bytes ≈ num_edges × out_channels × dtype_size  (per-edge intermediates)
            # node_bytes ≈ num_nodes × out_channels × dtype_size  (checkpoint input cost)
            # net ≈ (num_edges - num_nodes) × out_channels × dtype_size
            # Simplified to (num_edges/num_nodes) × out_channels × dtype_size so the
            # ratio is always positive for real graphs (degree ≥ 1).
            out_ch   = out.size(-1)
            elt_size = out.element_size()
            score    = (num_edges / max(num_nodes, 1)) * out_ch * elt_size

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
