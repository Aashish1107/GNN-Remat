"""
heuristic.py
------------
Auto-selects which MessagePassing layers are worth checkpointing based on
the ratio of memory saved to the estimated recompute cost.

A layer is selected when:

    output_bytes / recompute_flops  >  threshold

i.e. "checkpointing this layer saves a lot of memory per unit of extra work."

Public API
----------
    score_layers(infos, x, edge_index)   -> list[ScoredLayer]
    select(infos, x, edge_index, ...)    -> list[LayerInfo]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn

from .detector import LayerInfo


#Data class

@dataclass
class ScoredLayer:
    """A LayerInfo augmented with profiling estimates."""

    info: LayerInfo
    output_bytes: int
    """Bytes the layer's output tensor occupies in GPU memory."""

    recompute_flops: int
    """Rough scatter-add FLOPs during a recompute (|E| * d)."""

    score: float
    """output_bytes / recompute_flops — higher = better candidate."""

    selected: bool = False


#Scoring

def score_layers(
    infos: List[LayerInfo],
    x: torch.Tensor,
    edge_index: torch.Tensor,
) -> List[ScoredLayer]:
    """
    Run one forward pass per candidate layer and record output size and
    estimated recompute cost.

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
    num_edges = edge_index.size(1)
    scored: List[ScoredLayer] = []

    for info in infos:
        module = info.module
        module.eval()

        # Resolve the correct input dimension for this layer.
        # GCNConv / SAGEConv / GATConv all expose .in_channels.
        in_ch = getattr(module, "in_channels", None)
        if in_ch is None:
            in_ch = x.size(-1)

        # Build a synthetic input tensor with the right feature dimension.
        probe_x = torch.zeros(x.size(0), in_ch)

        with torch.no_grad():
            try:
                out = module(probe_x, edge_index)
            except Exception:
                # Skip layers that can't run (e.g. unusual signatures)
                continue

        output_bytes   = out.numel() * out.element_size()
        # Scatter-add cost: each edge contributes one addition per feature
        recompute_flops = max(num_edges * out.size(-1), 1)
        score           = output_bytes / recompute_flops

        scored.append(
            ScoredLayer(
                info=info,
                output_bytes=output_bytes,
                recompute_flops=recompute_flops,
                score=score,
            )
        )

    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


#Selection

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
        All MessagePassing layers detected in the model.
    x : torch.Tensor
        Sample node feature matrix used for one-shot profiling.
    edge_index : torch.Tensor
        Sample graph connectivity used for one-shot profiling.
    threshold : float
        Minimum score (output_bytes / recompute_flops) required for a
        layer to be selected.  Default 1.0 means "save at least 1 byte
        per FLOP of recompute".
    top_k : int or None
        If set, return at most the top-k highest-scoring layers regardless
        of threshold.

    Returns
    -------
    list[LayerInfo]
        Layers recommended for checkpointing.
    """
    scored = score_layers(infos, x, edge_index)

    candidates = [s for s in scored if s.score >= threshold]

    if top_k is not None:
        candidates = candidates[:top_k]

    for s in candidates:
        s.selected = True

    return [s.info for s in candidates]
