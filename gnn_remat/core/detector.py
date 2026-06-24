"""
detector.py
-----------
Scans a PyTorch model and identifies which submodules are PyG MessagePassing
(aggregation) layers — the memory bottleneck GNN-Remat targets.

Public API
----------
    detect(model)                  -> list[LayerInfo]
    filter_by_name(infos, names)   -> list[LayerInfo]
    filter_by_type(infos, types)   -> list[LayerInfo]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Type

import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ── Data class ───────────────────────────────────────────────────────────────

@dataclass
class LayerInfo:
    """Metadata about a single MessagePassing layer found in a model."""

    name: str
    """Dotted attribute path, e.g. 'encoder.conv1'."""

    module: MessagePassing
    """Reference to the actual module."""

    parent: nn.Module
    """Direct parent module — used by transform.py to swap the layer in-place."""

    attr: str
    """Attribute name on parent, i.e.  parent.<attr> is module."""

    def __repr__(self) -> str:
        return (
            f"LayerInfo(name={self.name!r}, "
            f"type={type(self.module).__name__})"
        )


# ── Detection ────────────────────────────────────────────────────────────────

def detect(model: nn.Module) -> List[LayerInfo]:
    """
    Walk every submodule of *model* and return one LayerInfo per
    MessagePassing leaf, in the order they appear in named_modules().

    Parameters
    ----------
    model : nn.Module
        Any PyTorch model, typically a PyG GNN.

    Returns
    -------
    list[LayerInfo]

    Example
    -------
    >>> infos = detect(my_gcn)
    >>> for info in infos:
    ...     print(info.name, type(info.module).__name__)
    conv1  GCNConv
    conv2  GCNConv
    """
    results: List[LayerInfo] = []

    for full_name, module in model.named_modules():
        if not isinstance(module, MessagePassing):
            continue

        # Resolve parent module and the final attribute name
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)

        results.append(
            LayerInfo(
                name=full_name,
                module=module,
                parent=parent,
                attr=parts[-1],
            )
        )

    return results


# ── Filters ──────────────────────────────────────────────────────────────────

def filter_by_name(
    infos: List[LayerInfo],
    names: Sequence[str],
) -> List[LayerInfo]:
    """Keep only layers whose dotted name is in *names*."""
    name_set = set(names)
    return [info for info in infos if info.name in name_set]


def filter_by_type(
    infos: List[LayerInfo],
    types: Sequence[Type[MessagePassing]],
) -> List[LayerInfo]:
    """Keep only layers whose type is (or subclasses) one of *types*."""
    return [
        info for info in infos
        if any(isinstance(info.module, t) for t in types)
    ]


# ── Checkpointing plan (Algorithm 1 output) ──────────────────────────────────

@dataclass
class LayerPlan:
    """One layer's entry in a CheckpointPlan."""
    name: str
    conv_type: str
    checkpoint: bool
    granularity: Optional[str] = None        # "aggr" | "module" | None
    chunk_nodes: Optional[int] = None
    est_savings_bytes: Optional[float] = None
    # ponytail: est_savings unpopulated; thread heuristic.score_layers scores
    #           through when the paper needs the savings column.
    reason: str = ""


@dataclass
class CheckpointPlan:
    """
    The explicit output of the checkpointing algorithm: one decision per
    MessagePassing layer.

    Algorithm (the decision pipeline whose result this records):
        detect MP layers
        -> select       (heuristic.select / mode: all|names|types|auto)
        -> granularity  (aggr for propagate-checkpoint, module for full-layer)
        -> chunk_nodes  (auto_chunk_size, or user value)
        -> apply
    """
    layers: List[LayerPlan]

    def __str__(self) -> str:
        head = (f"{'layer':<24} {'type':<16} {'ckpt':<5} "
                f"{'gran':<7} {'chunk':<8} reason")
        out = [head, "-" * len(head)]
        for p in self.layers:
            out.append(
                f"{p.name:<24} {p.conv_type:<16} "
                f"{'yes' if p.checkpoint else 'no':<5} "
                f"{(p.granularity or '-'):<7} "
                f"{(str(p.chunk_nodes) if p.chunk_nodes else '-'):<8} {p.reason}"
            )
        return "\n".join(out)


def plan_of(model: nn.Module) -> CheckpointPlan:
    """
    Derive the checkpointing plan from an already-transformed model.

    Single source of truth: reads what gnn_remat / the DSL actually applied
    (RematMessagePassing -> aggr, _RematConv -> module, plain MP -> skipped),
    so the reported plan can never drift from runtime behaviour.  Per-layer
    granularity (e.g. mixed via when_type rules) is reflected automatically.
    """
    from .remat_mp import RematMessagePassing  # local imports avoid import cycle
    from .wrapper import _RematConv

    inner_ids: set = set()           # inner convs of _RematConv — don't double count
    rows: List[LayerPlan] = []
    for name, mod in model.named_modules():
        if isinstance(mod, _RematConv):
            inner = mod.conv
            inner_ids.add(id(inner))
            rows.append(LayerPlan(name, type(inner).__name__, True,
                                  "module", reason="module checkpoint"))
        elif isinstance(mod, RematMessagePassing) and getattr(mod, "_is_remat", False):
            base = type(mod).__mro__[2].__name__   # XConvRemat -> RematMP -> XConv
            rows.append(LayerPlan(name, base, True, "aggr",
                                  getattr(mod, "_chunk_nodes", None),
                                  reason="propagate checkpoint"))
        elif isinstance(mod, MessagePassing) and id(mod) not in inner_ids:
            rows.append(LayerPlan(name, type(mod).__name__, False,
                                  reason="not selected"))
    return CheckpointPlan(rows)
