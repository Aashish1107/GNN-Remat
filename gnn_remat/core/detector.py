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
from typing import List, Sequence, Type

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
