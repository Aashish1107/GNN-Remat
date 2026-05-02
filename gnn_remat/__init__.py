"""
gnn_remat
---------
Aggregation-granular rematerialization for PyTorch Geometric.

Public API
----------
    gnn_remat(model, ...)    apply rematerialization to a model
    remove_remat(model)      strip all remat wrappers (for export/inference)
    detect(model)            inspect which layers would be wrapped
    LayerInfo                metadata dataclass returned by detect()

Modes
-----
    "all"   (default) — wrap every MessagePassing layer
    "auto"  — use the heuristic to pick only cost-effective layers
    "names" — wrap layers by explicit dotted name list
    "types" — wrap layers by MessagePassing subclass

Quick start
-----------
    from gnn_remat import gnn_remat

    model = MyGCN()
    model = gnn_remat(model)                          # all layers, default mode
    model = gnn_remat(model, layers=["conv1"])         # by name
    model = gnn_remat(model, mode="auto", x=x,         # heuristic
                      edge_index=edge_index)
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .core.detector import LayerInfo, detect as _detect
from .core.detector import filter_by_name, filter_by_type
from .core.heuristic import select as _heuristic_select
from .core.transform import apply as _apply
from .core.transform import remove as _remove

__version__ = "0.1.0"
__all__     = ["gnn_remat", "remove_remat", "detect", "LayerInfo"]

logger = logging.getLogger(__name__)

def gnn_remat(
    model: nn.Module,
    *,
    mode: str = "all",
    layers: Optional[Sequence[str]] = None,
    layer_types: Optional[Sequence[Type[MessagePassing]]] = None,
    x: Optional[torch.Tensor] = None,
    edge_index: Optional[torch.Tensor] = None,
    heuristic_threshold: float = 1.0,
    heuristic_top_k: Optional[int] = None,
    verbose: bool = False,
) -> nn.Module:
    """
    Apply aggregation-granular rematerialization to *model*.

    Returns a new model — the original is never modified.

    Parameters
    ----------
    model : nn.Module
        Any PyG model containing MessagePassing layers.

    mode : {"all", "auto", "names", "types"}
        How to select which layers to checkpoint:

        * ``"all"``    (default) — every MessagePassing layer.
        * ``"names"``  — only the layers listed in *layers*.
        * ``"types"``  — only layers whose class is in *layer_types*.
        * ``"auto"``   — heuristic: select layers where memory saving
                         justifies recompute cost.  Requires *x* and
                         *edge_index*.

    layers : list[str], optional
        Dotted layer names for mode="names", e.g. ["conv1", "enc.conv2"].

    layer_types : list[type], optional
        MessagePassing subclasses for mode="types", e.g. [GATConv].

    x : torch.Tensor, optional
        Sample node features for mode="auto" profiling.

    edge_index : torch.Tensor, optional
        Sample graph connectivity for mode="auto" profiling.

    heuristic_threshold : float
        Score cutoff for mode="auto".  Default 1.0.

    heuristic_top_k : int, optional
        Maximum layers to select in mode="auto".

    verbose : bool
        If True, log each layer that gets wrapped.

    Returns
    -------
    nn.Module
        Deep copy of *model* with selected layers checkpointed.

    Raises
    ------
    ValueError
        If an unsupported mode is requested, or required args are missing.

    Examples
    --------
    >>> model = gnn_remat(model)                          # wrap all
    >>> model = gnn_remat(model, layers=["conv1"])         # by name
    >>> model = gnn_remat(model, layer_types=[GATConv])    # by type
    >>> model = gnn_remat(model, mode="auto",              # heuristic
    ...                   x=x, edge_index=edge_index)
    """
    if verbose:
        logging.basicConfig(level=logging.INFO)

    all_infos = _detect(model)

    if not all_infos:
        logger.warning(
            "gnn_remat: no MessagePassing layers found in model — "
            "returning model unchanged."
        )
        return model

    #Select target layers based on mode
    if mode == "all":
        targets = all_infos

    elif mode == "names":
        if not layers:
            raise ValueError("mode='names' requires layers=[...] to be set.")
        targets = filter_by_name(all_infos, layers)

    elif mode == "types":
        if not layer_types:
            raise ValueError("mode='types' requires layer_types=[...] to be set.")
        targets = filter_by_type(all_infos, layer_types)

    elif mode == "auto":
        if x is None or edge_index is None:
            raise ValueError(
                "mode='auto' requires sample x and edge_index tensors "
                "for one-shot profiling."
            )
        targets = _heuristic_select(
            all_infos,
            x=x,
            edge_index=edge_index,
            threshold=heuristic_threshold,
            top_k=heuristic_top_k,
        )

    else:
        raise ValueError(
            f"Unknown mode {mode!r}. "
            "Choose from: 'all', 'names', 'types', 'auto'."
        )

    if not targets:
        logger.warning(
            "gnn_remat: mode=%r selected 0 layers — returning model unchanged.",
            mode,
        )
        return model

    return _apply(model, targets)


#Convenience wrappers

def remove_remat(model: nn.Module) -> nn.Module:
    """
    Strip all _RematConv wrappers from *model* (returns a new model).

    Useful before exporting / running inference when you want a plain model.
    """
    return _remove(model)


def detect(model: nn.Module) -> List[LayerInfo]:
    """
    Return a list of LayerInfo for every MessagePassing layer in *model*.

    Use this to inspect what gnn_remat() would wrap before committing.

    Example
    -------
    >>> for info in detect(model):
    ...     print(info.name, type(info.module).__name__)
    """
    return _detect(model)
