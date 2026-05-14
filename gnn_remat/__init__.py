"""
gnn_remat
---------
Aggregation-granular rematerialization for PyTorch Geometric.

Three usage styles
------------------
Functional (one-liner):
    from gnn_remat import gnn_remat
    model = gnn_remat(model)                 # propagate-level checkpoint, all layers
    model = gnn_remat(model, granularity="module")  # full-layer checkpoint

DSL / declarative (class decorator):
    import gnn_remat.core.dsl as remat

    @remat.checkpoint
    class MyGAT(nn.Module): ...

    @remat.checkpoint(granularity="aggr", layers=["conv1"])
    class MyGCN(nn.Module): ...

Layer annotation (policy at definition):
    class MyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = remat.layer(GCNConv(16, 64))
            self.conv2 = remat.layer(GATConv(64, 8, heads=4))

Composable rules:
    @remat.checkpoint(rules=[
        remat.when_type(GATConv, granularity="aggr"),
        remat.when_type(GCNConv, skip=True),
    ])
    class MyMixedGNN(nn.Module): ...
"""

from __future__ import annotations

import copy, logging
from typing import List, Optional, Sequence, Type

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .core.detector import LayerInfo, detect as _detect
from .core.detector import filter_by_name, filter_by_type
from .core.heuristic import select as _heuristic_select
from .core.dsl import _apply_to_model, AGGR, MODULE
from .core.wrapper import _RematConv
from .core.remat_mp import RematMessagePassing

__version__ = "0.2.0"
__all__     = ["gnn_remat", "remove_remat", "detect", "LayerInfo"]

logger = logging.getLogger(__name__)

def gnn_remat(
    model: nn.Module,
    *,
    mode: str = "all",
    granularity: str = "aggr",
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

    mode : {"all", "auto", "names", "types"}
        How to select which layers to checkpoint:

        * ``"all"``    (default) — every MessagePassing layer.
        * ``"names"``  — only the layers listed in *layers*.
        * ``"types"``  — only layers whose class is in *layer_types*.
        * ``"auto"``   — heuristic: select layers where memory saving
                         justifies recompute cost.  Requires *x* and
                         *edge_index*.

    granularity : {"aggr", "module"}
        What to checkpoint:

        "aggr"    (default, recommended)
            Checkpoints only the scatter-aggregation step inside propagate().
            Linear projections and attention coefficients stay in the autograd
            graph — they are NOT recomputed. This is the novel contribution:
            finer granularity than torch.utils.checkpoint.

        "module"
            Checkpoints the entire MessagePassing layer (linear + attention +
            scatter). Same as torch.utils.checkpoint. Provided for comparison.
                                              
    layers : list[str], optional
        Dotted layer names for mode="names", e.g. ["conv1", "enc.conv2"].

    layer_types : list[type], optional
        MessagePassing subclasses for mode="types", e.g. [GATConv].

    x : torch.Tensor, optional
        Sample node features for mode="auto" profiling.

    edge_index : torch.Tensor, optional
        Sample graph connectivity for mode="auto" profiling.

    heuristic_threshold : float
        Minimum estimated bytes saved for a layer to be selected in
        mode="auto".  Default 1.0 (any positive estimate qualifies).
        Pass a larger value, e.g. 1e6, to require at least 1 MB savings.

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
    >>> model = gnn_remat(model)          
    >>> model = gnn_remat(model, granularity="module")  # module-level baseline
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
        targets = None  # _apply_to_model treats None as "all"

    elif mode == "names":
        if not layers:
            raise ValueError("mode='names' requires layers=[...] to be set.")
        targets = list(layers)

    elif mode == "types":
        if not layer_types:
            raise ValueError("mode='types' requires layer_types=[...] to be set.")
        filtered     = filter_by_type(all_infos, layer_types)
        targets = [i.name for i in filtered]

    elif mode == "auto":
        if x is None or edge_index is None:
            raise ValueError(
                "mode='auto' requires sample x and edge_index tensors "
                "for one-shot profiling."
            )
        selected = _heuristic_select(
            all_infos,
            x=x,
            edge_index=edge_index,
            threshold=heuristic_threshold,
            top_k=heuristic_top_k,
        )
        targets = [i.name for i in selected]
        

    else:
        raise ValueError(
            f"Unknown mode {mode!r}. "
            "Choose from: 'all', 'names', 'types', 'auto'."
        )

    gran= AGGR if granularity == "aggr" else MODULE
    return _apply_to_model(model, gran, layers=targets)


#Convenience wrappers

def remove_remat(model: nn.Module) -> nn.Module:
    """Strip all GNN-Remat wrappers, returning a plain model."""
    model = copy.deepcopy(model)
    for name, mod in list(model.named_modules()):
        # Strip both kinds of wrapper
        is_remat_conv = isinstance(mod, _RematConv)
        is_remat_mp   = isinstance(mod, RematMessagePassing) and \
                        getattr(mod, "_is_remat", False)
        if not (is_remat_conv or is_remat_mp):
            continue
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        # Restore original conv (unwrap _RematConv) or revert class
        if is_remat_conv:
            setattr(parent, parts[-1], mod.conv)
        else:
            # Revert dynamic class to base class
            base_cls = type(mod).__mro__[2]  # skip RematMP, ConvRemat -> ConvClass
            mod.__class__ = base_cls
    return model


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
