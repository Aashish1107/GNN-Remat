"""
transform.py
------------
The compiler pass: takes a model and a list of target LayerInfos, then
swaps each targeted MessagePassing layer with a _RematConv wrapper.

This is the only module that mutates a model. It always operates on a
deep copy so the caller's original model is unchanged.

Public API
----------
    apply(model, layer_infos)   -> nn.Module   (new model, original unchanged)
    remove(model)               -> nn.Module   (strip all _RematConv wrappers)
"""

from __future__ import annotations

import copy
import logging
from typing import List

import torch.nn as nn

from .detector import LayerInfo, detect
from .wrapper import _RematConv, wrap

logger = logging.getLogger(__name__)


#Apply

def apply(
    model: nn.Module,
    layer_infos: List[LayerInfo],
) -> nn.Module:
    """
    Return a deep copy of *model* where every layer in *layer_infos* has
    been replaced by a _RematConv checkpoint wrapper.

    Parameters
    ----------
    model : nn.Module
        Original model (not modified).
    layer_infos : list[LayerInfo]
        Layers to wrap, typically from detector.detect() filtered by
        name / type / heuristic.

    Returns
    -------
    nn.Module
        New model with wrappers applied.  Parameters are shared with
        the originals inside each _RematConv.conv.

    Example
    -------
    >>> infos   = detect(model)
    >>> remat   = apply(model, infos)
    >>> out     = remat(x, edge_index)
    """
    new_model = copy.deepcopy(model)

    # Re-detect on the copy so LayerInfo.parent/.attr point at the copy
    name_to_info = {info.name: info for info in layer_infos}
    fresh_infos  = detect(new_model)

    wrapped_count = 0
    for fresh_info in fresh_infos:
        if fresh_info.name not in name_to_info:
            continue

        setattr(fresh_info.parent, fresh_info.attr, wrap(fresh_info.module))
        wrapped_count += 1
        logger.info(
            "GNN-Remat: wrapped '%s' (%s)",
            fresh_info.name,
            type(fresh_info.module).__name__,
        )

    if wrapped_count == 0:
        logger.warning(
            "GNN-Remat: no layers were wrapped — "
            "check that layer names match the model."
        )

    return new_model


#Remove

def remove(model: nn.Module) -> nn.Module:
    """
    Return a deep copy of *model* with all _RematConv wrappers stripped,
    restoring the original MessagePassing layers.

    Useful for exporting / inference when you want a plain model.

    Parameters
    ----------
    model : nn.Module

    Returns
    -------
    nn.Module
    """
    new_model = copy.deepcopy(model)

    for name, module in list(new_model.named_modules()):
        if not isinstance(module, _RematConv):
            continue

        parts  = name.split(".")
        parent = new_model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], module.conv)
        logger.info("GNN-Remat: unwrapped '%s'", name)

    return new_model
