"""
wrapper.py
----------
_RematConv: wraps a single MessagePassing module so its output activations
are freed immediately after the forward pass and recomputed on demand
during backpropagation using torch.utils.checkpoint.

Public API
----------
    wrap(module)   -> _RematConv
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt
from torch_geometric.nn import MessagePassing


#Wrapper

class _RematConv(nn.Module):
    """
    Thin checkpoint wrapper around a single MessagePassing layer.

    During the forward pass the layer runs normally, but PyTorch does NOT
    retain the output tensor in the autograd graph.  During the backward
    pass torch.utils.checkpoint re-runs the layer to reconstruct the
    activations it needs.

    Parameters
    ----------
    conv : MessagePassing
        The aggregation layer to wrap.

    Notes
    -----
    * use_reentrant=False is required for correctness with nested
      checkpoints and is the modern default in PyTorch >= 2.0.
    * All non-tensor arguments (e.g. edge_index, size) pass through
      transparently.
    """

    def __init__(self, conv: MessagePassing) -> None:
        super().__init__()
        self.conv = conv

    def forward(self, *args, **kwargs):
        # torch.utils.checkpoint does not support keyword arguments
        # directly, so we bind them into a closure.
        if kwargs:
            def _run(*a):
                return self.conv(*a, **kwargs)
        else:
            _run = self.conv

        return ckpt.checkpoint(_run, *args, use_reentrant=False)

    def __repr__(self) -> str:
        return f"_RematConv({self.conv!r})"


#Factory

def wrap(module: MessagePassing) -> _RematConv:
    """
    Return a _RematConv that checkpoints *module*.

    The original module is stored as ``wrapper.conv`` and its parameters
    remain part of the model's parameter tree.

    Parameters
    ----------
    module : MessagePassing

    Returns
    -------
    _RematConv
    """
    return _RematConv(module)
