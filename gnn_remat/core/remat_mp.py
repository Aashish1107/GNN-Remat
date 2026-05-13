"""
remat_mp.py
-----------
RematMessagePassing: overrides aggregate() — the scatter step — with a
torch.utils.checkpoint call. Linear projections and attention coefficients
computed in message() are kept in the normal autograd graph.

This is what makes GNN-Remat different from module checkpointing:
  module checkpoint  ->  recomputes: linear + attention + scatter
  GNN-Remat          ->  recomputes: scatter ONLY  (this file)

Public API
----------
    make_remat_conv(conv)   ->  conv with checkpointed aggregate()
    RematMessagePassing     mixin for advanced use
"""
from __future__ import annotations
import torch
import torch.utils.checkpoint as ckpt
from torch_geometric.nn import MessagePassing


class RematMessagePassing(MessagePassing):
    """
    MessagePassing mixin that checkpoints only aggregate().

    propagate() calls three things in order:
        message()    compute per-edge messages  (linear proj, attention)
        aggregate()  scatter messages to nodes  <- the memory bottleneck
        update()     post-process node embeddings

    By overriding only aggregate(), steps 1 and 3 stay in the normal
    autograd graph. Only the [num_nodes, out_channels] aggregated tensor
    is freed and recomputed during backward.
    """
    _is_remat: bool = True

    def aggregate(self, inputs: torch.Tensor, index: torch.Tensor,
                  ptr=None, dim_size=None) -> torch.Tensor:
        """Checkpoint the scatter step; keep messages in autograd graph."""
        def _scatter(inp, idx, p, d):
            return super(RematMessagePassing, self).aggregate(
                inp, idx, ptr=p, dim_size=d
            )
        if self.training and inputs.requires_grad:
            return ckpt.checkpoint(_scatter, inputs, index, ptr, dim_size, use_reentrant=False)
        return _scatter(inputs, index, ptr, dim_size)


def make_remat_conv(conv: MessagePassing) -> MessagePassing:
    """
    Return a copy of conv whose aggregate() is checkpointed.

    Builds a dynamic subclass:
        <ConvName>Remat(RematMessagePassing, <ConvClass>)
    MRO ensures RematMessagePassing.aggregate() fires first, while all
    other methods (message, __init__, attention logic) come from ConvClass.

    No weight copying — instance attributes are transplanted directly.
    """
    base_cls  = type(conv)
    remat_cls = type(
        f"{base_cls.__name__}Remat",
        (RematMessagePassing, base_cls),
        {"_is_remat": True},
    )
    # Transplant the instance into the new class without re-running __init__
    new_conv = object.__new__(remat_cls)
    new_conv.__dict__.update(conv.__dict__)
    return new_conv
