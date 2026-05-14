"""
remat_mp.py
-----------
RematMessagePassing: overrides propagate() — the full message-passing step —
with a torch.utils.checkpoint call.  Linear projections applied BEFORE
propagate() (e.g. GATConv.lin_src / lin_dst) are kept in the normal autograd
graph and are NOT recomputed.

WHY propagate() and not aggregate()
--------------------------------------
aggregate()-level checkpoint was tested and found to INCREASE memory for all
three models (GCN, GraphSAGE, GAT).  The reason:

  scatter_add backward computes d_messages = gather(d_agg, index).
  It needs only d_agg and the edge index — NOT the messages tensor.
  So baseline PyTorch never saves messages in ctx.saved_tensors.

  When aggregate() is checkpointed with use_reentrant=False, the checkpoint
  explicitly saves its *inputs* (messages [num_edges, out]) into
  ctx.saved_tensors.  This ADDS ~51 MB per layer (at 5K nodes, degree 10,
  out=256) while only "freeing" agg [num_nodes, out] = ~5 MB, which the
  next layer's linear already keeps alive as its own saved input.
  Net result: ~+46 MB per layer, ~+138 MB for a 3-layer model.

propagate()-level checkpoint:
  - Saves: the propagate() *inputs* (node features in kwargs, e.g.
    x_proj [num_nodes, out]) into ctx.saved_tensors.  Small node-level
    tensors (~5 MB/layer for out=256) vs the edge-level tensors freed.
  - Frees: the large per-edge intermediates that GAT's autograd retains:
      alpha [num_edges, heads]         ~0.8 MB / layer
      x_j   [num_edges, heads, F/H]   ~51  MB / layer
    For a 3-layer GAT: ~155 MB freed vs ~8 MB added → ~147 MB net savings.
  - Recomputes: message() + aggregate() during backward.  For GCN/SAGE the
    scatter-only recompute is cheap; for GAT attention is recomputed but only
    propagate() — NOT lin_src/lin_dst — so it is faster than module-level.
  - Trade-off: GCN/SAGE at small graphs (~5K nodes) show a small overhead
    (~+5 MB/layer) because they have no large per-edge tensors to free.
    At 50K+ nodes the freed edge tensors dominate and savings are real.

Public API
----------
    make_remat_conv(conv)   ->  conv with checkpointed propagate()
    RematMessagePassing     mixin for advanced use
"""
from __future__ import annotations
import torch
import torch.utils.checkpoint as ckpt
from torch_geometric.nn import MessagePassing


def _flatten_kwargs(kwargs: dict) -> tuple[list[torch.Tensor], dict]:
    """
    Extract all tensors from kwargs into a flat list for ckpt.checkpoint.

    Returns (flat_tensors, spec) where spec lets _unflatten_kwargs reconstruct
    the original dict.  Non-tensor values are stored in spec and restored
    verbatim; they are never passed through ckpt.checkpoint.
    """
    flat: list[torch.Tensor] = []
    spec: dict = {}
    for key, val in kwargs.items():
        if isinstance(val, torch.Tensor):
            spec[key] = ("t", len(flat))
            flat.append(val)
        elif isinstance(val, (tuple, list)):
            indices: list[int | None] = []
            orig: list = list(val)
            for v in orig:
                if isinstance(v, torch.Tensor):
                    indices.append(len(flat))
                    flat.append(v)
                else:
                    indices.append(None)
            spec[key] = ("s", type(val), orig, indices)
        else:
            spec[key] = ("v", val)
    return flat, spec


def _unflatten_kwargs(flat: list[torch.Tensor], spec: dict) -> dict:
    """Reconstruct the kwargs dict from a flat tensor list and spec."""
    out: dict = {}
    for key, info in spec.items():
        kind = info[0]
        if kind == "t":
            out[key] = flat[info[1]]
        elif kind == "s":
            _, orig_type, orig, indices = info
            seq = [flat[i] if i is not None else orig[j]
                   for j, i in enumerate(indices)]
            out[key] = orig_type(seq)
        else:
            out[key] = info[1]
    return out


class RematMessagePassing(MessagePassing):
    """
    MessagePassing mixin that checkpoints propagate().

    propagate() calls three steps:
        message()    compute per-edge messages  (linear proj, attention)
        aggregate()  scatter messages to nodes
        update()     post-process node embeddings

    The checkpoint frees all per-edge intermediates (attention coefficients,
    x_j, weighted messages) that the baseline autograd graph retains inside
    propagate().  Only the propagate() *inputs* (node-feature kwargs, typically
    [num_nodes, out]) are saved — these are small regardless of graph density.

    Gate: checkpointing is skipped when not training, when kwargs contain no
    float tensors requiring grad, or when the flat tensor list is empty.
    """
    _is_remat: bool = True

    def propagate(self, edge_index, size=None, **kwargs):
        """Checkpoint the full propagate step; linear projections before it stay in graph."""
        flat, spec = _flatten_kwargs(kwargs)

        any_float_grad = any(
            t.is_floating_point() and t.requires_grad for t in flat
        )

        if not (self.training and flat and any_float_grad):
            return super().propagate(edge_index, size=size, **kwargs)

        def _prop(*tensors):
            kw = _unflatten_kwargs(list(tensors), spec)
            return super(RematMessagePassing, self).propagate(
                edge_index, size=size, **kw
            )

        return ckpt.checkpoint(_prop, *flat, use_reentrant=False)


def make_remat_conv(conv: MessagePassing) -> MessagePassing:
    """
    Return a copy of *conv* whose propagate() is checkpointed.

    Builds a dynamic subclass:
        <ConvName>Remat(RematMessagePassing, <ConvClass>)
    MRO ensures RematMessagePassing.propagate() fires first, while all
    other methods (message, __init__, attention logic) come from ConvClass.

    No weight copying — instance attributes are transplanted directly so
    the new object shares the same parameter tensors as the original.
    """
    base_cls  = type(conv)
    remat_cls = type(
        f"{base_cls.__name__}Remat",
        (RematMessagePassing, base_cls),
        {"_is_remat": True},
    )
    new_conv = object.__new__(remat_cls)
    new_conv.__dict__.update(conv.__dict__)
    return new_conv
