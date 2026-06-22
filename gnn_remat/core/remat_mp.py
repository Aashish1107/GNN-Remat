"""
remat_mp.py
-----------
RematMessagePassing: overrides propagate() with torch.utils.checkpoint,
freeing per-edge intermediates and recomputing them on demand during backward.

WHY propagate() and not aggregate()
--------------------------------------
aggregate()-level checkpoint INCREASES memory for GCN/SAGE/GAT because
scatter_add's backward only needs d_agg + edge_index (not the messages),
so baseline PyTorch never saves messages.  Checkpointing aggregate() forces
those messages into ctx.saved_tensors, adding ~46 MB/layer net.

propagate()-level checkpoint:
  Saves : propagate *inputs* (node-feature kwargs, e.g. x_proj [N, out])
          into ctx.saved_tensors — small regardless of graph density.
  Frees : large per-edge intermediates autograd retained inside propagate():
            alpha [E, heads]        ~0.8 MB/layer
            x_j   [E, heads, F/H]  ~51  MB/layer
  Recomputes: message() + aggregate() on backward.

SAR-Inspired: Destination-Node Chunked Propagation
----------------------------------------------------
Ported from SAR (Mostafa 2021) to a single-GPU setting.

Process edges in destination-node ranges so at most one range's worth
of edge tensors is live at a time.  Each destination node's COMPLETE
neighbourhood falls in exactly one range, ensuring correctness for all
aggregation types (sum, mean, max, and attention softmax).

Key detail — per-edge kwargs slicing:
  Newer PyG versions (≥ 2.x) pre-compute per-edge tensors BEFORE calling
  propagate().  For example, GATConv computes alpha [E, H] via
  edge_updater() and passes it as a kwarg to propagate().  If we feed a
  27-edge chunk but still supply the full alpha [E=197, H], GATConv's
  message() does `alpha.unsqueeze(-1) * x_j` and crashes with a size
  mismatch.

  Fix (_slice_edge_kwargs): any kwarg tensor whose leading dimension equals
  num_edges_total is assumed to be per-edge and is sliced with the chunk mask
  before being passed to propagate().

Closure-safety note:
  _run_chunk() is a standalone method so chunk_ei / chunk-specific state
  is captured as function-parameter bindings, not by loop-variable reference.

Public API
----------
    make_remat_conv(conv, chunk_nodes=None)   ->  conv with checkpointed propagate()
    RematMessagePassing                           mixin for advanced use
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.utils.checkpoint as ckpt
from torch_geometric.nn import MessagePassing


# ── kwargs helpers ─────────────────────────────────────────────────────────────

def _flatten_kwargs(kwargs: dict) -> tuple[list[torch.Tensor], dict]:
    """
    Extract all tensors from kwargs into a flat list for ckpt.checkpoint.

    Returns (flat_tensors, spec) where spec lets _unflatten_kwargs reconstruct
    the original dict.  Non-tensor values are stored in spec verbatim.
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


# ── Core mixin ────────────────────────────────────────────────────────────────

class RematMessagePassing(MessagePassing):
    """
    MessagePassing mixin with two complementary memory optimisation modes.

    Mode 1 — Propagate-level checkpoint (original):
        Checkpoints the full propagate() so per-edge intermediates are freed
        after the forward pass and recomputed during backward.
        Linear projections applied before propagate() are NOT recomputed.

    Mode 2 — Destination-node chunked checkpoint (SAR-inspired):
        Processes edges in chunk_nodes-wide destination-node windows.
        Each chunk is independently checkpointed.
        Per-edge kwargs (e.g. pre-computed attention alpha) are sliced to
        the chunk before being passed to propagate(), preventing size mismatches.
        Reduces peak memory from O(E×F) to O(chunk_edges×F).
        Activated when _chunk_nodes is set via make_remat_conv / gnn_remat.

    Gate: checkpointing is skipped in eval() and when no float tensor
    requires a gradient, so inference pays zero overhead.
    """
    _is_remat: bool = True
    _chunk_nodes: Optional[int] = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def propagate(self, edge_index, size=None, **kwargs):
        chunk_nodes = getattr(self, '_chunk_nodes', None)

        # SAR path: chunk by destination-node ranges
        if (chunk_nodes is not None
                and self.training
                and edge_index.size(1) > chunk_nodes):
            return self._propagate_chunked(
                edge_index, size=size, chunk_nodes=chunk_nodes, **kwargs
            )

        # Standard propagate-level checkpoint (original behaviour)
        flat, spec = _flatten_kwargs(kwargs)

        any_float_grad = any(
            t.is_floating_point() and t.requires_grad for t in flat
        )

        if not (self.training and flat and any_float_grad):
            return super(RematMessagePassing, self).propagate(
                edge_index, size=size, **kwargs
            )

        def _prop(*tensors):
            kw = _unflatten_kwargs(list(tensors), spec)
            return super(RematMessagePassing, self).propagate(
                edge_index, size=size, **kw
            )

        return ckpt.checkpoint(_prop, *flat, use_reentrant=False)

    # ── Per-edge kwargs slicing ───────────────────────────────────────────────

    @staticmethod
    def _slice_edge_kwargs(kwargs: dict, mask: torch.Tensor,
                           num_edges_total: int) -> dict:
        """
        Slice any per-edge tensors in kwargs to match the current chunk.

        A tensor is considered per-edge when its leading dimension equals
        num_edges_total.  Node-level tensors and scalars are passed through
        unchanged.

        This handles the case (PyG ≥ 2.x GATConv, TransformerConv) where
        attention weights alpha [E, H] are pre-computed by edge_updater()
        BEFORE propagate() is called, then passed as a propagate() kwarg.
        Without slicing, chunk_ei has k edges but alpha still has E rows,
        causing a size mismatch inside message().
        """
        sliced: dict = {}
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                if val.size(0) == num_edges_total:
                    sliced[key] = val[mask]
                else:
                    sliced[key] = val
            elif isinstance(val, (tuple, list)):
                new_items = []
                for item in val:
                    if (isinstance(item, torch.Tensor)
                            and item.size(0) == num_edges_total):
                        new_items.append(item[mask])
                    else:
                        new_items.append(item)
                sliced[key] = type(val)(new_items)
            else:
                sliced[key] = val
        return sliced

    # ── SAR-inspired chunked propagation ──────────────────────────────────────

    def _propagate_chunked(self, edge_index, size=None, chunk_nodes=5_000, **kwargs):
        """
        Destination-node chunked propagation (SAR-inspired).

        Each window [dst_start, dst_end) holds ALL edges pointing to those
        destination nodes, ensuring every aggregation type is computed over
        the complete neighbourhood:

          add / sum  : partial sums are additive           ✓
          mean       : full neighbourhood per chunk         ✓
          max        : full neighbourhood per chunk         ✓
          attention  : softmax denominator spans full neigh ✓

        Per-edge kwargs (e.g. pre-computed alpha) are sliced to each chunk
        before the checkpoint call to prevent size mismatches.

        Memory per chunk:
          edge tensors  : O(chunk_edges × F)  — freed by checkpoint after backward
          scatter output: O(num_nodes × F)    — one temporary, freed after clone
          node features : O(num_nodes × F_in) — shared input, not duplicated
        """
        # Resolve graph dimensions
        num_dst = (
            int(size[1]) if size is not None and size[1] is not None
            else int(edge_index[1].max()) + 1
        )
        num_src = (
            int(size[0]) if size is not None and size[0] is not None
            else int(edge_index[0].max()) + 1
        )
        num_edges_total = edge_index.size(1)
        full_size = (num_src, num_dst)

        pieces: list[tuple[int, Optional[torch.Tensor]]] = []
        out_dtype = out_device = out_tail = None

        for dst_start in range(0, num_dst, chunk_nodes):
            dst_end   = min(dst_start + chunk_nodes, num_dst)
            chunk_len = dst_end - dst_start

            # All edges whose destination falls in [dst_start, dst_end)
            mask     = (edge_index[1] >= dst_start) & (edge_index[1] < dst_end)
            chunk_ei = edge_index[:, mask]

            if chunk_ei.size(1) == 0:
                pieces.append((chunk_len, None))
                continue

            # Slice per-edge kwargs to this chunk (fixes PyG >= 2.x GAT alpha mismatch)
            chunk_kwargs = self._slice_edge_kwargs(kwargs, mask, num_edges_total)

            # Flatten chunk-specific kwargs for the checkpoint call
            flat_chunk, spec_chunk = _flatten_kwargs(chunk_kwargs)
            any_float_grad = any(
                t.is_floating_point() and t.requires_grad for t in flat_chunk
            )

            if self.training and flat_chunk and any_float_grad:
                # _run_chunk captures chunk_ei + spec_chunk by parameter binding
                chunk_full = self._run_chunk(
                    chunk_ei, full_size, flat_chunk, spec_chunk
                )
            else:
                chunk_full = super(RematMessagePassing, self).propagate(
                    chunk_ei, size=full_size, **chunk_kwargs
                )

            # Extract the relevant slice and clone to free the [num_dst, F] buffer
            chunk_slice = chunk_full[dst_start:dst_end].clone()
            del chunk_full

            if out_dtype is None:
                out_dtype  = chunk_slice.dtype
                out_device = chunk_slice.device
                out_tail   = chunk_slice.shape[1:]

            pieces.append((chunk_len, chunk_slice))

        # Fill empty chunks (isolated nodes) with zeros, then cat
        resolved: list[torch.Tensor] = []
        for (chunk_len, tensor) in pieces:
            if tensor is None:
                if out_dtype is None:
                    raise RuntimeError(
                        "_propagate_chunked: all edge chunks are empty — "
                        "the graph has no edges."
                    )
                tensor = torch.zeros(
                    chunk_len, *out_tail,
                    dtype=out_dtype, device=out_device,
                )
            resolved.append(tensor)

        # CatBackward correctly fans the upstream gradient to each chunk's checkpoint
        return torch.cat(resolved, dim=0)

    def _run_chunk(
        self,
        chunk_ei: torch.Tensor,
        full_size: tuple[int, int],
        flat_chunk: list[torch.Tensor],
        spec_chunk: dict,
    ) -> torch.Tensor:
        """
        Checkpointed single-chunk propagate.

        Separate method so chunk_ei and spec_chunk are bound as function
        parameters (each call binds its own values) rather than as loop
        variables (which would all refer to the last iteration's value when
        the checkpoint re-runs during backward).
        """
        def _prop(*tensors):
            kw = _unflatten_kwargs(list(tensors), spec_chunk)
            return super(RematMessagePassing, self).propagate(
                chunk_ei, size=full_size, **kw
            )

        return ckpt.checkpoint(_prop, *flat_chunk, use_reentrant=False)


# ── Factory ────────────────────────────────────────────────────────────────────

def make_remat_conv(
    conv: MessagePassing,
    chunk_nodes: Optional[int] = None,
) -> MessagePassing:
    """
    Return a copy of *conv* whose propagate() is checkpointed.

    Builds a dynamic subclass:
        <ConvName>Remat(RematMessagePassing, <ConvClass>)
    MRO ensures RematMessagePassing.propagate() fires first, while all
    other methods (message, __init__, attention logic) come from ConvClass.

    No weight copying — instance attributes are transplanted directly.

    Parameters
    ----------
    conv : MessagePassing
    chunk_nodes : int, optional
        If set, activate SAR-inspired chunked propagation.  Only
        chunk_nodes destination-nodes' worth of edges are materialised at
        once.  Use auto_chunk_size() from heuristic.py for a good value.
        Default None uses the standard single-shot checkpoint only.
    """
    base_cls  = type(conv)
    remat_cls = type(
        f"{base_cls.__name__}Remat",
        (RematMessagePassing, base_cls),
        {"_is_remat": True},
    )
    new_conv = object.__new__(remat_cls)
    new_conv.__dict__.update(conv.__dict__)
    new_conv._chunk_nodes = chunk_nodes
    return new_conv
