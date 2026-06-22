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

        Edges are sorted by destination ONCE per forward (O(E log E)), so each
        window is a contiguous slice located via searchsorted — instead of a
        fresh boolean mask over all E edges per window (O(num_chunks × E)).

        When the layer's node-feature kwarg is the recognised `x` (GCN, SAGE,
        GAT), destination indices are remapped to a local [0, chunk_len) range
        and the dst features are sliced to the window, so each chunk's scatter
        output is [chunk_len, F] rather than the full [num_dst, F].  This removes
        the O(num_dst × F) memory floor that otherwise caps the saving (and makes
        very small chunks regress).  Layers with other destination-indexed
        kwargs (e.g. TransformerConv's per-dst query) fall back to the safe
        full-size scatter + slice, still benefiting from the faster windowing.

        Reduces peak edge memory from O(E×F) to O(chunk_edges×F); with the
        local-remap path the output term drops from O(num_dst×F) to
        O(chunk_len×F).  Activated when _chunk_nodes is set.

    Gate: checkpointing is skipped in eval() and when no float tensor
    requires a gradient, so inference pays zero overhead.
    """
    _is_remat: bool = True
    _chunk_nodes: Optional[int] = None

    # ── Main entry point ──────────────────────────────────────────────────────

    def propagate(self, edge_index, size=None, **kwargs):
        # Fast path: inference pays zero checkpoint overhead.  Checked first so
        # eval() does not even flatten kwargs.
        if not self.training:
            return super(RematMessagePassing, self).propagate(
                edge_index, size=size, **kwargs
            )

        # SAR path: chunk by destination-node ranges
        chunk_nodes = getattr(self, '_chunk_nodes', None)
        if chunk_nodes is not None and edge_index.size(1) > chunk_nodes:
            return self._propagate_chunked(
                edge_index, size=size, chunk_nodes=chunk_nodes, **kwargs
            )

        # Standard propagate-level checkpoint (original behaviour)
        flat, spec = _flatten_kwargs(kwargs)
        any_float_grad = any(
            t.is_floating_point() and t.requires_grad for t in flat
        )
        if not (flat and any_float_grad):
            return super(RematMessagePassing, self).propagate(
                edge_index, size=size, **kwargs
            )

        def _prop(*tensors):
            kw = _unflatten_kwargs(list(tensors), spec)
            return super(RematMessagePassing, self).propagate(
                edge_index, size=size, **kw
            )

        return ckpt.checkpoint(_prop, *flat, use_reentrant=False)

    # ── Per-edge kwargs handling (sort-based windowing) ───────────────────────

    @staticmethod
    def _permute_edge_kwargs(kwargs: dict, perm: torch.Tensor,
                             num_edges_total: int) -> dict:
        """
        Reorder per-edge tensors by *perm* once, so later chunks are contiguous
        slices.  A tensor is per-edge when its leading dim equals
        num_edges_total (e.g. GAT/Transformer alpha [E, H] pre-computed by
        edge_updater()).  Node tensors and scalars pass through unchanged.
        """
        out: dict = {}
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                out[key] = val[perm] if val.size(0) == num_edges_total else val
            elif isinstance(val, (tuple, list)):
                out[key] = type(val)(
                    item[perm] if (isinstance(item, torch.Tensor)
                                   and item.size(0) == num_edges_total)
                    else item
                    for item in val
                )
            else:
                out[key] = val
        return out

    @staticmethod
    def _slice_edge_kwargs_range(kwargs: dict, lo: int, hi: int,
                                 num_edges_total: int) -> dict:
        """Slice already-permuted per-edge tensors to the contiguous [lo, hi)."""
        out: dict = {}
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                out[key] = val[lo:hi] if val.size(0) == num_edges_total else val
            elif isinstance(val, (tuple, list)):
                out[key] = type(val)(
                    item[lo:hi] if (isinstance(item, torch.Tensor)
                                    and item.size(0) == num_edges_total)
                    else item
                    for item in val
                )
            else:
                out[key] = val
        return out

    @staticmethod
    def _can_local_remap(kwargs: dict, num_src: int, num_dst: int,
                         num_edges_total: int) -> bool:
        """
        True when destination indices can be safely remapped to a local
        [0, chunk_len) range — i.e. the only node-indexed kwarg is the
        recognised `x` (single tensor or src/dst tuple).

        Any *other* node-indexed tensor (leading dim == num_src or num_dst,
        and not == num_edges_total) is something we cannot classify as
        source- vs destination-indexed by shape alone (e.g. TransformerConv's
        per-dst `query` collides in size with per-src `key`/`value` on a
        homogeneous graph).  In that case we fall back to full-size scatter.
        """
        def _is_unhandled_node_tensor(t: object) -> bool:
            return (isinstance(t, torch.Tensor)
                    and t.dim() > 0
                    and t.size(0) != num_edges_total
                    and t.size(0) in (num_src, num_dst))

        for key, val in kwargs.items():
            if key == "x":
                continue  # handled explicitly by _localize_node_kwargs
            if _is_unhandled_node_tensor(val):
                return False
            if isinstance(val, (tuple, list)):
                if any(_is_unhandled_node_tensor(v) for v in val):
                    return False
        return True

    @staticmethod
    def _localize_node_kwargs(kwargs: dict, dst_start: int, dst_end: int,
                              num_src: int, num_dst: int) -> dict:
        """
        Slice the destination side of the `x` node-feature kwarg to the window
        so the chunk's scatter output is [chunk_len, F].

        Source features stay full (messages gather by global source index);
        only the destination features are sliced:
          * x Tensor  → (x_full_src, x[dst_start:dst_end])
          * x (xs, xd) → (xs, xd[dst_start:dst_end])
        """
        out = dict(kwargs)
        x = out.get("x", None)
        if isinstance(x, torch.Tensor):
            # Single tensor ⇒ homogeneous graph ⇒ src and dst share features.
            out["x"] = (x, x[dst_start:dst_end])
        elif isinstance(x, (tuple, list)) and len(x) == 2:
            x_src, x_dst = x[0], x[1]
            if isinstance(x_dst, torch.Tensor):
                x_dst = x_dst[dst_start:dst_end]
            out["x"] = (x_src, x_dst)
        return out

    # ── SAR-inspired chunked propagation ──────────────────────────────────────

    def _propagate_chunked(self, edge_index, size=None, chunk_nodes=5_000, **kwargs):
        """
        Destination-node chunked propagation (SAR-inspired).

        Each window [dst_start, dst_end) holds ALL edges pointing to those
        destination nodes, ensuring every aggregation type is computed over the
        complete neighbourhood:

          add / sum  : partial sums are additive            ✓
          mean       : full neighbourhood per chunk          ✓
          max        : full neighbourhood per chunk          ✓
          attention  : softmax denominator spans full neigh  ✓

        Implementation (two optimisations over the naive version):

          1. Sort edges by destination ONCE (O(E log E)).  Each window is then
             a contiguous slice located with searchsorted, instead of a fresh
             boolean mask over all E edges per window (O(num_chunks × E)).

          2. Local destination remap (when _can_local_remap): destination
             indices are shifted to [0, chunk_len) and dst features sliced to
             the window, so each chunk's scatter output is [chunk_len, F].
             Otherwise (e.g. TransformerConv) fall back to a full [num_dst, F]
             scatter that is sliced afterwards — correct, just less memory-thrifty.

        Memory per chunk (local-remap path):
          edge tensors  : O(chunk_edges × F)  — freed by checkpoint after backward
          scatter output: O(chunk_len  × F)   — no full [num_dst, F] buffer
          node features : O(num_src × F_in)    — shared source input, not duplicated
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

        # ── (1) sort edges by destination once; windows become contiguous ──────
        perm       = torch.argsort(edge_index[1])
        sorted_ei  = edge_index[:, perm]
        sorted_dst = sorted_ei[1]
        perm_kwargs = self._permute_edge_kwargs(kwargs, perm, num_edges_total)

        # Edge boundaries for every window, computed in one searchsorted + sync.
        node_bounds = list(range(0, num_dst, chunk_nodes)) + [num_dst]
        edge_bounds = torch.searchsorted(
            sorted_dst,
            torch.tensor(node_bounds, device=sorted_dst.device,
                         dtype=sorted_dst.dtype),
        ).tolist()

        # ── (2) decide path once ───────────────────────────────────────────────
        local = self._can_local_remap(kwargs, num_src, num_dst, num_edges_total)

        pieces: list[tuple[int, Optional[torch.Tensor]]] = []
        out_dtype = out_device = out_tail = None

        for w in range(len(node_bounds) - 1):
            dst_start, dst_end = node_bounds[w], node_bounds[w + 1]
            lo, hi             = edge_bounds[w], edge_bounds[w + 1]
            chunk_len          = dst_end - dst_start

            if hi == lo:                       # isolated dst nodes — no edges
                pieces.append((chunk_len, None))
                continue

            chunk_ei     = sorted_ei[:, lo:hi]
            chunk_kwargs = self._slice_edge_kwargs_range(
                perm_kwargs, lo, hi, num_edges_total
            )

            if local:
                # Shift dst indices into [0, chunk_len) and slice dst features.
                chunk_ei = torch.stack(
                    [chunk_ei[0], chunk_ei[1] - dst_start], dim=0
                )
                chunk_kwargs = self._localize_node_kwargs(
                    chunk_kwargs, dst_start, dst_end, num_src, num_dst
                )
                run_size = (num_src, chunk_len)
            else:
                run_size = (num_src, num_dst)

            flat_chunk, spec_chunk = _flatten_kwargs(chunk_kwargs)
            any_float_grad = any(
                t.is_floating_point() and t.requires_grad for t in flat_chunk
            )

            if flat_chunk and any_float_grad:
                out = self._run_chunk(chunk_ei, run_size, flat_chunk, spec_chunk)
            else:
                out = super(RematMessagePassing, self).propagate(
                    chunk_ei, size=run_size, **chunk_kwargs
                )

            if local:
                piece = out                    # already [chunk_len, F]
            else:
                piece = out[dst_start:dst_end].clone()   # free [num_dst, F]
                del out

            if out_dtype is None:
                out_dtype, out_device, out_tail = (
                    piece.dtype, piece.device, piece.shape[1:]
                )
            pieces.append((chunk_len, piece))

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
                    chunk_len, *out_tail, dtype=out_dtype, device=out_device,
                )
            resolved.append(tensor)

        # CatBackward fans the upstream gradient to each chunk's checkpoint.
        return torch.cat(resolved, dim=0)

    def _run_chunk(
        self,
        chunk_ei: torch.Tensor,
        run_size: tuple[int, int],
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
                chunk_ei, size=run_size, **kw
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
