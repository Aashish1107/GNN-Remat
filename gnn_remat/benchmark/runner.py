"""
runner.py
---------
CLI entry point for GNN-Remat benchmarks.

Usage
-----
    # Single run
    python -m gnn_remat.benchmark.runner --model gat --nodes 5000
    python -m gnn_remat.benchmark.runner --all --nodes 5000

    # With SAR-inspired chunked propagation (adds a 4th benchmark condition)
    python -m gnn_remat.benchmark.runner --model gat --nodes 50000 --chunk-nodes 5000
    python -m gnn_remat.benchmark.runner --model gat --nodes 50000 --chunk-nodes auto

    # Scale sweep: 5K → 10K → 25K → 50K → 100K nodes in one table
    python -m gnn_remat.benchmark.runner --model gat --scale
    python -m gnn_remat.benchmark.runner --model gat --scale --chunk-nodes auto

    # OOM crossover: double nodes until baseline OOMs but GNN-Remat fits (CUDA only)
    python -m gnn_remat.benchmark.runner --model gat --find-limit
    python -m gnn_remat.benchmark.runner --model gat --find-limit --max-nodes 500000
"""
from __future__ import annotations

import gc
import sys
import argparse
import traceback

import torch

from .models import build
from .profiler import compare


_SCALE_NODES = [5_000, 10_000, 25_000, 50_000, 100_000]
_ALL_MODELS  = ["gcn", "graphsage", "gat", "transformer"]


def _load_dataset(name: str):
    """
    Load a real graph for benchmarking: ogbn-* (OGB) or a torch_geometric
    dataset (reddit, flickr).  Returns (x, edge_index, in_channels, num_classes)
    on CPU; the profiler moves them to the device.
    """
    name_l = name.lower()
    orig_load = torch.load                      # OGB needs weights_only=False
    def _patched(f, *a, **k):
        k.setdefault("weights_only", False)
        return orig_load(f, *a, **k)
    torch.load = _patched
    try:
        if name_l.startswith("ogbn-"):
            from ogb.nodeproppred import PygNodePropPredDataset
            ds = PygNodePropPredDataset(name=name_l)
            data = ds[0]
            return data.x, data.edge_index, data.x.size(1), ds.num_classes
        import torch_geometric.datasets as D
        cls = {"reddit": "Reddit", "flickr": "Flickr"}.get(name_l)
        if cls is None:
            raise ValueError(
                f"Unknown --dataset {name!r}. Use ogbn-* (e.g. ogbn-arxiv, "
                f"ogbn-products), reddit, or flickr."
            )
        ds = getattr(D, cls)(root=f"data/{cls}")
        data = ds[0]
        return data.x, data.edge_index, data.x.size(1), ds.num_classes
    finally:
        torch.load = orig_load


def _make_graph(num_nodes: int, avg_degree: int, num_features: int, device):
    """Synthesise a random Erdos-Renyi graph for benchmarking."""
    x        = torch.randn(num_nodes, num_features, device=device)
    num_edges = num_nodes * avg_degree
    src      = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst      = torch.randint(0, num_nodes, (num_edges,), device=device)
    return x, torch.stack([src, dst])


def _build_kwargs(model_name: str, args) -> dict:
    kw = dict(
        in_channels=args.features,
        hidden=args.hidden,
        out_channels=args.classes,
        num_layers=args.layers,
    )
    if model_name in ("gat", "transformer"):
        kw["heads"] = args.heads
    return kw


def _chunk_nodes_type(value: str):
    """Custom argparse type: accepts an integer or the string 'auto'."""
    if value.lower() == "auto":
        return "auto"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"--chunk-nodes must be a positive integer or 'auto', got: {value!r}"
        )


def _resolve_chunk_nodes(args, num_nodes: int):
    """
    Resolve --chunk-nodes to an actual integer (or None).

    'auto' → calls auto_chunk_size() with the model dimensions from args.
    int    → returned unchanged.
    None   → no chunking.
    """
    raw = getattr(args, "chunk_nodes", None)
    if raw is None:
        return None
    if raw == "auto":
        from gnn_remat import auto_chunk_size
        chunk = auto_chunk_size(
            num_nodes=num_nodes,
            out_channels=args.hidden,
            num_heads=args.heads,
            avg_degree=float(args.degree),
        )
        print(f"  [auto chunk_nodes] = {chunk:,}  "
              f"(hidden={args.hidden}, heads={args.heads}, degree={args.degree})")
        return chunk
    return raw  # already an int


def _cleanup():
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass  # CUDA context already corrupted — ignore


def _is_cuda_context_error(exc: BaseException) -> bool:
    """True when a CUDA OOM has corrupted the CUDA context (AcceleratorError)."""
    cls_name = type(exc).__name__
    msg      = str(exc).lower()
    return (
        "AcceleratorError" in cls_name
        or "cuda error" in msg
        or "cudaerror" in msg
    )


# ── Single run ────────────────────────────────────────────────────────────────

def run_one(model_name: str, args, device):
    x = edge_index = None
    if args.dataset:                       # real graph overrides synthetic dims
        x, edge_index, args.features, args.classes = _load_dataset(args.dataset)
        args.nodes = x.size(0)

    chunk_nodes = _resolve_chunk_nodes(args, args.nodes)

    print(f"\n{'='*64}")
    src = f"dataset={args.dataset}" if args.dataset else "synthetic"
    print(f"  Model: {model_name.upper()}  |  {src}  nodes={args.nodes:,}  "
          f"features={args.features}  layers={args.layers}")
    if chunk_nodes is not None:
        print(f"  Chunked propagation: chunk_nodes={chunk_nodes:,}")
    if args.amp:
        print(f"  AMP: bf16 autocast")
    print(f"{'='*64}")

    model = build(model_name, **_build_kwargs(model_name, args))
    if x is None:
        graph_device = torch.device("cpu")  # build on CPU; profiler moves it
        try:
            x, edge_index = _make_graph(args.nodes, args.degree, args.features,
                                        graph_device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and not isinstance(
                e, getattr(torch.cuda, "OutOfMemoryError", ())
            ):
                raise
            print(f"  [skip] could not allocate graph tensors: {e}")
            return None

    result = compare(model, x, edge_index, num_epochs=args.epochs,
                     device=device, chunk_nodes=chunk_nodes, amp=args.amp)
    print(result.summary())
    return result


# ── Scale sweep ───────────────────────────────────────────────────────────────

def run_scale_sweep(model_name: str, args, device):
    """
    Profile the model at each node count in _SCALE_NODES and print a single
    table showing how memory savings grow with graph size.
    """
    using_chunk = getattr(args, "chunk_nodes", None) is not None

    if device.type == "cpu":
        print("  Note: running scale sweep on CPU — memory readings will all be 0.0 MB "
              "(CPU allocator has no peak counter). Use CUDA for meaningful results.")

    width = 97 if using_chunk else 80
    print(f"\n{'='*width}")
    print(f"  Scale sweep — {model_name.upper()}  "
          f"features={args.features}  hidden={args.hidden}  "
          f"layers={args.layers}  degree={args.degree}  heads={args.heads}")
    if using_chunk:
        raw = args.chunk_nodes
        chunk_label = "auto" if raw == "auto" else f"{raw:,}"
        print(f"  Chunked propagation: chunk_nodes={chunk_label}")
    print(f"{'='*width}")

    if using_chunk:
        col = (f"{'Nodes':>8}  {'Edges':>9}  {'Baseline':>11}  "
               f"{'Module ckpt':>12}  {'GNN-Remat':>11}  "
               f"{'Chunked':>11}  {'Chunk saving':>13}")
    else:
        col = (f"{'Nodes':>8}  {'Edges':>9}  {'Baseline':>11}  "
               f"{'Module ckpt':>12}  {'GNN-Remat':>11}  {'Remat saving':>13}")
    print(col)
    print("─" * width)

    for n_nodes in _SCALE_NODES:
        chunk_nodes = _resolve_chunk_nodes(args, n_nodes)
        model   = build(model_name, **_build_kwargs(model_name, args))
        x, ei   = _make_graph(n_nodes, args.degree, args.features, torch.device("cpu"))
        n_edges = n_nodes * args.degree

        try:
            result = compare(model, x, ei, num_epochs=args.epochs,
                             device=device, chunk_nodes=chunk_nodes, amp=args.amp)
        except Exception as exc:
            del model, x, ei
            if _is_cuda_context_error(exc):
                print(f"{n_nodes:>8,}  {n_edges:>9,}  "
                      f"(CUDA context corrupted after OOM — stopping sweep)")
            else:
                traceback.print_exc()
                print(f"{n_nodes:>8,}  — unexpected error, see above")
            break
        finally:
            _cleanup()

        def _mem(r):
            return f"{r.peak_memory_mb:>7.0f} MB" if r.ok else "        OOM"

        ref = result.baseline

        # GNN-Remat saving vs baseline
        if ref.ok and result.gnn_remat.ok:
            pct    = (1 - result.gnn_remat.peak_memory_mb / ref.peak_memory_mb) * 100
            saving = f"{pct:>+.1f}%"
        elif not ref.ok and result.gnn_remat.ok:
            saving = "baseline OOM"
        else:
            saving = "—"

        if using_chunk and result.chunked_remat is not None:
            # Chunk saving vs baseline (or gnn_remat when baseline OOM'd)
            chunk_ref = ref if ref.ok else result.gnn_remat
            if chunk_ref.ok and result.chunked_remat.ok:
                pct_c       = (1 - result.chunked_remat.peak_memory_mb
                               / chunk_ref.peak_memory_mb) * 100
                chunk_saving = f"{pct_c:>+.1f}%"
            elif not ref.ok and result.chunked_remat.ok:
                chunk_saving = "base OOM"
            else:
                chunk_saving = "—"

            print(f"{n_nodes:>8,}  {n_edges:>9,}  {_mem(result.baseline)}  "
                  f"{_mem(result.module_ckpt)}   {_mem(result.gnn_remat)}  "
                  f"{_mem(result.chunked_remat)}  {chunk_saving:>13}")
        else:
            print(f"{n_nodes:>8,}  {n_edges:>9,}  {_mem(result.baseline)}  "
                  f"{_mem(result.module_ckpt)}   {_mem(result.gnn_remat)}  {saving:>13}")

    print()


# ── OOM limit search ──────────────────────────────────────────────────────────

def find_oom_limit(model_name: str, args, device):
    """
    Double node count until baseline OOMs but GNN-Remat still fits.
    This is the headline "OOM-to-fits" result: GNN-Remat enables training at
    graph sizes that are unreachable with the unmodified model.

    When --chunk-nodes is also supplied, a 4th column shows whether the chunked
    variant fits at even larger graphs.
    """
    if device.type == "cpu":
        print("  [skip] --find-limit requires CUDA. "
              "CPU peak memory is not constrained the same way.")
        return

    using_chunk = getattr(args, "chunk_nodes", None) is not None
    width = 70 if using_chunk else 56

    print(f"\n{'='*64}")
    print(f"  OOM crossover search — {model_name.upper()}")
    print(f"  Config: features={args.features}  hidden={args.hidden}  "
          f"layers={args.layers}  heads={args.heads}  degree={args.degree}")
    if using_chunk:
        raw = args.chunk_nodes
        print(f"  Chunked propagation: chunk_nodes={'auto' if raw == 'auto' else f'{raw:,}'}")
    print(f"  Starting at {args.nodes:,} nodes, doubling each step "
          f"(cap: {args.max_nodes:,})")
    print(f"{'='*64}")
    if using_chunk:
        print(f"  {'Nodes':>9}  {'Edges':>10}  {'Baseline':>12}  "
              f"{'GNN-Remat':>12}  {'Chunked':>10}")
        print(f"  {'─'*width}")
    else:
        print(f"  {'Nodes':>9}  {'Edges':>10}  {'Baseline':>12}  {'GNN-Remat':>12}")
        print(f"  {'─'*width}")

    n_nodes = args.nodes
    found   = False

    while n_nodes <= args.max_nodes:
        chunk_nodes = _resolve_chunk_nodes(args, n_nodes)
        model   = build(model_name, **_build_kwargs(model_name, args))
        x, ei   = _make_graph(n_nodes, args.degree, args.features, torch.device("cpu"))
        n_edges = n_nodes * args.degree

        try:
            result = compare(model, x, ei, num_epochs=2, device=device,
                             chunk_nodes=chunk_nodes, amp=args.amp)
        except Exception as exc:
            del model, x, ei
            if _is_cuda_context_error(exc):
                print(f"  {n_nodes:>9,}  (CUDA context corrupted after OOM — stopping)")
            else:
                traceback.print_exc()
            break
        finally:
            _cleanup()

        base_ok    = result.baseline.ok
        remat_ok   = result.gnn_remat.ok
        chunked_ok = result.chunked_remat is not None and result.chunked_remat.ok

        base_str  = (f"{result.baseline.peak_memory_mb:>8.0f} MB"
                     if base_ok else "         OOM")
        remat_str = (f"{result.gnn_remat.peak_memory_mb:>8.0f} MB"
                     if remat_ok else "         OOM")

        if using_chunk and result.chunked_remat is not None:
            chunk_str = (f"{result.chunked_remat.peak_memory_mb:>6.0f} MB"
                         if chunked_ok else "       OOM")
            print(f"  {n_nodes:>9,}  {n_edges:>10,}  {base_str}  {remat_str}  {chunk_str}")
        else:
            print(f"  {n_nodes:>9,}  {n_edges:>10,}  {base_str}  {remat_str}")

        # Crossover: baseline OOM but at least one remat variant fits
        fits_variant = remat_ok or chunked_ok
        if not base_ok and fits_variant:
            best = result.chunked_remat if chunked_ok else result.gnn_remat
            label = "GNN-Remat+Chunk" if chunked_ok else "GNN-Remat"
            print(f"\n  {label} enables training at {n_nodes:,} nodes ({n_edges:,} edges).")
            print(f"    Baseline  : OOM")
            print(f"    {label:<15}: {best.peak_memory_mb:.0f} MB  "
                  f"({best.epoch_time_ms:.1f} ms/step)")
            found = True
            break

        if not base_ok and not fits_variant:
            print(f"\n  All conditions OOM at {n_nodes:,} nodes.")
            print("  Try smaller --hidden or --heads, or a GPU with more VRAM.")
            break

        n_nodes *= 2

    if not found and n_nodes > args.max_nodes:
        print(f"\n  Baseline did not OOM up to {args.max_nodes:,} nodes.")
        print("  Increase --max-nodes, --hidden, or --heads to stress the GPU more.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv=None):
    # Box-drawing chars in the tables crash the Windows cp1252 console; force UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        description="GNN-Remat benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",      default="gat",
                        help="gcn | graphsage | gat | transformer")
    parser.add_argument("--all",        action="store_true",
                        help="Run all four models (gcn, graphsage, gat, transformer)")
    parser.add_argument("--nodes",      type=int, default=5_000,
                        help="Number of nodes (single run or --find-limit start point)")
    parser.add_argument("--features",   type=int, default=128,
                        help="Node feature dimension")
    parser.add_argument("--hidden",     type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("--classes",    type=int, default=40,
                        help="Output classes")
    parser.add_argument("--layers",     type=int, default=3,
                        help="Number of GNN layers")
    parser.add_argument("--degree",     type=int, default=10,
                        help="Average node degree")
    parser.add_argument("--epochs",     type=int, default=5,
                        help="Training steps per benchmark condition")
    parser.add_argument("--heads",      type=int, default=4,
                        help="Attention heads (GAT and Transformer only)")
    parser.add_argument("--scale",      action="store_true",
                        help=f"Sweep node counts {_SCALE_NODES} and print a scale table")
    parser.add_argument("--find-limit", action="store_true",
                        help="Double nodes until baseline OOMs but GNN-Remat fits (CUDA only)")
    parser.add_argument("--max-nodes",  type=int, default=500_000,
                        help="Node count cap for --find-limit")
    parser.add_argument("--chunk-nodes", type=_chunk_nodes_type, default=None,
                        metavar="N|auto",
                        help="Enable SAR-inspired chunked propagation. Pass an integer "
                             "(e.g. 5000) or 'auto' to compute from GPU memory. "
                             "Adds a 4th benchmark condition 'GNN-Remat+Chunk'.")
    parser.add_argument("--amp", action="store_true",
                        help="bf16 autocast for every condition (stacks with remat/chunk).")
    parser.add_argument("--dataset", default=None, metavar="NAME",
                        help="Real graph instead of synthetic: ogbn-arxiv, "
                             "ogbn-products, reddit, flickr (single run; ignores --scale).")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_run = _ALL_MODELS if args.all else [args.model]

    overview: list[tuple[str, str]] = []
    for name in models_to_run:
        try:
            if args.scale and not args.dataset:
                run_scale_sweep(name, args, device)
                overview.append((name, "scale sweep done"))
            elif args.find_limit and not args.dataset:
                find_oom_limit(name, args, device)
                overview.append((name, "limit search done"))
            else:  # single run (always for --dataset, which is one fixed graph)
                result = run_one(name, args, device)
                if result is None:
                    overview.append((name, "skipped (graph alloc failed)"))
                else:
                    rows = [result.baseline, result.module_ckpt, result.gnn_remat]
                    if result.chunked_remat is not None:
                        rows.append(result.chunked_remat)
                    total    = len(rows)
                    ok       = sum(1 for r in rows if r.ok)
                    oom_labs = [r.label for r in rows if not r.ok]
                    if oom_labs:
                        overview.append((name, f"{ok}/{total} ok · OOM: {', '.join(oom_labs)}"))
                    else:
                        overview.append((name, f"{total}/{total} ok"))
        except Exception as e:
            traceback.print_exc()
            overview.append((name, f"error: {type(e).__name__}"))

    if len(models_to_run) > 1:
        print(f"\n{'─'*64}\nRun summary")
        for name, status in overview:
            print(f"  {name.upper():<12} {status}")


if __name__ == "__main__":
    main()
