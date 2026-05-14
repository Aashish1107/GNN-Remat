"""
runner.py
---------
CLI entry point for GNN-Remat benchmarks.

Usage
-----
    # Single run
    python -m gnn_remat.benchmark.runner --model gat --nodes 5000
    python -m gnn_remat.benchmark.runner --all --nodes 5000

    # Scale sweep: 5K → 10K → 25K → 50K → 100K nodes in one table
    python -m gnn_remat.benchmark.runner --model gat --scale
    python -m gnn_remat.benchmark.runner --all --scale

    # OOM crossover: double nodes until baseline OOMs but GNN-Remat fits (CUDA only)
    python -m gnn_remat.benchmark.runner --model gat --find-limit
    python -m gnn_remat.benchmark.runner --model gat --find-limit --max-nodes 500000
"""
from __future__ import annotations

import gc
import argparse
import traceback

import torch

from .models import build
from .profiler import compare


_SCALE_NODES = [5_000, 10_000, 25_000, 50_000, 100_000]
_ALL_MODELS  = ["gcn", "graphsage", "gat", "transformer"]


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
    print(f"\n{'='*64}")
    print(f"  Model: {model_name.upper()}  |  nodes={args.nodes:,}  "
          f"features={args.features}  layers={args.layers}")
    print(f"{'='*64}")

    model = build(model_name, **_build_kwargs(model_name, args))
    graph_device = torch.device("cpu")  # always build graph on CPU; profiler moves it
    try:
        x, edge_index = _make_graph(args.nodes, args.degree, args.features, graph_device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" not in str(e).lower() and not isinstance(
            e, getattr(torch.cuda, "OutOfMemoryError", ())
        ):
            raise
        print(f"  [skip] could not allocate graph tensors: {e}")
        return None

    result = compare(model, x, edge_index, num_epochs=args.epochs, device=device)
    print(result.summary())
    return result


# ── Scale sweep ───────────────────────────────────────────────────────────────

def run_scale_sweep(model_name: str, args, device):
    """
    Profile the model at each node count in _SCALE_NODES and print a single
    table showing how memory savings grow with graph size.
    """
    if device.type == "cpu":
        print("  Note: running scale sweep on CPU — memory readings will all be 0.0 MB "
              "(CPU allocator has no peak counter). Use CUDA for meaningful results.")

    print(f"\n{'='*80}")
    print(f"  Scale sweep — {model_name.upper()}  "
          f"features={args.features}  hidden={args.hidden}  "
          f"layers={args.layers}  degree={args.degree}  heads={args.heads}")
    print(f"{'='*80}")

    col = f"{'Nodes':>8}  {'Edges':>9}  {'Baseline':>11}  " \
          f"{'Module ckpt':>12}  {'GNN-Remat':>11}  {'Remat saving':>13}"
    print(col)
    print("─" * 80)

    for n_nodes in _SCALE_NODES:
        model   = build(model_name, **_build_kwargs(model_name, args))
        x, ei   = _make_graph(n_nodes, args.degree, args.features, torch.device("cpu"))
        n_edges = n_nodes * args.degree

        try:
            result = compare(model, x, ei, num_epochs=args.epochs, device=device)
        except Exception as exc:
            del model, x, ei
            if _is_cuda_context_error(exc):
                print(f"{n_nodes:>8,}  {n_edges:>9,}  (CUDA context corrupted after OOM — stopping sweep)")
            else:
                traceback.print_exc()
                print(f"{n_nodes:>8,}  — unexpected error, see above")
            break
        finally:
            _cleanup()

        def _mem(r):
            return f"{r.peak_memory_mb:>7.0f} MB" if r.ok else "        OOM"

        ref = result.baseline
        if ref.ok and result.gnn_remat.ok:
            pct = (1 - result.gnn_remat.peak_memory_mb / ref.peak_memory_mb) * 100
            saving = f"{pct:>+.1f}%"
        elif not ref.ok and result.gnn_remat.ok:
            saving = "baseline OOM"
        else:
            saving = "—"

        print(f"{n_nodes:>8,}  {n_edges:>9,}  {_mem(result.baseline)}  "
              f"{_mem(result.module_ckpt)}   {_mem(result.gnn_remat)}  {saving:>13}")

    print()


# ── OOM limit search ──────────────────────────────────────────────────────────

def find_oom_limit(model_name: str, args, device):
    """
    Double node count until baseline OOMs but GNN-Remat still fits.
    This is the headline "OOM-to-fits" result: GNN-Remat enables training at
    graph sizes that are unreachable with the unmodified model.
    """
    if device.type == "cpu":
        print("  [skip] --find-limit requires CUDA. "
              "CPU peak memory is not constrained the same way.")
        return

    print(f"\n{'='*64}")
    print(f"  OOM crossover search — {model_name.upper()}")
    print(f"  Config: features={args.features}  hidden={args.hidden}  "
          f"layers={args.layers}  heads={args.heads}  degree={args.degree}")
    print(f"  Starting at {args.nodes:,} nodes, doubling each step "
          f"(cap: {args.max_nodes:,})")
    print(f"{'='*64}")
    print(f"  {'Nodes':>9}  {'Edges':>10}  {'Baseline':>12}  {'GNN-Remat':>12}")
    print(f"  {'─'*52}")

    n_nodes = args.nodes
    found   = False

    while n_nodes <= args.max_nodes:
        model   = build(model_name, **_build_kwargs(model_name, args))
        x, ei   = _make_graph(n_nodes, args.degree, args.features, torch.device("cpu"))
        n_edges = n_nodes * args.degree

        try:
            result = compare(model, x, ei, num_epochs=2, device=device)
        except Exception as exc:
            del model, x, ei
            if _is_cuda_context_error(exc):
                print(f"  {n_nodes:>9,}  (CUDA context corrupted after OOM — stopping)")
            else:
                traceback.print_exc()
            break
        finally:
            _cleanup()

        base_ok  = result.baseline.ok
        remat_ok = result.gnn_remat.ok

        base_str  = f"{result.baseline.peak_memory_mb:>8.0f} MB"  if base_ok  else "         OOM"
        remat_str = f"{result.gnn_remat.peak_memory_mb:>8.0f} MB" if remat_ok else "         OOM"
        print(f"  {n_nodes:>9,}  {n_edges:>10,}  {base_str}  {remat_str}")

        if not base_ok and remat_ok:
            print(f"\n  GNN-Remat enables training at {n_nodes:,} nodes ({n_edges:,} edges).")
            print(f"    Baseline  : OOM")
            print(f"    GNN-Remat : {result.gnn_remat.peak_memory_mb:.0f} MB  "
                  f"({result.gnn_remat.epoch_time_ms:.1f} ms/step)")
            found = True
            break

        if not base_ok and not remat_ok:
            print(f"\n  Both OOM at {n_nodes:,} nodes.")
            print("  Try smaller --hidden or --heads, or a GPU with more VRAM.")
            break

        n_nodes *= 2

    if not found and n_nodes > args.max_nodes:
        print(f"\n  Baseline did not OOM up to {args.max_nodes:,} nodes.")
        print("  Increase --max-nodes, --hidden, or --heads to stress the GPU more.")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main(argv=None):
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
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_run = _ALL_MODELS if args.all else [args.model]

    overview: list[tuple[str, str]] = []
    for name in models_to_run:
        try:
            if args.scale:
                run_scale_sweep(name, args, device)
                overview.append((name, "scale sweep done"))
            elif args.find_limit:
                find_oom_limit(name, args, device)
                overview.append((name, "limit search done"))
            else:
                result = run_one(name, args, device)
                if result is None:
                    overview.append((name, "skipped (graph alloc failed)"))
                else:
                    rows     = [result.baseline, result.module_ckpt, result.gnn_remat]
                    ok       = sum(1 for r in rows if r.ok)
                    oom_labs = [r.label for r in rows if not r.ok]
                    if oom_labs:
                        overview.append((name, f"{ok}/3 ok · OOM: {', '.join(oom_labs)}"))
                    else:
                        overview.append((name, "3/3 ok"))
        except Exception as e:
            traceback.print_exc()
            overview.append((name, f"error: {type(e).__name__}"))

    if len(models_to_run) > 1:
        print(f"\n{'─'*64}\nRun summary")
        for name, status in overview:
            print(f"  {name.upper():<12} {status}")


if __name__ == "__main__":
    main()
