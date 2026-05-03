"""
runner.py
---------
CLI entry point for GNN-Remat benchmarks.

Usage
-----
    python -m gnn_remat.benchmark.runner --model gcn --nodes 5000 --features 128 --layers 3
    python -m gnn_remat.benchmark.runner --model gat --nodes 5000 --features 64 --layers 2 --heads 4
    python -m gnn_remat.benchmark.runner --all
"""
from __future__ import annotations

import argparse
import sys
import torch

from .models import build
from .profiler import compare


def _make_graph(num_nodes: int, avg_degree: int, num_features: int, device):
    """Synthesise a random graph for benchmarking."""
    x = torch.randn(num_nodes, num_features, device=device)
    num_edges = num_nodes * avg_degree
    src = torch.randint(0, num_nodes, (num_edges,), device=device)
    dst = torch.randint(0, num_nodes, (num_edges,), device=device)
    return x, torch.stack([src, dst])


def run_one(model_name: str, args, device):
    print(f"\n{'='*64}")
    print(f"  Model: {model_name.upper()}  |  nodes={args.nodes}  "
          f"features={args.features}  layers={args.layers}")
    print(f"{'='*64}")

    model = build(
        model_name,
        in_channels=args.features,
        hidden=args.hidden,
        out_channels=args.classes,
        num_layers=args.layers,
    )
    x, edge_index = _make_graph(args.nodes, args.degree, args.features, device)

    result = compare(model, x, edge_index, num_epochs=args.epochs, device=device)
    print(result.summary())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="GNN-Remat benchmark runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",    default="gcn",  help="gcn | graphsage | gat")
    parser.add_argument("--all",      action="store_true", help="Run all three models")
    parser.add_argument("--nodes",    type=int, default=5000,  help="Number of nodes")
    parser.add_argument("--features", type=int, default=128,   help="Node feature dimension")
    parser.add_argument("--hidden",   type=int, default=256,   help="Hidden dimension")
    parser.add_argument("--classes",  type=int, default=40,    help="Output classes")
    parser.add_argument("--layers",   type=int, default=3,     help="Number of GNN layers")
    parser.add_argument("--degree",   type=int, default=10,    help="Avg node degree")
    parser.add_argument("--epochs",   type=int, default=5,     help="Benchmark epochs")
    parser.add_argument("--heads",    type=int, default=4,     help="Attention heads (GAT only)")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    models_to_run = ["gcn", "graphsage", "gat"] if args.all else [args.model]
    for name in models_to_run:
        run_one(name, args, device)


if __name__ == "__main__":
    main()
