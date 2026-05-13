"""
examples/train_ogbn_arxiv.py
-----------------------------
Minimal full-batch training loop on ogbn-arxiv showing GNN-Remat in action.

Usage
-----
    pip install ogb
    python examples/train_ogbn_arxiv.py --model gcn --layers 3
    python examples/train_ogbn_arxiv.py --model gat --remat auto
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from gnn_remat import gnn_remat
from gnn_remat.benchmark.models import build


def load_data(device):
    try:
        from ogb.nodeproppred import PygNodePropPredDataset
    except ImportError:
        raise SystemExit("Install ogb first:  pip install ogb")

    # Fix for PyTorch 2.6+ compatibility with OGB by monkey-patching torch.load
    original_load = torch.load
    def patched_load(f, *args, **kwargs):
        # Force weights_only=False for OGB datasets
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(f, *args, **kwargs)
    
    torch.load = patched_load
    try:
        dataset = PygNodePropPredDataset(name="ogbn-arxiv")
    finally:
        torch.load = original_load
    data    = dataset[0]
    split   = dataset.get_idx_split()

    x          = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y          = data.y.squeeze().to(device)
    train_mask = split["train"].to(device)
    val_mask   = split["valid"].to(device)

    return x, edge_index, y, train_mask, val_mask, dataset.num_classes


def train(model, x, edge_index, y, train_mask, optimizer):
    model.train()
    optimizer.zero_grad()
    out  = model(x, edge_index)
    loss = F.cross_entropy(out[train_mask], y[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, x, edge_index, y, val_mask):
    model.eval()
    out  = model(x, edge_index)
    pred = out.argmax(dim=-1)
    return (pred[val_mask] == y[val_mask]).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="gcn",  help="gcn | graphsage | gat")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--remat",  default="all",
                        help="off | all | auto  (GNN-Remat mode)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    x, edge_index, y, train_mask, val_mask, num_classes = load_data(device)
    print(f"Graph: {x.size(0):,} nodes  |  {edge_index.size(1):,} edges  "
          f"|  {x.size(1)} features")

    model = build(
        args.model,
        in_channels=x.size(1),
        hidden=args.hidden,
        out_channels=num_classes,
        num_layers=args.layers,
    ).to(device)

    if args.remat != "off":
        kwargs = dict(mode=args.remat)
        if args.remat == "auto":
            kwargs.update(x=x[:1000], edge_index=edge_index)
        model = gnn_remat(model, verbose=True, **kwargs)
        print(f"GNN-Remat applied (mode={args.remat})")
    else:
        print("No rematerialization (baseline)")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, args.epochs + 1):
        t0   = time.perf_counter()
        loss = train(model, x, edge_index, y, train_mask, optimizer)
        ms   = (time.perf_counter() - t0) * 1000
        if epoch % 10 == 0:
            acc = evaluate(model, x, edge_index, y, val_mask)
            print(f"Epoch {epoch:03d}  loss={loss:.4f}  val_acc={acc:.4f}  {ms:.0f}ms/ep")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        print(f"\nPeak GPU memory: {peak:.1f} MB")


if __name__ == "__main__":
    main()
