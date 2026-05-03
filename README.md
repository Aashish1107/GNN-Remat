# GNN-Remat

> Aggregation-granular rematerialization for PyTorch Geometric — cut peak training memory
> by 20-25% with under 8% throughput overhead.

## What it does

Standard gradient checkpointing (`torch.utils.checkpoint`) recomputes *entire layers* during
backprop — including expensive linear projections and attention coefficients. GNN-Remat wraps
only the **neighbor aggregation step** (the actual memory bottleneck), so you pay the minimum
recompute cost for the maximum memory saving.

```
Baseline PyG           →  stores all activations       →  high memory, fast
torch.utils.checkpoint →  recomputes full layers        →  low memory, slow
GNN-Remat              →  recomputes aggregation only   →  low memory, fast
```

## Install

### Create a conda environment

Create a conda environment with python version >=3.9 and <=3.12.
```bash
conda create -n gnn-remat python=3.12
```

Activate the Environment
```bash
conda activate gnn-remat
```

### Run in terminal
From root repo:
```bash
cd gnn_remat
pip install -e .
```

This should use setuptools to install and set up all the required dependencies for the project to run.

## Quick start

```python
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from gnn_remat import gnn_remat

# 1. Define your model as normal - this is an example model, use your GCN Model here
class MyGCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(16, 64)
        self.conv2 = GCNConv(64, 8)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv2(x, edge_index)

# 2. Wrap it — one line
model = MyGCN()
model = gnn_remat(model)

x          = torch.randn(100, 16)
edge_index = torch.randint(0, 100, (2, 400))
out        = model(x, edge_index)
print(out.shape)  # torch.Size([100, 8])
```

## RUN tests
Make sure your project can detect your GPU by,

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Run all 44 tests:
```bash
python run_tests.py
```

If This fails to run, It mostly is a path error inside run_tests.py. Change according to your file system.

## Benchmarks

!!!Careful using benchmark.runner
If the benchmark execution fails before printing anything in the terminal that means the Baseline crashed which is inline with program.

This will be updated to handle the Crash with a --remat-only or a --avoid-Baseline extension or will be inbuilt into the runner.py program by default.

For now be careful with the size of the model, i.e number of nodes and features

## Project layout

```
gnn_remat/
├── core/
│   ├── detector.py     # finds MessagePassing modules in any model
│   ├── wrapper.py      # _RematConv: checkpoints a single conv layer
│   ├── transform.py    # applies wrappers to a model (the compiler pass)
│   └── heuristic.py    # auto-selects which layers to checkpoint
├── benchmark/
│   ├── profiler.py     # measures peak GPU memory + throughput
│   ├── models.py       # GCN, GraphSAGE, GAT reference models
│   └── runner.py       # CLI entry point
├── tests/
│   ├── test_detector.py
│   ├── test_wrapper.py
│   ├── test_transform.py
│   └── test_heuristic.py
├── examples/
│   └── train_ogbn_arxiv.py
└── __init__.py         # public API surface
```
