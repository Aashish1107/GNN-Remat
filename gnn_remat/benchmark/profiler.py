"""
profiler.py
-----------
Measures peak GPU (or CPU) memory and training throughput for a GNN model
under three conditions: baseline, module-level checkpoint, and GNN-Remat.

Public API
----------
    profile(model, x, edge_index, ...)    -> ProfileResult
    compare(model, x, edge_index, ...)    -> CompareResult
    CompareResult.summary()               -> str (human-readable table)
"""
from __future__ import annotations

import gc
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


#Result data classes

@dataclass
class ProfileResult:
    """Metrics for a single model+condition."""
    label: str
    peak_memory_mb: float
    epoch_time_ms: float
    epochs_run: int

    @property
    def throughput(self) -> float:
        """Epochs per second."""
        return self.epochs_run / (self.epoch_time_ms * self.epochs_run / 1000)


@dataclass
class CompareResult:
    """Side-by-side comparison of three conditions."""
    baseline: ProfileResult
    module_ckpt: ProfileResult
    gnn_remat: ProfileResult

    def summary(self) -> str:
        rows = [self.baseline, self.module_ckpt, self.gnn_remat]
        lines = [
            f"{'Condition':<22}  {'Peak Mem (MB)':>14}  {'Epoch (ms)':>11}  {'Mem saved':>10}",
            "-" * 64,
        ]
        base_mem = self.baseline.peak_memory_mb
        for r in rows:
            saved = (1 - r.peak_memory_mb / base_mem) * 100 if base_mem > 0 else 0
            lines.append(
                f"{r.label:<22}  {r.peak_memory_mb:>14.1f}  "
                f"{r.epoch_time_ms:>11.1f}  {saved:>9.1f}%"
            )
        return "\n".join(lines)


#Core profiling logic

def _reset_memory():
    """Clear CUDA memory stats and run GC."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_memory_mb() -> float:
    """Return peak allocated memory in MB (GPU if available, else 0)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def profile(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    label: str = "model",
    num_epochs: int = 5,
    warmup: int = 1,
    criterion: Optional[Callable] = None,
    device: Optional[torch.device] = None,
) -> ProfileResult:
    """
    Run *num_epochs* of forward + backward and record peak memory and time.

    Parameters
    ----------
    model : nn.Module
    x, edge_index : torch.Tensor
        Graph data. Moved to *device* automatically.
    label : str
        Name shown in summary tables.
    num_epochs : int
        Training steps to average over.
    warmup : int
        Steps to run before measurement begins.
    criterion : callable, optional
        Loss function(output, target). Defaults to output.sum().
    device : torch.device, optional
        Defaults to CUDA if available, else CPU.

    Returns
    -------
    ProfileResult
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).train()
    x = x.to(device)
    edge_index = edge_index.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def _step():
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = out.sum() if criterion is None else criterion(out)
        loss.backward()
        optimizer.step()

    # Warmup
    for _ in range(warmup):
        _step()

    # Measured run
    _reset_memory()
    t0 = time.perf_counter()
    for _ in range(num_epochs):
        _step()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return ProfileResult(
        label=label,
        peak_memory_mb=_peak_memory_mb(),
        epoch_time_ms=elapsed_ms / num_epochs,
        epochs_run=num_epochs,
    )


def compare(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_epochs: int = 5,
    device: Optional[torch.device] = None,
) -> CompareResult:
    """
    Profile *model* under three conditions and return a CompareResult.

    Conditions
    ----------
    1. Baseline   — vanilla model, no checkpointing
    2. ModuleCkpt — torch.utils.checkpoint wrapping entire layers
    3. GNN-Remat  — aggregation-granular checkpointing (this project)

    Parameters
    ----------
    model : nn.Module
    x, edge_index : torch.Tensor
    num_epochs : int
    device : torch.device, optional

    Returns
    -------
    CompareResult
    """
    import copy
    from gnn_remat import gnn_remat
    from torch_geometric.nn import MessagePassing

    # ── Condition 1: Baseline ────────────────────────────────────────────────
    base_result = profile(
        copy.deepcopy(model), x, edge_index,
        label="Baseline (PyG)",
        num_epochs=num_epochs,
        device=device,
    )

    # ── Condition 2: Module-level checkpoint ─────────────────────────────────
    module_model = copy.deepcopy(model)
    for name, mod in list(module_model.named_modules()):
        if isinstance(mod, MessagePassing):
            parts = name.split(".")
            parent = module_model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            original = getattr(parent, parts[-1])
            # Wrap with vanilla torch checkpoint at module level
            class _ModuleCkpt(nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m
                def forward(self, *args):
                    return torch.utils.checkpoint.checkpoint(
                        self.m, *args, use_reentrant=False
                    )
            setattr(parent, parts[-1], _ModuleCkpt(original))

    mod_result = profile(
        module_model, x, edge_index,
        label="Module checkpoint",
        num_epochs=num_epochs,
        device=device,
    )

    # ── Condition 3: GNN-Remat ───────────────────────────────────────────────
    remat_model = gnn_remat(copy.deepcopy(model))
    remat_result = profile(
        remat_model, x, edge_index,
        label="GNN-Remat",
        num_epochs=num_epochs,
        device=device,
    )

    return CompareResult(
        baseline=base_result,
        module_ckpt=mod_result,
        gnn_remat=remat_result,
    )
