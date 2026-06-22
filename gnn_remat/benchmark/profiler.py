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
import math
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
    status: str = "ok"

    @property
    def ok(self)-> bool:
        return self.status == "ok"

    @property
    def throughput(self) -> float:
        if not self.ok or self.epoch_time_ms <= 0:
            return float("nan")
        """Epochs per second."""
        return self.epochs_run / (self.epoch_time_ms * self.epochs_run / 1000)


@dataclass
class CompareResult:
    """Side-by-side comparison of three (or four) conditions."""
    baseline: ProfileResult
    module_ckpt: ProfileResult
    gnn_remat: ProfileResult
    chunked_remat: Optional[ProfileResult] = None  # set when chunk_nodes is passed

    def summary(self) -> str:
        rows = [self.baseline, self.module_ckpt, self.gnn_remat]
        if self.chunked_remat is not None:
            rows.append(self.chunked_remat)

        # Use the first successful row as the reference for "Mem saved".
        # Normally that's the baseline; when baseline OOMs we fall back to
        # module-checkpoint so the GNN-Remat saving is still legible.
        reference = next((r for r in rows if r.ok), None)
        ref_is_baseline = reference is self.baseline
        saved_header = "Mem saved" if (reference is None or ref_is_baseline) else "Mem saved*"

        lines = [
            f"{'Condition':<24}  {'Peak Mem (MB)':>14}  "
            f"{'Epoch (ms)':>11}  {saved_header:>10}",
            "-" * 66,
        ]
        for r in rows:
            if not r.ok:
                lines.append(
                    f"{r.label:<24}  {'OOM':>14}  {'OOM':>11}  {'—':>10}"
                )
                continue
            if reference is None or reference.peak_memory_mb <= 0:
                lines.append(
                    f"{r.label:<24}  {r.peak_memory_mb:>14.1f}  "
                    f"{r.epoch_time_ms:>11.1f}  {'—':>10}"
                )
            else:
                saved = (1 - r.peak_memory_mb / reference.peak_memory_mb) * 100
                lines.append(
                    f"{r.label:<24}  {r.peak_memory_mb:>14.1f}  "
                    f"{r.epoch_time_ms:>11.1f}  {saved:>9.1f}%"
                )

        if reference is not None and not ref_is_baseline:
            lines.append(
                f"  * Mem saved is vs. {reference.label} "
                f"(baseline OOM'd at this configuration)."
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

def _is_oom(exc: BaseException) -> bool:
    if hasattr(torch.cuda, "OutOfMemoryError") and isinstance(
        exc, torch.cuda.OutOfMemoryError
    ):
        return True
    if not isinstance(exc, (RuntimeError, MemoryError)):
        return False
    msg = str(exc).lower()
    return any(needle in msg for needle in (
        "out of memory",
        "cuda error: out of memory",
        "cublas_status_alloc_failed",
        "cudnn_status_not_initialized",
        "not enough memory",
        "alloc_cpu",
        "cannot allocate memory",
    ))

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
 
    model_dev: Optional[nn.Module] = None
    x_dev: Optional[torch.Tensor] = None
    ei_dev: Optional[torch.Tensor] = None
    optimizer: Optional[torch.optim.Optimizer] = None
 
    try:
        model_dev = model.to(device).train()
        x_dev = x.to(device)
        ei_dev = edge_index.to(device)
        optimizer = torch.optim.Adam(model_dev.parameters(), lr=1e-3)
 
        def _step():
            optimizer.zero_grad()
            out = model_dev(x_dev, ei_dev)
            loss = out.sum() if criterion is None else criterion(out)
            loss.backward()
            optimizer.step()
 
        # Warmup (not measured). OOM here is still OOM — same handling.
        for _ in range(warmup):
            _step()
 
        # Measured run.
        _reset_memory()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(num_epochs):
            _step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
 
        return ProfileResult(
            label=label,
            peak_memory_mb=_peak_memory_mb(),
            epoch_time_ms=elapsed_ms / num_epochs,
            epochs_run=num_epochs,
            status="ok",
        )
 
    except BaseException as exc:
        if not _is_oom(exc):
            raise
        # Drop refs before empty_cache, otherwise the allocator still
        # holds the blocks and the next condition inherits the pressure.
        model_dev = None
        x_dev = None
        ei_dev = None
        optimizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ProfileResult(
            label=label,
            peak_memory_mb=float("nan"),
            epoch_time_ms=float("nan"),
            epochs_run=0,
            status="OOM",
        )
 
    finally:
        # Belt-and-suspenders cleanup on both the happy path and any
        # non-OOM error. Setting locals to None drops profile()'s refs;
        # the caller may still hold a reference to the original model.
        model_dev = None
        x_dev = None
        ei_dev = None
        optimizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def _between_conditions():
    """Aggressive memory reclamation between benchmark conditions."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compare(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    num_epochs: int = 5,
    device: Optional[torch.device] = None,
    chunk_nodes: Optional[int] = None,
) -> CompareResult:
    """
    Profile *model* under three (or four) conditions and return a CompareResult.

    Conditions
    ----------
    1. Baseline        — vanilla model, no checkpointing
    2. ModuleCkpt      — torch.utils.checkpoint wrapping entire layers
    3. GNN-Remat       — propagate-level checkpoint (this project)
    4. GNN-Remat+Chunk — condition 3 + SAR-inspired chunked propagation
                         (only when chunk_nodes is not None)

    Parameters
    ----------
    model : nn.Module
    x, edge_index : torch.Tensor
    num_epochs : int
    device : torch.device, optional
    chunk_nodes : int, optional
        If set, adds a 4th condition using destination-node chunked propagation.
        Use auto_chunk_size() from gnn_remat to compute a good value, or pass
        an explicit integer.

    Returns
    -------
    CompareResult
    """
    import copy
    from gnn_remat import gnn_remat
    from torch_geometric.nn import MessagePassing

    # ── Condition 1: Baseline ────────────────────────────────────────────────
    base_model = copy.deepcopy(model)
    base_result = profile(
        base_model, x, edge_index,
        label="Baseline (PyG)",
        num_epochs=num_epochs,
        device=device,
    )
    del base_model
    _between_conditions()

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
    del module_model
    _between_conditions()

    # ── Condition 3: GNN-Remat (propagate-level checkpoint only) ─────────────
    remat_model = gnn_remat(copy.deepcopy(model))
    remat_result = profile(
        remat_model, x, edge_index,
        label="GNN-Remat",
        num_epochs=num_epochs,
        device=device,
    )
    del remat_model
    _between_conditions()

    # ── Condition 4: GNN-Remat + chunked propagation (optional) ──────────────
    chunked_result: Optional[ProfileResult] = None
    if chunk_nodes is not None:
        chunked_model = gnn_remat(copy.deepcopy(model), chunk_nodes=chunk_nodes)
        chunked_result = profile(
            chunked_model, x, edge_index,
            label=f"GNN-Remat+Chunk",
            num_epochs=num_epochs,
            device=device,
        )
        del chunked_model
        _between_conditions()

    return CompareResult(
        baseline=base_result,
        module_ckpt=mod_result,
        gnn_remat=remat_result,
        chunked_remat=chunked_result,
    )
