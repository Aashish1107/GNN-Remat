"""
core/replacer.py
----------------
Compile-time step 2: given a list of (name, module) pairs from the detector,
swap each one with a RematConv wrapper inside the model.

The replacer modifies the model in-place on a deep copy so the original
model is never mutated.

Example
-------
  from gnn_remat.core.replacer import apply_remat
  new_model, report = apply_remat(model, candidates)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .wrapper import RematConv


@dataclass
class RematReport:
    """
    Summary of what the replacer changed.

    Attributes
    ----------
    wrapped : list[str]
        Names of layers that were successfully wrapped.
    skipped : list[str]
        Names of layers that were found but not wrapped (e.g. already wrapped).
    """
    wrapped: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = ["GNN-Remat transform report"]
        lines.append(f"  Wrapped ({len(self.wrapped)}): {', '.join(self.wrapped) or 'none'}")
        lines.append(f"  Skipped ({len(self.skipped)}): {', '.join(self.skipped) or 'none'}")
        return "\n".join(lines)


def apply_remat(
    model: nn.Module,
    candidates: list[tuple[str, MessagePassing]],
    *,
    use_reentrant: bool = False,
) -> tuple[nn.Module, RematReport]:
    """
    Deep-copy ``model`` and replace each candidate layer with a RematConv.

    Parameters
    ----------
    model : nn.Module
        Original model (not mutated).
    candidates : list of (name, module)
        Output of ``detect_mp_layers()``.
    use_reentrant : bool
        Forwarded to RematConv (torch.utils.checkpoint setting).

    Returns
    -------
    new_model : nn.Module
        Transformed model with RematConv wrappers installed.
    report : RematReport
        Human-readable summary of what changed.
    """
    new_model = copy.deepcopy(model)
    report = RematReport()

    for name, _ in candidates:
        # Navigate to the parent module using dot-separated name
        parts = name.split(".")
        parent = _get_submodule(new_model, parts[:-1])
        attr = parts[-1]
        original = getattr(parent, attr)

        # Safety: skip if already wrapped (idempotent)
        if isinstance(original, RematConv):
            report.skipped.append(name)
            continue

        setattr(parent, attr, RematConv(original, use_reentrant=use_reentrant))
        report.wrapped.append(name)

    return new_model, report


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_submodule(model: nn.Module, parts: list[str]) -> nn.Module:
    """Traverse a dot-path list to return a nested submodule."""
    node = model
    for part in parts:
        node = getattr(node, part)
    return node
