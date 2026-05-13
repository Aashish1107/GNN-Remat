"""
dsl.py
------
The DSL surface of GNN-Remat. Provides decorator and annotation syntax
that makes the memory policy look like a language construct, not just
a function call.

This is what justifies calling this an embedded DSL:
- You declare intent with an annotation on the class or layer
- The system rewrites execution based on that declaration
- The user never writes memory management code

Public API
----------
    @remat.checkpoint                        class decorator
    @remat.checkpoint(granularity="aggr")    parameterised decorator
    @remat.checkpoint(layers=["conv1"])      selective decorator
    remat.checkpoint.apply(model)            imperative apply

Examples
--------
    import gnn_remat.core.dsl as remat

    # 1. Annotate a whole model class (all MP layers checkpointed)
    @remat.checkpoint
    class MyGCN(nn.Module): ...

    # 2. Annotate with options
    @remat.checkpoint(granularity="aggr", layers=["conv1"])
    class MyGAT(nn.Module): ...

    # 3. Apply imperatively to an existing instance
    model = remat.checkpoint.apply(model)
    model = remat.checkpoint.apply(model, granularity="module")
"""
from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Type

import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .remat_mp import make_remat_conv
from .wrapper  import wrap


# ── Granularity constants (the DSL's vocabulary) ──────────────────────────────
AGGR   = "aggr"    # checkpoint only scatter aggregation  (novel, this project)
MODULE = "module"  # checkpoint full layer                (baseline comparison)


# ── Internal transform ────────────────────────────────────────────────────────

def _apply_to_model(
    model:       nn.Module,
    granularity: str            = AGGR,
    layers:      Optional[List[str]] = None,
    layer_types: Optional[List[Type[MessagePassing]]] = None,
) -> nn.Module:
    """
    Deep-copy model and replace selected MessagePassing layers.

    granularity="aggr"   ->  make_remat_conv()  (checkpoint scatter only)
    granularity="module" ->  wrap()              (checkpoint full layer)
    """
    model = copy.deepcopy(model)

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, MessagePassing):
            continue
        if layers      and name  not in layers:
            continue
        if layer_types and not any(isinstance(mod, t) for t in layer_types):
            continue

        # Navigate to parent
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        if granularity == AGGR:
            replacement = make_remat_conv(mod)
        else:
            replacement = wrap(mod)

        setattr(parent, parts[-1], replacement)

    return model


# ── Checkpoint descriptor (the DSL entry-point) ───────────────────────────────

class _CheckpointDescriptor:
    """
    Callable object that works as both a bare decorator and a
    parameterised decorator factory, plus an imperative .apply() method.

    Usage patterns
    --------------
    @remat.checkpoint                    # bare decorator on a class
    @remat.checkpoint(granularity=...)   # parameterised decorator on a class
    remat.checkpoint.apply(model)        # imperative on an instance
    """

    # Default policy
    _default_granularity = AGGR

    def __call__(self, cls_or_granularity=None, *,
                 granularity: str = AGGR,
                 layers: Optional[List[str]] = None,
                 layer_types: Optional[List[Type]] = None):
        """
        Works in two modes:
          @remat.checkpoint                ->  cls_or_granularity is the class
          @remat.checkpoint(granularity=.) ->  cls_or_granularity is None,
                                               returns a decorator
        """
        # Bare decorator: @remat.checkpoint
        if isinstance(cls_or_granularity, type):
            cls = cls_or_granularity
            return self._decorate_class(cls, granularity, layers, layer_types)

        # Parameterised: @remat.checkpoint(granularity="aggr")
        # cls_or_granularity is the first keyword-ish positional arg, ignore
        def _decorator(cls):
            return self._decorate_class(cls, granularity, layers, layer_types)
        return _decorator

    def _decorate_class(self, cls, granularity, layers, layer_types):
        """
        Wrap cls.__init__ so that every instance is automatically transformed
        after construction.
        """
        original_init = cls.__init__

        def patched_init(self_instance, *args, **kwargs):
            original_init(self_instance, *args, **kwargs)
            transformed = _apply_to_model(
                self_instance, granularity, layers, layer_types
            )
            # Copy transformed submodules back into self
            self_instance.__dict__.update(transformed.__dict__)

        cls.__init__    = patched_init
        cls._remat_dsl  = True   # mark for introspection
        return cls

    def apply(
        self,
        model:       nn.Module,
        granularity: str = AGGR,
        layers:      Optional[List[str]] = None,
        layer_types: Optional[List[Type]] = None,
    ) -> nn.Module:
        """
        Imperatively apply rematerialization to a model instance.

        Parameters
        ----------
        model : nn.Module
        granularity : {"aggr", "module"}
            "aggr"   — checkpoint scatter aggregation only  (default)
            "module" — checkpoint entire MessagePassing layer
        layers : list[str], optional
            Dotted layer names to target. Default: all MessagePassing layers.
        layer_types : list[type], optional
            MessagePassing subclasses to target.

        Returns
        -------
        nn.Module
            New model (original unchanged).
        """
        return _apply_to_model(model, granularity, layers, layer_types)


# ── Module-level singleton — this is what users import ───────────────────────
checkpoint = _CheckpointDescriptor()
