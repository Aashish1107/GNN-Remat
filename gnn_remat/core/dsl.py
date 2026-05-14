"""
dsl.py
------
The DSL surface of GNN-Remat.  Provides three complementary syntax styles
that make memory policy look like a language construct, not library boilerplate.

Three ways to express the same intent
--------------------------------------
1. Class decorator (whole model at once):

        import gnn_remat.core.dsl as remat

        @remat.checkpoint
        class MyGAT(nn.Module): ...

        @remat.checkpoint(granularity="aggr", layers=["conv1"])
        class MyGCN(nn.Module): ...

2. Layer annotation (policy at the point of definition):

        class MyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = remat.layer(GCNConv(16, 64))
                self.conv2 = remat.layer(GATConv(64, 8, heads=4), granularity="module")

3. Composable rules (type- or name-based, mixed policies):

        @remat.checkpoint(rules=[
            remat.when_type(GATConv, granularity="aggr"),
            remat.when_type(GCNConv, skip=True),
        ])
        class MyMixedGNN(nn.Module): ...

        # Imperatively:
        model = remat.checkpoint.apply(model, rules=[
            remat.when_name("enc.conv", granularity="aggr"),
        ])

Public API
----------
    remat.checkpoint            class decorator (bare or parameterised)
    remat.checkpoint.apply()    imperative apply to an instance
    remat.layer(conv, ...)      wrap a single conv at definition time
    remat.when_type(cls, ...)   rule matching on layer type
    remat.when_name(name, ...)  rule matching on dotted layer name
"""
from __future__ import annotations

import copy
from typing import List, Optional, Sequence, Type

import torch.nn as nn
from torch_geometric.nn import MessagePassing

from .remat_mp import make_remat_conv
from .wrapper  import wrap


# ── Granularity constants ─────────────────────────────────────────────────────

AGGR   = "aggr"    # checkpoint propagate()            (novel, this project)
MODULE = "module"  # checkpoint full layer forward()   (baseline comparison)


# ── Composable rules ──────────────────────────────────────────────────────────

class _Rule:
    """Base class for DSL composition rules."""

    granularity: str = AGGR
    is_skip: bool    = False

    def matches(self, name: str, module: nn.Module) -> bool:
        raise NotImplementedError


class _WhenType(_Rule):
    """Matches layers by their Python class (or any subclass)."""

    def __init__(self, conv_type: type, *, granularity: str = AGGR, skip: bool = False):
        self._type      = conv_type
        self.granularity = granularity
        self.is_skip     = skip

    def matches(self, name: str, module: nn.Module) -> bool:
        return isinstance(module, self._type)


class _WhenName(_Rule):
    """Matches a layer by its exact dotted attribute path."""

    def __init__(self, layer_name: str, *, granularity: str = AGGR, skip: bool = False):
        self._name       = layer_name
        self.granularity = granularity
        self.is_skip     = skip

    def matches(self, name: str, module: nn.Module) -> bool:
        return name == self._name


def when_type(conv_type: type, *, granularity: str = AGGR, skip: bool = False) -> _Rule:
    """
    DSL rule: apply policy to all layers of *conv_type*.

    Parameters
    ----------
    conv_type : type
        A MessagePassing subclass, e.g. GATConv.
    granularity : {"aggr", "module"}
        Checkpoint granularity.  Ignored when skip=True.
    skip : bool
        If True, explicitly exclude matching layers from any checkpointing.

    Examples
    --------
    >>> remat.when_type(GATConv, granularity="aggr")
    >>> remat.when_type(GCNConv, skip=True)
    """
    return _WhenType(conv_type, granularity=granularity, skip=skip)


def when_name(layer_name: str, *, granularity: str = AGGR, skip: bool = False) -> _Rule:
    """
    DSL rule: apply policy to the layer with the given dotted name.

    Parameters
    ----------
    layer_name : str
        Dotted attribute path, e.g. "conv1" or "encoder.conv2".
    granularity : {"aggr", "module"}
    skip : bool

    Examples
    --------
    >>> remat.when_name("conv1", granularity="aggr")
    >>> remat.when_name("head.readout", skip=True)
    """
    return _WhenName(layer_name, granularity=granularity, skip=skip)


# ── Layer-level annotation ────────────────────────────────────────────────────

def layer(conv: nn.Module, granularity: str = AGGR) -> nn.Module:
    """
    Wrap a single MessagePassing layer at the point of definition.

    Lets you express the memory policy inline instead of as a post-hoc
    transformation of the whole model.

    Parameters
    ----------
    conv : MessagePassing
        The layer to wrap.  Non-MessagePassing modules are returned unchanged.
    granularity : {"aggr", "module"}
        "aggr"   — checkpoint propagate() only  (default, recommended)
        "module" — checkpoint the full layer forward

    Returns
    -------
    nn.Module
        Checkpointed version of *conv*, or *conv* itself if not a MP layer.

    Example
    -------
    >>> class MyModel(nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.conv1 = remat.layer(GCNConv(16, 64))
    ...         self.conv2 = remat.layer(GATConv(64, 8, heads=4))
    """
    if not isinstance(conv, MessagePassing):
        return conv
    return make_remat_conv(conv) if granularity == AGGR else wrap(conv)


# ── Internal transform ────────────────────────────────────────────────────────

def _apply_to_model(
    model:       nn.Module,
    granularity: str                       = AGGR,
    layers:      Optional[List[str]]       = None,
    layer_types: Optional[List[type]]      = None,
    rules:       Optional[List[_Rule]]     = None,
) -> nn.Module:
    """
    Deep-copy model and replace selected MessagePassing layers.

    Priority: rules > layer_types > layers > (all layers when none specified).

    granularity="aggr"   ->  make_remat_conv()  (checkpoint propagate)
    granularity="module" ->  wrap()              (checkpoint full layer)
    """
    model = copy.deepcopy(model)

    for name, mod in list(model.named_modules()):
        if not isinstance(mod, MessagePassing):
            continue

        if rules is not None:
            # First matching rule wins.
            matched: Optional[_Rule] = None
            for rule in rules:
                if rule.matches(name, mod):
                    matched = rule
                    break
            if matched is None:
                continue          # no rule matched — leave layer untouched
            if matched.is_skip:
                continue          # explicitly excluded
            effective_gran = matched.granularity
        else:
            if layers      and name not in layers:
                continue
            if layer_types and not any(isinstance(mod, t) for t in layer_types):
                continue
            effective_gran = granularity

        # Navigate to parent module
        parts  = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)

        replacement = make_remat_conv(mod) if effective_gran == AGGR else wrap(mod)
        setattr(parent, parts[-1], replacement)

    return model


# ── Checkpoint descriptor ─────────────────────────────────────────────────────

class _CheckpointDescriptor:
    """
    Callable object that works as both a bare decorator and a
    parameterised decorator factory, plus an imperative .apply() method.

    Usage patterns
    --------------
    @remat.checkpoint                              bare decorator on a class
    @remat.checkpoint(granularity="aggr")          parameterised decorator
    @remat.checkpoint(rules=[when_type(GATConv)])  rule-based decorator
    remat.checkpoint.apply(model)                  imperative on an instance
    """

    def __call__(
        self,
        cls_or_none=None,
        *,
        granularity: str                   = AGGR,
        layers:      Optional[List[str]]   = None,
        layer_types: Optional[List[type]]  = None,
        rules:       Optional[List[_Rule]] = None,
    ):
        # Bare decorator: @remat.checkpoint
        if isinstance(cls_or_none, type):
            return self._decorate_class(
                cls_or_none, granularity, layers, layer_types, rules
            )
        # Parameterised: @remat.checkpoint(granularity=...) — return a decorator
        def _decorator(cls):
            return self._decorate_class(cls, granularity, layers, layer_types, rules)
        return _decorator

    def _decorate_class(self, cls, granularity, layers, layer_types, rules):
        """Wrap cls.__init__ so every instance is transformed after construction."""
        original_init = cls.__init__

        def patched_init(self_instance, *args, **kwargs):
            original_init(self_instance, *args, **kwargs)
            transformed = _apply_to_model(
                self_instance, granularity, layers, layer_types, rules
            )
            self_instance.__dict__.update(transformed.__dict__)

        cls.__init__   = patched_init
        cls._remat_dsl = True
        return cls

    def apply(
        self,
        model:       nn.Module,
        granularity: str                   = AGGR,
        layers:      Optional[List[str]]   = None,
        layer_types: Optional[List[type]]  = None,
        rules:       Optional[List[_Rule]] = None,
    ) -> nn.Module:
        """
        Imperatively apply rematerialization to a model instance.

        Parameters
        ----------
        model : nn.Module
        granularity : {"aggr", "module"}
        layers : list[str], optional
        layer_types : list[type], optional
        rules : list[_Rule], optional
            If provided, overrides layers/layer_types.  Use when_type() or
            when_name() to build rules.

        Returns
        -------
        nn.Module  (original unchanged)
        """
        return _apply_to_model(model, granularity, layers, layer_types, rules)


# ── Module-level singleton ────────────────────────────────────────────────────

checkpoint = _CheckpointDescriptor()
