# SPDX-License-Identifier: MIT
"""Adapter between the migration / verifier and the recipes module (W3, ``singlet.v2.recipes``).

The migration only relies on this small surface of the recipes module::

    RECIPES                                                        # dict recipe_id -> recipe (optional, informational)
    plan(member_name, legacy_bytes, ctx) -> (recipe_id, params, needs) | None
    regenerate(recipe_id, params, ctx) -> bytes                    # exact legacy bytes (may raise if only the matrix is reproducible)
    regenerate_matrix(recipe_id, params, ctx) -> Matrix            # optional: for derived ``.1pz`` members (canonical criterion)
    cells_columns(member_name, legacy_bytes, ctx) -> {name: (dtype, ndarray)}   # optional

Anything missing is treated as "no recipe": the member is stored (kind 3). ``load_recipes()`` returns an adapter around
``singlet.v2.recipes`` if it imports, always followed by the tiny built-in set (``vocab-json-v1`` for the study-level
``feature_vocab.json``) so that the migration works end to end without W3's module.
"""
from __future__ import annotations

import importlib
from typing import Optional

from . import builtin_recipes


class Recipes:
    def __init__(self, mods):
        self.mods = [m for m in mods if m is not None]

    # ------------------------------------------------------------------ plan
    def plan(self, member: str, legacy_bytes: bytes, ctx):
        for m in self.mods:
            fn = getattr(m, "plan", None)
            if fn is None:
                continue
            p = fn(member, legacy_bytes, ctx)
            if p is not None:
                rid, params, needs = p[0], p[1], list(p[2]) if len(p) > 2 and p[2] is not None else []
                return rid, dict(params or {}), needs
        return None

    def cells_columns(self, member: str, legacy_bytes: bytes, ctx) -> dict:
        out = {}
        for m in self.mods:
            fn = getattr(m, "cells_columns", None)
            if fn is None:
                continue
            r = fn(member, legacy_bytes, ctx)
            if r:
                for k, v in r.items():
                    out.setdefault(k, v)
        return out

    # ------------------------------------------------------------------ regenerate
    def _owner(self, recipe_id: str):
        for m in self.mods:
            reg = getattr(m, "RECIPES", None)
            if reg is not None and recipe_id in reg:
                return m
        return None

    def regenerate_bytes(self, recipe_id: str, params: dict, ctx) -> bytes:
        own = self._owner(recipe_id)
        for m in ([own] if own is not None else []) + [x for x in self.mods if x is not own]:
            fn = getattr(m, "regenerate", None)
            if fn is None:
                continue
            try:
                return bytes(fn(recipe_id, params, ctx))
            except LookupError:
                continue
        raise LookupError("no recipe module can regenerate %r" % recipe_id)

    def regenerate_member_matrix(self, recipe_id: str, params: dict, ctx):
        """Canonical matrix of a derived ``.1pz`` member (a recipes ``Matrix`` / ``Csc``)."""
        own = self._owner(recipe_id)
        for m in ([own] if own is not None else []) + [x for x in self.mods if x is not own]:
            fn = getattr(m, "regenerate_matrix", None)
            if fn is None:
                continue
            try:
                return fn(recipe_id, params, ctx)
            except LookupError:
                continue
        # no matrix-level entry point: regenerate the container bytes and decode them
        from .legacy import decode_1pz_bytes
        return decode_1pz_bytes(self.regenerate_bytes(recipe_id, params, ctx))

    def ids(self) -> list:
        out = []
        for m in self.mods:
            out.extend(sorted(getattr(m, "RECIPES", {}) or {}))
        return out


def load_recipes(spec=None) -> Recipes:
    """``None`` -> ``singlet.v2.recipes`` if importable + built-ins; ``"none"`` -> built-ins only; a module / object -> that + built-ins."""
    mods = []
    if spec is None:
        try:
            mods.append(importlib.import_module("singlet.v2.recipes"))
        except ImportError:
            pass
    elif isinstance(spec, str):
        if spec != "none":
            mods.append(importlib.import_module(spec))
    elif isinstance(spec, Recipes):
        return spec
    else:
        mods.append(spec)
    mods.append(builtin_recipes)
    return Recipes(mods)
