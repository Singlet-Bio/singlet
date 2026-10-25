# SPDX-License-Identifier: MIT
"""Verify-or-store recipes for the legacy bundle members (owner: W3; consumer: the migration / verifier of W2).

Interface (agreed with W2, see bench/compression/notes/m_w3.md)::

    RECIPES: dict[str, Recipe]
    plan(member_name, legacy_bytes, ctx) -> Plan | None            Plan = (recipe_id, params, needs)
    regenerate(recipe_id, params, ctx) -> bytes
    regenerate_matrix(recipe_id, params, ctx) -> Matrix            derived .1pz members (canonical-matrix criterion)
    cells_columns(member_name, legacy_bytes, ctx) -> {name: (dtype, ndarray)}   per-cell content moved to the cells table

``plan`` verifies its own proposal against ``legacy_bytes`` (bytes, or the decoded matrix for derived ``.1pz`` members) before
returning it, so a returned plan is a statement "regenerate() reproduces this member on this sample".  Everything is pure numpy.

Recipe ids (versioned; the implementation of an id never changes, a change of behaviour is a new id):

  psi-v1-rint | psi-v1-trunc | psi-v1-stub   splice_psi.1pz from sj_counts.1pz           recipes_psi.py
  cellcalls-v1                               cell_calls.tsv from the cells table          recipes_cells.py
  tsvcols-v1                                 per-cell TSVs from typed cells columns       recipes_tables.py
  whitelist-sorted-v1                        STARsolo raw/barcodes.tsv                    recipes_star.py
  features-v1                                STARsolo raw|filtered/features.tsv           recipes_star.py
  mtx-text-v1                                STARsolo matrix.mtx text from a CSC          recipes_star.py
  mtx-filter-v1                              STARsolo filtered matrix.mtx / barcodes.tsv  recipes_star.py
  ambient-v1                                 ambient_profile.tsv from an integer vector   recipes_extra.py
  geneexpr-v1                                gene_expression.tsv                          recipes_extra.py
  splice-events-v1                           splice_events.tsv                            recipes_extra.py
  vocab-json-v1                              study-level feature_vocab.json               recipes_extra.py
  stub-v1                                    the 151-byte 0 x 0 .1pz stub                 recipes_extra.py
"""

from __future__ import annotations

import collections
import logging

from . import recipes_cells as rcc
from . import recipes_extra as rex
from . import recipes_psi as rpsi
from . import recipes_star as rst
from . import recipes_tables as rtb
from .recipes_ctx import Matrix

log = logging.getLogger("singlet.store.recipes")

Plan = collections.namedtuple("Plan", "recipe_id params needs")

NotByteExact = rpsi.NotByteExact


class Recipe:
    def __init__(self, rid, doc, regenerate, regenerate_matrix=None):
        self.id = rid
        self.doc = doc
        self.regenerate = regenerate
        self.regenerate_matrix = regenerate_matrix

    def __repr__(self):
        return "Recipe(%s)" % self.id


def _mtx_text_regenerate(rid, params, ctx):
    member = rst.RAW_MTX if params.get("m", "raw") == "raw" else rst.FIL_MTX
    return rst.regenerate_mtx_text(rid, params, ctx, member)


def _stub_matrix(rid, params, ctx):
    import numpy as np

    return Matrix(
        np.zeros(1, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.uint32),
        (0, 0),
        [],
        [],
        {},
    )


RECIPES = {}


def _reg(rid, doc, regenerate, regenerate_matrix=None):
    RECIPES[rid] = Recipe(rid, doc, regenerate, regenerate_matrix)


for _r in ("rint", "trunc", "stub"):
    _reg(
        "psi-v1-" + _r,
        "splice_psi.1pz from sj_counts.1pz, PSI stored as %s"
        % ("the constant 0x0 stub" if _r == "stub" else _r + " of the float PSI"),
        rpsi.regenerate_psi,
        rpsi.regenerate_matrix_psi,
    )
_reg("cellcalls-v1", "cell_calls.tsv from the cells table", rcc.regenerate_cell_calls)
_reg("tsvcols-v1", "per-cell TSV from typed cells-table columns", rtb.regenerate_tsv)
_reg(
    "whitelist-sorted-v1",
    "STARsolo raw/barcodes.tsv = the sorted protocol whitelist",
    rst.regenerate_whitelist,
)
_reg(
    "features-v1",
    "STARsolo features.tsv from the gene table (vocab, positional or permutation order)",
    rst.regenerate_features,
)
_reg("mtx-text-v1", "STARsolo matrix.mtx text from a decoded CSC", _mtx_text_regenerate)
_reg(
    "mtx-filter-v1",
    "STARsolo filtered matrix.mtx / barcodes.tsv from raw + called set",
    rst.regenerate_filter,
)
_reg("ambient-v1", "ambient_profile.tsv from the integer vector ambient_k", rex.regenerate_ambient)
_reg(
    "geneexpr-v1",
    "gene_expression.tsv (count, effective length, TPM, FPKM)",
    rex.regenerate_geneexpr,
)
_reg(
    "splice-events-v1", "splice_events.tsv from the sj junction names", rex.regenerate_splice_events
)
_reg(
    "vocab-json-v1", "feature_vocab.json from the vocabulary gene table", rex.regenerate_vocab_json
)
_reg("stub-v1", "the 151-byte 0 x 0 .1pz stub", rex.regenerate_stub, _stub_matrix)

# member name -> planners tried in order
_PLANNERS = {
    "splice_psi.1pz": [rpsi.plan_splice_psi],
    "cell_calls.tsv": [rcc.plan_cell_calls],
    rst.RAW_BC: [rst.plan_whitelist],
    rst.FIL_BC: [rst.plan_filter],
    rst.RAW_FEAT: [rst.plan_features],
    rst.FIL_FEAT: [rst.plan_features],
    rst.RAW_MTX: [rst.plan_mtx_text],
    rst.FIL_MTX: [rst.plan_filter, rst.plan_mtx_text],
    "ambient_profile.tsv": [rex.plan_ambient],
    "gene_expression.tsv": [rex.plan_geneexpr],
    "splice_events.tsv": [rex.plan_splice_events],
    "feature_vocab.json": [rex.plan_vocab_json],
}
for _m in rtb.PROFILES:
    _PLANNERS[_m] = [rtb.plan_tsv]

LAST_ERRORS = {}  # member -> last exception text of a planner (diagnostics)


def _planners(member):
    ps = list(_PLANNERS.get(member, []))
    if member.endswith(".1pz") and member != "splice_psi.1pz":
        ps.append(rex.plan_stub)
    return ps


def plan(member_name, legacy_bytes, ctx):
    """First verified plan for a member, or None.  A planner that raises is skipped (the error is kept in LAST_ERRORS)."""
    if legacy_bytes is None:
        return None
    for fn in _planners(member_name):
        try:
            p = fn(member_name, legacy_bytes, ctx)
        except (
            Exception
        ) as e:  # a recipe bug must never break a migration; the member is stored instead
            LAST_ERRORS[member_name] = "%s: %s" % (type(e).__name__, e)
            log.debug("planner %s failed on %s: %s", getattr(fn, "__name__", fn), member_name, e)
            continue
        if p is None:
            continue
        rid, params, needs = p
        if fn is rst.plan_mtx_text:
            params = dict(params, m="raw" if member_name == rst.RAW_MTX else "filtered")
        return Plan(rid, dict(params), list(needs))
    return None


def regenerate(recipe_id, params, ctx):
    r = RECIPES.get(recipe_id)
    if r is None:
        raise LookupError("unknown recipe %r" % recipe_id)
    return r.regenerate(recipe_id, params, ctx)


def regenerate_matrix(recipe_id, params, ctx):
    r = RECIPES.get(recipe_id)
    if r is None or r.regenerate_matrix is None:
        raise LookupError("recipe %r has no matrix-level regeneration" % recipe_id)
    return r.regenerate_matrix(recipe_id, params, ctx)


def cells_columns(member_name, legacy_bytes, ctx):
    """Typed per-cell columns that make ``member_name`` derivable ({} when the member has none)."""
    if legacy_bytes is None:
        return {}
    try:
        if member_name == "cell_calls.tsv":
            return rcc.cells_columns_cell_calls(member_name, legacy_bytes, ctx)
        if member_name in rtb.PROFILES:
            return rtb.cells_columns_tsv(member_name, legacy_bytes, ctx)
    except Exception as e:
        LAST_ERRORS[member_name] = "%s: %s" % (type(e).__name__, e)
        log.debug("cells_columns failed on %s: %s", member_name, e)
    return {}
