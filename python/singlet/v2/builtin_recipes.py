# SPDX-License-Identifier: MIT
"""Built-in stand-in recipes (W2). Only what the migration needs to run end to end when ``singlet.v2.recipes`` (W3) is absent or
does not cover a member: the study-level ``feature_vocab.json`` from the vocabulary pack.

W3's recipes take precedence (the adapter asks them first); the ids here are versioned so that a member derived by a built-in recipe
stays regenerable after W3's module is replaced.
"""

from __future__ import annotations

RECIPES = {
    "vocab-json-v1": "feature_vocab.json = json.dumps(gene table, indent=2) of the exon vocabulary pack"
}


def plan(member, legacy_bytes, ctx):
    if member != "feature_vocab.json" or ctx.sample_id:
        return None
    pk = ctx.pack_for("exon") if hasattr(ctx, "pack_for") else None
    if pk is None:
        return None
    try:
        ok = pk.vocab_json() == legacy_bytes
    except Exception:
        return None
    return ("vocab-json-v1", {"pack": pk.id.hex()}, ["pack:genes"]) if ok else None


def regenerate(recipe_id, params, ctx):
    if recipe_id != "vocab-json-v1":
        raise LookupError(recipe_id)
    return ctx.packs.get(params["pack"]).vocab_json()
