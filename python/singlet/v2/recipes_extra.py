# SPDX-License-Identifier: MIT
"""ambient-v1, geneexpr-v1, splice-events-v1, vocab-json-v1, stub-v1 (owner: W3).

  ambient-v1       ambient_profile.tsv from an integer vector k (UMIs of the ambient pool per exon row):
                   value = (k + 0.5) * (1 / sum(k + 0.5)) printed %.6g (ambient_correction.h:estimate_ambient)
  geneexpr-v1      gene_expression.tsv: count (exon + intron gene sums), effective_length (merged exon bp), TPM, FPKM
                   with the normaliser total_mapped (= pileup_stats.json mapped_reads) as a param (tpm_fpkm.h)
  splice-events-v1 splice_events.tsv from the junction names of sj_counts.1pz (splice_psi.h:write_splice_events)
  vocab-json-v1    study-level feature_vocab.json from the gene table of the vocabulary pack
  stub-v1          the constant 151-byte 0x0 .1pz the pipeline writes for an output it failed to produce
"""
from __future__ import annotations

import json
import re

import numpy as np

from . import recipes_psi as rpsi
from . import recipes_pzw as pzw
from . import recipes_qc as rqc

AMBIENT_VECTOR = "ambient_k"


# ---------------------------------------------------------------------------------------------------
# ambient-v1
# ---------------------------------------------------------------------------------------------------
def render_ambient(names, k) -> bytes:
    k = np.asarray(k, dtype=np.float64)
    h = k + 0.5
    inv = 1.0 / float(np.cumsum(h)[-1]) if len(h) else 0.0
    v = h * inv
    return ("feature\tambient_fraction\n" + "".join("%s\t%.6g\n" % (n, x) for n, x in zip(names, v.tolist()))).encode("utf-8")


def infer_ambient_k(vals):
    """Candidate integer vectors k (generator) with sum(k) == T for the T consistent with the printed values."""
    v = np.asarray(vals, dtype=np.float64)
    n = len(v)
    if n == 0 or not (v > 0).all():
        return
    vmin = float(v.min())
    for kmin in range(0, 4):
        D = (kmin + 0.5) / vmin
        T0 = int(round(D - 0.5 * n))
        w = int(D * 1e-6) + 4
        for dT in sorted(range(-w, w + 1), key=abs):
            T = T0 + dT
            if T < 0:
                continue
            k = np.rint(v * (T + 0.5 * n) - 0.5)
            if (k < 0).any() or int(k.sum()) != T:
                continue
            yield k.astype(np.int64)


def plan_ambient(member, legacy_bytes, ctx):
    names = ctx.vocab_names("exon")
    if names is None:
        return None
    try:
        s = legacy_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not s.startswith("feature\tambient_fraction\n") or not s.endswith("\n"):
        return None
    rows = [r.split("\t") for r in s[len("feature\tambient_fraction\n"):-1].split("\n")] if len(s) > 25 else []
    if len(rows) != len(names) or any(len(r) != 2 for r in rows) or [r[0] for r in rows] != list(names):
        return None
    try:
        vals = [float(r[1]) for r in rows]
    except ValueError:
        return None
    k = None
    for cand in infer_ambient_k(vals):
        if render_ambient(names, cand) == legacy_bytes:
            k = cand
            break
    if k is None:
        return None
    dt = np.uint16 if k.max() < 65536 else np.uint32
    if hasattr(ctx, "provide_vector"):
        ctx.provide_vector(AMBIENT_VECTOR, k.astype(dt))
    return ("ambient-v1", {}, ["pack:exon", "vector:" + AMBIENT_VECTOR])


def regenerate_ambient(recipe_id, params, ctx) -> bytes:
    return render_ambient(ctx.vocab_names("exon"), ctx.vector(AMBIENT_VECTOR))


# ---------------------------------------------------------------------------------------------------
# geneexpr-v1
# ---------------------------------------------------------------------------------------------------
GE_HEADER = "gene_id\tgene_name\teffective_length\tcount\tTPM\tFPKM\n"


def gene_tables(ctx):
    """(genes [(id, name)], effective_length[int64], count[int64]) from the vocabulary and the matrices."""
    ex = ctx.matrix("exon_counts.1pz")
    it = ctx.matrix("intron_counts.1pz")
    genes = ctx.genes()
    ex_names = ctx.vocab_names("exon")
    gid, _, _ = rqc.split_row_names(ex_names, False)
    gix = {g: i for i, (g, _) in enumerate(genes)}
    eg = np.array([gix[g] for g in gid], dtype=np.int64)
    G = len(genes)
    pat = re.compile(r":(\d+)-(\d+)$")
    span = np.array([int(m.group(2)) - int(m.group(1)) for m in (pat.search(n) for n in ex_names)], dtype=np.int64)
    ln = np.bincount(eg, weights=np.maximum(span, 0), minlength=G).astype(np.int64)
    cnt = np.zeros(G, dtype=np.float64)
    cnt += np.bincount(eg, weights=np.bincount(ex.indices.astype(np.int64), weights=ex.data.astype(np.float64), minlength=ex.shape[0]), minlength=G)
    if it is not None and it.shape[0] > 0:
        igid, _, _ = rqc.split_row_names(it.rownames if it.rownames is not None else ctx.vocab_names("intron"), True)
        ig = np.array([gix[g] for g in igid], dtype=np.int64)
        cnt += np.bincount(ig, weights=np.bincount(it.indices.astype(np.int64), weights=it.data.astype(np.float64), minlength=it.shape[0]), minlength=G)
    return genes, np.maximum(ln, 1), cnt.astype(np.int64)


def render_geneexpr(genes, ln, cnt, total_mapped) -> bytes:
    lnf = ln.astype(np.float64)
    c = cnt.astype(np.float64)
    pos = cnt > 0
    rpk = np.where(pos, c / lnf, 0.0)
    s = float(np.cumsum(rpk)[-1]) if len(rpk) else 0.0
    tpm = rpk * (1.0e6 / s) if s > 0.0 else np.zeros(len(rpk))
    if total_mapped > 0:
        fpkm = np.where(pos, (c * 1.0e9) / (lnf * float(total_mapped)), 0.0)
    else:
        fpkm = np.zeros(len(rpk))
    out = [GE_HEADER]
    out.extend("%s\t%s\t%d\t%d\t%.6f\t%.6f\n" % (g, nm, l, k, t, f)
               for (g, nm), l, k, t, f in zip(genes, ln.tolist(), cnt.tolist(), tpm.tolist(), fpkm.tolist()))
    return "".join(out).encode("utf-8")


def _total_mapped_candidates(ctx, legacy_bytes, ln_cnt):
    """Implied normaliser: FPKM = count * 1e9 / (length * total) -> total.  Use the most expressed row."""
    tot = None
    pb = ctx.member_bytes("pileup_stats.json")
    if pb:
        try:
            tot = int(json.loads(pb).get("mapped_reads"))
        except Exception:
            tot = None
    return tot


def plan_geneexpr(member, legacy_bytes, ctx):
    try:
        genes, ln, cnt = gene_tables(ctx)
    except Exception:
        return None
    # implied total from the legacy FPKM column
    try:
        s = legacy_bytes.decode("utf-8")
        rows = [r.split("\t") for r in s[:-1].split("\n")[1:]]
        if len(rows) != len(genes):
            return None
        fp = np.array([float(r[5]) for r in rows])
    except Exception:
        return None
    cands = []
    t = _total_mapped_candidates(ctx, legacy_bytes, None)
    if t:
        cands.append(t)
    pos = np.flatnonzero((cnt > 0) & (fp > 0))
    if len(pos):
        # use the row with the largest FPKM (best relative precision of the 6 printed decimals)
        i = pos[np.argmax(fp[pos])]
        est = cnt[i] * 1e9 / (ln[i] * fp[i])
        base = int(round(est))
        cands.extend(base + d for d in sorted(range(-int(est * 1e-6) - 3, int(est * 1e-6) + 4), key=abs))
    elif (cnt == 0).all() or (fp == 0).all():
        cands.append(0)
    seen = set()
    for tm in cands:
        if tm in seen or tm < 0:
            continue
        seen.add(tm)
        if render_geneexpr(genes, ln, cnt, tm) == legacy_bytes:
            return ("geneexpr-v1", {"total_mapped": int(tm)}, ["pack:genes", "pack:exon", "matrix:exon_counts.1pz", "matrix:intron_counts.1pz"])
    return None


def regenerate_geneexpr(recipe_id, params, ctx) -> bytes:
    genes, ln, cnt = gene_tables(ctx)
    return render_geneexpr(genes, ln, cnt, params["total_mapped"])


# ---------------------------------------------------------------------------------------------------
# splice-events-v1
# ---------------------------------------------------------------------------------------------------
SE_HEADER = "event_id\tevent_type\tn_junctions\tjunction_names\n"


def render_splice_events(names) -> bytes:
    out = [SE_HEADER]
    for eid, typ, members in rpsi.build_events(names):
        out.append("%s\t%s\t%d\t%s\n" % (eid, typ, len(members), ",".join(names[j] for j in members.tolist())))
    return "".join(out).encode("utf-8")


def plan_splice_events(member, legacy_bytes, ctx):
    sj = ctx.matrix("sj_counts.1pz")
    if sj is None or sj.rownames is None:
        return None
    if render_splice_events(sj.rownames) != legacy_bytes:
        return None
    return ("splice-events-v1", {}, ["matrix:sj_counts.1pz"])


def regenerate_splice_events(recipe_id, params, ctx) -> bytes:
    return render_splice_events(ctx.matrix("sj_counts.1pz").rownames)


# ---------------------------------------------------------------------------------------------------
# vocab-json-v1
# ---------------------------------------------------------------------------------------------------
def render_vocab_json(build, schema, genes) -> bytes:
    return json.dumps({"schema_version": schema, "reference_build": build, "n_genes": len(genes),
                       "genes": [{"gene_id": a, "gene_name": b} for a, b in genes]}, indent=2).encode("utf-8")


def plan_vocab_json(member, legacy_bytes, ctx):
    if member != "feature_vocab.json" or ctx.sample_id:
        return None
    genes = ctx.genes()
    if genes is None:
        return None
    try:
        j = json.loads(legacy_bytes)
        build, schema = j["reference_build"], j["schema_version"]
    except Exception:
        return None
    if render_vocab_json(build, schema, genes) != legacy_bytes:
        return None
    params = {"build": build, "schema": schema}
    pk = ctx.pack_for("exon") if hasattr(ctx, "pack_for") else None
    if pk is not None:
        params["pack"] = pk.id.hex()          # informational: lets the manifest list the pack this file depends on
    return ("vocab-json-v1", params, ["pack:genes"])


def regenerate_vocab_json(recipe_id, params, ctx) -> bytes:
    genes = None
    if "pack" in params and hasattr(ctx, "packs"):
        pk = ctx.packs.get(params["pack"])
        if pk is not None:
            if "schema" not in params:                               # params written by W2's stand-in recipe
                return pk.vocab_json()
            genes = pk.gene_table()
    if genes is None:
        genes = ctx.genes()
    return render_vocab_json(params["build"], params["schema"], genes)


# ---------------------------------------------------------------------------------------------------
# stub-v1
# ---------------------------------------------------------------------------------------------------
def plan_stub(member, legacy_bytes, ctx):
    if legacy_bytes == pzw.STUB_1PZ:
        return ("stub-v1", {}, [])
    return None


def regenerate_stub(recipe_id, params, ctx) -> bytes:
    return pzw.STUB_1PZ
