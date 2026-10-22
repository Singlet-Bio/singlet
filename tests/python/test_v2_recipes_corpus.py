# SPDX-License-Identifier: MIT
"""Golden tests of the v2 migration recipes (W3) on the local compression-study corpora.

Every claim is "the recipe regenerates the legacy member byte for byte" (for derived ``.1pz`` members: the decoded canonical
matrix, and the container bytes whenever a writer with the right zstd build exists).  Tests skip when the corpus is absent.

    python -m pytest tests/python/test_v2_recipes_corpus.py            # a handful of samples per recipe (minutes)
    SINGLET_V2_FULL=1 python -m pytest tests/python/test_v2_recipes_corpus.py   # every sample (about an hour on the laptop)
    python tests/python/test_v2_recipes_corpus.py [--full] [--tsv out.tsv]   # the pass-rate table of notes/m_w3.md
    python tests/python/test_v2_recipes_corpus.py --fetch [dir ...]          # (re)fetch corpus_m_w3 by range request
    python tests/python/test_v2_recipes_corpus.py --psi                      # which zstd build reproduces each splice_psi.1pz

Sources: ``bench/compression/corpus/bundles/GSE145370.singlet`` (14 samples, gen A), ``corpus_w4_corpus/<id>`` (39 samples),
``corpus_w4_mtx/<GSE.GSM>`` (4 gen-B STARsolo samples) and ``corpus_m_w3/<id>`` (extra members W3 fetched by range request:
per-cell tables, splice_events, pileup_stats, ...).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from singlet.v2 import recipes as R  # noqa: E402
from singlet.v2 import recipes_ctx as CTX  # noqa: E402
from singlet.v2 import recipes_pzw as PW  # noqa: E402

CORPUS = ROOT / "bench" / "compression"
BUNDLE = CORPUS / "corpus" / "bundles" / "GSE145370.singlet"
W4C = CORPUS / "corpus_w4_corpus"
MTX = CORPUS / "corpus_w4_mtx"
EXTRA = CORPUS / "corpus_m_w3"
FULL = os.environ.get("SINGLET_V2_FULL") == "1"

MTX_SAMPLES = [
    "GSE171161.GSM5220137",
    "GSE309702.GSM9277757",
    "GSE263963.GSM8208009",
    "GSE311511.GSM9326201",
]
GSE145370_GSMS = ["GSM43174%02d" % i for i in range(23, 37)]
W4C_SAMPLES = [
    "L_a",
    "L_b",
    "bn_dropseq",
    "bn_v2",
    "bn_v3_a",
    "bn_v3_b",
    "h_arc",
    "h_bd",
    "h_celseq2",
    "h_dropseq",
    "h_indrop",
    "h_mars",
    "h_microwell",
    "h_quartz",
    "h_split",
    "h_v2_a",
    "h_v2_b",
    "h_v2_c",
    "h_v3_a",
    "h_v3_b",
    "h_v3_c",
    "h_visium",
    "m_arc",
    "m_celseq2",
    "m_dropseq",
    "m_mars",
    "m_v2_a",
    "m_v2_b",
    "m_v3_a",
    "m_v3_b",
    "m_v3_c",
    "m_visium",
    "mq_v2",
    "mq_v3",
    "r_v2",
    "r_v3",
    "x_stub",
    "z_v2",
    "z_v3_a",
    "z_v3_b",
]
STUB_PSI = {"L_b", "h_v2_c", "h_v3_b", "h_v3_c", "x_stub"}

# members tried by the migration flow, in an order that exercises the dependencies in both directions
FLOW_MEMBERS = [
    "splice_psi.1pz",
    "cell_calls.tsv",
    "cell_qc_metrics.tsv",
    "doublet_scores.tsv",
    "cell_cycle_scores.tsv",
    "ambient_contamination.tsv",
    "auto_barcodes.tsv",
    "donor/donor_assignments.tsv",
    "mt/mt_summary.tsv",
    "spatial_coordinates.tsv",
    "visium_qc.tsv",
    "star_Solo.out/Gene/raw/barcodes.tsv",
    "star_Solo.out/Gene/filtered/barcodes.tsv",
    "star_Solo.out/Gene/raw/features.tsv",
    "star_Solo.out/Gene/filtered/features.tsv",
    "star_Solo.out/Gene/filtered/matrix.mtx",
    "star_Solo.out/Gene/raw/matrix.mtx",
    "ambient_profile.tsv",
    "gene_expression.tsv",
    "splice_events.tsv",
    "mt_heteroplasmy.1pz",
    "vdj_gene_usage.1pz",
]


# ------------------------------------------------------------------------------------------------------------ sources
def have(path) -> bool:
    return Path(path).exists()


def ctx_gse145370(gsm):
    if not have(BUNDLE):
        pytest.skip("GSE145370.singlet not present")
    return CTX.DirContext(CTX.ZipSource(BUNDLE, "samples/%s/" % gsm), sample_id=gsm)


def ctx_w4c(name):
    d = W4C / name
    if not have(d):
        pytest.skip("corpus_w4_corpus/%s not present" % name)
    srcs = [d] + ([EXTRA / name] if have(EXTRA / name) else [])
    return CTX.DirContext(srcs, sample_id=name)


def ctx_mtx(name):
    d = MTX / name
    if not have(d):
        pytest.skip("corpus_w4_mtx/%s not present" % name)
    srcs = [d] + ([EXTRA / name] if have(EXTRA / name) else [])
    return CTX.DirContext(srcs, sample_id=name)


# ------------------------------------------------------------------------------------------------------------ the flow
class Result(dict):
    pass


def run_flow(ctx, members=None, record_cells=True):
    """Emulate the migration: candidate cells columns for every member, then plan + regenerate per member.

    Returns {member: Result(plan, ok, bytes_ok, matrix_ok, t_plan, t_regen, params_bytes, cols)}"""
    members = [m for m in (members or FLOW_MEMBERS) if ctx.member_bytes(m) is not None]
    cand = {}
    for m in members:
        cols = R.cells_columns(m, ctx.member_bytes(m), ctx)
        cand[m] = cols
        base = ctx.cells() or {}
        ctx.add_cells({k: v for k, v in cols.items() if k not in base})
    out = {}
    for m in members:
        lb = ctx.member_bytes(m)
        r = Result(
            member=m,
            size=len(lb),
            plan=None,
            ok=None,
            bytes_ok=None,
            matrix_ok=None,
            t_plan=0.0,
            t_regen=0.0,
            params_bytes=0,
            cols=sorted(cand[m]),
        )
        t = time.time()
        p = R.plan(m, lb, ctx)
        r["t_plan"] = time.time() - t
        r["plan"] = p
        if p is not None:
            r["params_bytes"] = len(json.dumps(p.params, separators=(",", ":")))
            t = time.time()
            if m.endswith(".1pz"):
                leg = CTX.decode_1pz_bytes(lb)
                got = R.regenerate_matrix(p.recipe_id, p.params, ctx)
                r["matrix_ok"] = bool(got.equals(leg))
                try:
                    r["bytes_ok"] = R.regenerate(p.recipe_id, p.params, ctx) == lb
                except R.NotByteExact:
                    r["bytes_ok"] = None
                r["ok"] = r["matrix_ok"]
            else:
                r["bytes_ok"] = R.regenerate(p.recipe_id, p.params, ctx) == lb
                r["ok"] = r["bytes_ok"]
            r["t_regen"] = time.time() - t
        out[m] = r
    return out


def assert_flow(res, expect_plan=(), expect_none=()):
    for m, r in res.items():
        if r["plan"] is not None:
            assert r["ok"], "%s: plan %s did not reproduce the member" % (m, r["plan"].recipe_id)
    for m in expect_plan:
        assert res[m]["plan"] is not None, "%s: expected a recipe (errors: %s)" % (
            m,
            R.LAST_ERRORS.get(m),
        )
    for m in expect_none:
        assert res[m]["plan"] is None


# ------------------------------------------------------------------------------------------------------------ gen A bundle
@pytest.mark.parametrize("gsm", GSE145370_GSMS if FULL else ["GSM4317429", "GSM4317432"])
def test_gse145370_psi_and_cellcalls(gsm):
    ctx = ctx_gse145370(gsm)
    res = run_flow(
        ctx, ["splice_psi.1pz", "cell_calls.tsv", "mt_heteroplasmy.1pz", "vdj_gene_usage.1pz"]
    )
    assert_flow(res, expect_plan=["splice_psi.1pz", "cell_calls.tsv", "mt_heteroplasmy.1pz"])
    p = res["splice_psi.1pz"]["plan"]
    assert p.recipe_id == "psi-v1-rint"  # pipeline 0.3.0 (git 8e1803b) stores nearbyint(PSI)
    assert res["splice_psi.1pz"]["matrix_ok"]
    if p.params["writer"] is not None:
        assert res["splice_psi.1pz"]["bytes_ok"]
    c = res["cell_calls.tsv"]["plan"]
    assert c.recipe_id == "cellcalls-v1" and c.params == {
        "method": "star_solo",
        "listed": "called",
        "order": "sorted",
        "pvalue": "zero",
    }
    assert res["mt_heteroplasmy.1pz"]["plan"].recipe_id == "stub-v1"  # the 0 x 0 mt stub
    assert res["mt_heteroplasmy.1pz"]["matrix_ok"] and res["mt_heteroplasmy.1pz"]["bytes_ok"]


def test_gse145370_psi_bytes_need_the_right_zstd():
    """The 14 shipped files were written with zstd 1.5.5/1.5.6; a 1.5.7 writer must not claim them."""
    ctx = ctx_gse145370("GSM4317429")
    lb = ctx.member_bytes("splice_psi.1pz")
    ws = PW.available_writers()
    matches = [
        w for w, wr in ws.items() if wr.write(R.regenerate_matrix("psi-v1-rint", {}, ctx)) == lb
    ]
    assert len(matches) <= 1
    p = R.plan("splice_psi.1pz", lb, ctx)
    assert p.params["writer"] == (matches[0] if matches else None)


# ------------------------------------------------------------------------------------------------------------ corpus_w4_corpus
DEFAULT_W4C = ["h_v3_a", "h_visium", "h_v3_b", "m_mars", "h_bd", "h_celseq2", "L_a", "x_stub"]


@pytest.mark.parametrize("name", W4C_SAMPLES if FULL else DEFAULT_W4C)
def test_w4c_psi_and_cellcalls(name):
    ctx = ctx_w4c(name)
    res = run_flow(ctx, ["splice_psi.1pz", "cell_calls.tsv", "mt_heteroplasmy.1pz"])
    assert_flow(res)
    psi = res["splice_psi.1pz"]
    assert psi["plan"] is not None
    if name in STUB_PSI:
        assert psi["plan"].recipe_id == "psi-v1-stub"
    else:
        assert psi["plan"].recipe_id in ("psi-v1-rint", "psi-v1-trunc")
        assert psi["matrix_ok"]
    cc = res["cell_calls.tsv"]
    if name == "x_stub":
        assert (
            cc["plan"] is None
        )  # barcodes of cell_calls.tsv are not matrix columns (empty matrices)
    else:
        assert cc["plan"] is not None and cc["plan"].recipe_id == "cellcalls-v1"
        method = cc["plan"].params["method"]
        assert method in ("star_solo", "emptydrops", "top_n_fallback")
        assert (name in ("h_bd", "h_indrop", "h_microwell", "m_dropseq")) == (
            method == "emptydrops"
        )
        assert (method == "top_n_fallback") == (cc["plan"].params["order"] == "explicit")


def test_w4c_cellcalls_variants_have_the_expected_columns():
    ctx = ctx_w4c("h_bd")
    cols = R.cells_columns("cell_calls.tsv", ctx.member_bytes("cell_calls.tsv"), ctx)
    assert {"cc_listed", "cc_pvalue"} <= set(cols)
    ctx = ctx_w4c("m_mars")
    cols = R.cells_columns("cell_calls.tsv", ctx.member_bytes("cell_calls.tsv"), ctx)
    assert "cc_pos" in cols and "cc_listed" not in cols


# ------------------------------------------------------------------------------------------------------------ per-cell tables (gen B)
DEFAULT_CELLS = ["m_mars", "m_celseq2", "h_visium"]
ALL_CELLS = ["m_mars", "m_celseq2", "h_visium", "m_visium", "m_v3_a", "m_dropseq", "h_microwell"]


@pytest.mark.parametrize("name", ALL_CELLS if FULL else DEFAULT_CELLS)
def test_per_cell_tables_non_10x_and_visium(name):
    ctx = ctx_w4c(name)
    if not have(EXTRA / name):
        pytest.skip("corpus_m_w3/%s not present" % name)
    members = [
        m
        for m in FLOW_MEMBERS
        if m.endswith(".tsv")
        and m not in ("gene_expression.tsv", "splice_events.tsv", "ambient_profile.tsv")
        and not m.startswith("star_Solo")
    ]
    res = run_flow(ctx, members + ["cell_calls.tsv"])
    assert_flow(
        res,
        expect_plan=[
            "cell_calls.tsv",
            "cell_qc_metrics.tsv",
            "doublet_scores.tsv",
            "auto_barcodes.tsv",
        ],
    )


@pytest.mark.parametrize("name", MTX_SAMPLES if FULL else MTX_SAMPLES[:1])
def test_gen_b_starsolo_sample(name):
    ctx = ctx_mtx(name)
    res = run_flow(ctx)
    assert_flow(
        res,
        expect_plan=[
            "splice_psi.1pz",
            "cell_calls.tsv",
            "cell_qc_metrics.tsv",
            "doublet_scores.tsv",
            "cell_cycle_scores.tsv",
            "ambient_contamination.tsv",
            "auto_barcodes.tsv",
            "star_Solo.out/Gene/raw/barcodes.tsv",
            "star_Solo.out/Gene/filtered/barcodes.tsv",
            "star_Solo.out/Gene/raw/features.tsv",
            "star_Solo.out/Gene/filtered/features.tsv",
            "star_Solo.out/Gene/raw/matrix.mtx",
            "star_Solo.out/Gene/filtered/matrix.mtx",
            "ambient_profile.tsv",
            "gene_expression.tsv",
            "splice_events.tsv",
        ],
    )
    ids = {m: r["plan"].recipe_id for m, r in res.items() if r["plan"] is not None}
    assert (
        ids["star_Solo.out/Gene/raw/matrix.mtx"] == "mtx-text-v1"
        and ids["star_Solo.out/Gene/filtered/matrix.mtx"] == "mtx-filter-v1"
    )
    assert (
        ids["star_Solo.out/Gene/raw/features.tsv"] == "features-v1"
        and ids["gene_expression.tsv"] == "geneexpr-v1"
    )
    # the mouse features order comes from the permutation resource, the human one from the positional rule
    f = res["star_Solo.out/Gene/raw/features.tsv"]["plan"].params
    assert f["order"] == ("pos" if ctx.reference_build.startswith("GRCh38") else "perm")
    # the ambient vector is the only extra information of ambient_profile.tsv (2 + N bytes of params/vector, not 22 MB of text)
    assert ctx.vector("ambient_k") is not None and ctx.vector("ambient_k").size == len(
        ctx.vocab_names("exon")
    )


def test_auxiliary_matrix_is_registered_once():
    """The filtered matrix is planned before the raw one (alphabetical member order): the aux matrix must still exist for both."""
    ctx = ctx_mtx(MTX_SAMPLES[0])
    members = [
        "cell_calls.tsv",
        "star_Solo.out/Gene/raw/barcodes.tsv",
        "star_Solo.out/Gene/filtered/matrix.mtx",
        "star_Solo.out/Gene/raw/matrix.mtx",
    ]
    res = run_flow(ctx, members)
    assert_flow(res, expect_plan=members)
    assert ctx.matrix("star_Solo.out/Gene/raw/matrix.mtx#csc") is not None


# ------------------------------------------------------------------------------------------------------------ study level
def test_vocab_json_study_level_gen_a():
    if not have(BUNDLE):
        pytest.skip("GSE145370.singlet not present")
    ctx = CTX.DirContext(
        [CTX.ZipSource(BUNDLE, "samples/GSM4317429/"), CTX.ZipSource(BUNDLE, "")], sample_id="x"
    )
    ctx.sample_id = ""  # study-level context: the vocabulary of the build
    lb = ctx.member_bytes("feature_vocab.json")
    p = R.plan("feature_vocab.json", lb, ctx)
    assert p is not None and p.recipe_id == "vocab-json-v1" and p.params["build"] == "GRCh38-2024-A"
    assert R.regenerate(p.recipe_id, p.params, ctx) == lb


@pytest.mark.parametrize("name", MTX_SAMPLES)
def test_vocab_json_study_level_gen_b(name):
    gse = name.split(".")[0]
    vj = MTX / gse / "feature_vocab.json"
    if not have(vj) or not have(MTX / name):
        pytest.skip("corpus_w4_mtx/%s not present" % gse)
    ctx = CTX.DirContext([MTX / name], sample_id="x")
    ctx.sample_id = ""
    lb = vj.read_bytes()
    p = R.plan("feature_vocab.json", lb, ctx)
    assert p is not None and p.recipe_id == "vocab-json-v1"
    assert R.regenerate(p.recipe_id, p.params, ctx) == lb


# ------------------------------------------------------------------------------------------------------------ table (script mode)
def sweep(full=False, tsv=None):
    rows = []
    targets = []
    for g in GSE145370_GSMS if full else GSE145370_GSMS[:2]:
        targets.append(
            (
                "GSE145370/" + g,
                lambda g=g: ctx_gse145370(g),
                ["splice_psi.1pz", "cell_calls.tsv", "mt_heteroplasmy.1pz", "vdj_gene_usage.1pz"],
            )
        )
    for n in W4C_SAMPLES if full else DEFAULT_W4C:
        targets.append(
            (
                n,
                lambda n=n: ctx_w4c(n),
                None
                if n in ALL_CELLS
                else [
                    "splice_psi.1pz",
                    "cell_calls.tsv",
                    "mt_heteroplasmy.1pz",
                    "vdj_gene_usage.1pz",
                ],
            )
        )
    for n in MTX_SAMPLES if full else MTX_SAMPLES[:1]:
        targets.append((n, lambda n=n: ctx_mtx(n), None))
    for name, mk, members in targets:
        try:
            ctx = mk()
        except pytest.skip.Exception:
            continue
        res = run_flow(ctx, members)
        for m, r in res.items():
            rows.append(
                (
                    name,
                    m,
                    r["size"],
                    r["plan"].recipe_id if r["plan"] else "-",
                    r["ok"],
                    r["bytes_ok"],
                    r["params_bytes"],
                    round(r["t_plan"], 2),
                    round(r["t_regen"], 2),
                    ",".join(r["cols"]),
                )
            )
            print("\t".join(str(x) for x in rows[-1]), flush=True)
    if tsv:
        with open(tsv, "w") as f:
            f.write(
                "sample\tmember\tbytes\trecipe\tok\tbytes_ok\tparams_bytes\tt_plan\tt_regen\tcells_columns\n"
            )
            for r in rows:
                f.write("\t".join(str(x) for x in r) + "\n")
    return rows


# ------------------------------------------------------------------------------------------------------------ psi writer table
def psi_table(limit=None):
    """Which writer (zstd build) reproduces each splice_psi.1pz, and whether rint and trunc are distinguishable (notes/m_w3.md section 5)."""
    from singlet.v2 import recipes_psi as RP

    ws = PW.available_writers()
    print("writers:", {k: v.label for k, v in ws.items()})
    names = [("GSE145370/" + g, lambda g=g: ctx_gse145370(g)) for g in GSE145370_GSMS]
    names += [(n, lambda n=n: ctx_w4c(n)) for n in W4C_SAMPLES] + [
        (n, lambda n=n: ctx_mtx(n)) for n in MTX_SAMPLES
    ]
    for name, mk in names[:limit]:
        try:
            ctx = mk()
        except pytest.skip.Exception:
            continue
        lb = ctx.member_bytes("splice_psi.1pz")
        if lb is None:
            continue
        p = R.plan("splice_psi.1pz", lb, ctx)
        if p is None or p.recipe_id == "psi-v1-stub":
            print(name, len(lb), p.recipe_id if p else "NO PLAN")
            continue
        sj = ctx.matrix("sj_counts.1pz")
        _, _, psi, _ = RP.compute_psi(sj)
        both = bool(np.array_equal(RP.quantise(psi, "rint"), RP.quantise(psi, "trunc")))
        m = RP.psi_matrix(sj, p.recipe_id.rsplit("-", 1)[1])
        match = [
            w
            for w, wr in ws.items()
            if wr.write(m, p.params["level"], p.params["chunk_cols"]) == lb
        ]
        print(
            name,
            len(lb),
            p.recipe_id,
            "writer",
            p.params["writer"],
            "matches",
            match,
            "rint==trunc" if both else "",
        )


# ------------------------------------------------------------------------------------------------------------ fetching corpus_m_w3
FETCH_SAMPLES = {  # corpus_m_w3 dir -> (GSE, GSM, also the large per-cell tables)
    "GSE309702.GSM9277757": ("GSE309702", "GSM9277757", True),
    "GSE171161.GSM5220137": ("GSE171161", "GSM5220137", True),
    "GSE263963.GSM8208009": ("GSE263963", "GSM8208009", True),
    "GSE311511.GSM9326201": ("GSE311511", "GSM9326201", True),
    "m_dropseq": ("GSE249398", "GSM7945758", False),
    "h_microwell": ("GSE134355", "GSM4008717", True),
    "m_mars": ("GSE222410", "GSM6922987", True),
    "m_celseq2": ("GSE207275", "GSM6283570", True),
    "h_visium": ("GSE233208", "GSM7412940", True),
    "m_visium": ("GSE218360", "GSM6742901", True),
    "m_v3_a": ("GSE241928", "GSM7745727", True),
}
FETCH_MEMBERS = [
    "auto_barcodes.tsv",
    "auto_barcodes_seg1.tsv",
    "auto_barcodes_seg2.tsv",
    "cell_cycle_scores.tsv",
    "cell_qc_metrics.tsv",
    "doublet_scores.tsv",
    "ambient_contamination.tsv",
    "ambient_contamination.json",
    "metrics_summary.csv",
    "gene_expression.tsv",
    "splice_events.tsv",
    "read_stats.tsv",
    "saturation_curve.tsv",
    "pileup_stats.json",
    "summary.json",
    "provenance.json",
    "spatial_coordinates.tsv",
    "visium_qc.tsv",
    "nonhost/nonhost_per_cell.tsv",
    "mt_variants.tsv",
    "donor_assignments.tsv",
    "sj_counts.1pz",
    "splice_psi.1pz",
    "mt_heteroplasmy.1pz",
    "star_Log.final.out",
    "star_Log.progress.out",
    "star_Solo.out/Barcodes.stats",
    "star_Solo.out/Gene/Summary.csv",
    "star_Solo.out/Gene/Features.stats",
]
FETCH_BIG = ["donor/donor_assignments.tsv", "mt/mt_summary.tsv"]


def fetch_extra(only=None):
    """Range-fetch the members the corpora lack (public GETs of data.singlet.bio, ~190 MB) into corpus_m_w3/<id>/ (``/`` written ``__``)."""
    import io
    import zipfile

    sys.path.insert(0, str(CORPUS / "tools"))
    import fetch_corpus as F

    for dirname, (gse, gsm, big) in FETCH_SAMPLES.items():
        if only and dirname not in only:
            continue
        hf = F.HttpFile(F.BASE.format(gse=gse))
        zf = zipfile.ZipFile(io.BufferedReader(hf, buffer_size=1 << 20))
        d = EXTRA / dirname
        d.mkdir(parents=True, exist_ok=True)
        for w in FETCH_MEMBERS + (FETCH_BIG if big else []):
            name = "samples/%s/%s" % (gsm, w)
            try:
                zi = zf.getinfo(name)
            except KeyError:
                continue
            dst = d / w.replace("/", "__")
            if dst.exists() and dst.stat().st_size == zi.file_size:
                continue
            dst.write_bytes(zf.read(name))
        print(dirname, "http MB", round(hf.bytes_fetched / 1e6, 1), flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--fetch" in a:
        fetch_extra([x for x in a if not x.startswith("--")] or None)
    elif "--psi" in a:
        psi_table()
    else:
        sweep(full="--full" in a, tsv=a[a.index("--tsv") + 1] if "--tsv" in a else None)
