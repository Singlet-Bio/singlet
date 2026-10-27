# SPDX-License-Identifier: MIT
"""Migration + verification tests for singlet.s1 (W2).

Layers:
  * synthetic bundles written with the legacy writer (no corpus needed): names modes, cells table, derived members through a fake
    recipes module (including a recipe that lies and one that cheats), auxiliary members, determinism, tampering;
  * corpus tests on bench/compression/corpus/bundles/GSE145370.singlet (a few samples by default, all 14 with SINGLET_S1_FULL=1) and the loose
    sample directories of corpus_w4_corpus / corpus_w4_mtx (a few by default, all with SINGLET_S1_FULL=1).
"""

from __future__ import annotations

import json
import os
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from singlet.s1 import cells as cells_mod  # noqa: E402
from singlet.s1 import container as C  # noqa: E402
from singlet.s1 import migrate as MG  # noqa: E402
from singlet.s1 import packs as P  # noqa: E402
from singlet.s1 import util  # noqa: E402
from singlet.s1 import verify as V  # noqa: E402
from singlet.s1.sample import S1Sample  # noqa: E402

# the legacy TP1Z writer is a compiled extension: skip the module (rather than fail collection) where it is not built
_pz = pytest.importorskip("singlet._pz", reason="singlet._pz is not built")

CORPUS = ROOT / "bench" / "compression"
BUNDLE = CORPUS / "corpus" / "bundles" / "GSE145370.singlet"
FULL = os.environ.get("SINGLET_S1_FULL") == "1"


# ------------------------------------------------------------------------------------------------------------ synthetic bundle
def write_1pz(path, m, n, density, seed, rownames, colnames, vt_dtype=np.uint16, kv=None):
    rs = np.random.RandomState(seed)
    cols = []
    for j in range(n):
        k = 0 if j % 11 == 5 else rs.binomial(m, density)
        cols.append(np.sort(rs.choice(m, size=k, replace=False)).astype(np.int32))
    indptr = np.concatenate([[0], np.cumsum([len(c) for c in cols])]).astype(np.int32)
    indices = np.concatenate(cols).astype(np.int32)
    data = np.minimum(
        rs.geometric(0.4, size=len(indices)).astype(np.int64), np.iinfo(vt_dtype).max
    ).astype(vt_dtype)
    assert _pz.write_1pz(
        str(path),
        indptr,
        indices,
        data,
        m,
        n,
        rownames=rownames,
        colnames=colnames,
        zstd_level=3,
        chunk_cols=64,
        threads=1,
        user_meta=kv,
    )
    return indptr, indices, data.astype(np.uint32)


def whitelist_barcodes(count, seed):
    store = P.default_store()
    w = store.whitelist_by_name("737K-august-2016")
    rs = np.random.RandomState(seed)
    return w.barcodes(np.sort(rs.choice(w.n, size=count, replace=False)))


def make_bundle(tmp_path, gse="GSE9001", gsms=("GSM9001", "GSM9002"), real_barcodes=True) -> Path:
    bundle = tmp_path / ("%s.singlet" % gse)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    with zipfile.ZipFile(bundle, "w", allowZip64=True) as z:
        z.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "gse_id": gse,
                    "gsm_ids": list(gsms),
                    "reference_build": "toy",
                    "created_at": "2026-01-01T00:00:00+00:00",
                },
                indent=2,
            ),
        )
        z.writestr(
            "study_meta.json", json.dumps({"gse_id": gse, "series_title": "toy é"}, indent=2)
        )
        z.writestr(
            "feature_vocab.json",
            json.dumps(
                {"schema_version": "1.0", "reference_build": "toy", "n_genes": 0, "genes": []},
                indent=2,
            ),
        )
        for si, g in enumerate(gsms):
            n = 140 + 10 * si
            bcs = whitelist_barcodes(n, si) if real_barcodes else ["CELL%04d" % i for i in range(n)]
            rows_e = ["G%03d_gene%d_chr1:%d-%d" % (i, i, 100 * i, 100 * i + 50) for i in range(240)]
            rows_i = [
                "G%03d_gene%d_intron_chr1:%d-%d" % (i, i, 100 * i + 50, 100 * i + 90)
                for i in range(200)
            ]
            rows_j = ["chr1:%d-%d" % (100 * i, 100 * i + 40 + (i % 3)) for i in range(300)]
            p = work / "m.1pz"
            ip, ix, dt = write_1pz(p, 240, n, 0.08, si, rows_e, bcs)
            z.write(p, "samples/%s/exon_counts.1pz" % g, zipfile.ZIP_STORED)
            write_1pz(p, 200, n, 0.06, si + 10, rows_i, bcs)
            z.write(p, "samples/%s/intron_counts.1pz" % g, zipfile.ZIP_STORED)
            write_1pz(p, 300, n, 0.05, si + 20, rows_j, bcs, vt_dtype=np.uint32)
            z.write(p, "samples/%s/sj_counts.1pz" % g, zipfile.ZIP_STORED)
            # splice_psi: a function of sj_counts (the fake recipe computes data // 2 + 1, clipped to uint8) - lies in sample 2
            sj = _pz.read_1pz(str(p))
            psi = (np.asarray(sj["data"]).astype(np.uint32) // 2 + 1).clip(0, 255).astype(np.uint8)
            if si == 1:
                psi[0] += 1  # the recipe will not reproduce this one
            assert _pz.write_1pz(
                str(p),
                np.asarray(sj["indptr"], np.int32),
                np.asarray(sj["indices"], np.int32),
                psi,
                300,
                n,
                rownames=rows_j,
                colnames=bcs,
                zstd_level=3,
                chunk_cols=64,
                threads=1,
            )
            z.write(p, "samples/%s/splice_psi.1pz" % g, zipfile.ZIP_STORED)
            # a matrix with different columns (inline column names) and key/values
            write_1pz(
                p,
                20,
                30,
                0.3,
                99,
                ["tcr%d (V)" % i for i in range(20)],
                ["OTHER%d" % i for i in range(30)],
                kv={"k2": "v2", "k1": "v1"},
            )
            z.write(p, "samples/%s/vdj_gene_usage.1pz" % g, zipfile.ZIP_STORED)
            z.writestr(
                "samples/%s/summary.json" % g,
                json.dumps(
                    {"protocol_name": "10xv2", "git_sha": "abc", "reference_build": "toy"}, indent=2
                ),
            )
            called = bcs[: n // 3]
            lines = ["barcode\tis_cell\tlog10_umi\tempty_drops_pvalue\tcall_method"]
            for b in called:
                lines.append("%s\tTRUE\t3.1000\t0.0000e+00\tstar_solo" % b)
            z.writestr("samples/%s/cell_calls.tsv" % g, "\n".join(lines) + "\n")
            if si == 0:
                z.writestr(
                    "samples/%s/ambient_profile.tsv" % g,
                    "".join("g%d\t%.6g\n" % (i, (i % 5 + 0.5) / 100) for i in range(50)),
                )
                z.writestr("samples/%s/star_Solo.out/Gene/Summary.csv" % g, "a,b\n1,2\n")
            z.writestr("samples/%s/log.txt" % g, ("line %d\n" % si) * 100)
    return bundle


class FakeRecipes:
    """A stand-in for singlet.s1.recipes exercising every path of the migration."""

    RECIPES = {
        "toy-psi-v1": 1,
        "toy-cc-v1": 1,
        "toy-ambient-v1": 1,
        "toy-cheat-v1": 1,
        "toy-liar-v1": 1,
    }

    # ---- plan / regenerate
    def plan(self, member, legacy_bytes, ctx):
        if member == "splice_psi.1pz":
            return ("toy-psi-v1", {}, ["matrix:sj_counts.1pz"])
        if member == "cell_calls.tsv":
            cells = ctx.cells()
            if cells is None:
                return None
            return ("toy-cc-v1", {}, ["cells:barcode", "cells:called", "cells:cc_log"])
        if member == "ambient_profile.tsv":
            k = np.arange(50, dtype=np.uint32) % 5
            ctx.provide_vector("ambient_k", k)
            return ("toy-ambient-v1", {"n": 50}, ["vector:ambient_k"])
        if member == "log.txt" and ctx.sample_id == "GSM9001":
            return (
                "toy-cheat-v1",
                {},
                [],
            )  # reads legacy-only state at regenerate time -> must be demoted by stage 2
        if member == "star_Solo.out/Gene/Summary.csv":
            return ("toy-liar-v1", {}, [])
        return None

    def _psi(self, ctx):
        from singlet.s1.recipes_ctx import Matrix

        sj = ctx.matrix("sj_counts.1pz")
        data = (np.asarray(sj.data).astype(np.uint32) // 2 + 1).clip(0, 255)
        return Matrix(
            np.asarray(sj.indptr),
            np.asarray(sj.indices),
            data,
            sj.shape,
            sj.rownames,
            sj.colnames,
            {},
        )

    def regenerate_matrix(self, recipe_id, params, ctx):
        if recipe_id == "toy-psi-v1":
            return self._psi(ctx)
        raise LookupError(recipe_id)

    def regenerate(self, recipe_id, params, ctx):
        if recipe_id == "toy-cc-v1":
            c = ctx.cells()
            lines = ["barcode\tis_cell\tlog10_umi\tempty_drops_pvalue\tcall_method"]
            for b, called in zip(c["barcode"], c["called"]):
                if called:
                    lines.append("%s\tTRUE\t3.1000\t0.0000e+00\tstar_solo" % b)
            return ("\n".join(lines) + "\n").encode()
        if recipe_id == "toy-ambient-v1":
            k = ctx.vector("ambient_k")
            return "".join(
                "g%d\t%.6g\n" % (i, (int(v) + 0.5) / 100) for i, v in enumerate(k)
            ).encode()
        if recipe_id == "toy-cheat-v1":
            return ctx.bundle.read(
                "samples/GSM9001/log.txt"
            )  # only the migration context has .bundle
        if recipe_id == "toy-liar-v1":
            return b"a,b\n1,3\n"
        raise LookupError(recipe_id)

    def cells_columns(self, member, legacy_bytes, ctx):
        if member == "cell_calls.tsv":
            n = len(ctx.cells()["barcode"])
            return {
                "cc_log": (np.float32, np.full(n, 3.1, dtype=np.float32)),
                "cc_unused": ("u8", np.zeros(n, np.uint8)),
            }
        return {}


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("synth")
    bundle = make_bundle(tmp)
    out = tmp / "out"
    opts = MG.Options(recipes=FakeRecipes(), created_at="2026-10-22T00:00:00Z", log=lambda *a: None)
    res = MG.migrate_bundle(str(bundle), str(out), None, opts)
    wr = MG.write_outputs(res, opts)
    return types.SimpleNamespace(tmp=tmp, bundle=bundle, out=out, res=res, wr=wr, opts=opts)


def ledger_of(out, gse="GSE9001"):
    return [
        json.loads(ln) for ln in open(out / "s1" / "ledger" / (gse + ".jsonl"), encoding="utf-8")
    ]


def by_member(lines, gsm):
    return {
        rec["name"]: rec
        for rec in lines
        if rec.get("gsm") == gsm and rec["type"] in ("member", "sample_member", "v2member")
    }


def test_synthetic_migration_verifies(synth):
    rep = V.verify(
        str(synth.bundle), str(synth.out), None, None, FakeRecipes(), log=lambda *a: None
    )
    assert rep.fail == 0, rep.items
    assert rep.ok > 100
    assert rep.by_kind["matrices"][0] >= 2 * 4 * 7 and rep.by_kind["derived"][0] >= 3


def test_synthetic_members_and_treatments(synth):
    lines = ledger_of(synth.out)
    m1, m2 = by_member(lines, "GSM9001"), by_member(lines, "GSM9002")
    # matrices are pz2, with packs/cells references where they apply
    for nm in ("exon_counts.1pz", "intron_counts.1pz", "sj_counts.1pz", "vdj_gene_usage.1pz"):
        assert m1[nm]["kind"] == "pz2" and m1[nm]["verify_mode"] == "canonical"
    # splice_psi: derived in sample 1, stored (as pz2 matrix) in sample 2 where the recipe does not reproduce it
    assert (
        m1["splice_psi.1pz"]["treatment"] == "recipe"
        and m1["splice_psi.1pz"]["recipe"] == "toy-psi-v1"
    )
    assert (
        m2["splice_psi.1pz"]["treatment"] == "matrix" and "rejected" in m2["splice_psi.1pz"]["note"]
    )
    # cell_calls derived from the cells table; cells table has the columns it reads and not the unused one
    assert (
        m1["cell_calls.tsv"]["recipe"] == "toy-cc-v1"
        and m2["cell_calls.tsv"]["recipe"] == "toy-cc-v1"
    )
    s = S1Sample(open(synth.out / synth.res["samples"][0]["key"], "rb").read())
    cells = s.cells()
    assert "cc_log" in cells.names and "cc_unused" not in cells.names
    assert cells.names[:6] == list(cells_mod.REQUIRED)
    # aux vector registered by a verified plan, stored as a non-legacy member
    assert (
        m1["ambient_profile.tsv"]["recipe"] == "toy-ambient-v1"
        and "vector:ambient_k" in m1
        and not m1["vector:ambient_k"].get("legacy", True)
    )
    assert s.reader.members["vector:ambient_k"].flags & C.FLAG_LEGACY == 0
    # the cheating recipe passed stage 1 and was demoted by stage 2; the liar never passed stage 1
    assert m1["log.txt"]["treatment"] in ("stored", "raw") and "stage-2" in m1["log.txt"]["note"]
    assert (
        m1["star_Solo.out/Gene/Summary.csv"]["treatment"] in ("stored", "raw")
        and "rejected" in m1["star_Solo.out/Gene/Summary.csv"]["note"]
    )
    assert any(
        rec["type"] == "sample" and rec["stage2_rounds"] >= 2
        for rec in lines
        if rec.get("gsm") == "GSM9001"
    )
    # derived members have no body
    for nm, mi in s.reader.members.items():
        if mi.kind == C.KIND_DERIVED:
            assert mi.len == 0 and mi.meta["recipe"]


def test_synthetic_names_modes(synth):
    s = S1Sample(open(synth.out / synth.res["samples"][0]["key"], "rb").read())
    ex = s.reader.members["exon_counts.1pz"].meta["names"]
    assert ex["cols"] == "cells" and ex["rows"] == "inline"  # toy build: no vocabulary pack applies
    assert (
        s.reader.members["vdj_gene_usage.1pz"].meta["names"]["cols"] == "inline"
    )  # other barcodes
    t = s.cells()
    assert s.reader.members["cells"].meta["whitelist"]  # real 10x barcodes -> whitelist indices
    assert t.barcodes == s.matrix("exon_counts.1pz").colnames
    assert s.matrix("vdj_gene_usage.1pz").kv == {"k1": "v1", "k2": "v2"}
    assert 0 < int(t.column("called").sum()) < t.n_rows
    # every matrix of the sample decodes identically with column-range access
    sub = s.matrix("sj_counts.1pz", cols=(10, 70))
    full = s.matrix("sj_counts.1pz")
    assert (sub.to_scipy() != full.to_scipy()[:, 10:70]).nnz == 0 and sub.colnames == full.colnames[
        10:70
    ]


def test_called_ed_columns_and_parameters(synth, tmp_path):
    from singlet.s1 import cellcall

    man = json.load(open(synth.wr["manifest"], encoding="utf-8"))
    for s in man["samples"]:
        cells = S1Sample((synth.out / s["key"]).read_bytes()).cells()
        meta = s["members"]["cells"]
        if not cellcall.available():  # library default: computed only where numba is installed
            assert "called_ed" not in cells.names and "called_ed" not in meta
            continue
        assert {"called_ed", "called_ed_fdr"} <= set(cells.names)
        cv2, called = np.asarray(cells.column("called_ed")), np.asarray(cells.column("called"))
        assert (cv2[called]).all()
        p = meta["called_ed"]
        assert p["method"] in ("singlet-emptydrops", "stored-call") and p["n_called_ed"] == int(
            cv2.sum()
        )
    # switched off: no columns, no parameters
    opts = MG.Options(
        recipes="none", created_at="2026-10-22T00:00:00Z", log=lambda *a: None, cell_call_ed=False
    )
    res = MG.migrate_bundle(str(synth.bundle), str(tmp_path / "off"), None, opts)
    for s in res["samples"]:
        cells = S1Sample((tmp_path / "off" / s["key"]).read_bytes()).cells()
        assert "called_ed" not in cells.names and "called_ed" not in s["members"]["cells"]


def test_synthetic_manifest_and_outputs(synth):
    man = json.load(open(synth.wr["manifest"], encoding="utf-8"))
    assert man["format"] == "singlet-s1" and man["gse"] == "GSE9001" and len(man["samples"]) == 2
    assert set(man["study_files"]) == {"manifest.json", "study_meta.json", "feature_vocab.json"}
    assert man["rev"] == synth.wr["rev"] and len(man["rev"]) == 16
    for s in man["samples"]:
        obj = (synth.out / s["key"]).read_bytes()
        assert len(obj) == s["bytes"] and s["key"].endswith(util.d16(obj).hex() + ".s1")
        ex = s["members"]["exon_counts.1pz"]
        assert (
            ex["kind"] == 1
            and ex["chunk_cols"] in (128, 256, 512, 1024, 2048, 4096)
            and ex["nnz"] > 0
            and ex["chunk_index"][0] + ex["chunk_index"][1] <= len(obj)
        )
        assert s["n_called"] < s["n_columns"] and s["stats"]["median_umi"] > 0
    # packs referenced are shipped next to the objects with an index
    ref = synth.out / "s1" / "ref"
    idx = json.load(open(ref / "index.json"))
    assert idx["packs"] and all((ref / (e["id"] + ".pack")).exists() for e in idx["packs"])


def test_synthetic_is_deterministic(synth, tmp_path):
    out2 = tmp_path / "out2"
    opts = MG.Options(recipes=FakeRecipes(), created_at="2026-10-22T00:00:00Z", log=lambda *a: None)
    res = MG.migrate_bundle(str(synth.bundle), str(out2), None, opts)
    wr = MG.write_outputs(res, opts)
    assert wr["rev"] == synth.wr["rev"]
    for s in synth.res["samples"]:
        assert (out2 / s["key"]).read_bytes() == (synth.out / s["key"]).read_bytes()
    assert (out2 / "s1" / "m" / "GSE9001.json").read_bytes() == (
        synth.out / "s1" / "m" / "GSE9001.json"
    ).read_bytes()


def test_without_recipes_everything_is_stored_and_still_verifies(synth, tmp_path):
    opts = MG.Options(recipes="none", created_at="2026-10-22T00:00:00Z", log=lambda *a: None)
    res = MG.migrate_bundle(str(synth.bundle), str(tmp_path), None, opts)
    MG.write_outputs(res, opts)
    lines = ledger_of(tmp_path)
    assert not any(rec.get("treatment") == "recipe" for rec in lines if rec["type"] == "member")
    rep = V.verify(str(synth.bundle), str(tmp_path), None, None, "none", log=lambda *a: None)
    assert rep.fail == 0, rep.items


def _copy_tree(src: Path, dst: Path):
    import shutil

    shutil.copytree(src, dst)


def _tamper(synth, tmp_path, fn):
    out = tmp_path / "t"
    _copy_tree(synth.out, out)
    man = json.load(open(out / "s1" / "m" / "GSE9001.json", encoding="utf-8"))
    return out, man, fn


def _verify_fail(synth, out, **kw):
    rep = V.verify(
        str(synth.bundle),
        str(out),
        None,
        None,
        FakeRecipes(),
        log=lambda *a: None,
        quick=kw.get("quick", True),
    )
    return rep


def test_verify_detects_flipped_matrix_byte(synth, tmp_path):
    out, man, _ = _tamper(synth, tmp_path, None)
    s = man["samples"][0]
    p = out / s["key"]
    b = bytearray(p.read_bytes())
    ex = s["members"]["exon_counts.1pz"]
    b[ex["off"] + ex["len"] - 40] ^= 0x01
    p.write_bytes(bytes(b))
    rep = _verify_fail(synth, out)
    assert rep.fail > 0
    assert (
        V.main(["--bundle", str(synth.bundle), "--store", str(out), "--recipes", "none", "--quick"])
        == 1
    )


def test_verify_detects_changed_stored_member(synth, tmp_path):
    out, man, _ = _tamper(synth, tmp_path, None)
    s = man["samples"][1]
    p = out / s["key"]
    b = bytearray(p.read_bytes())
    mm = s["members"]["summary.json"]
    # replace the stored frame by another valid frame of the same length is hard; flip a payload byte instead
    b[mm["off"] + 10] ^= 0x01
    p.write_bytes(bytes(b))
    assert _verify_fail(synth, out).fail > 0


def test_verify_detects_missing_object_and_bad_rev(synth, tmp_path):
    out, man, _ = _tamper(synth, tmp_path, None)
    (out / man["samples"][0]["key"]).unlink()
    assert _verify_fail(synth, out).fail > 0
    out2 = tmp_path / "t2"
    _copy_tree(synth.out, out2)
    mp = out2 / "s1" / "m" / "GSE9001.json"
    m = json.load(open(mp, encoding="utf-8"))
    m["rev"] = "0" * 16
    json.dump(m, open(mp, "w", encoding="utf-8"))
    assert _verify_fail(synth, out2).fail > 0


def test_verify_detects_legacy_change(synth, tmp_path):
    # the legacy bundle gains a member the .s1 output does not know -> reported
    b2 = tmp_path / "GSE9001.singlet"
    with zipfile.ZipFile(synth.bundle) as zi, zipfile.ZipFile(b2, "w") as zo:
        for i in zi.infolist():
            data = zi.read(i.filename)
            if i.filename == "samples/GSM9001/log.txt":
                data = data.replace(b"line", b"lime")
            zo.writestr(i, data)
    rep = V.verify(
        str(b2), str(synth.out), None, None, FakeRecipes(), log=lambda *a: None, quick=True
    )
    assert rep.fail > 0


def test_verify_detects_extra_legacy_member(synth, tmp_path):
    b2 = tmp_path / "GSE9001.singlet"
    with zipfile.ZipFile(synth.bundle) as zi, zipfile.ZipFile(b2, "w") as zo:
        for i in zi.infolist():
            zo.writestr(i, zi.read(i.filename))
        zo.writestr("samples/GSM9001/new.tsv", "x\n")
    assert (
        V.verify(
            str(b2), str(synth.out), None, None, FakeRecipes(), log=lambda *a: None, quick=True
        ).fail
        > 0
    )


def test_unsafe_ids_in_an_untrusted_bundle_are_refused(tmp_path):
    p = tmp_path / "GSE1.singlet"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("manifest.json", "{}")
        z.writestr("samples/../../evil/summary.json", "{}")
    with pytest.raises(ValueError):
        MG.migrate_bundle(
            str(p), str(tmp_path / "out"), None, MG.Options(recipes="none", log=lambda *a: None)
        )
    q = tmp_path / "GSE2.singlet"
    with zipfile.ZipFile(q, "w") as z:
        z.writestr("manifest.json", json.dumps({"gse_id": "../x"}))
    with pytest.raises(ValueError):
        MG.migrate_bundle(
            str(q), str(tmp_path / "out2"), None, MG.Options(recipes="none", log=lambda *a: None)
        )


# ------------------------------------------------------------------------------------------------------------ corpus tests
needs_bundle = pytest.mark.skipif(not BUNDLE.exists(), reason="corpus bundle missing")


@needs_bundle
def test_corpus_bundle_subset(tmp_path):
    samples = None if FULL else ["GSM4317423", "GSM4317429"]
    opts = MG.Options(recipes="none", created_at="2026-10-22T00:00:00Z", log=lambda *a: None)
    res = MG.migrate_bundle(str(BUNDLE), str(tmp_path), None, opts, samples)
    MG.write_outputs(res, opts)
    t = res["totals"]
    assert t["s1_bytes"] < t["legacy_zip_bytes"] * 0.5
    # vocabulary packs are used (rows are not stored) and feature_vocab.json is derived
    s0 = S1Sample(open(tmp_path / res["samples"][0]["key"], "rb").read())
    assert s0.reader.members["exon_counts.1pz"].meta["names"]["rows"]["member"] == "exon"
    assert s0.reader.members["intron_counts.1pz"].meta["names"]["rows"]["member"] == "intron"
    assert "recipe" in res["study_files"]["feature_vocab.json"]
    # random access: a column range of a pz2 member with names from the vocabulary pack and the cells table
    full = s0.matrix("exon_counts.1pz")
    sub = s0.matrix("exon_counts.1pz", cols=(1000, 1500))
    assert (
        sub.rownames == full.rownames
        and sub.colnames == full.colnames[1000:1500] == s0.cells().barcodes[1000:1500]
    )
    assert (sub.to_scipy() != full.to_scipy()[:, 1000:1500]).nnz == 0
    rep = V.verify(
        str(BUNDLE), str(tmp_path), None, samples, "none", log=lambda *a: None, quick=not FULL
    )
    assert rep.fail == 0, rep.items
    assert rep.by_kind["matrices"][0] >= 7 * 6 * len(res["samples"])


LOOSE = (
    sorted(p for p in (CORPUS / "corpus_w4_corpus").glob("*") if (p / "exon_counts.1pz").exists())
    if (CORPUS / "corpus_w4_corpus").exists()
    else []
)
LOOSE_QUICK = [p for p in LOOSE if p.name in ("h_v3_a", "x_stub", "z_v2", "bn_dropseq", "m_visium")]


@pytest.mark.skipif(not LOOSE, reason="corpus_w4_corpus missing")
@pytest.mark.parametrize("d", LOOSE if FULL else LOOSE_QUICK, ids=lambda p: p.name)
def test_corpus_w4_samples(d, tmp_path):
    opts = MG.Options(recipes="none", created_at="2026-10-22T00:00:00Z", log=lambda *a: None)
    res = MG.migrate_bundle(str(d), str(tmp_path), None, opts)
    MG.write_outputs(res, opts)
    rep = V.verify(str(d), str(tmp_path), None, None, "none", log=lambda *a: None, quick=True)
    assert rep.fail == 0, rep.items


MTX = (
    [p for p in (CORPUS / "corpus_w4_mtx").glob("GSE*.GSM*")]
    if (CORPUS / "corpus_w4_mtx").exists()
    else []
)
MTX_QUICK = [p for p in MTX if p.name in ("GSE171161.GSM5220137",)]


@pytest.mark.skipif(not MTX, reason="corpus_w4_mtx missing")
@pytest.mark.parametrize("d", MTX if FULL else MTX_QUICK, ids=lambda p: p.name)
def test_corpus_mtx_gen_b_members(d, tmp_path):
    opts = MG.Options(recipes="none", created_at="2026-10-22T00:00:00Z", log=lambda *a: None)
    res = MG.migrate_bundle(str(d), str(tmp_path), None, opts)
    MG.write_outputs(res, opts)
    rep = V.verify(str(d), str(tmp_path), None, None, "none", log=lambda *a: None, quick=True)
    assert rep.fail == 0, rep.items
    names = {rec["name"] for rec in ledger_of(tmp_path, res["gse"]) if rec["type"] == "member"}
    assert "star_Solo.out/Gene/raw/matrix.mtx" in names
