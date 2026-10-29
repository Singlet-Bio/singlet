# SPDX-License-Identifier: MIT
"""singlet.s1.junctions: the shared splice-junction table (docs/format-s1.md section 12) on synthetic data: names round trip, segment
coding, annotation from a GTF and motifs from a FASTA, append-only versions with stable ids, id <-> coordinate lookups, and a sample's
rows as table ids + local junctions + the legacy permutation (exact)."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

pytest.importorskip("zstandard")

from singlet.s1 import junctions as J  # noqa: E402
from singlet.s1 import packs as P  # noqa: E402

BUILD = "TEST-1"


def _random_names(rng, n, chroms=("chr1", "chr2", "chrX", "chr10")):
    out = set()
    while len(out) < n:
        c = chroms[int(rng.integers(len(chroms)))]
        s = int(rng.integers(1, 5_000_000))
        out.add(f"{c}:{s}-{s + int(rng.integers(50, 200_000))}")
    return sorted(out)


def test_names_round_trip_and_refusals():
    names = ["chr7:26402-31059", "chr7:31606-35334", "chrX:5-900", "GL000194.1:100-250"]
    j = J.parse_names(names)
    assert j.names() == names and j.chroms == ["chr7", "chrX", "GL000194.1"]
    for bad in ("chr1:10", "chr1:010-20", "chr1:a-20", "nocolon"):
        with pytest.raises(J.JunctionError):
            J.parse_names([bad])


def test_segment_body_round_trip_with_attributes():
    rng = np.random.default_rng(1)
    j = J.parse_names(_random_names(rng, 5000))
    j = J.Junctions(
        j.chroms,
        j.chrom,
        j.start,
        j.end,
        rng.integers(0, 256, len(j)),
        rng.integers(0, 40000, len(j)),
        rng.integers(0, 256, len(j)),
    )
    seg = J.Segment(J.build_segment(j, build=BUILD, id_lo=0, tier="core"))
    o = j.genomic_order()
    jj = j.take(o)
    got = seg.j
    assert got.names() == jj.names()
    assert (
        np.array_equal(got.attr, jj.attr)
        and np.array_equal(got.gene, jj.gene)
        and np.array_equal(got.n_tx, jj.n_tx)
    )
    assert seg.header["coords"] == J.COORDS and seg.header["n"] == len(j)


def _write_gtf(path):
    # one + strand gene with two transcripts sharing intron 1, one - strand gene; GTF is 1-based closed
    rows = [
        ("chr1", "+", "G1", "T1", [(101, 200), (301, 400), (501, 600)]),
        ("chr1", "+", "G1", "T2", [(101, 200), (301, 400), (701, 800)]),
        ("chr1", "-", "G2", "T3", [(1001, 1100), (1201, 1300)]),
    ]
    with open(path, "w") as fh:
        fh.write("#!genome-build TEST\n")
        for chrom, sd, g, t, exons in rows:
            for s, e in exons:
                fh.write(
                    f'{chrom}\tTEST\texon\t{s}\t{e}\t.\t{sd}\t.\tgene_id "{g}.1"; transcript_id "{t}";\n'
                )


def test_gtf_annotation_and_motifs(tmp_path):
    gtf = tmp_path / "a.gtf"
    _write_gtf(gtf)
    ann = J.read_gtf(str(gtf), gene_index={"G1": 0, "G2": 1})
    # 0-based [first intron base, first exon base after): exon (101,200) -> intron starts at 200; next exon at 301 -> ends at 300
    assert ann.introns[("chr1", 200, 300)] == (1, 2)  # gene index 0 + 1, two transcripts
    assert ann.introns[("chr1", 400, 500)] == (1, 1) and ann.introns[("chr1", 400, 700)] == (1, 1)
    assert ann.strand[("chr1", 1100, 1200)] == 2
    seq = ["N"] * 1500
    seq[200:202] = "GT"
    seq[298:300] = "AG"  # + strand canonical
    seq[1100:1102] = "CT"
    seq[1198:1200] = "AC"  # - strand canonical, read on +
    genome = {"chr1": "".join(seq)}
    j = J.annotate(
        J.parse_names(["chr1:200-300", "chr1:1100-1200", "chr1:200-500", "chr1:250-350"]),
        ann,
        genome,
    )
    assert j.annotated.tolist() == [True, True, False, False]
    assert j.motif.tolist() == [1, 2, 0, 0] and j.strand.tolist() == [1, 2, 0, 0]
    assert j.n_tx.tolist() == [2, 1, 0, 0] and j.gene.tolist() == [1, 2, 0, 0]
    # an unannotated pairing of annotated ends keeps the donor / acceptor bits
    assert (
        (j.attr[2] & J.DONOR_ANN) and (j.attr[2] & J.ACCEPTOR_ANN) and not (j.attr[3] & J.DONOR_ANN)
    )


def test_versions_are_append_only_with_stable_ids():
    rng = np.random.default_rng(2)
    names = _random_names(rng, 3000)
    a = J.parse_names(names[:2000])
    b = J.parse_names(names[2000:])
    s0 = J.build_segment(a, build=BUILD, id_lo=0, tier="core")
    v1 = J.build_version([s0], build=BUILD, version=1)
    s1 = J.build_segment(b, build=BUILD, id_lo=2000, tier="rare")
    v2 = J.build_version([s0, s1], build=BUILD, version=2, previous=v1)
    t1, t2 = J.JunctionTable(v1, [s0]), J.JunctionTable(v2, [s0, s1])
    ids = np.arange(2000)
    assert t1.names(ids) == t2.names(
        ids
    )  # every id of version 1 means the same junction in version 2
    assert sorted(t2.names(np.arange(3000))) == sorted(names)
    with pytest.raises(J.JunctionError, match="contiguous"):
        J.build_version([s1], build=BUILD, version=3)
    other = J.build_version(
        [J.build_segment(b, build=BUILD, id_lo=0, tier="core")], build=BUILD, version=1
    )
    with pytest.raises(J.JunctionError, match="keep every segment"):
        J.build_version([s0, s1], build=BUILD, version=2, previous=other)
    with pytest.raises(J.JunctionError, match="outside"):
        t1.lookup([2000])


def test_lookup_find_and_segments_fetched_on_demand():
    rng = np.random.default_rng(3)
    names = _random_names(rng, 4000)
    s0 = J.build_segment(J.parse_names(names[:3000]), build=BUILD, id_lo=0, tier="core")
    s1 = J.build_segment(J.parse_names(names[3000:]), build=BUILD, id_lo=3000, tier="rare")
    v = J.build_version([s0, s1], build=BUILD, version=1)
    fetched = []
    store = {P.pack_id(s1).hex(): s1}
    t = J.JunctionTable(v, [s0], fetch=lambda h: fetched.append(h) or store[h])
    q = rng.integers(0, 4000, 5000)
    j = t.lookup(q)
    assert fetched == [
        P.pack_id(s1).hex()
    ]  # the rare segment is read once, when an id in its range is asked for
    assert np.array_equal(t.find(j), q)
    unknown = J.parse_names(["chr1:1-2", "chrZ:5-10"])
    assert t.find(unknown).tolist() == [-1, -1]
    bad = J.JunctionTable(v, [s0], fetch=lambda h: s0)
    with pytest.raises(J.JunctionError, match="digest"):
        bad.lookup([3500])


def test_sample_rows_round_trip_to_legacy_order():
    rng = np.random.default_rng(4)
    table_names = _random_names(rng, 2000)
    seg = J.build_segment(J.parse_names(table_names), build=BUILD, id_lo=0, tier="core")
    t = J.JunctionTable(J.build_version([seg], build=BUILD, version=1), [seg])
    own = _random_names(np.random.default_rng(5), 300, chroms=("chr3", "chrUn_x"))
    legacy = list(rng.permutation(table_names[:700] + own))  # the pipeline's first-seen order
    rows = J.sample_rows(t, legacy)
    assert (
        np.diff(rows.ids.astype(np.int64)) > 0
    ).all()  # strictly ascending: table ids, then LOCAL_BASE + k
    assert int((rows.ids >= J.LOCAL_BASE).sum()) == 300 and len(rows.local) == 300
    loc = J.decode_local(J.encode_local(rows.local))
    perm = J.decode_perm(J.encode_perm(rows.legacy_perm))
    v2names = J.row_names(t, rows.ids, loc)
    back = [None] * len(legacy)
    for r, k in enumerate(perm.tolist()):
        back[k] = v2names[r]
    assert back == legacy


def test_client_reads_junction_rows_stored_as_table_ids(tmp_path, monkeypatch):
    """A study whose junction matrices store table ids (+ a .local member) loads with the same names and values as inline names."""
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import _s1_synth as synth
    from singlet import _pz2
    from singlet.s1 import client

    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    client.clear_memory_cache()
    try:
        s = synth.Synth(str(tmp_path / "study")).build(extras=True, junction_table=True)
    except IndexError:  # pragma: no cover - packs not shipped
        pytest.skip("vocabulary packs are not available")
    st = client.open_study(synth.GSE, base=str(tmp_path / "study"))
    sm = st.samples["GSM9000001"]
    assert "sj_counts.1pz.local" in sm.members
    with open(tmp_path / "study" / sm.key, "rb") as fh:
        obj = fh.read()
    m = sm.members["sj_counts.1pz"]
    assert _pz2.Member(obj[m.off : m.off + m.len]).header["row_names_kind"] == "pack_subset"
    ad = st.to_anndata(["GSM9000001", "GSM9000003"], modalities="splice_junctions")
    names = list(ad.uns["obsm_features"]["splice_junctions"])
    assert sorted(names) == sorted(
        set(s.truth["GSM9000001"]["sj_names"]) | set(s.truth["GSM9000003"]["sj_names"])
    )
    for g, rows in (("GSM9000001", slice(0, 40)), ("GSM9000003", slice(40, None))):
        t = s.truth[g]
        called = np.flatnonzero(t["called"])
        cols = called[np.argsort(np.array(t["barcodes"], dtype=object)[called], kind="stable")]
        pos = [names.index(x) for x in t["sj_names"]]
        got = ad.obsm["splice_junctions"][rows][:, pos]
        assert (got != t["sj"][:, cols].T).nnz == 0
    client.clear_memory_cache()


def test_migration_with_a_junction_table_regenerates_every_legacy_member(tmp_path):
    """migrate_bundle with Options(junction_table=...): sj_counts rows become table ids (+ .local, + .legacy_order) and verify
    still finds every legacy member bit-exact, including splice_psi, which a recipe derives from sj_counts."""
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    pytest.importorskip("singlet._pz", reason="singlet._pz is not built")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import test_s1_migrate as TM
    from singlet import _pz2
    from singlet.s1 import migrate as MG
    from singlet.s1 import verify as V

    bundle = TM.make_bundle(tmp_path)
    # the bundle's junctions are chr1:100i-(100i+40+i%3), i < 300: the table holds the first 200, the rest stay local
    tdir = tmp_path / "table"
    tdir.mkdir()
    names = ["chr1:%d-%d" % (100 * i, 100 * i + 40 + (i % 3)) for i in range(200)]
    seg = J.build_segment(J.parse_names(names), build="toy", id_lo=0, tier="core")
    ver = J.build_version([seg], build="toy", version=1)
    for b in (seg, ver):
        (tdir / (P.pack_id(b).hex() + ".pack")).write_bytes(b)
    store = P.PackStore(dirs=[P.PKG_DIR, str(tdir)])
    table = store.get(P.pack_id(ver).hex())
    out = tmp_path / "out"
    opts = MG.Options(
        recipes=TM.FakeRecipes(),
        created_at="2026-10-22T00:00:00Z",
        log=lambda *a: None,
        packs=store,
        junction_table=table,
    )
    res = MG.migrate_bundle(str(bundle), str(out), None, opts)
    MG.write_outputs(res, opts)
    rep = V.verify(str(bundle), str(out), None, None, TM.FakeRecipes(), log=lambda *a: None)
    assert rep.fail == 0, (
        [it for it in rep.items if not it[0]][:5] if hasattr(rep, "items") else rep
    )
    man = __import__("json").load(open(out / "s1" / "m" / "GSE9001.json"))
    s0 = man["samples"][0]
    mem = s0["members"]
    assert "sj_counts.1pz.local" in mem and "sj_counts.1pz.legacy_order" in mem
    obj = (out / s0["key"]).read_bytes()
    m = mem["sj_counts.1pz"]
    h = _pz2.Member(obj[m["off"] : m["off"] + m["len"]]).header
    assert h["row_names_kind"] == "pack_subset" and h["digest"] == m["digest"]
    idx = json_ids = None  # noqa: F841
    ref = {
        e["id"]: e["kind"]
        for e in __import__("json").load(open(out / "s1" / "ref" / "index.json"))["packs"]
    }
    assert (
        ref[P.pack_id(ver).hex()] == "junction-version"
        and ref[P.pack_id(seg).hex()] == "junction-segment"
    )


def test_legacy_http_reader_retries_a_cut_response(monkeypatch):
    """A response cut short by the network (http.client.IncompleteRead) is retried, not fatal (the census lost 16 studies to it)."""
    import http.client

    from singlet.s1 import legacy as L

    calls = []

    class Resp:
        status = 206

        def __init__(self, ok):
            self.ok = ok

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            if not self.ok:
                raise http.client.IncompleteRead(b"ab", 3)
            return b"abcde"

        def getheaders(self):
            return [("Content-Range", "bytes 0-4/10")]

    def fake_urlopen(req, timeout=None):
        calls.append(1)
        return Resp(len(calls) > 1)

    monkeypatch.setattr(L.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    src = L._HttpSource("https://example.invalid/x.singlet")
    assert src.read(0, 5) == b"abcde" and len(calls) == 2


# ---------------------------------------------------------------------------------------------------------- migration wiring
def _toy_table(tdir, version=1, n=200):
    names = ["chr1:%d-%d" % (100 * i, 100 * i + 40 + (i % 3)) for i in range(n)]
    seg = J.build_segment(J.parse_names(names), build="toy", id_lo=0, tier="core")
    ver = J.build_version([seg], build="toy", version=version)
    for b in (seg, ver):
        (tdir / (P.pack_id(b).hex() + ".pack")).write_bytes(b)
    return seg, ver


def test_published_table_picks_the_newest_version_of_the_build(tmp_path):
    seg, v1 = _toy_table(tmp_path, version=1)
    v2 = J.build_version([seg], build="toy", version=2, previous=v1)
    (tmp_path / (P.pack_id(v2).hex() + ".pack")).write_bytes(v2)
    other = J.build_version(
        [J.build_segment(J.parse_names(["chr2:1-50"]), build="other", id_lo=0, tier="core")],
        build="other",
        version=9,
    )
    (tmp_path / (P.pack_id(other).hex() + ".pack")).write_bytes(other)
    store = P.PackStore(dirs=[str(tmp_path)])
    index = [
        P.index_entry(b, "junction-version", {"build": bl})
        for b, bl in ((v1, "toy"), (v2, "toy"), (other, "other"))
    ]
    t = J.published_table("toy", store, index)
    assert t.id == P.pack_id(v2) and t.header["version"] == 2
    assert (
        J.published_table("GRCh38-2024-A", store, index) is None
    )  # nothing published for that build: inline names
    dup = J.build_version([seg], build="toy", version=2, extra={"note": "a second v2"})
    (tmp_path / (P.pack_id(dup).hex() + ".pack")).write_bytes(dup)
    store.rescan()
    with pytest.raises(J.JunctionError, match="ambiguous"):
        J.published_table(
            "toy", store, index + [P.index_entry(dup, "junction-version", {"build": "toy"})]
        )


def test_migrate_with_auto_records_the_table_and_verifies(tmp_path, monkeypatch):
    """--junction-table auto: the table published for the study's build is used, recorded in the manifest (packs.junctions)
    and shipped with the study's packs; verify passes on every legacy member."""
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    pytest.importorskip("singlet._pz", reason="singlet._pz is not built")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import json

    import test_s1_migrate as TM
    from singlet.s1 import migrate as MG
    from singlet.s1 import verify as V

    bundle = TM.make_bundle(tmp_path)  # reference build "toy"
    tdir = tmp_path / "ref"
    tdir.mkdir()
    seg, ver = _toy_table(tdir)
    store = P.PackStore(dirs=[P.PKG_DIR, str(tdir)])
    monkeypatch.setattr(
        J,
        "fetch_index",
        lambda remote=None: [P.index_entry(ver, "junction-version", {"build": "toy"})],
    )
    logs = []
    out = tmp_path / "out"
    opts = MG.Options(
        recipes=TM.FakeRecipes(),
        created_at="2026-10-22T00:00:00Z",
        log=logs.append,
        packs=store,
        junction_table="auto",
    )
    res = MG.migrate_bundle(str(bundle), str(out), None, opts)
    MG.write_outputs(res, opts)
    assert any("junction table for toy: version 1" in str(x) for x in logs)
    man = json.load(open(out / "s1" / "m" / "GSE9001.json"))
    assert man["packs"]["junctions"] == P.pack_id(ver).hex()
    assert all("sj_counts.1pz.legacy_order" in s["members"] for s in man["samples"])
    shipped = {
        e["id"]: e["kind"] for e in json.load(open(out / "s1" / "ref" / "index.json"))["packs"]
    }
    assert (
        shipped[P.pack_id(ver).hex()] == "junction-version"
        and shipped[P.pack_id(seg).hex()] == "junction-segment"
    )
    rep = V.verify(str(bundle), str(out), None, None, TM.FakeRecipes(), log=lambda *a: None)
    assert rep.fail == 0


def test_migrate_without_a_published_table_keeps_inline_names(tmp_path, monkeypatch):
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    pytest.importorskip("singlet._pz", reason="singlet._pz is not built")
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import json

    import test_s1_migrate as TM
    from singlet.s1 import migrate as MG

    bundle = TM.make_bundle(tmp_path)
    monkeypatch.setattr(J, "fetch_index", lambda remote=None: [])
    out = tmp_path / "out"
    opts = MG.Options(
        recipes=TM.FakeRecipes(),
        created_at="2026-10-22T00:00:00Z",
        log=lambda *a: None,
        junction_table="auto",
    )
    res = MG.migrate_bundle(str(bundle), str(out), None, opts)
    MG.write_outputs(res, opts)
    man = json.load(open(out / "s1" / "m" / "GSE9001.json"))
    assert "junctions" not in man["packs"] and not any(
        "sj_counts.1pz.local" in s["members"] for s in man["samples"]
    )
    # a table for another build is refused, not silently applied
    other = J.JunctionTable(
        J.build_version(
            [J.build_segment(J.parse_names(["chr1:1-50"]), build="b2", id_lo=0, tier="core")],
            build="b2",
            version=1,
        )
    )
    with pytest.raises(ValueError, match="is for b2"):
        MG.migrate_bundle(
            str(bundle),
            str(tmp_path / "o2"),
            None,
            MG.Options(recipes=TM.FakeRecipes(), log=lambda *a: None, junction_table=other),
        )


def test_migrate_cli_and_campaign_default(tmp_path):
    """The CLI flag parses (none / auto / a pack path); the campaign passes --junction-table auto unless told otherwise."""
    from singlet.s1 import migrate as MG

    assert (
        MG._junction_arg("none") is None
        and MG._junction_arg("auto") == "auto"
        and callable(MG._junction_arg("abcd" * 8))
    )
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
    import s1_campaign as camp

    a = camp.build_parser().parse_args(["run", "--mode", "dry"])
    assert a.junction_table == "auto"
    b = camp.build_parser().parse_args(["run", "--mode", "dry", "--junction-table", "none"])
    assert b.junction_table == "none"
