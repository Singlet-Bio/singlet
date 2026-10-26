# SPDX-License-Identifier: MIT
"""Release policy (Research Alpha) in the .s1 transcode: singlet.s1.policy, migrate/verify --policy, the per-cell labels step, the
sample provenance block, and scripts/s1_campaign.py --alpha-policy.

The synthetic bundle carries every member class the policy withholds (genotype, mtDNA alleles, ancestry call, ambient, doublets,
per-cell non-host, legacy PSI) plus one excluded sample; the checks are that nothing of either reaches any output byte, that the kept
members still verify against the legacy bundle, and that the manifest names the policy version and generic layer names only.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from singlet.s1 import policy as POL  # noqa: E402

_pz = pytest.importorskip("singlet._pz", reason="singlet._pz is not built")


def _load(name, path):
    if name not in sys.modules:
        s = importlib.util.spec_from_file_location(name, path)
        m = importlib.util.module_from_spec(s)
        sys.modules[name] = m
        s.loader.exec_module(m)
    return sys.modules[name]


TM = _load("test_s1_migrate", Path(__file__).with_name("test_s1_migrate.py"))
from singlet.s1 import migrate as MG  # noqa: E402
from singlet.s1 import verify as V  # noqa: E402

GSE = "GSE9001"
KEEP, DROP = "GSM9001", "GSM9002"


# ------------------------------------------------------------------------------------------------ rules
@pytest.mark.parametrize(
    "rel, layer",
    [
        ("donor/snp_ad.1pz", "genotype"),
        ("donor/snp_dp.1pz", "genotype"),
        ("donor/donor0.vcf", "genotype"),
        ("donor/donor3_coverage.tsv", "genotype"),
        ("donor/donor_assignments.tsv", None),
        ("donor_assignments.tsv", None),
        ("ase_counts.tsv", "genotype"),
        ("snp_counts.tsv", "genotype"),
        ("calls.vcf.gz", "genotype"),
        ("star_Aligned.sortedByCoord.out.bam", "genotype"),
        ("mt/mt_events.1pz", "mtdna_alleles"),
        ("mt/donor0_mt_consensus.fa", "mtdna_alleles"),
        ("mt/donor0_mt_variants.vcf", "genotype"),  # a VCF either way: withheld
        ("mt/mt_summary.tsv", None),
        ("mt_heteroplasmy.1pz", "mtdna_alleles"),
        ("mt_variants.tsv", "mtdna_alleles"),
        ("mt_reference.fa", None),
        ("ancestry_call.json", "ancestry_call"),
        ("ambient_contamination.tsv", "ambient"),
        ("ambient_contamination.json", "ambient"),
        ("ambient_profile.tsv", "ambient"),
        ("doublet_scores.tsv", "doublet_scores"),
        ("nonhost/nonhost_per_cell.1pz", "nonhost_per_cell"),
        ("nonhost/nonhost_per_cell.tsv", "nonhost_per_cell"),
        ("nonhost/nonhost_summary.json", None),
        ("nonhost/nonhost_em_abundance.tsv", None),
        ("splice_psi.1pz", "splice_psi_legacy"),
        ("sj_counts.1pz", None),
        ("exon_counts.1pz", None),
        ("cell_calls.tsv", None),
        ("sex_call.json", "sex_call_legacy"),
        ("sex_call_extra.json", None),
        ("summary.json", None),
        ("pileup_stats.json", None),
        ("star_Solo.out/Gene/raw/matrix.mtx", None),
    ],
)
def test_omitted_layer(rel, layer):
    assert POL.omitted_layer(rel) == layer
    assert layer is None or layer in POL.OMITTED_LAYERS


def test_policy_withholds_everything_the_restricted_rules_hold_back():
    names = [
        "donor/snp_ad.1pz",
        "donor/x.tsv",
        "mt/mt_events.1pz",
        "mt/anything",
        "ase_counts.tsv",
        "a/snp_b",
        "x.bcf",
        "y.vcf.gz",
        "donor/donor_assignments.tsv",
        "mt/mt_summary.tsv",
        "summary.json",
    ]
    assert POL.is_restricted_superset_ok(names) == []


def test_policy_file_is_validated(tmp_path):
    p = tmp_path / "p.json"
    p.write_text(
        json.dumps({"version": "v", "withheld_studies": ["GSE1"], "excluded_samples": ["nope"]})
    )
    with pytest.raises(ValueError):
        POL.Policy.load(str(p))
    p.write_text(
        json.dumps({"version": "v", "withheld_studies": ["GSE1"], "excluded_samples": ["GSM2"]})
    )
    pol = POL.Policy.load(str(p))
    assert pol.withholds("GSE1") and pol.excludes("GSM2") and len(pol.sha256) == 64
    blk = pol.manifest_block()
    assert blk == {"name": "alpha", "version": "v", "omitted_layers": list(POL.OMITTED_LAYERS)}


# ------------------------------------------------------------------------------------------------ synthetic bundle with every class
def _policy_bundle(tmp_path: Path, leak: bool = False) -> Path:
    b = TM.make_bundle(tmp_path, gse=GSE, gsms=(KEEP, DROP))
    # rewrite the study files with sample lists, then add the members the policy withholds (zipfile "a" appends; the old
    # study files are replaced by writing a fresh archive)
    src = zipfile.ZipFile(b)
    items = [
        (i, src.read(i.filename))
        for i in src.infolist()
        if i.filename not in ("manifest.json", "study_meta.json", "feature_vocab.json")
    ]
    src.close()
    extra = {
        "donor/snp_ad.1pz": b"germline-ad" * 10,
        "donor/donor0_coverage.tsv": b"chrom\tpos\tref\talt\tdp\tad\n1\t100\tA\tG\t3\t1\n",
        "donor/donor_assignments.tsv": b"cell\tcell_status\n",
        "mt/mt_events.1pz": b"mt-events" * 10,
        "mt/mt_summary.tsv": b"cell\tn_events\n",
        "mt_variants.tsv": b"pos\tref\talt\n73\tA\tG\n",
        "ancestry_call.json": b'{"group": "EUR"}',
        "doublet_scores.tsv": b"cell\tscore\n",
        "ambient_contamination.tsv": b"cell\trho\n",
        "nonhost/nonhost_per_cell.tsv": b"cell\ttaxon\n",
        "nonhost/nonhost_summary.json": b'{"screened": true}',
        "sex_call.json": b'{"sex": "male"}',
        "pileup_stats.json": json.dumps(
            {
                "total_reads": 1000,
                "exon_hits": 600,
                "wrong_strand": 40,
                "chrm_reads": 10,
                "mapped_reads": 900,
                "reverse_strand": False,
            }
        ).encode(),
        "star_Log.final.out": b"                          Number of input reads |\t1234\n",
    }
    names = [f"samples/{g}/{r}" for g in (KEEP, DROP) for r in extra] + [
        i.filename for i, _ in items
    ]
    with zipfile.ZipFile(b, "w", allowZip64=True) as z:
        z.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "gse_id": GSE,
                    "gsm_ids": [KEEP, DROP],
                    "n_gsms": 2,
                    "reference_build": "toy",
                    "included_files": {
                        KEEP: ["exon_counts.1pz", "doublet_scores.tsv"],
                        DROP: ["exon_counts.1pz"],
                    },
                    "checksums": {n: "0" * 64 for n in names},
                },
                indent=2,
            ),
        )
        z.writestr(
            "study_meta.json",
            json.dumps(
                {
                    "gse_id": GSE,
                    "series_title": "toy",
                    "gsm_meta": {
                        KEEP: {"gsm_id": KEEP, "organism": "Homo sapiens", "sample_source": "PBMC"},
                        DROP: {"gsm_id": DROP, "organism": "Homo sapiens", "sample_source": "PBMC"},
                    },
                },
                indent=2,
            ),
        )
        # the toy rows are G000..G239 (exon) / G000..G199 (intron); a few carry real symbols so the sex model has something
        sym = [
            "XIST",
            "RPS4Y1",
            "DDX3Y",
            "UTY",
            "KDM5D",
            "CD3E",
            "CD3D",
            "MS4A1",
            "CD79A",
            "LYZ",
            "CD14",
            "NKG7",
            "GNLY",
        ]
        genes = [
            {"gene_id": "G%03d" % i, "gene_name": sym[i] if i < len(sym) else "GENE%d" % i}
            for i in range(240)
        ]
        z.writestr(
            "feature_vocab.json",
            json.dumps(
                {"schema_version": "1.0", "reference_build": "toy", "n_genes": 240, "genes": genes},
                indent=2,
            ),
        )
        for info, data in items:
            if leak and info.filename == f"samples/{KEEP}/summary.json":
                d = json.loads(data)
                d["paired_with"] = DROP
                data = json.dumps(d).encode()
            z.writestr(info, data)
        for g in (KEEP, DROP):
            for r, d in extra.items():
                z.writestr(f"samples/{g}/{r}", d)
    return b


@pytest.fixture
def pol(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(
        json.dumps(
            {"version": "alpha-test-1", "withheld_studies": ["GSE7"], "excluded_samples": [DROP]}
        )
    )
    return POL.Policy.load(str(p)), p


def _all_bytes(d: Path) -> bytes:
    return b"".join(p.read_bytes() for p in sorted(d.rglob("*")) if p.is_file())


def test_policy_transcode_drops_and_still_verifies(tmp_path, pol):
    policy, ppath = pol
    b = _policy_bundle(tmp_path)
    out = tmp_path / "out"
    opts = MG.Options(
        recipes=TM.FakeRecipes(),
        created_at="2026-10-26T00:00:00Z",
        log=lambda *a: None,
        policy=policy,
    )
    res = MG.migrate_bundle(str(b), str(out), None, opts)
    MG.write_outputs(res, opts)
    man = json.loads((out / "s1" / "m" / f"{GSE}.json").read_text("utf-8"))
    assert [s["gsm"] for s in man["samples"]] == [KEEP]
    assert man["release_policy"] == {
        "name": "alpha",
        "version": "alpha-test-1",
        "omitted_layers": list(POL.OMITTED_LAYERS),
    }
    members = set(man["samples"][0]["members"])
    for r in (
        "donor/snp_ad.1pz",
        "donor/donor0_coverage.tsv",
        "mt/mt_events.1pz",
        "mt_variants.tsv",
        "ancestry_call.json",
        "doublet_scores.tsv",
        "ambient_contamination.tsv",
        "ambient_profile.tsv",
        "nonhost/nonhost_per_cell.tsv",
        "splice_psi.1pz",
        "sex_call.json",
    ):
        assert r not in members, r
    for r in (
        "donor/donor_assignments.tsv",
        "mt/mt_summary.tsv",
        "nonhost/nonhost_summary.json",
        "sj_counts.1pz",
        "exon_counts.1pz",
    ):
        assert r in members, r
    # no output byte names the excluded sample; the .s1 manifest's study files were rewritten without it
    s1_bytes = _all_bytes(out / "s1")
    assert DROP.encode() not in s1_bytes and b"germline-ad" not in s1_bytes
    lm = json.loads(MG.M.study_file_bytes(man["study_files"]["manifest.json"]))
    assert lm["gsm_ids"] == [KEEP] and lm["n_gsms"] == 1 and lm["release_policy"] == "alpha-test-1"
    assert all(DROP not in k and "doublet" not in k and "snp_" not in k for k in lm["checksums"])
    assert lm["included_files"] == {KEEP: ["exon_counts.1pz"]}
    sm = json.loads(MG.M.study_file_bytes(man["study_files"]["study_meta.json"]))
    assert list(sm["gsm_meta"]) == [KEEP]
    # provenance and junction support
    prov = man["samples"][0]["provenance"]
    assert prov["reads_used"] == 1234 and prov["reads_used_source"] == "star_Log.final.out"
    assert (
        prov["strand"]["wrong_strand_fraction"] == pytest.approx(40 / 640, abs=1e-5)
        and prov["strand"]["class"] == "ok"
    )
    assert (
        prov["release_policy"] == "alpha-test-1"
        and prov["library_type"]["verdict"] == "gene_expression"
    )
    js = man["samples"][0]["junction_support"]
    assert (
        js["n_junctions"] == 300
        and 0 <= js["n_supported"] <= 300
        and js["min_molecules"] == MG.JUNCTION_SUPPORT_MIN
    )
    # the private audit names what was dropped; it sits next to, never inside, the .s1 tree
    audit = json.loads((out / "policy-audit.json").read_text())
    assert audit["excluded_samples"] == [DROP] and audit["omitted_members"]["genotype"] == 2
    # verify against the policy's view passes; against the whole legacy bundle it fails (members and a sample are missing)
    rep = V.verify(
        str(b), str(out), None, None, TM.FakeRecipes(), log=lambda *a: None, policy=policy
    )
    assert rep.fail == 0, rep.items
    assert rep.by_kind["policy"][0] > 0
    rep2 = V.verify(str(b), str(out), None, None, TM.FakeRecipes(), log=lambda *a: None)
    assert rep2.fail > 0


def test_withheld_study_writes_nothing(tmp_path):
    b = _policy_bundle(tmp_path)
    p = tmp_path / "w.json"
    p.write_text(json.dumps({"version": "v", "withheld_studies": [GSE], "excluded_samples": []}))
    out = tmp_path / "out"
    rc = MG.main(["--bundle", str(b), "--out", str(out), "--policy", str(p), "--recipes", "none"])
    assert rc == 3 and not (out / "s1" / "s").exists() or not any((out / "s1" / "s").iterdir())


def test_all_samples_excluded_is_withheld(tmp_path):
    b = _policy_bundle(tmp_path)
    p = tmp_path / "w.json"
    p.write_text(
        json.dumps({"version": "v", "withheld_studies": [], "excluded_samples": [KEEP, DROP]})
    )
    opts = MG.Options(recipes="none", log=lambda *a: None, policy=POL.Policy.load(str(p)))
    with pytest.raises(MG.PolicyWithheld):
        MG.migrate_bundle(str(b), str(tmp_path / "out"), None, opts)


def test_kept_member_naming_an_excluded_sample_is_refused(tmp_path, pol):
    policy, _ = pol
    b = _policy_bundle(tmp_path, leak=True)
    opts = MG.Options(recipes="none", log=lambda *a: None, policy=policy)
    with pytest.raises(ValueError, match="excluded sample"):
        MG.migrate_bundle(str(b), str(tmp_path / "out"), None, opts)


# ------------------------------------------------------------------------------------------------ labels inside the transcode
def test_labels_are_written_as_referenced_sidecars_and_verified(tmp_path, pol):
    pytest.importorskip("pyarrow")
    pytest.importorskip("sklearn")
    from singlet.s1.labels_step import LabelOptions

    policy, _ = pol
    b = _policy_bundle(tmp_path)
    out = tmp_path / "out"
    opts = MG.Options(
        recipes=TM.FakeRecipes(),
        created_at="2026-10-26T00:00:00Z",
        log=lambda *a: None,
        policy=policy,
        labels=LabelOptions(timeout_s=600),
    )
    res = MG.migrate_bundle(str(b), str(out), None, opts)
    MG.write_outputs(res, opts)
    man = json.loads((out / "s1" / "m" / f"{GSE}.json").read_text("utf-8"))
    s = man["samples"][0]
    ent = s["annotations"]["a2026-10"]
    assert man["annotations_default"] == "a2026-10"
    assert (out / ent["key"]).is_file() and ent["n_rows"] == s["n_called"]
    assert ent["cells_digest"] == s["members"]["cells"]["digest"] and ent["cell_type"][
        "default_levels"
    ] == [1, 2]
    led = [json.loads(x) for x in (out / "s1" / "ledger" / f"{GSE}.jsonl").read_text().splitlines()]
    lab = [x for x in led if x["type"] == "labels"]
    assert (
        len(lab) == 1 and lab[0]["status"] == "ok" and lab[0]["coverage"]["cells"] == s["n_called"]
    )
    # the labels are also columns of the sample object's own cells table, equal to the sidecar on the called cells
    import pyarrow.parquet as pq

    from singlet.s1.labels_step import CELLS_LABEL_COLUMNS
    from singlet.s1.sample import S1Sample

    t = S1Sample((out / s["key"]).read_bytes(), MG.P.default_store()).cells()
    assert all(t.has(n) for n, _ in CELLS_LABEL_COLUMNS)
    assert s["members"]["cells"]["labels"]["annot_version"] == "a2026-10"
    sc = pq.read_table(out / ent["key"]).to_pydict()
    called = np.flatnonzero(np.asarray(t.column("called"), bool))
    assert list(called) == sc["col"]
    assert [t.column("sex_chrom")[i] for i in called] == [x or "" for x in sc["sex_chrom"]]
    assert [t.column("cell_type_cl")[i] for i in called] == [x or "" for x in sc["cell_type_cl"]]
    uncalled = np.flatnonzero(~np.asarray(t.column("called"), bool))
    assert all(t.column("sex_chrom")[i] == "" for i in uncalled)
    rep = V.verify(
        str(b), str(out), None, None, TM.FakeRecipes(), log=lambda *a: None, policy=policy
    )
    assert rep.fail == 0, rep.items
    assert rep.by_kind["labels"][0] >= 5
    # the campaign accepts the sidecar key class and checks it against the manifest
    camp = _load("s1_campaign", ROOT / "scripts" / "s1_campaign.py")
    plan = camp.collect_objects(out / "s1", GSE, 8 << 20)
    assert any(camp.RE_ANNOT.match(o.key) for o in plan.objects)
    assert camp.object_headers(ent["key"])[1].endswith("immutable")


def test_labels_over_budget_leave_the_object_unlabelled(tmp_path, pol):
    pytest.importorskip("pyarrow")
    from singlet.s1.labels_step import LabelOptions

    policy, _ = pol
    b = _policy_bundle(tmp_path)
    out = tmp_path / "out"
    opts = MG.Options(
        recipes=TM.FakeRecipes(),
        log=lambda *a: None,
        policy=policy,
        labels=LabelOptions(max_gb=1e-9),
    )
    res = MG.migrate_bundle(str(b), str(out), None, opts)
    MG.write_outputs(res, opts)
    man = json.loads((out / "s1" / "m" / f"{GSE}.json").read_text("utf-8"))
    assert "annotations" not in man["samples"][0] and "annotations_default" not in man
    led = [json.loads(x) for x in (out / "s1" / "ledger" / f"{GSE}.jsonl").read_text().splitlines()]
    lab = [x for x in led if x["type"] == "labels"]
    assert lab[0]["status"] == "over_budget" and "budget" in lab[0]["reason"]


def test_species_without_a_model_get_no_sidecar(tmp_path, pol):
    from singlet.s1 import labels_step as LS

    r = LS.label_sample(
        gsm=KEEP,
        gse=GSE,
        exon=None,
        intron=None,
        basis_barcodes=["A"],
        called=np.array([True]),
        cells_digest="",
        vocab={},
        study_meta={"gsm_meta": {KEEP: {"organism": "Danio rerio"}}},
        donor_tsv=None,
        depth_members={},
        opts=LS.LabelOptions(),
    )
    assert r["status"] == "skipped" and "no label model" in r["reason"] and "sidecar" not in r


# ------------------------------------------------------------------------------------------------ the 14 bundles with a bad CRC
def test_member_with_a_wrong_central_directory_crc_is_read_on_its_local_header(tmp_path):
    from singlet.s1.legacy import LegacyBundle, LegacyError

    p = tmp_path / "b.singlet"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", b'{"gse_id": "GSE1"}')
        z.writestr("samples/GSM1/summary.json", b"{}")
    data = bytearray(p.read_bytes())
    cd = data.rfind(
        b"PK\x01\x02", 0, len(data)
    )  # last central entry is summary.json; find manifest's (the first)
    cd = data.find(b"PK\x01\x02")
    (crc,) = struct.unpack_from("<I", data, cd + 16)
    struct.pack_into("<I", data, cd + 16, crc ^ 0xFFFF)
    p.write_bytes(bytes(data))
    with pytest.raises(zipfile.BadZipFile):
        zipfile.ZipFile(p).read("manifest.json")
    b = LegacyBundle.open(str(p))
    assert b.read("manifest.json") == b'{"gse_id": "GSE1"}' and b.crc_repaired == ["manifest.json"]
    assert b.entries["manifest.json"].crc32 == zlib.crc32(b'{"gse_id": "GSE1"}')
    # a member whose data matches neither CRC still fails
    data2 = bytearray(p.read_bytes())
    loc = data2.find(b"PK\x03\x04")
    struct.pack_into("<I", data2, loc + 14, 0x12345678)
    p.write_bytes(bytes(data2))
    with pytest.raises(LegacyError):
        LegacyBundle.open(str(p)).read("manifest.json")


# ------------------------------------------------------------------------------------------------ campaign
_C = _load("test_s1_campaign_common", Path(__file__).with_name("test_s1_campaign_common.py"))
camp = _C.load_script("s1_campaign")


@pytest.fixture
def env(tmp_path, monkeypatch):
    return _C.build_env(tmp_path, monkeypatch, camp)


def test_campaign_skips_withheld_studies_and_writes_nothing_for_them(env, tmp_path):
    gses = _C.seed(env.root, 3)
    p = tmp_path / "pol.json"
    p.write_text(
        json.dumps(
            {"version": "alpha-test-2", "withheld_studies": [gses[1]], "excluded_samples": []}
        )
    )
    assert env.run("--alpha-policy", str(p), "--audit-dir", str(tmp_path / "audit")) == 0
    rows = env.rows_of()
    assert rows[gses[1]]["status"] == "withheld" and rows[gses[0]]["status"] == "ok"
    keys = [k for k, _ in env.store.list_keys("singlet-data", "s1/")]
    assert not any(gses[1] in k for k in keys)  # no manifest, no ledger
    assert not (env.rows / f"{gses[1]}.json").exists()  # and no catalog row
    assert env.ncalls("migrate") == 2


def test_campaign_under_a_policy_transcodes_bundles_that_still_hold_restricted_members(
    env, tmp_path
):
    """The policy (not the restrict split) is what keeps them out: the study is no longer blocked, and the collect step still refuses
    a transcoder output that let a restricted member through (the stand-in transcoder copies every member)."""
    gses = _C.seed(env.root, 2, restricted="GSE000002")
    p = tmp_path / "pol.json"
    p.write_text(
        json.dumps({"version": "alpha-test-2", "withheld_studies": [], "excluded_samples": []})
    )
    env.run("--alpha-policy", str(p))
    rows = env.rows_of()
    assert rows[gses[0]]["status"] == "ok" and rows[gses[1]]["restricted_members"] == 2
    assert (
        rows[gses[1]]["status"] == "collect_failed"
        and "restricted members" in rows[gses[1]]["error"]
    )
    assert not any(gses[1] in k for k, _ in env.store.list_keys("singlet-data", "s1/m/"))


def test_cell_index_dictionary_takes_levels_from_the_label_path():
    """A query for a compartment or lineage matches deeper calls through the labeller's path, without an ontology file."""
    from singlet.s1 import atlas

    d = atlas.make_dict(
        ["CL:0000084", "CL:0000236"],
        None,
        None,
        {"CL:0000084": {"CL:0000988", "CL:0000542"}, "CL:0000236": {"CL:0000988"}},
    )
    by = {e["id"]: e for e in d}
    assert by["CL:0000084"]["anc"] == ["CL:0000542", "CL:0000988"] and by["CL:0000236"]["anc"] == [
        "CL:0000988"
    ]
    assert atlas.make_dict(["CL:0000084"], None)[1]["anc"] == []
