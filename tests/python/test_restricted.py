# SPDX-License-Identifier: MIT
"""Per-donor germline members stay out of the public bundle: the rules, the packer and redact_bundle.

Like test_pack_gse.py this builds synthetic pipeline output (hand-written TP1Z headers) and needs no compiled codec.
"""

from __future__ import annotations

import json
import shutil
import struct
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from singlet import bundle as bundle_mod
from singlet._integrity import verify_bundle
from singlet._restricted import (
    PUBLIC_PROFILE,
    RESTRICTED_PROFILE,
    is_restricted_member,
    is_restricted_path,
    public_ancestry_call,
    redact_bundle,
    restore_bundle,
    restricted_summary,
)

GSE = "GSE000088"
GOOD = "GSM0000881"


def _pz_bytes(m: int, n: int, nnz: int, pad: int = 600) -> bytes:
    """A TP1Z file with the given header dims (the payload is never decoded)."""
    header = struct.pack("<IHBBIIQ", 0x5A315054, 1, 0, 0, m, n, nnz)
    return header + b"\0" * (96 - len(header)) + b"\0" * pad


def _write_sample(results: Path, gsm: str, *, n_called: int, exon: bytes) -> Path:
    out = results / gsm / "out"
    out.mkdir(parents=True)
    (out / "pileup_stats.json").write_text("{}")
    summary = {"n_cells_called": n_called, "reference_build": "GRCh38", "status": "success"}
    (out / "summary.json").write_text(json.dumps(summary))
    (out / "gene_expression.tsv").write_text("gene_id\tgene_name\nENSG01\tAAA\nENSG02\tBBB\n")
    (out / "cell_calls.tsv").write_text("barcode\tis_cell\nAAAC\tTrue\nAAAG\tFalse\n")
    (out / "exon_counts.1pz").write_bytes(exon)
    return out


def _catalog(*gsms: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gse_id": [GSE] * len(gsms),
            "gsm_id": list(gsms),
            "organism": ["Homo sapiens"] * len(gsms),
        }
    )


RESTRICTED = [
    "donor/snp_ad.1pz",
    "donor/snp_dp.1pz",
    "donor/donor0.vcf",
    "donor/donor0_coverage.tsv",
    "ase_counts.tsv",
    "mt/donor0_mt_consensus.fa",
    "mt/donor0_mt_variants.vcf",
    "mt/mt_events.1pz",
]
PUBLIC = [
    "exon_counts.1pz",
    "summary.json",
    "donor/donor_assignments.tsv",
    "mt/mt_summary.tsv",
    "mt_heteroplasmy.1pz",
    "mt_variants.tsv",
    "sex_call.json",
    "ancestry_call.json",
    "doublet_scores.tsv",
]
ANCESTRY = {
    "ancestry": "EUR",
    "confidence": 0.93,
    "n_informative_aims": 31,
    "n_covered_aims": 40,
    "aim_panel_size": 51,
    "populations": {"EUR": 0.93, "AFR": 0.02},
    "log_likelihoods": {"EUR": -12.5, "AFR": -40.1},
}


def _add_donor_outputs(out: Path) -> None:
    for rel in RESTRICTED:
        f = out / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(_pz_bytes(2, 2, 1) if rel.endswith(".1pz") else b"chr1\t100\tC\tT\t0/1\n")
    (out / "donor" / "donor_assignments.tsv").write_text("barcode\tdonor\nAAAC\tdonor0\n")
    (out / "mt").mkdir(exist_ok=True)
    (out / "mt" / "mt_summary.tsv").write_text("barcode\tdonor_id\nAAAC\tdonor0\n")
    (out / "sex_call.json").write_text('{"sex": "female"}')
    (out / "ancestry_call.json").write_text(json.dumps(ANCESTRY, indent=2))


@pytest.fixture
def results(tmp_path):
    r = tmp_path / "results"
    out = _write_sample(r, GOOD, n_called=1, exon=_pz_bytes(2, 2, 3))
    _add_donor_outputs(out)
    return r


def _names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as zf:
        return {n for n in zf.namelist() if not n.endswith("/")}


class TestRules:
    @pytest.mark.parametrize("rel", RESTRICTED)
    def test_restricted(self, rel):
        assert is_restricted_member(rel)
        assert is_restricted_path(f"samples/GSM1/{rel}")

    @pytest.mark.parametrize("rel", PUBLIC)
    def test_public(self, rel):
        assert not is_restricted_member(rel)
        assert not is_restricted_path(f"samples/GSM1/{rel}")

    def test_new_outputs_under_donor_and_mt_are_private(self):
        assert is_restricted_member("donor/anything_new.tsv")
        assert is_restricted_member("mt/anything_new.tsv")

    def test_private_dir_is_restricted(self):
        for rel in (
            "private/snp_ad.1pz",
            "private/ase_counts.tsv",
            "private/mt_allele_counts.mtx.gz",
            "private/donor/donor0.vcf",
            "private/mt/donor0_mt_consensus.fa",
        ):
            assert is_restricted_member(rel)
            assert is_restricted_path(f"samples/GSM1/{rel}")
        assert not is_restricted_member("donor_assignments.tsv")

    def test_study_level_files_are_public(self):
        for n in ("manifest.json", "study_meta.json", "feature_vocab.json"):
            assert not is_restricted_path(n)

    def test_ancestry_call_loses_the_likelihoods_only(self):
        out = json.loads(public_ancestry_call(json.dumps(ANCESTRY).encode()))
        assert "log_likelihoods" not in out
        assert (
            out["ancestry"] == "EUR"
            and out["confidence"] == 0.93
            and out["populations"]["EUR"] == 0.93
        )
        low = b'{"ancestry": "insufficient_data", "confidence": 0.0}'
        assert public_ancestry_call(low) == low
        assert public_ancestry_call(b"not json") == b"not json"


def _full_manifest(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read("manifest.json"))


class TestPack:
    def test_restricted_members_go_to_the_sidecar_only(self, results, tmp_path):
        pub, side = tmp_path / "pub.singlet", tmp_path / "side.singlet"
        bundle_mod.pack_gse(
            GSE,
            results,
            "unused",
            pub,
            verbose=False,
            private_out_path=side,
            _catalog_df=_catalog(GOOD),
        )
        pub_names, side_names = _names(pub), _names(side)
        assert not [n for n in pub_names if is_restricted_path(n)]
        for rel in RESTRICTED:
            assert f"samples/{GOOD}/{rel}" in side_names
            assert f"samples/{GOOD}/{rel}" not in pub_names
        # nothing is stored twice, except the trimmed vs untrimmed ancestry call and each file's manifest
        assert pub_names & side_names == {"manifest.json", f"samples/{GOOD}/ancestry_call.json"}
        # the public bundle verifies as it is; the sidecar verifies too and carries the original manifest
        assert verify_bundle(pub)["profile"] == PUBLIC_PROFILE
        sman = verify_bundle(side)
        assert sman["profile"] == RESTRICTED_PROFILE
        assert set(sman["original_manifest"]["checksums"]) == (pub_names | side_names) - {
            "manifest.json"
        }
        with zipfile.ZipFile(pub) as zf:
            assert "log_likelihoods" not in json.loads(
                zf.read(f"samples/{GOOD}/ancestry_call.json")
            )
        with zipfile.ZipFile(side) as zf:
            assert "log_likelihoods" in json.loads(zf.read(f"samples/{GOOD}/ancestry_call.json"))

    def test_without_a_private_path_they_are_not_packed_anywhere(self, results, tmp_path):
        pub = tmp_path / "pub.singlet"
        bundle_mod.pack_gse(GSE, results, "unused", pub, verbose=False, _catalog_df=_catalog(GOOD))
        assert not [n for n in _names(pub) if is_restricted_path(n)]
        assert verify_bundle(pub)["profile"] == PUBLIC_PROFILE

    def test_internal_readers_see_both_files_as_one_bundle(self, results, tmp_path):
        pub, side = tmp_path / "pub.singlet", tmp_path / "side.singlet"
        bundle_mod.pack_gse(
            GSE,
            results,
            "unused",
            pub,
            verbose=False,
            private_out_path=side,
            _catalog_df=_catalog(GOOD),
        )
        with bundle_mod.SingletBundle.open(pub) as b:
            assert not b.has("donor_snp_ad", GOOD) and not b.has("ase_counts", GOOD)
            assert b.has("exon_counts", GOOD)
        with bundle_mod.SingletBundle.open(pub, restricted=side) as b:
            assert b.has("donor_snp_ad", GOOD) and b.has("ase_counts", GOOD)
            assert "donor/donor0.vcf" in b.list_files(GOOD)
            assert b.has("exon_counts", GOOD)


def _complete(results, tmp_path) -> Path:
    """A bundle as the old packer wrote it: everything in out/, original manifest (rebuilt from public + sidecar)."""
    pub, side, full = tmp_path / "p.singlet", tmp_path / "s.singlet", tmp_path / "complete.singlet"
    bundle_mod.pack_gse(
        GSE,
        results,
        "unused",
        pub,
        verbose=False,
        private_out_path=side,
        _catalog_df=_catalog(GOOD),
    )
    restore_bundle(pub, side, full)
    for p in (pub, side):
        p.unlink()
    return full


class TestSplit:
    def test_restore_reproduces_the_complete_bundle(self, results, tmp_path):
        full = _complete(results, tmp_path)
        man = verify_bundle(full)  # all members against the original manifest
        assert "profile" not in man
        assert restricted_summary(full)["members"] == len(RESTRICTED)

    def test_split_then_restore_round_trips(self, results, tmp_path):
        full = _complete(results, tmp_path)
        before = full.read_bytes()
        pub, side, back = (
            tmp_path / "pub.singlet",
            tmp_path / "side.singlet",
            tmp_path / "back.singlet",
        )
        stats = redact_bundle(full, pub, sidecar=side)
        assert full.read_bytes() == before  # the source is never modified
        assert stats["removed"] == len(RESTRICTED) and stats["sidecar_bytes"] > 0
        assert not [n for n in _names(pub) if is_restricted_path(n)]
        assert verify_bundle(pub)["profile"] == PUBLIC_PROFILE
        assert verify_bundle(side)["profile"] == RESTRICTED_PROFILE
        restore_bundle(pub, side, back)
        assert verify_bundle(back) == _full_manifest(full)
        with zipfile.ZipFile(full) as a, zipfile.ZipFile(back) as b:
            assert {n: a.read(n) for n in a.namelist() if n != "manifest.json"} == {
                n: b.read(n) for n in b.namelist() if n != "manifest.json"
            }

    def test_a_corrupt_source_is_refused_and_leaves_nothing_behind(self, results, tmp_path):
        full = _complete(results, tmp_path)
        bad = tmp_path / "bad.singlet"
        with zipfile.ZipFile(full) as zin, zipfile.ZipFile(bad, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename.endswith("donor/snp_ad.1pz"):
                    data = data[:-1] + b"X"
                zout.writestr(info, data)
        with pytest.raises(RuntimeError, match="checksum"):
            redact_bundle(bad, tmp_path / "o.singlet", sidecar=tmp_path / "s.singlet")
        assert not list(tmp_path.glob("o.singlet*")) and not list(tmp_path.glob("s.singlet*"))

    def test_refuses_to_overwrite_its_source(self, results, tmp_path):
        full = _complete(results, tmp_path)
        with pytest.raises(ValueError):
            redact_bundle(full, full)


class TestMigrationScript:
    """scripts/restrict_published_bundles.py against directory-backed buckets."""

    @pytest.fixture
    def mig(self):
        import importlib.util

        path = Path(__file__).resolve().parents[2] / "scripts" / "restrict_published_bundles.py"
        spec = importlib.util.spec_from_file_location("restrict_published_bundles", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def _published(self, results, tmp_path) -> tuple:
        root = tmp_path / "r2"
        target = root / "singlet-data" / "data" / GSE / f"{GSE}.singlet"
        target.parent.mkdir(parents=True)
        shutil.move(str(_complete(results, tmp_path)), str(target))
        sidecar = root / "singlet-restricted" / "restricted" / GSE / f"{GSE}.restricted.singlet"
        base = [
            "--gse", GSE,
            "--private", "singlet-restricted",
            "--ledger", str(tmp_path / "ledger.jsonl"),
            "--local-root", str(root),
            "--scratch", str(tmp_path),
        ]  # fmt: skip
        return target, sidecar, base

    def test_migrate_then_restore(self, results, tmp_path, mig):
        target, sidecar, base = self._published(results, tmp_path)
        original = target.read_bytes()
        ledger = tmp_path / "ledger.jsonl"

        # a dry run writes nothing
        assert mig.main(["migrate", *base]) == 0
        assert target.read_bytes() == original and not sidecar.exists() and not ledger.exists()

        assert mig.main(["migrate", *base, "--apply"]) == 0
        assert not [n for n in _names(target) if is_restricted_path(n)]
        assert verify_bundle(target)["profile"] == PUBLIC_PROFILE
        assert verify_bundle(sidecar)["profile"] == RESTRICTED_PROFILE
        done = [json.loads(line) for line in ledger.read_text().splitlines()][-1]
        assert done["stage"] == "done" and done["ok"]

        # idempotent: a second pass finds nothing left to move
        assert mig.main(["migrate", *base, "--apply"]) == 0

        # public + sidecar read as one bundle
        with bundle_mod.SingletBundle.open(target, restricted=sidecar) as b:
            assert b.has("donor_snp_ad", GOOD)

        assert mig.main(["restore", *base, "--apply"]) == 0
        assert restricted_summary(target)["members"] == len(RESTRICTED)
        assert verify_bundle(target)

    def test_a_sidecar_that_does_not_read_back_leaves_the_public_object_alone(
        self, results, tmp_path, mig, monkeypatch
    ):
        target, sidecar, base = self._published(results, tmp_path)
        original = target.read_bytes()
        real = mig.LocalStore.upload

        def corrupting(self, path, bucket, key):
            real(self, path, bucket, key)
            if bucket == "singlet-restricted":
                with open(self._p(bucket, key), "r+b") as fh:
                    fh.seek(10)
                    fh.write(b"\xff\xff")

        monkeypatch.setattr(mig.LocalStore, "upload", corrupting)
        assert mig.main(["migrate", *base, "--apply"]) == 1
        assert target.read_bytes() == original


class TestMigrationSelection:
    def test_all_and_shard_cover_every_study_once_in_accession_order(self, tmp_path):
        import importlib.util

        path = Path(__file__).resolve().parents[2] / "scripts" / "restrict_published_bundles.py"
        spec = importlib.util.spec_from_file_location("restrict_published_bundles", path)
        mig = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mig)

        root = tmp_path / "r2"
        sizes = {"GSE5": 10, "GSE10": 20, "GSE3": 30, "GSE2": 50, "GSE4": 40}
        for gse, n in sizes.items():
            f = root / "singlet-data" / "data" / gse / f"{gse}.singlet"
            f.parent.mkdir(parents=True)
            f.write_bytes(b"x" * n)
        store = mig.LocalStore(root)

        class A:
            gse = None
            gse_list = None
            public = "singlet-data"
            shard = None

        assert list(mig.gses_from(A, store)) == ["GSE2", "GSE3", "GSE4", "GSE5", "GSE10"]
        parts = []
        for i in range(2):
            A.shard = f"{i}/2"
            parts.append(list(mig.gses_from(A, store)))
        assert sorted(parts[0] + parts[1]) == sorted(sizes) and not set(parts[0]) & set(parts[1])
