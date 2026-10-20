# SPDX-License-Identifier: MIT
"""Per-donor germline members stay out of the public bundle: the rules, the packer and redact_bundle.

Like test_pack_gse.py this builds synthetic pipeline output and needs no compiled codec.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest
from singlet import bundle as bundle_mod
from singlet._integrity import verify_bundle
from singlet._restricted import (
    FULL_PROFILE,
    PUBLIC_PROFILE,
    is_restricted_member,
    is_restricted_path,
    public_ancestry_call,
    redact_bundle,
    restricted_summary,
)
from test_pack_gse import GOOD, GSE, _catalog, _pz_bytes, _write_sample

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


class TestPack:
    def test_public_bundle_never_holds_restricted_members(self, results, tmp_path):
        pub, priv = tmp_path / "pub.singlet", tmp_path / "priv.singlet"
        bundle_mod.pack_gse(
            GSE,
            results,
            "unused",
            pub,
            verbose=False,
            private_out_path=priv,
            _catalog_df=_catalog(GOOD),
        )
        pub_names, priv_names = _names(pub), _names(priv)
        assert not [n for n in pub_names if is_restricted_path(n)]
        for rel in RESTRICTED:
            assert f"samples/{GOOD}/{rel}" in priv_names
            assert f"samples/{GOOD}/{rel}" not in pub_names
        for rel in PUBLIC:
            if (results / GOOD / "out" / rel).exists():
                assert (
                    f"samples/{GOOD}/{rel}" in pub_names and f"samples/{GOOD}/{rel}" in priv_names
                )
        # both pass the client's integrity check, and say which one they are
        assert verify_bundle(pub)["profile"] == PUBLIC_PROFILE
        assert verify_bundle(priv)["profile"] == FULL_PROFILE
        with zipfile.ZipFile(pub) as zf:
            assert "log_likelihoods" not in json.loads(
                zf.read(f"samples/{GOOD}/ancestry_call.json")
            )
        with zipfile.ZipFile(priv) as zf:
            assert "log_likelihoods" in json.loads(zf.read(f"samples/{GOOD}/ancestry_call.json"))

    def test_without_a_private_path_they_are_not_packed_anywhere(self, results, tmp_path):
        pub = tmp_path / "pub.singlet"
        bundle_mod.pack_gse(GSE, results, "unused", pub, verbose=False, _catalog_df=_catalog(GOOD))
        assert not [n for n in _names(pub) if is_restricted_path(n)]
        assert verify_bundle(pub)["profile"] == PUBLIC_PROFILE


class TestRedact:
    def _legacy(self, results, tmp_path) -> Path:
        """A bundle as the old packer wrote it: everything in out/ included."""
        old = tmp_path / "old.singlet"
        priv = tmp_path / "priv.singlet"
        bundle_mod.pack_gse(
            GSE,
            results,
            "unused",
            tmp_path / "ignored.singlet",
            verbose=False,
            private_out_path=priv,
            _catalog_df=_catalog(GOOD),
        )
        # the private bundle is the old layout apart from its profile marker
        with zipfile.ZipFile(priv) as zin, zipfile.ZipFile(old, "w", zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "manifest.json":
                    m = json.loads(data)
                    m.pop("profile")
                    data = json.dumps(m).encode()
                zout.writestr(info, data)
        return old

    def test_redacted_copy_matches_a_public_pack(self, results, tmp_path):
        old = self._legacy(results, tmp_path)
        before = old.read_bytes()
        assert restricted_summary(old)["members"] == len(RESTRICTED)
        dst = tmp_path / "red.singlet"
        stats = redact_bundle(old, dst)
        assert stats["removed"] == len(RESTRICTED) and stats["kept"] > 0
        assert old.read_bytes() == before  # the source is never modified
        assert not [n for n in _names(dst) if is_restricted_path(n)]
        man = verify_bundle(dst)  # checksums of every member verified
        assert man["profile"] == PUBLIC_PROFILE
        assert man["gsm_ids"] == [GOOD]
        assert set(man["included_files"][GOOD]) == {
            n.split("/", 2)[2] for n in _names(dst) if n.startswith("samples/")
        }
        with zipfile.ZipFile(dst) as zf:
            assert "log_likelihoods" not in json.loads(
                zf.read(f"samples/{GOOD}/ancestry_call.json")
            )

    def test_refuses_to_overwrite_its_source(self, results, tmp_path):
        old = self._legacy(results, tmp_path)
        with pytest.raises(ValueError):
            redact_bundle(old, old)


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

    def _published(self, results, root: Path) -> Path:
        priv = root / "tmp.singlet"
        bundle_mod.pack_gse(
            GSE,
            results,
            "unused",
            root / "ignored.singlet",
            verbose=False,
            private_out_path=priv,
            _catalog_df=_catalog(GOOD),
        )
        target = root / "singlet-data" / "data" / GSE / f"{GSE}.singlet"
        target.parent.mkdir(parents=True)
        # legacy layout: the complete bundle, no profile marker
        with zipfile.ZipFile(priv) as zin, zipfile.ZipFile(target, "w") as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename == "manifest.json":
                    m = json.loads(data)
                    m.pop("profile")
                    data = json.dumps(m).encode()
                zout.writestr(info, data)
        priv.unlink()
        return target

    def test_copy_then_redact_then_rollback(self, results, tmp_path, mig):
        root = tmp_path / "r2"
        target = self._published(results, root)
        original = target.read_bytes()
        ledger = tmp_path / "ledger.jsonl"
        base = [
            "--gse",
            GSE,
            "--private",
            "singlet-private",
            "--ledger",
            str(ledger),
            "--local-root",
            str(root),
            "--scratch",
            str(tmp_path),
        ]

        # a redact run before any verified copy changes nothing
        assert mig.main(["redact", *base, "--apply"]) == 1
        assert target.read_bytes() == original

        # dry runs write nothing
        assert mig.main(["copy", *base]) == 0
        assert not (root / "singlet-private").exists() and not ledger.exists()

        assert mig.main(["copy", *base, "--apply"]) == 0
        assert (root / "singlet-private" / "full" / GSE / f"{GSE}.singlet").read_bytes() == original
        assert target.read_bytes() == original  # public untouched by the copy phase

        assert mig.main(["redact", *base, "--apply"]) == 0
        assert not [n for n in _names(target) if is_restricted_path(n)]
        assert verify_bundle(target)["profile"] == PUBLIC_PROFILE
        # idempotent: a second pass finds nothing to do
        assert mig.main(["redact", *base, "--apply"]) == 0

        assert mig.main(["rollback", *base, "--apply"]) == 0
        assert target.read_bytes() == original
