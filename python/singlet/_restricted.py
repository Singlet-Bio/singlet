"""Members of a ``.singlet`` bundle that are not part of the public product.

Some pipeline outputs carry per-donor germline information: genotype calls, read depth over a SNP panel,
allele-specific counts, donor mitochondrial sequence. The pipeline still computes them. The packer writes them to a
*restricted sidecar* in private storage and leaves them out of the public bundle, so the two files together hold
exactly what the old single bundle held (no member is stored twice, so storage does not grow).

The rules are an allowlist per directory, so a new output under ``donor/`` or ``mt/`` is private until someone
decides otherwise. Paths are relative to ``samples/<GSM>/``. ``functions/_shared/restricted.ts`` in
singlet-website applies the same rules to what the site lists and serves; keep the two in step.

Sidecar layout (a ``.singlet``-style zip, same member paths as the original, ``manifest.json`` profile
``restricted-1``)::

    manifest.json     checksums of the sidecar's members + ``original_manifest`` (the manifest of the complete bundle)
    samples/<GSM>/…   the restricted members, byte for byte, and the untrimmed ``ancestry_call.json``
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Stays public inside donor/ (barcode → donor label) and inside mt/ (per-cell event counts).
_DONOR_PUBLIC = frozenset({"donor_assignments.tsv"})
_MT_PUBLIC = frozenset({"mt_summary.tsv"})
_SAMPLE_RE = re.compile(r"^samples/GSM\d+/")
_VARIANT_FILE_RE = re.compile(r"\.(vcf|bcf)(\.gz)?$", re.IGNORECASE)
_CHUNK = 8 * 1024 * 1024

#: ``profile`` in the manifest of a public bundle.
PUBLIC_PROFILE = "public-1"
#: ``profile`` in the manifest of a restricted sidecar.
RESTRICTED_PROFILE = "restricted-1"


def is_restricted_member(rel: str) -> bool:
    """True when ``rel`` (a path inside ``samples/<GSM>/``) is held back from the public bundle."""
    rel = rel.replace("\\", "/")
    if rel.startswith("donor/"):
        return rel[len("donor/") :] not in _DONOR_PUBLIC
    if rel.startswith("mt/"):
        return rel[len("mt/") :] not in _MT_PUBLIC
    if rel == "ase_counts.tsv":
        return True
    name = rel.rsplit("/", 1)[-1]
    return name.startswith("snp_") or bool(_VARIANT_FILE_RE.search(name))


def is_restricted_path(arc_name: str) -> bool:
    """:func:`is_restricted_member` for a full archive path (``samples/<GSM>/…``)."""
    m = _SAMPLE_RE.match(arc_name)
    if m:
        return is_restricted_member(arc_name[m.end() :])
    if arc_name.startswith("samples/"):
        return False
    return is_restricted_member(arc_name)


def is_ancestry_call(arc_name: str) -> bool:
    return arc_name.rsplit("/", 1)[-1] == "ancestry_call.json"


_MT_ALLELE_RE = re.compile(r"_mt_(consensus|variants)")
_ALPHA_PRIVATE = frozenset(
    {"ase_counts.tsv", "mt_heteroplasmy.1pz", "mt_variants.tsv", "ancestry_call.json"}
)


class AlphaPolicy:
    """What a Research Alpha public bundle may hold; mirrors ``classify`` in the data-guard's ``policy.ts``.

    ``withheld_studies`` are not served at all; ``excluded_samples`` (GSMs) are left out of every bundle.
    """

    def __init__(
        self, withheld_studies=(), excluded_samples=(), version: Optional[str] = None
    ) -> None:
        self.withheld = frozenset(withheld_studies)
        self.excluded = frozenset(excluded_samples)
        self.version = version

    @classmethod
    def load(cls, path: Union[str, Path]) -> "AlphaPolicy":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d.get("withheld_studies", []), d.get("excluded_samples", []), d.get("version"))

    @staticmethod
    def is_private_member(rel: str) -> bool:
        """Germline or mtDNA allele data, for a path relative to ``samples/<GSM>/`` (or a bare member name)."""
        if rel.startswith("donor/"):
            return rel[len("donor/") :] not in _DONOR_PUBLIC
        if rel.startswith("mt/"):
            return rel[len("mt/") :] not in _MT_PUBLIC
        if rel in _ALPHA_PRIVATE:
            return True
        name = rel.rsplit("/", 1)[-1]
        return name.startswith("snp_") or bool(
            _VARIANT_FILE_RE.search(name) or _MT_ALLELE_RE.search(name)
        )

    def classify(self, arc_name: str) -> str:
        """``"excluded"``, ``"private"`` or ``"ok"`` for a full archive path."""
        m = _SAMPLE_RE.match(arc_name)
        if m and _gsm_of(arc_name) in self.excluded:
            return "excluded"
        rel = arc_name[m.end() :] if m else arc_name
        return "private" if self.is_private_member(rel) else "ok"

    def public_manifest(self, man: dict) -> dict:
        """``man`` without the excluded samples (``gsm_ids``, ``n_gsms``, ``excluded_samples``)."""
        out = dict(man)
        out["gsm_ids"] = [g for g in man.get("gsm_ids", []) if g not in self.excluded]
        out["n_gsms"] = len(out["gsm_ids"])
        if isinstance(man.get("excluded_samples"), list):
            out["excluded_samples"] = [
                e
                for e in man["excluded_samples"]
                if not (isinstance(e, dict) and e.get("gsm_id") in self.excluded)
            ]
        return out

    def public_study_meta(self, data: bytes) -> bytes:
        """``study_meta.json`` without the excluded samples' ``gsm_meta`` entries (``data`` itself if none)."""
        try:
            meta = json.loads(data)
        except ValueError:
            return data
        gm = meta.get("gsm_meta") if isinstance(meta, dict) else None
        if not isinstance(gm, dict) or not set(gm) & self.excluded:
            return data
        meta["gsm_meta"] = {g: v for g, v in gm.items() if g not in self.excluded}
        return json.dumps(meta, indent=2).encode()

    def listed_excluded(self, man: dict, names) -> List[str]:
        """Excluded GSMs that ``man`` lists or that have members among ``names``."""
        seen = set(man.get("gsm_ids", []))
        seen.update(
            e.get("gsm_id") for e in man.get("excluded_samples") or [] if isinstance(e, dict)
        )
        seen.update(_gsm_of(n) for n in names)
        return sorted(g for g in seen if g in self.excluded)


def public_ancestry_call(data: bytes) -> bytes:
    """The public form of ``ancestry_call.json``: the group, its confidence and the coverage counts.

    The per-population log-likelihoods are sums over the individual panel genotypes, so they are dropped.
    """
    try:
        call = json.loads(data)
    except ValueError:
        return data
    if not isinstance(call, dict) or "log_likelihoods" not in call:
        return data
    call.pop("log_likelihoods")
    return (json.dumps(call, indent=2) + "\n").encode()


def _sha256_stream(fh) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: fh.read(_CHUNK), b""):
        h.update(block)
    return h.hexdigest()


def _gsm_of(arc_name: str) -> Optional[str]:
    m = _SAMPLE_RE.match(arc_name)
    return arc_name[len("samples/") : m.end() - 1] if m else None


def _included(checksums: Dict[str, str], gsm_ids: List[str]) -> Dict[str, list]:
    out: Dict[str, list] = {g: [] for g in gsm_ids}
    for name in checksums:
        m = _SAMPLE_RE.match(name)
        if m:
            out.setdefault(_gsm_of(name) or "", []).append(name[m.end() :])
    return {g: sorted(v) for g, v in out.items()}


def sidecar_manifest(original: dict, checksums: Dict[str, str]) -> dict:
    """``manifest.json`` for a sidecar holding ``checksums``, remembering ``original`` (the complete bundle's)."""
    gsm_ids = list(original.get("gsm_ids", []))
    return {
        "schema_version": original.get("schema_version", "1.0"),
        "gse_id": original.get("gse_id"),
        "gsm_ids": gsm_ids,
        "n_gsms": len(gsm_ids),
        "reference_build": original.get("reference_build"),
        "created_at": original.get("created_at"),
        "singlet_version": original.get("singlet_version"),
        "profile": RESTRICTED_PROFILE,
        "included_files": _included(checksums, gsm_ids),
        "checksums": checksums,
        "original_manifest": original,
    }


def _copy_member(
    zin: zipfile.ZipFile, info: zipfile.ZipInfo, zout: zipfile.ZipFile, name: str
) -> str:
    """Stream one member across without recompressing more than its own method needs; returns its SHA-256."""
    zi = zipfile.ZipInfo(name, date_time=info.date_time)
    zi.compress_type = info.compress_type
    h = hashlib.sha256()
    with zin.open(info) as fin, zout.open(zi, "w", force_zip64=True) as fout:
        for block in iter(lambda: fin.read(_CHUNK), b""):
            h.update(block)
            fout.write(block)
    return h.hexdigest()


def _put_bytes(zout: zipfile.ZipFile, name: str, data: bytes, compress_type: int) -> str:
    zi = zipfile.ZipInfo(name)
    zi.compress_type = compress_type
    zout.writestr(zi, data)
    return hashlib.sha256(data).hexdigest()


def redact_bundle(
    src: Union[str, Path],
    dst: Union[str, Path],
    *,
    sidecar: Union[str, Path, None] = None,
    verify: bool = True,
    alpha: Optional[AlphaPolicy] = None,
) -> Dict[str, int]:
    """Split the complete bundle ``src`` into a public bundle ``dst`` and, with ``sidecar``, a restricted sidecar.

    ``src`` is not touched. Restricted members go to the sidecar, ``ancestry_call.json`` is trimmed in ``dst`` (the
    original goes to the sidecar), and ``manifest.json`` is rewritten in both. Members are streamed. Every member
    read from ``src`` is checked against the checksum in its manifest, so a corrupt source is refused rather than
    split. With ``verify`` both outputs are read back: every member against its checksum, nothing restricted in
    ``dst``, and public ∪ sidecar equal to the original member for member.

    Without ``sidecar`` the restricted members are dropped (the public form only).

    With ``alpha`` the Research Alpha rules decide what is private (``ancestry_call.json`` goes to the sidecar whole),
    every member of an excluded sample goes to the sidecar, and the public ``manifest.json`` and ``study_meta.json`` no
    longer list those samples (the original ``study_meta.json`` goes to the sidecar).

    Returns counts: ``kept``, ``removed``, ``removed_bytes`` (compressed), ``bytes_in``, ``bytes_out``,
    ``sidecar_bytes``, ``excluded_samples``.
    """
    src, dst = Path(src), Path(dst)
    sidecar = Path(sidecar) if sidecar is not None else None
    if len({src.resolve(), dst.resolve(), *([sidecar.resolve()] if sidecar else [])}) != (
        3 if sidecar else 2
    ):
        raise ValueError("src, dst and sidecar must be different files")
    for p in (dst, sidecar):
        if p is not None:
            p.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    stmp = sidecar.with_name(sidecar.name + ".part") if sidecar else None
    stats = {
        "kept": 0,
        "removed": 0,
        "removed_bytes": 0,
        "bytes_in": src.stat().st_size,
        "bytes_out": 0,
        "sidecar_bytes": 0,
        "excluded_samples": 0,
    }
    pub_sums: Dict[str, str] = {}
    res_sums: Dict[str, str] = {}

    def _discard() -> None:
        for p in (tmp, stmp):
            if p is not None:
                p.unlink(missing_ok=True)

    try:
        with zipfile.ZipFile(src, "r") as zin:
            old = json.loads(zin.read("manifest.json"))
            vouched = old.get("checksums") or {}
            if alpha is not None:
                stats["excluded_samples"] = len(alpha.listed_excluded(old, zin.namelist()))
                if old.get("gsm_ids") and not alpha.public_manifest(old)["gsm_ids"]:
                    raise ValueError(
                        f"{src.name}: every sample is excluded; the study should be withheld"
                    )
            with zipfile.ZipFile(tmp, "w", allowZip64=True) as zpub:
                zres = zipfile.ZipFile(stmp, "w", allowZip64=True) if stmp else None
                try:
                    for info in zin.infolist():
                        name = info.filename
                        if name.endswith("/") or name == "manifest.json":
                            continue
                        private = (
                            alpha.classify(name) != "ok"
                            if alpha is not None
                            else is_restricted_path(name)
                        )
                        if alpha is not None and name == "study_meta.json":
                            data = zin.read(name)
                            digest = hashlib.sha256(data).hexdigest()
                            pub = alpha.public_study_meta(data)
                            pub_sums[name] = _put_bytes(zpub, name, pub, info.compress_type)
                            if pub != data and zres is not None:
                                res_sums[name] = _put_bytes(zres, name, data, info.compress_type)
                            stats["kept"] += 1
                        elif private:
                            stats["removed"] += 1
                            stats["removed_bytes"] += info.compress_size
                            if zres is not None:
                                res_sums[name] = _copy_member(zin, info, zres, name)
                                digest = res_sums[name]
                            else:
                                with zin.open(info) as fh:
                                    digest = _sha256_stream(fh)
                        elif is_ancestry_call(name):
                            data = zin.read(name)
                            digest = hashlib.sha256(data).hexdigest()
                            pub = public_ancestry_call(data)
                            pub_sums[name] = _put_bytes(zpub, name, pub, info.compress_type)
                            if pub != data and zres is not None:
                                res_sums[name] = _put_bytes(zres, name, data, info.compress_type)
                            stats["kept"] += 1
                        else:
                            pub_sums[name] = _copy_member(zin, info, zpub, name)
                            digest = pub_sums[name]
                            stats["kept"] += 1
                        if name in vouched and vouched[name] != digest:
                            raise RuntimeError(
                                f"{src.name}: {name} does not match its checksum in the source manifest"
                            )
                    pub_man = alpha.public_manifest(old) if alpha is not None else dict(old)
                    gsm_ids = list(pub_man.get("gsm_ids", []))
                    pub_man["included_files"] = _included(pub_sums, gsm_ids)
                    pub_man["checksums"] = pub_sums
                    pub_man["profile"] = PUBLIC_PROFILE
                    if alpha is not None and alpha.version:
                        pub_man["alpha_policy"] = alpha.version
                    zpub.writestr(
                        zipfile.ZipInfo("manifest.json"),
                        json.dumps(pub_man, indent=2).encode(),
                        compress_type=zipfile.ZIP_DEFLATED,
                    )
                    if zres is not None:
                        zres.writestr(
                            zipfile.ZipInfo("manifest.json"),
                            json.dumps(sidecar_manifest(old, res_sums), indent=2).encode(),
                            compress_type=zipfile.ZIP_DEFLATED,
                        )
                finally:
                    if zres is not None:
                        zres.close()
        if verify:
            _verify_pair(tmp, stmp, old, alpha)
    except BaseException:
        _discard()
        raise

    shutil.move(str(tmp), str(dst))
    stats["bytes_out"] = dst.stat().st_size
    if sidecar is not None and stmp is not None:
        shutil.move(str(stmp), str(sidecar))
        stats["sidecar_bytes"] = sidecar.stat().st_size
    return stats


def _check_zip(path: Path, expect_profile: str) -> dict:
    """Read every member of ``path`` back against the checksums in its own manifest; returns the manifest."""
    with zipfile.ZipFile(path, "r") as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"{path.name} is corrupt at {bad}")
        man = json.loads(z.read("manifest.json"))
        if man.get("profile") != expect_profile:
            raise RuntimeError(f"{path.name} is not a {expect_profile} archive")
        names = {n for n in z.namelist() if not n.endswith("/") and n != "manifest.json"}
        if set(man["checksums"]) != names:
            raise RuntimeError(f"manifest.json of {path.name} does not match its members")
        for name, digest in man["checksums"].items():
            with z.open(name) as fh:
                if _sha256_stream(fh) != digest:
                    raise RuntimeError(f"{name} in {path.name} does not match its checksum")
    return man


def _verify_pair(
    public: Path, sidecar: Optional[Path], original: dict, alpha: Optional[AlphaPolicy] = None
) -> None:
    pub = _check_zip(public, PUBLIC_PROFILE)
    if alpha is not None:
        leaked = [n for n in pub["checksums"] if alpha.classify(n) != "ok"]
        with zipfile.ZipFile(public) as z:
            meta = (
                json.loads(z.read("study_meta.json"))
                if "study_meta.json" in pub["checksums"]
                else {}
            )
        listed = set(alpha.listed_excluded(pub, pub["checksums"]))
        listed |= (
            set(pub.get("included_files") or {}) | set(meta.get("gsm_meta") or {})
        ) & alpha.excluded
        if listed:
            raise RuntimeError(f"public archive still lists excluded sample {sorted(listed)[0]}")
    else:
        leaked = [n for n in pub["checksums"] if is_restricted_path(n)]
    if leaked:
        raise RuntimeError(f"public archive still holds {leaked[0]}")
    if sidecar is None:
        return
    res = _check_zip(sidecar, RESTRICTED_PROFILE)
    want = original.get("checksums") or {}
    if not want:
        return
    # public ∪ sidecar (sidecar wins, it holds the untrimmed ancestry call) must be the original, member for member.
    merged = {**pub["checksums"], **res["checksums"]}
    if merged != want:
        missing = sorted(set(want) - set(merged))
        extra = sorted(set(merged) - set(want))
        differ = sorted(n for n in set(merged) & set(want) if merged[n] != want[n])
        raise RuntimeError(
            f"public + sidecar differ from the original (missing {missing[:3]}, extra {extra[:3]}, differing {differ[:3]})"
        )


def restore_bundle(
    public: Union[str, Path], sidecar: Union[str, Path], dst: Union[str, Path]
) -> Dict[str, int]:
    """Rebuild the complete bundle from a public bundle and its sidecar (members and manifest as they were).

    The result is not byte-identical to the file that was split (member order and timestamps differ) but holds
    the same members with the same checksums and the original ``manifest.json``; it is verified before it is
    moved into place.
    """
    public, sidecar, dst = Path(public), Path(sidecar), Path(dst)
    tmp = dst.with_name(dst.name + ".part")
    n = 0
    try:
        with zipfile.ZipFile(public) as zp, zipfile.ZipFile(sidecar) as zr:
            res_man = json.loads(zr.read("manifest.json"))
            original = res_man["original_manifest"]
            in_sidecar = set(res_man["checksums"])
            with zipfile.ZipFile(tmp, "w", allowZip64=True) as zout:
                for info in zp.infolist():
                    if info.filename.endswith("/") or info.filename in (
                        "manifest.json",
                        *in_sidecar,
                    ):
                        continue
                    _copy_member(zp, info, zout, info.filename)
                    n += 1
                for info in zr.infolist():
                    if info.filename == "manifest.json" or info.filename.endswith("/"):
                        continue
                    _copy_member(zr, info, zout, info.filename)
                    n += 1
                zout.writestr(
                    zipfile.ZipInfo("manifest.json"),
                    json.dumps(original, indent=2).encode(),
                    compress_type=zipfile.ZIP_DEFLATED,
                )
        with zipfile.ZipFile(tmp) as z:
            names = {x for x in z.namelist() if not x.endswith("/") and x != "manifest.json"}
            want: Dict[str, Any] = original.get("checksums") or {}
            if want and names != set(want):
                raise RuntimeError(
                    "restored archive does not hold the members the original manifest lists"
                )
            for name, digest in want.items():
                with z.open(name) as fh:
                    if _sha256_stream(fh) != digest:
                        raise RuntimeError(f"{name} does not match its checksum after restore")
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    shutil.move(str(tmp), str(dst))
    return {"members": n, "bytes_out": dst.stat().st_size}


def restricted_summary(
    src: Union[str, Path, Any], alpha: Optional[AlphaPolicy] = None
) -> Dict[str, Optional[int]]:
    """How many members and compressed bytes of ``src`` are restricted (read-only; central directory only).

    With ``alpha`` it uses the Research Alpha rules and also reads ``manifest.json`` to count the excluded samples the
    bundle lists (``excluded_samples``).
    """
    n = size = 0
    out: Dict[str, Optional[int]] = {}
    with zipfile.ZipFile(src, "r") as z:
        for info in z.infolist():
            name = info.filename
            if name.endswith("/"):
                continue
            if alpha.classify(name) != "ok" if alpha is not None else is_restricted_path(name):
                n += 1
                size += info.compress_size
        if alpha is not None:
            man = json.loads(z.read("manifest.json")) if "manifest.json" in z.namelist() else {}
            out["excluded_samples"] = len(alpha.listed_excluded(man, z.namelist()))
    return {"members": n, "compressed_bytes": size, **out}
