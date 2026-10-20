"""Members of a ``.singlet`` bundle that are not part of the public product.

Some pipeline outputs carry per-donor germline information: genotype calls, read depth over a SNP panel,
allele-specific counts, donor mitochondrial sequence. The pipeline still computes them and the packer still
writes them, but into a *private* bundle. The public bundle leaves them out.

The rules are an allowlist per directory, so a new output under ``donor/`` or ``mt/`` is private until someone
decides otherwise. Paths are relative to ``samples/<GSM>/``. ``functions/_shared/restricted.ts`` in
singlet-website applies the same rules to what the site lists and serves; keep the two in step.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Union

# Stays public inside donor/ (barcode → donor label) and inside mt/ (per-cell event counts).
_DONOR_PUBLIC = frozenset({"donor_assignments.tsv"})
_MT_PUBLIC = frozenset({"mt_summary.tsv"})
_SAMPLE_RE = re.compile(r"^samples/GSM\d+/")
_VARIANT_FILE_RE = re.compile(r"\.(vcf|bcf)(\.gz)?$", re.IGNORECASE)

#: Written into the manifest of a public bundle.
PUBLIC_PROFILE = "public-1"
#: Written into the manifest of the private bundle (the complete one).
FULL_PROFILE = "full-1"


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


def _sha256_stream(fh, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: fh.read(chunk), b""):
        h.update(block)
    return h.hexdigest()


def redact_bundle(
    src: Union[str, Path],
    dst: Union[str, Path],
    *,
    verify: bool = True,
) -> Dict[str, int]:
    """Write the public form of the bundle ``src`` to ``dst``; ``src`` is not touched.

    Restricted members are left out, ``ancestry_call.json`` is replaced by its public form, and
    ``manifest.json`` is rewritten so that ``included_files`` and ``checksums`` describe exactly what ``dst``
    holds. Members are streamed, never held whole in memory. With ``verify`` the written archive is read back:
    every member against its checksum, and nothing restricted left.

    Returns counts: ``kept``, ``removed``, ``removed_bytes`` (compressed) and ``bytes_in`` / ``bytes_out``.
    """
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve():
        raise ValueError("dst must not be src")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    stats = {
        "kept": 0,
        "removed": 0,
        "removed_bytes": 0,
        "bytes_in": src.stat().st_size,
        "bytes_out": 0,
    }
    checksums: Dict[str, str] = {}

    with zipfile.ZipFile(src, "r") as zin:
        old = json.loads(zin.read("manifest.json"))
        with zipfile.ZipFile(tmp, "w", allowZip64=True) as zout:
            for info in zin.infolist():
                name = info.filename
                if name.endswith("/") or name == "manifest.json":
                    continue
                if is_restricted_path(name):
                    stats["removed"] += 1
                    stats["removed_bytes"] += info.compress_size
                    continue
                zi = zipfile.ZipInfo(name, date_time=info.date_time)
                zi.compress_type = info.compress_type
                if name.endswith("/ancestry_call.json") or name == "ancestry_call.json":
                    data = public_ancestry_call(zin.read(name))
                    checksums[name] = hashlib.sha256(data).hexdigest()
                    zout.writestr(zi, data)
                else:
                    h = hashlib.sha256()
                    with zin.open(info) as fin, zout.open(zi, "w", force_zip64=True) as fout:
                        for block in iter(lambda: fin.read(8 * 1024 * 1024), b""):
                            h.update(block)
                            fout.write(block)
                    checksums[name] = h.hexdigest()
                stats["kept"] += 1

            included: Dict[str, list] = {}
            for name in checksums:
                m = _SAMPLE_RE.match(name)
                if m:
                    gsm = name[len("samples/") : m.end() - 1]
                    included.setdefault(gsm, []).append(name[m.end() :])
            manifest = dict(old)
            manifest["included_files"] = {
                g: sorted(included.get(g, [])) for g in old.get("gsm_ids", included)
            }
            manifest["checksums"] = checksums
            manifest["profile"] = PUBLIC_PROFILE
            zout.writestr(
                zipfile.ZipInfo("manifest.json"),
                json.dumps(manifest, indent=2).encode(),
                compress_type=zipfile.ZIP_DEFLATED,
            )

    if verify:
        with zipfile.ZipFile(tmp, "r") as z:
            bad = z.testzip()
            if bad:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"redacted archive is corrupt at {bad}")
            names = [n for n in z.namelist() if not n.endswith("/")]
            leaked = [n for n in names if is_restricted_path(n)]
            if leaked:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"redacted archive still holds {leaked[0]}")
            man = json.loads(z.read("manifest.json"))
            listed = set(man["checksums"])
            if listed != {n for n in names if n != "manifest.json"}:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    "manifest.json does not match the members of the redacted archive"
                )
            for name, digest in man["checksums"].items():
                with z.open(name) as fh:
                    if _sha256_stream(fh) != digest:
                        tmp.unlink(missing_ok=True)
                        raise RuntimeError(f"{name} does not match its checksum")

    shutil.move(str(tmp), str(dst))
    stats["bytes_out"] = dst.stat().st_size
    return stats


def restricted_summary(src: Union[str, Path]) -> Dict[str, Optional[int]]:
    """How many members and compressed bytes of ``src`` are restricted (read-only; central directory only)."""
    n = size = 0
    with zipfile.ZipFile(src, "r") as z:
        for info in z.infolist():
            if not info.filename.endswith("/") and is_restricted_path(info.filename):
                n += 1
                size += info.compress_size
    return {"members": n, "compressed_bytes": size}
