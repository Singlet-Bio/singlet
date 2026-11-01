# SPDX-License-Identifier: MIT
"""Release policy applied while a legacy bundle is transcoded to .s1 objects (Research Alpha, docs/s1-migration.md, Research Alpha).

A policy file is JSON::

    {"version": "alpha-2026-10-26b", "withheld_studies": ["GSE..."], "excluded_samples": ["GSM..."], ...}

What it does to one study:

* a withheld study is not transcoded at all (:meth:`Policy.withholds`);
* an excluded sample is dropped before anything is read: no .s1 object, no manifest entry, no ledger line;
* members of the layers in :data:`OMITTED_LAYERS` are dropped from every public object (:func:`omitted_layer`);
* the study files that list samples or members (``manifest.json``, ``study_meta.json``) are rewritten without the dropped
  samples and members, so no excluded accession reaches a public object.

The .s1 manifest records only ``{"name", "version", "omitted_layers"}``: the policy version and the generic names of the layers it
withholds, never an excluded accession and never a per-study count of excluded samples. What was dropped per study is returned to
the caller (:class:`PolicyReport`) for the private audit trail.

:func:`apply_to_bundle` filters a :class:`~singlet.s1.legacy.LegacyBundle` in place, and :class:`FilteredLegacy` gives the same
view to the verifier, so the transcoder and the verifier compare exactly the members the policy keeps; every kept member is the
legacy member byte for byte (the rewritten study files are the only exception, and they are checked to be the legacy JSON minus
the dropped entries).
"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

#: generic layer names recorded in the manifest, in a fixed order
OMITTED_LAYERS = (
    "genotype",  # donor/* except donor_assignments.tsv, snp_*, *.vcf/.bcf, ase_counts.tsv, aligned reads (BAM/CRAM)
    "mtdna_alleles",  # mt/* (mt_summary.tsv included: its per-cell mts_* columns), mt_heteroplasmy.1pz, mt_variants.tsv, *_mt_consensus*, *_mt_variants*
    "ancestry_call",  # ancestry_call.json (50-marker call; replaced by the validated per-cell label)
    "ambient",  # ambient_contamination.*, ambient_profile.tsv
    "doublet_scores",  # doublet_scores.tsv, and the doublet columns of cell_qc_metrics.tsv (qc_doublet_score)
    "nonhost_per_cell",  # nonhost/*per_cell* (the sample-level nonhost summary and abundance stay)
    "splice_psi_legacy",  # splice_psi.1pz (0/1 values; clients compute true PSI from sj_counts, D8)
    "sex_call_legacy",  # sex_call.json (legacy pipeline call; replaced by the validated per-cell sex_chrom labels)
)

_VARIANT = re.compile(r"\.(vcf|bcf)(\.gz)?$", re.IGNORECASE)
_READS = re.compile(r"\.(bam|bai|cram|crai|sam)$", re.IGNORECASE)
_GSM = re.compile(r"GSM\d+")


def omitted_layer(rel: str) -> Optional[str]:
    """The layer a sample member (path inside ``samples/<GSM>/``) belongs to when the policy withholds it, else None.

    This is a superset of :func:`singlet._restricted.is_restricted_member` (tested), so a bundle that still holds restricted members
    can be transcoded under the policy without any of them reaching a public object."""
    rel = rel.replace("\\", "/")
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith("donor/"):
        return None if rel == "donor/donor_assignments.tsv" else "genotype"
    if (
        rel == "ase_counts.tsv"
        or name.startswith("snp_")
        or _VARIANT.search(name)
        or _READS.search(name)
    ):
        return "genotype"
    if rel.startswith("mt/"):
        return "mtdna_alleles"  # mt_summary.tsv too: its per-cell columns (mts_*) are mtDNA-derived
    if (
        rel in ("mt_heteroplasmy.1pz", "mt_variants.tsv")
        or "_mt_consensus" in name
        or "_mt_variants" in name
    ):
        return "mtdna_alleles"
    if name == "ancestry_call.json":
        return "ancestry_call"
    if name.startswith("ambient_contamination.") or name == "ambient_profile.tsv":
        return "ambient"
    if name in ("doublet_scores.tsv", "doublet_scores_engine.tsv"):
        return "doublet_scores"
    if rel.startswith("nonhost/") and "per_cell" in name:
        return "nonhost_per_cell"
    if rel == "splice_psi.1pz":
        return "splice_psi_legacy"
    if name == "sex_call.json":
        return "sex_call_legacy"
    return None


@dataclass
class Policy:
    version: str
    withheld: Set[str]
    excluded: Set[str]
    sha256: str = ""
    name: str = "alpha"

    @classmethod
    def load(cls, path: str) -> "Policy":
        raw = open(path, "rb").read()
        d = json.loads(raw.decode("utf-8"))
        v = d.get("version")
        if not isinstance(v, str) or not v:
            raise ValueError("policy file has no version")
        ws, es = d.get("withheld_studies"), d.get("excluded_samples")
        if not isinstance(ws, list) or not isinstance(es, list):
            raise ValueError("policy file needs lists withheld_studies and excluded_samples")
        bad = [x for x in ws if not re.fullmatch(r"GSE\d+", str(x))] + [
            x for x in es if not re.fullmatch(r"GSM\d+", str(x))
        ]
        if bad:
            raise ValueError("policy file has malformed accessions, e.g. %r" % bad[:3])
        return cls(v, set(ws), set(es), hashlib.sha256(raw).hexdigest())

    def withholds(self, gse: str) -> bool:
        return gse in self.withheld

    def excludes(self, gsm: str) -> bool:
        return gsm in self.excluded

    def manifest_block(self) -> dict:
        """What the public manifest says about the policy: version and generic layer names only."""
        return {"name": self.name, "version": self.version, "omitted_layers": list(OMITTED_LAYERS)}


@dataclass
class PolicyReport:
    """What the policy dropped from one study (private: for the audit trail on the compute node, never published)."""

    gse: str
    version: str
    excluded_samples: List[str] = field(default_factory=list)
    omitted_members: Dict[str, int] = field(default_factory=dict)  # layer -> number of members
    omitted_bytes: int = 0
    rewritten_study_files: List[str] = field(default_factory=list)
    kept_samples: int = 0

    def to_json(self) -> dict:
        return {
            "gse": self.gse,
            "policy_version": self.version,
            "excluded_samples": sorted(self.excluded_samples),
            "omitted_members": dict(sorted(self.omitted_members.items())),
            "omitted_bytes": self.omitted_bytes,
            "rewritten_study_files": self.rewritten_study_files,
            "kept_samples": self.kept_samples,
        }

    def public_summary(self) -> dict:
        """Counts that name no accession, for the (non-public) campaign ledger."""
        return {
            "policy_version": self.version,
            "omitted_layers_present": sorted(self.omitted_members),
        }


# ------------------------------------------------------------------------------------------------ study files
def _rewrite_manifest_json(
    data: bytes, drop_gsm: Callable[[str], bool], drop_path: Callable[[str], bool], version: str
) -> bytes:
    m = json.loads(data.decode("utf-8"))
    if not isinstance(m, dict):
        return data
    if isinstance(m.get("gsm_ids"), list):
        m["gsm_ids"] = [g for g in m["gsm_ids"] if not drop_gsm(str(g))]
        if "n_gsms" in m:
            m["n_gsms"] = len(m["gsm_ids"])
    inc = m.get("included_files")
    if isinstance(inc, dict):
        m["included_files"] = {
            g: [r for r in (v if isinstance(v, list) else []) if omitted_layer(str(r)) is None]
            for g, v in inc.items()
            if not drop_gsm(str(g))
        }
    cs = m.get("checksums")
    if isinstance(cs, dict):
        m["checksums"] = {k: v for k, v in cs.items() if not drop_path(k)}
    m["release_policy"] = version
    return (json.dumps(m, indent=2) + "\n").encode("utf-8")


def _rewrite_study_meta(data: bytes, drop_gsm: Callable[[str], bool]) -> bytes:
    m = json.loads(data.decode("utf-8"))
    if not isinstance(m, dict):
        return data
    gm = m.get("gsm_meta")
    if isinstance(gm, dict):
        m["gsm_meta"] = {g: v for g, v in gm.items() if not drop_gsm(str(g))}
    return (json.dumps(m, indent=2) + "\n").encode("utf-8")


def strip_doublet_columns(data: bytes) -> bytes:
    """``cell_qc_metrics.tsv`` without its doublet columns (``doublet_score``, ``is_doublet``, ...): doublet calls are a withheld
    layer, and this table would otherwise carry them into the public cells table as ``qc_doublet_score``. Other bytes unchanged."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    lines = text.split("\n")
    head = [h.rstrip("\r") for h in lines[0].split("\t")]
    drop = {i for i, h in enumerate(head) if "doublet" in h.strip().lower()}
    if not drop:
        return data
    out = []
    for ln in lines:
        if not ln:
            out.append(ln)
            continue
        cr = ln.endswith("\r")
        f = (ln[:-1] if cr else ln).split("\t")
        out.append("\t".join(x for i, x in enumerate(f) if i not in drop) + ("\r" if cr else ""))
    return "\n".join(out).encode("utf-8")


def _is_qc_table(path: str) -> bool:
    s = _sample_of(path)
    return s is not None and s[1].rsplit("/", 1)[-1] == "cell_qc_metrics.tsv"


def _sample_of(path: str) -> Optional[Tuple[str, str]]:
    if not path.startswith("samples/"):
        return None
    parts = path.split("/", 2)
    if len(parts) < 3:
        return None
    return parts[1], parts[2]


def plan(names: Iterable[str], policy: Policy) -> Tuple[Dict[str, str], Set[str]]:
    """-> ({dropped path: reason}, {excluded GSM present}) for the archive member names of one bundle."""
    drop: Dict[str, str] = {}
    excl: Set[str] = set()
    for p in names:
        s = _sample_of(p)
        if s is None:
            continue
        g, rel = s
        if policy.excludes(g):
            drop[p] = "excluded_sample"
            excl.add(g)
            continue
        lay = omitted_layer(rel)
        if lay is not None:
            drop[p] = lay
    return drop, excl


def rewrite_study_files(
    study: Dict[str, bytes], drop: Dict[str, str], excl: Set[str], policy: Policy
) -> Dict[str, bytes]:
    """{name: new bytes} for the study files that must change (unchanged files are not returned)."""
    out: Dict[str, bytes] = {}

    def dg(g: str) -> bool:
        return g in excl or policy.excludes(g)

    def dp(path: str) -> bool:
        s = _sample_of(path)
        return path in drop or (s is not None and (dg(s[0]) or omitted_layer(s[1]) is not None))

    for name, data in study.items():
        try:
            if name == "manifest.json":
                new = _rewrite_manifest_json(data, dg, dp, policy.version)
            elif name == "study_meta.json":
                new = _rewrite_study_meta(data, dg)
            else:
                continue
        except (ValueError, UnicodeDecodeError):
            continue  # not JSON: left alone; leak_check below still refuses an excluded accession inside it
        if new != data:
            out[name] = new
    return out


def leak_check(blobs: Dict[str, bytes], excl: Set[str]) -> List[str]:
    """Names of blobs whose bytes still contain an excluded accession of this study (must be empty)."""
    if not excl:
        return []
    pat = re.compile(
        rb"(?<![0-9])(?:" + b"|".join(re.escape(g.encode()) for g in sorted(excl)) + rb")(?![0-9])"
    )
    return [n for n, b in blobs.items() if pat.search(b)]


# ------------------------------------------------------------------------------------------------ bundle views
def apply_to_bundle(
    bundle, policy: Policy, gse: str, *, text_scan_max: int = 64 << 20
) -> PolicyReport:
    """Filter a :class:`~singlet.s1.legacy.LegacyBundle` in place: drop excluded samples and withheld layers, rewrite the study
    files. Raises ``ValueError`` when an excluded accession would survive in a kept study file or small text member."""
    rep = PolicyReport(gse, policy.version)
    drop, excl = plan(list(bundle.entries), policy)
    for p, why in drop.items():
        e = bundle.entries.pop(p)
        if why != "excluded_sample":
            rep.omitted_members[why] = rep.omitted_members.get(why, 0) + 1
        rep.omitted_bytes += e.csize
    rep.excluded_samples = sorted(excl)
    for p in [p for p in bundle.entries if _is_qc_table(p)]:
        old = bundle.read(p)
        new = strip_doublet_columns(old)
        if new != old:
            bundle.override(p, new)
            rep.omitted_members["doublet_scores"] = rep.omitted_members.get("doublet_scores", 0) + 1
    study = {p: bundle.read(p) for p in bundle.study_paths()}
    new = rewrite_study_files(study, drop, excl, policy)
    for name, data in new.items():
        bundle.override(name, data)
        rep.rewritten_study_files.append(name)
    study.update(new)
    leaks = leak_check(study, excl)
    small = {}
    for (
        p,
        e,
    ) in (
        bundle.entries.items()
    ):  # sample members that are text and small: provenance, summaries, tables
        if (
            p.startswith("samples/")
            and e.size <= text_scan_max
            and not p.endswith((".1pz", ".mtx"))
        ):
            small[p] = bundle.read(p)
    leaks += leak_check(small, excl)
    if leaks:
        raise ValueError(
            "%s: %d kept members still name an excluded sample (first: %s)"
            % (gse, len(leaks), leaks[0])
        )
    rep.kept_samples = len(bundle.samples())
    return rep


class FilteredLegacy:
    """The verifier's legacy view (``names() / read() / info()``) with the policy applied the same way."""

    def __init__(self, inner, policy: Policy):
        self.inner = inner
        self.policy = policy
        names = list(inner.names())
        self.drop, self.excl = plan(names, policy)
        self._names = [n for n in names if n not in self.drop and not n.endswith("/")]
        study = {n: inner.read(n) for n in self._names if not n.startswith("samples/")}
        self.override = rewrite_study_files(study, self.drop, self.excl, policy)
        for n in self._names:
            if _is_qc_table(n):
                old = inner.read(n)
                new = strip_doublet_columns(old)
                if new != old:
                    self.override[n] = new

    def names(self):
        return list(self._names)

    def read(self, n):
        if n in self.override:
            return self.override[n]
        if n in self.drop:
            raise KeyError(n)
        return self.inner.read(n)

    def info(self, n):
        if n in self.override:
            d = self.override[n]
            return len(d), zlib.crc32(d) & 0xFFFFFFFF
        return self.inner.info(n)


def is_restricted_superset_ok(names: Iterable[str]) -> List[str]:
    """Members :func:`singlet._restricted.is_restricted_member` holds back that :func:`omitted_layer` would keep (must be empty)."""
    from singlet._restricted import is_restricted_member

    return [n for n in names if is_restricted_member(n) and omitted_layer(n) is None]
