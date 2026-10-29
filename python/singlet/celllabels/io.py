# SPDX-License-Identifier: MIT
"""Read what one sample's annotation needs: called cells' gene counts, their ``col`` in store column order, sample metadata, donor
labels, and (restricted, optional) per-donor SNP depths.

Two sources:

* a per-sample study (``s1/m/<GSE>.json`` on data.singlet.bio): :func:`singlet.s1.client.cells` with ``where="called"`` returns the called
  cells with their ``column`` in the sample object, which is ``col`` directly;
* a legacy ``.singlet`` bundle on disk (studies not yet migrated): ``col`` is the cell's position among the columns of
  ``exon_counts.1pz`` (``intron_counts.1pz`` when the exon matrix is empty), the order :mod:`singlet.s1.migrate` gives the
  ``cells`` table, so the labels line up with the .s1 object once the study is migrated.

The restricted sidecar (``restricted/<GSE>/<GSE>.restricted.singlet`` in the private bucket) is only opened when a path to it is
given, and only ``donor/donor<k>_coverage.tsv`` is read from it: depth and allele counts per SNP, used in memory by
:mod:`singlet.celllabels.ancestry` and dropped.
"""

from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import scipy.sparse as sps


@dataclass
class SampleInput:
    gsm: str
    gse: str
    X: sps.csr_matrix  # called cells x genes, raw counts (exon + intron)
    gene_names: List[str]
    col: np.ndarray  # int64 per cell
    barcodes: List[str]
    species: Optional[str]  # 'human' | 'mouse' | None (other organisms: no cell type / sex model)
    tissue: str  # free text used for the guide's tissue context
    disease: str
    meta: dict = field(default_factory=dict)
    donor: Optional[np.ndarray] = (
        None  # donor label per cell ('' = unassigned / doublet), None = no demultiplexing
    )
    donor_sites: Dict[str, "object"] = field(
        default_factory=dict
    )  # donor label -> DataFrame(chrom,pos,ref,alt,ad,dp); restricted
    source: str = ""
    gene_ids: List[str] = field(
        default_factory=list
    )  # Ensembl ids of the columns of ``X`` (version suffix as stored)

    @property
    def barcode_digest(self) -> str:
        """SHA-256 of the called barcodes joined by newlines in ``col`` order (lets the ingest check the alignment)."""
        order = np.argsort(self.col, kind="stable")
        return hashlib.sha256("\n".join(self.barcodes[i] for i in order).encode()).hexdigest()


def species_of(organism: str) -> Optional[str]:
    o = (organism or "").lower()
    if "homo sapiens" in o or o == "human":
        return "human"
    if "mus musculus" in o or o == "mouse":
        return "mouse"
    return None


_TISSUE_KEYS = re.compile(
    r"^\s*(tissue|tissue type|organ|organ part|source|source name|anatomical site|site|body site|region|brain region)\s*$",
    re.I,
)


def _characteristics(meta: dict) -> List[tuple]:
    out = []
    for part in re.split(r"\s*;;\s*", str(meta.get("sample_characteristics") or "")):
        k, sep, v = part.partition(":")
        if sep:
            out.append((k.strip(), v.strip()))
    return out


def _organ(text: str) -> bool:
    from .guide import tissue_groups

    return bool(tissue_groups(text) - {"tumor", "embryo"})


def tissue_text(meta: dict, series_title: str = "") -> str:
    """Free text naming the sample's tissue: the GEO source name and the tissue-like characteristics (underscores as
    spaces). When they name no organ, the diagnosis often does ('esophagus carcinoma'), and failing that the series title
    ('A single-cell atlas of the human bladder')."""
    parts = [str(meta.get(k) or "") for k in ("sample_source", "tissue") if meta.get(k)]
    parts += [v for k, v in _characteristics(meta) if _TISSUE_KEYS.match(k)]
    text = " ; ".join(p.replace("_", " ") for p in parts if p)
    for extra in (disease_text(meta), series_title):
        if _organ(text):
            break
        if extra and _organ(extra):
            text = f"{text} ; {extra}" if text else extra
    return text


def disease_text(meta: dict) -> str:
    hits = [
        f"{k}: {v}"
        for k, v in _characteristics(meta)
        if re.match(
            r"(disease|diagnosis|condition|tumou?r|cancer|subject status|status)\b", k, re.I
        )
    ]
    return " ; ".join(hits).replace("_", " ")


def _donor_labels(df, barcodes: List[str]) -> Optional[np.ndarray]:
    """Donor label per barcode from a donor_assignments table ('' for doublets/unassigned); None when there is one donor."""
    if df is None or len(df) == 0:
        return None
    bc_col = "cell" if "cell" in df.columns else df.columns[0]
    lab_col = (
        "cell_status"
        if "cell_status" in df.columns
        else ("donor_id" if "donor_id" in df.columns else df.columns[1])
    )
    m = dict(zip(df[bc_col].astype(str), df[lab_col].astype(str)))
    lab = np.array([m.get(b, "") for b in barcodes], dtype=object)
    lab[~np.char.startswith(lab.astype(str), "donor")] = ""
    if len({x for x in lab if x}) < 2:
        return None
    return lab


_COV = re.compile(r"(?:^|/)donor/(donor\d+)_coverage\.tsv$")


def read_donor_sites(sidecar: Path, gsm: str) -> Dict[str, "object"]:
    """Per-donor SNP depth tables of one sample from a restricted sidecar (``donor/donor<k>_coverage.tsv``)."""
    import pandas as pd

    out: Dict[str, object] = {}
    with zipfile.ZipFile(sidecar) as z:
        for name in z.namelist():
            if not name.startswith(f"samples/{gsm}/"):
                continue
            m = _COV.search(name)
            if not m:
                continue
            df = pd.read_csv(
                io.BytesIO(z.read(name)),
                sep="\t",
                usecols=["chrom", "pos", "ref", "alt", "dp", "ad"],
                dtype={
                    "chrom": str,
                    "pos": np.int64,
                    "ref": str,
                    "alt": str,
                    "dp": np.int64,
                    "ad": np.int64,
                },
            )
            out[m.group(1)] = df
    return out


def read_legacy(
    bundle_path: Union[str, Path], gsm: str, *, sidecar: Optional[Path] = None
) -> SampleInput:
    """One sample of a legacy ``.singlet`` bundle on disk."""
    from singlet.bundle import SingletBundle

    bundle_path = Path(bundle_path)

    b = SingletBundle(bundle_path)
    try:
        a = b.raw_counts(gsm)
        barcodes = [str(x) for x in a.obs_names]
        basis = None
        for member in ("exon_counts.1pz", "intron_counts.1pz"):
            if member in set(b.list_files(gsm)) and not b._is_empty_pz(gsm, member):
                basis = [str(x) for x in b.read(member, gsm).obs_names]
                break
        if basis is None:
            raise ValueError(f"{gsm}: no count matrix to take the column order from")
        pos = {bc: i for i, bc in enumerate(basis)}
        col = np.array([pos.get(bc, -1) for bc in barcodes], np.int64)
        if (col < 0).any():
            raise ValueError(
                f"{gsm}: {(col < 0).sum()} called barcodes are not columns of the count matrix"
            )
        meta = dict(b.study_meta.get("gsm_meta", {}).get(gsm, {}))
        series_title = str(b.study_meta.get("series_title") or "")
        gse = str(b.manifest.get("gse_id") or "")
        donor = None
        if b.has("donor_assignments", gsm):
            donor = _donor_labels(b.read("donor_assignments", gsm), barcodes)
        X = sps.csr_matrix(a.X)
        genes = [str(g) for g in a.var["gene_name"]]
        gene_ids = [str(g) for g in a.var_names]
    finally:
        b.close()
    s = SampleInput(
        gsm=gsm,
        gse=gse,
        X=X,
        gene_names=genes,
        col=col,
        barcodes=barcodes,
        species=species_of(meta.get("organism", "")),
        tissue=tissue_text(meta, series_title),
        disease=disease_text(meta),
        meta=meta,
        donor=donor,
        source=f"legacy:{bundle_path.name}",
        gene_ids=gene_ids,
    )
    if sidecar is not None:
        s.donor_sites = read_donor_sites(sidecar, gsm)
    return s


def read_s1(
    gse: str, gsm: str, *, base: Optional[str] = None, sidecar: Optional[Path] = None
) -> SampleInput:
    """One sample of a per-sample study (data.singlet.bio by default)."""
    from singlet.s1 import client as C
    from singlet.s1.bundle import S1Bundle

    a = C.cells(gse, where="called", samples=[gsm], base=base)
    col = a.obs["column"].to_numpy(np.int64)
    barcodes = [n.split("_", 1)[1] if n.startswith(f"{gsm}_") else n for n in a.obs_names]
    st = C.open_study(gse, base=base)
    meta = dict(st.study_meta.get("gsm_meta", {}).get(gsm, {}))
    series_title = str(st.study_meta.get("series_title") or "")
    donor = None
    try:
        vb = S1Bundle(st)
        if "donor_assignments" in vb.modalities(gsm):
            donor = _donor_labels(vb.read("donor_assignments", gsm), barcodes)
    except (KeyError, ValueError):
        donor = None
    s = SampleInput(
        gsm=gsm,
        gse=gse,
        X=sps.csr_matrix(a.X),
        gene_names=[str(g) for g in a.var["gene_name"]],
        col=col,
        barcodes=barcodes,
        species=species_of(meta.get("organism", "")),
        tissue=tissue_text(meta, series_title),
        disease=disease_text(meta),
        meta=meta,
        donor=donor,
        source=f"store:{st.rev}",
        gene_ids=[str(g) for g in a.var_names],
    )
    if sidecar is not None:
        s.donor_sites = read_donor_sites(sidecar, gsm)
    return s


read_v2 = read_store = read_s1  # DUAL: old names, removed in the cleanup
