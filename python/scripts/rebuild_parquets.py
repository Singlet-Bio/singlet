#!/usr/bin/env python3
"""Rebuild sample_index.parquet and catalog_v1.parquet from Supabase.

Run periodically to keep offline data fresh:
    python3 scripts/rebuild_parquets.py

Outputs:
    singlet/data/sample_index.parquet  — full sample index with tissue/cell_type
    singlet/data/catalog_v1.parquet    — series-level catalog
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Supabase client
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    sys.exit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables")

OUTPUT_DIR = Path(__file__).parent.parent / "singlet" / "data"


# ═══════════════════════════════════════════════════════════════
# TISSUE NORMALIZATION
# ═══════════════════════════════════════════════════════════════
TISSUE_NORMALIZE = {
    "brain": "brain",
    "cerebral": "brain",
    "cortex": "brain",
    "hippocampus": "brain",
    "cerebellum": "brain",
    "frontal cortex": "brain",
    "temporal lobe": "brain",
    "leptomening": "brain",
    "hypothalamus": "brain",
    "striatum": "brain",
    "thalamus": "brain",
    "amygdala": "brain",
    "blood": "blood",
    "pbmc": "blood",
    "peripheral blood": "blood",
    "whole blood": "blood",
    "cord blood": "blood",
    "serum": "blood",
    "bone marrow": "bone marrow",
    "bm aspirate": "bone marrow",
    "lung": "lung",
    "airway": "lung",
    "bronchial": "lung",
    "alveolar": "lung",
    "pulmonary": "lung",
    "bronchoalveolar": "lung",
    "liver": "liver",
    "hepat": "liver",
    "kidney": "kidney",
    "renal": "kidney",
    "nephron": "kidney",
    "heart": "heart",
    "cardiac": "heart",
    "myocardium": "heart",
    "ventricle": "heart",
    "skin": "skin",
    "dermis": "skin",
    "epidermis": "skin",
    "cutaneous": "skin",
    "intestine": "intestine",
    "colon": "intestine",
    "gut": "intestine",
    "ileum": "intestine",
    "duodenum": "intestine",
    "jejunum": "intestine",
    "cecum": "intestine",
    "rectum": "intestine",
    "colorectal": "intestine",
    "colonic": "intestine",
    "pancreas": "pancreas",
    "islet": "pancreas",
    "pancreatic": "pancreas",
    "breast": "breast",
    "mammary": "breast",
    "spleen": "spleen",
    "splenic": "spleen",
    "thymus": "thymus",
    "thymic": "thymus",
    "lymph node": "lymph node",
    "tonsil": "lymph node",
    "retina": "retina",
    "eye": "eye",
    "cornea": "eye",
    "ocular": "eye",
    "muscle": "muscle",
    "skeletal muscle": "muscle",
    "myofiber": "muscle",
    "adipose": "adipose",
    "fat": "adipose",
    "stomach": "stomach",
    "gastric": "stomach",
    "ovary": "ovary",
    "ovarian": "ovary",
    "testis": "testis",
    "testes": "testis",
    "testicul": "testis",
    "prostate": "prostate",
    "prostatic": "prostate",
    "bladder": "bladder",
    "placenta": "placenta",
    "villi": "placenta",
    "trophoblast": "placenta",
    "endometri": "endometrium",
    "uterus": "endometrium",
    "uterine": "endometrium",
    "tumor": "tumor",
    "cancer": "tumor",
    "carcinoma": "tumor",
    "melanoma": "tumor",
    "glioblastoma": "tumor",
    "glioma": "tumor",
    "lymphoma": "tumor",
    "leukemia": "tumor",
    "sarcoma": "tumor",
    "adenocarcinoma": "tumor",
    "neoplasm": "tumor",
    "organoid": "organoid",
    "embryo": "embryo",
    "embryonic": "embryo",
    "fetal": "embryo",
    "cell line": "cell line",
    "cell culture": "cell line",
    "csf": "CSF",
    "cerebrospinal": "CSF",
    "dorsal root ganglion": "nervous system",
    "spinal cord": "nervous system",
    "ganglio": "nervous system",
    "nerve": "nervous system",
    "synovial": "joint",
    "cartilage": "joint",
    "joint": "joint",
    "thyroid": "thyroid",
    "adrenal": "adrenal",
    "esophag": "esophagus",
    "nasal": "nasal",
    "sinus": "nasal",
    "olfactory": "nasal",
    "oral": "oral cavity",
    "gingiv": "oral cavity",
    "saliv": "oral cavity",
    "bone": "bone",
}


# ═══════════════════════════════════════════════════════════════
# CELL TYPE NORMALIZATION
# ═══════════════════════════════════════════════════════════════
CT_NORMALIZE = {
    "pbmc": "PBMC",
    "peripheral blood mononuclear": "PBMC",
    "t cell": "T cells",
    "t-cell": "T cells",
    "t lymphocyte": "T cells",
    "cd4": "CD4+ T cells",
    "cd4+": "CD4+ T cells",
    "cd4 t": "CD4+ T cells",
    "cd8": "CD8+ T cells",
    "cd8+": "CD8+ T cells",
    "cd8 t": "CD8+ T cells",
    "car-t": "CAR-T cells",
    "car t": "CAR-T cells",
    "cart cell": "CAR-T cells",
    "b cell": "B cells",
    "b-cell": "B cells",
    "b lymphocyte": "B cells",
    "nk cell": "NK cells",
    "natural killer": "NK cells",
    "monocyte": "monocytes",
    "macrophage": "macrophages",
    "dendritic": "dendritic cells",
    "neutrophil": "neutrophils",
    "mast cell": "mast cells",
    "immune": "immune cells",
    "fibroblast": "fibroblasts",
    "stromal": "stromal cells",
    "mesenchymal": "mesenchymal cells",
    "epithelial": "epithelial cells",
    "endothelial": "endothelial cells",
    "neuron": "neurons",
    "astrocyte": "astrocytes",
    "microglia": "microglia",
    "oligodendrocyte": "oligodendrocytes",
    "hepatocyte": "hepatocytes",
    "cardiomyocyte": "cardiomyocytes",
    "stem cell": "stem cells",
    "ipsc": "stem cells (iPSC)",
    "ips cell": "stem cells (iPSC)",
    "induced pluripotent": "stem cells (iPSC)",
    "hesc": "stem cells (hESC)",
    "embryonic stem": "stem cells (hESC)",
    "hpsc": "stem cells (hESC)",
    "organoid": "organoid",
    "tumor": "tumor cells",
    "cancer": "tumor cells",
    "carcinoma": "tumor cells",
    "lymphoma": "tumor cells",
    "leukemia": "tumor cells",
    "melanoma": "tumor cells",
    "glioblastoma": "tumor cells",
    "myeloma": "tumor cells",
    "sarcoma": "tumor cells",
    "plasma cell": "plasma cells",
    "platelet": "platelets",
    "erythrocyte": "erythrocytes",
    "red blood": "erythrocytes",
    "spermat": "spermatocytes",
    "lymphocyte": "lymphocytes",
    "bone marrow": "bone marrow cells",
    "mait": "MAIT cells",
    "regulatory t": "regulatory T cells",
    "treg": "regulatory T cells",
    "gamma delta": "gamma-delta T cells",
}

CT_KEYWORDS = [
    ("k562", "cell line (K562)"),
    ("jurkat", "cell line (Jurkat)"),
    ("hek293", "cell line (HEK293)"),
    ("hek 293", "cell line (HEK293)"),
    ("hela", "cell line (HeLa)"),
    ("thp-1", "cell line (THP-1)"),
    ("thp1", "cell line (THP-1)"),
    ("a549", "cell line (A549)"),
    ("mcf7", "cell line (MCF7)"),
    ("mcf-7", "cell line (MCF7)"),
    ("u937", "cell line (U937)"),
    ("caco-2", "cell line (Caco-2)"),
    ("caco2", "cell line (Caco-2)"),
    ("nhdf", "fibroblasts"),
]

CL_NORMALIZE = {
    "k562": "cell line (K562)",
    "jurkat": "cell line (Jurkat)",
    "hek293": "cell line (HEK293)",
    "hek 293": "cell line (HEK293)",
    "293t": "cell line (HEK293)",
    "hela": "cell line (HeLa)",
    "thp-1": "cell line (THP-1)",
    "thp1": "cell line (THP-1)",
    "a549": "cell line (A549)",
    "u937": "cell line (U937)",
    "mcf7": "cell line (MCF7)",
    "mcf-7": "cell line (MCF7)",
    "caco-2": "cell line (Caco-2)",
    "caco2": "cell line (Caco-2)",
    "raji": "cell line",
    "u2os": "cell line",
    "hepg2": "cell line",
    "sk-n": "cell line",
    "sh-sy5y": "cell line",
}

CT_TISSUE_WORDS = frozenset(
    {
        "brain",
        "lung",
        "liver",
        "kidney",
        "heart",
        "skin",
        "muscle",
        "blood",
        "bone",
        "spleen",
        "thymus",
        "colon",
        "intestine",
        "stomach",
        "pancreas",
        "breast",
        "ovary",
        "testis",
        "prostate",
        "bladder",
        "retina",
        "adipose",
        "placenta",
        "tonsil",
        "lymph",
        "cord",
        "cerebral",
        "cortex",
        "hippocampus",
        "tumor tissue",
        "biopsy",
        "fresh tissue",
        "frozen tissue",
        "dissociated",
        "sorted",
        "whole",
        "single cell",
        "suspension",
        "digest",
        "fraction",
    }
)


def _match_tissue(text: str):
    for key in sorted(TISSUE_NORMALIZE.keys(), key=len, reverse=True):
        if key in text:
            return TISSUE_NORMALIZE[key]
    return None


def _match_cell_type(text: str):
    words = set(text.split())
    if words and words.issubset(CT_TISSUE_WORDS):
        return None
    for kw, ct in CT_KEYWORDS:
        if kw in text:
            return ct
    for key in sorted(CT_NORMALIZE.keys(), key=len, reverse=True):
        if key in text:
            return CT_NORMALIZE[key]
    return None


def normalize_tissue(row):
    """Extract tissue from characteristics, source, then title."""
    chars = row.get("characteristics")
    if chars and isinstance(chars, dict):
        for key in ("tissue", "tissue type", "organ", "tissue/organ"):
            val = chars.get(key, "")
            if val:
                t = _match_tissue(str(val).lower())
                if t:
                    return t
    source = row.get("source", "")
    if source and pd.notna(source):
        t = _match_tissue(str(source).lower())
        if t:
            return t
    title = row.get("title", "")
    if title and pd.notna(title):
        t = _match_tissue(str(title).lower())
        if t:
            return t
    return None


def normalize_cell_type(row: dict):
    """Extract cell type from characteristics, then source."""
    chars = row.get("characteristics")
    if chars and isinstance(chars, dict):
        for key in ("cell type", "cell_type", "celltype", "cell type/tissue"):
            val = chars.get(key, "")
            if val:
                ct = _match_cell_type(str(val).lower())
                if ct:
                    return ct
        for key in ("cell line", "cell_line", "cellline"):
            val = chars.get(key, "")
            if val:
                v = str(val).lower().strip()
                for cl_key, cl_type in CL_NORMALIZE.items():
                    if cl_key in v:
                        return cl_type
                return "cell line"
    source = row.get("source", "")
    if source and pd.notna(source):
        ct = _match_cell_type(str(source).lower())
        if ct:
            return ct
    return None


def fetch_all_samples() -> pd.DataFrame:
    """Paginated fetch of all samples from Supabase."""
    sb = create_client(SUPABASE_URL, SUPABASE_KEY)
    all_rows = []
    offset = 0
    while True:
        r = sb.table("samples").select("*").range(offset, offset + 999).execute()
        all_rows.extend(r.data)
        if len(r.data) < 1000:
            break
        offset += 1000
    return pd.DataFrame(all_rows)


def parse_characteristics(val):
    if pd.isna(val) or val is None:
        return None
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def main():
    print("Fetching samples from Supabase...")
    df = fetch_all_samples()
    print(f"  Total: {len(df)}, SUCCESS: {(df['status'] == 'SUCCESS').sum()}")

    # Parse characteristics
    df["characteristics"] = df["characteristics"].apply(parse_characteristics)

    # Normalize tissue and cell_type
    print("Normalizing tissue...")
    df["tissue"] = df.apply(normalize_tissue, axis=1)

    print("Normalizing cell_type...")
    df["cell_type"] = df.apply(normalize_cell_type, axis=1)

    success = df[df["status"] == "SUCCESS"]
    print(
        f"  Tissue: {success['tissue'].notna().sum()}/{len(success)} ({100 * success['tissue'].notna().sum() / len(success):.0f}%)"
    )
    print(
        f"  Cell type: {success['cell_type'].notna().sum()}/{len(success)} ({100 * success['cell_type'].notna().sum() / len(success):.0f}%)"
    )

    # Build sample_index.parquet
    sample_cols = [
        "gsm_id",
        "gse_id",
        "organism",
        "status",
        "failure_category",
        "protocol",
        "mapping_rate",
        "cells_called",
        "median_genes",
        "median_umis",
        "mt_pct",
        "doublet_rate",
        "wall_time_s",
        "title",
        "source",
        "tissue",
        "cell_type",
    ]
    sample_index = df[[c for c in sample_cols if c in df.columns]].copy()
    sample_path = OUTPUT_DIR / "sample_index.parquet"
    sample_index.to_parquet(sample_path, index=False)
    print(f"\nSaved: {sample_path} ({len(sample_index)} rows)")

    # Build catalog_v1.parquet (SUCCESS series only)
    catalog = (
        success.groupby("gse_id")
        .agg(
            organism=("organism", "first"),
            n_samples=("gsm_id", "count"),
            n_cells=("cells_called", "sum"),
            protocol=("protocol", "first"),
            tissue=("tissue", "first"),
            cell_type=("cell_type", "first"),
        )
        .reset_index()
    )
    catalog["n_cells"] = catalog["n_cells"].astype(int)
    catalog = catalog.sort_values("n_cells", ascending=False)
    catalog_path = OUTPUT_DIR / "catalog_v1.parquet"
    catalog.to_parquet(catalog_path, index=False)
    print(f"Saved: {catalog_path} ({len(catalog)} series)")

    # Summary stats
    total_cells = int(success["cells_called"].sum())
    print(f"\n{'═' * 50}")
    print(f"Total samples: {len(df):,}")
    print(f"SUCCESS: {len(success):,}")
    print(f"Total cells: {total_cells:,}")
    print(f"Series: {len(catalog)}")
    print(
        f"Species: {df['organism'].dropna().loc[lambda s: (s != 'unknown') & (s.str.strip() != '')].nunique()}"
    )
    print(f"Tissues: {success['tissue'].nunique()} categories")
    print(f"Cell types: {success['cell_type'].nunique()} categories")
    print(f"{'═' * 50}")


if __name__ == "__main__":
    main()
