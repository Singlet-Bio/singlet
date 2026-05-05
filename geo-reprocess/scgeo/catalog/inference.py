"""
Protocol inference from metadata using heuristics.

Applies rule-based pattern matching to infer single-cell protocols
(10x Chromium, Smart-seq, Drop-seq, etc.) from combined metadata fields.
"""
import logging
import re
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for protocol inference
_RE_SMARTSEQ = re.compile(r"smart-?seq|smartseq", re.IGNORECASE)
_RE_10X = re.compile(r"\b10x\b|chromium|10x\s*genomics", re.IGNORECASE)
_RE_HIC = re.compile(r"\bhi-c\b", re.IGNORECASE)
_RE_CHIPSEQ = re.compile(r"chip-?seq", re.IGNORECASE)
_RE_BISULFITE = re.compile(r"bisulfite|methyl-?seq|\bwgbs\b|\brrbs\b", re.IGNORECASE)
_RE_ATAC = re.compile(r"\bsc-?atac\b|\bsn-?atac\b|\batac-?seq\b", re.IGNORECASE)
_RE_ATAC_10X = re.compile(r"\b10x\b|chromium", re.IGNORECASE)
_RE_SMARTSEQ3 = re.compile(r"smart-?seq\s*3|smartseq3|\bss3\b|\bht\s*ss3\b", re.IGNORECASE)
_RE_SMARTSEQ2 = re.compile(r"smart-?seq\s*2|smartseq2", re.IGNORECASE)
_RE_CELSEQ2 = re.compile(r"cel-?seq\s*2|celseq2", re.IGNORECASE)
_RE_CELSEQ = re.compile(r"cel-?seq", re.IGNORECASE)
_RE_SCIRNA = re.compile(r"sci-?rna|sci-?seq", re.IGNORECASE)
_RE_MARSSEQ = re.compile(r"mars-?seq", re.IGNORECASE)
_RE_STRTSEQ = re.compile(r"strt-?seq", re.IGNORECASE)
_RE_QUARTZSEQ = re.compile(r"quartz-?seq", re.IGNORECASE)
_RE_ICELL8 = re.compile(r"icell8|wafergen", re.IGNORECASE)
_RE_FLUIDIGM = re.compile(r"fluidigm", re.IGNORECASE)
_RE_TAKARA = re.compile(r"takara", re.IGNORECASE)
_RE_TAKARA_SMART = re.compile(r"smart|smar|v4|cdna|plate", re.IGNORECASE)
_RE_NEXTERA = re.compile(r"nextera", re.IGNORECASE)
_RE_NEXTERA_PLATE = re.compile(r"plate|cdna|single.cell|scrna|single.nucleus", re.IGNORECASE)
_RE_MULTIOME = re.compile(r"multiome|10x.{0,50}atac.{0,50}gex|10x.{0,50}gex.{0,50}atac", re.IGNORECASE)
_RE_10X_5PRIME = re.compile(r"10x.{0,50}5'|chromium.{0,50}5'|5'\s*gene\s*expression", re.IGNORECASE)
_RE_10XV3 = re.compile(r"10x.{0,50}v3|chromium.{0,50}v3|v3\.1|10x\s+v3", re.IGNORECASE)
_RE_10XV2 = re.compile(r"10x.{0,50}v2|chromium.{0,50}v2|10x\s+v2", re.IGNORECASE)
_RE_10XV4 = re.compile(r"10x.{0,50}v4|chromium.{0,50}v4|10x\s+v4", re.IGNORECASE)
_RE_DROPSEQ = re.compile(r"drop-?seq", re.IGNORECASE)
_RE_INDROP = re.compile(r"indrop", re.IGNORECASE)
_RE_SPLITSEQ = re.compile(r"split-?seq", re.IGNORECASE)
_RE_SEQWELL = re.compile(r"seq-?well", re.IGNORECASE)
_RE_PARSE = re.compile(r"parse|evercode", re.IGNORECASE)
_RE_RHAPSODY = re.compile(r"rhapsody|bd\s*rhapsody", re.IGNORECASE)
_RE_SLIDESEQ = re.compile(r"slide-?seq", re.IGNORECASE)
_RE_MICROWELL = re.compile(r"microwell", re.IGNORECASE)
_RE_DDSEQ = re.compile(r"ddseq|dd-?seq", re.IGNORECASE)
_RE_SURECELL = re.compile(r"surecell|sure\s*cell", re.IGNORECASE)
_RE_DNBELAB = re.compile(r"dnbelab|dnb-?seq", re.IGNORECASE)
_RE_CITESEQ = re.compile(r"cite-?seq|citeseq", re.IGNORECASE)
_RE_VISIUM = re.compile(r"\bvisium\b", re.IGNORECASE)
_RE_SCOPESEQ = re.compile(r"scope-?seq", re.IGNORECASE)
_RE_SNRNA = re.compile(r"snrna|single.?nucleus", re.IGNORECASE)
_RE_FACS = re.compile(r"facs|sorted|sorting", re.IGNORECASE)
_RE_PLATE = re.compile(r"plate|well|384|96", re.IGNORECASE)
_RE_CDNA = re.compile(r"cdna|rna|transcri", re.IGNORECASE)


def _lower(val) -> str:
    """Safe lowercase conversion."""
    if pd.isna(val):
        return ""
    return str(val).lower()


def infer_protocol(row: pd.Series) -> Tuple[str, str]:
    """
    Infer single-cell protocol from combined metadata fields.
    
    Args:
        row: DataFrame row with metadata columns
        
    Returns:
        (protocol, confidence) tuple where confidence is 'high', 'medium', or 'low'
    """
    # Assemble all text sources
    text_fields = []
    for col in [
        "title", "series_title", "summary", "overall_design",
        "sample_extract_protocol", "sample_data_processing",
        "sample_source", "sample_characteristics",
        "library_name",
    ]:
        if col in row.index:
            text_fields.append(_lower(row[col]))
    
    combined = " ".join(text_fields)
    lib_source = _lower(row.get("library_source", ""))
    lib_strategy = _lower(row.get("library_strategy", ""))
    
    # Compute avg read length if available
    read_count = row.get("read_count", 0)
    base_count = row.get("base_count", 0)
    if pd.notna(read_count) and pd.notna(base_count) and float(read_count) > 0:
        avg_len = float(base_count) / float(read_count)
    else:
        avg_len = 0
    
    # Pre-compute presence flags
    has_smartseq = bool(_RE_SMARTSEQ.search(combined))
    has_10x = bool(_RE_10X.search(combined))
    
    # ===== NON-RNA MODALITY FLAGS (check first) =====
    
    if lib_strategy == "hi-c" or _RE_HIC.search(combined):
        return "hi_c", "high"
    
    if lib_strategy == "chip-seq" or _RE_CHIPSEQ.search(combined):
        return "chipseq", "high"
    
    if lib_strategy == "bisulfite-seq" or _RE_BISULFITE.search(combined):
        return "methylation", "high"
    
    if lib_strategy == "atac-seq" or _RE_ATAC.search(combined):
        if _RE_ATAC_10X.search(combined):
            return "10x_atac", "medium"
        return "scATAC", "medium"
    
    # ===== SPATIAL PROTOCOLS (before droplet — "10x Visium" must not match 10x first) =====
    
    if _RE_VISIUM.search(combined):
        return "visium", "high"
    
    if _RE_SLIDESEQ.search(combined):
        return "slideseq", "high"
    
    # ===== PLATE-BASED / FULL-LENGTH PROTOCOLS =====
    
    if _RE_SMARTSEQ3.search(combined):
        return "smartseq3", "high"
    
    if _RE_SMARTSEQ2.search(combined):
        return "smartseq2", "high"
    if has_smartseq:
        return "smartseq2", "high"
    
    if _RE_CELSEQ2.search(combined):
        return "celseq2", "high"
    if _RE_CELSEQ.search(combined):
        return "celseq", "medium"
    
    if _RE_SCIRNA.search(combined):
        return "scirna", "high"
    
    if _RE_MARSSEQ.search(combined):
        return "marsseq", "high"
    
    if _RE_STRTSEQ.search(combined):
        return "strtseq", "high"
    
    if _RE_QUARTZSEQ.search(combined):
        return "quartzseq", "high"
    
    if _RE_ICELL8.search(combined):
        return "icell8", "high"
    
    if _RE_FLUIDIGM.search(combined):
        return "smartseq2", "medium"
    
    if _RE_TAKARA.search(combined):
        if _RE_TAKARA_SMART.search(combined):
            return "smartseq2", "medium"
        if lib_strategy == "rna-seq":
            return "smartseq2", "low"
    
    if _RE_NEXTERA.search(combined):
        if _RE_NEXTERA_PLATE.search(combined):
            return "smartseq2", "medium"
        if lib_strategy == "rna-seq":
            return "smartseq2", "low"
    
    # ===== DROPLET-BASED PROTOCOLS =====
    
    if _RE_MULTIOME.search(combined):
        return "10x_multiome", "medium"
    
    if _RE_10X_5PRIME.search(combined):
        return "10xv3_5prime", "medium"
    
    if _RE_10XV3.search(combined):
        if avg_len > 90:
            return "10xv3", "low"
        return "10xv3", "high"
    
    if _RE_10XV2.search(combined):
        if avg_len > 90:
            return "10xv2", "low"
        return "10xv2", "high"
    
    if _RE_10XV4.search(combined):
        if avg_len > 90:
            return "10xv4", "low"
        return "10xv4", "high"
    
    if has_10x:
        if avg_len > 90:
            return "10x_suspect", "low"
        if 50 < avg_len <= 90:
            return "10xv3", "medium"
        return "10xv3", "low"
    
    if _RE_DROPSEQ.search(combined):
        return "dropseq", "high"
    
    if _RE_INDROP.search(combined):
        return "indrop", "high"
    
    if _RE_SPLITSEQ.search(combined):
        return "splitseq", "high"
    
    if _RE_SEQWELL.search(combined):
        return "seqwell", "high"
    
    if _RE_PARSE.search(combined):
        return "parse", "medium"
    
    if _RE_RHAPSODY.search(combined):
        return "bd_rhapsody", "high"
    
    if _RE_MICROWELL.search(combined):
        return "microwell", "medium"
    
    if _RE_DDSEQ.search(combined):
        return "ddseq", "high"
    
    if _RE_SURECELL.search(combined):
        return "surecell", "high"
    
    if _RE_DNBELAB.search(combined):
        return "dnbelab", "high"
    
    if _RE_CITESEQ.search(combined):
        return "citeseq", "medium"
    
    if _RE_SCOPESEQ.search(combined):
        return "scopeseq", "high"
    
    # ===== INDIRECT / INFERRED PROTOCOLS =====
    
    if _RE_SNRNA.search(combined):
        if has_10x:
            if avg_len > 90:
                return "10x_suspect", "low"
            return "10xv3", "medium"
        return "snRNA_unknown", "low"
    
    if _RE_FACS.search(combined):
        if _RE_PLATE.search(combined):
            if _RE_CDNA.search(combined):
                return "plate_based", "low"
    
    if "single cell" in lib_source or "transcriptomic single cell" in lib_source:
        return "unknown_sc", "low"
    
    return "unknown", "low"


def infer_protocols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply protocol inference to a DataFrame.
    
    Args:
        df: DataFrame with metadata columns
        
    Returns:
        DataFrame with added 'protocol_inferred' and 'protocol_confidence' columns
    """
    logger.info("Inferring protocols...")
    
    protocols = df.apply(infer_protocol, axis=1, result_type="expand")
    df = df.copy()
    df["protocol_inferred"] = protocols[0]
    df["protocol_confidence"] = protocols[1]
    
    # Log summary
    logger.info(f"Total samples: {len(df):,}")
    logger.info("Top protocols:")
    for proto, count in df["protocol_inferred"].value_counts().head(10).items():
        pct = count / len(df) * 100
        logger.info(f"  {proto:20s}: {count:>7,} ({pct:5.1f}%)")
    
    logger.info("Confidence levels:")
    for conf, count in df["protocol_confidence"].value_counts().items():
        pct = count / len(df) * 100
        logger.info(f"  {conf:10s}: {count:>7,} ({pct:5.1f}%)")
    
    return df
