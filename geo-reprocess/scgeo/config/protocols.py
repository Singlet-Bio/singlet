"""Protocol and chemistry configuration for sc-geo.

Maps GEO protocol annotations to simpleaf chemistry strings
and defines protocol detection thresholds.
"""
from typing import Dict, Optional, Set

# Protocol → simpleaf chemistry mapping
# Geometry strings follow FGDL: https://hackmd.io/@PI7Og0l1ReeBZu_pjQGUQQ/rJMgmvr13
# Verified against scg_lib_structs: https://teichlab.github.io/scg_lib_structs/
PROTOCOL_CHEMISTRY: Dict[str, Optional[str]] = {
    # 10x Chromium (builtin chemistries in simpleaf)
    "10xv2":         "10xv2",
    "10xv3":         "10xv3",
    "10xv4":         "10xv4-3p",
    "10xv3_5prime":  "10xv3",
    "10x_multiome":  "10xv3",
    # Droplet protocols (custom geometry strings)
    "dropseq":       "1{b[12]u[8]x:}2{r:}",
    "seqwell":       "1{b[12]u[8]x:}2{r:}",
    "dnbelab":       "1{b[10]u[10]x:}2{r:}",
    # Complex barcoding with positional skips for linker regions.
    # IMPORTANT: Use x[N] (positional skip), NOT f[SEQ] (anchor match), because
    # simpleaf strips f[] elements before passing geometry to piscem, which
    # causes barcode extraction at wrong positions.
    # Note: simpleaf ≤0.19.x cannot parse b[N-M] range notation; use max length b[M]
    "indrop":        "1{b[8]x[22]b[8]u[6]x:}2{r:}",       # linker: GAGTGATTGCTTGTGACGCCTT (22bp)
    "bd_rhapsody":   "1{b[9]x[12]b[9]x[13]b[9]u[8]x:}2{r:}",  # linkers: ACTGGCCTGCGA (12bp), GGTAGCGGTGACA (13bp)
    # ddSEQ Single-Cell 3' RNA-Seq: R1 = UMI(8) + BC1(7) + PB(0-4) + linker(10) + BC2(7) + linker(10) + BC3(7) + linker(12)
    # NOTE: Variable Phase Block (0-4bp) means only PB=0 barcodes are extracted correctly.
    # Bio-Rad's Omnition software handles this natively; simpleaf FGDL is an approximation.
    "ddseq":         "1{u[8]b[7]x[10]b[7]x[10]b[7]x:}2{r:}",  # PB=0 variant, 49bp R1 min
    # sci-RNA-seq3: R1(34bp) = hairpin_BC(10) + CAGAGC(6) + UMI(8) + RT_BC(10)
    # NOTE: sci-RNA-seq also uses i5/i7 index reads for additional barcode levels.
    # Without I1/I2 index reads, only the RT barcode level is captured.
    "scirna":        "1{b[10]x[6]u[8]b[10]x:}2{r:}",       # sci-RNA-seq3 geometry
    "splitseq":      "1{r:}2{u[10]b[8]x[30]b[8]x[30]b[8]x:}",  # linkers: 30bp each (SPLiT-seq specific)
    # Parse Biosciences (microSPLiT chemistry): barcodes on R2, cDNA on R1.
    # R2 structure: UMI(10bp) + R3 BC(8bp) + Round3 linker(30bp) + R2 BC(8bp)
    #             + Round2 linker(22bp) + R1 BC(8bp) + polyT.  Min R2 = 86bp.
    # IMPORTANT: Use x[N] (positional skip), NOT f[SEQ] (anchor match), because
    # simpleaf strips f[] elements before passing geometry to piscem, which
    # causes barcode extraction at wrong positions.
    # Linker sequences for reference (from Teichlab scg_lib_structs):
    #   Round3 linker RC (30bp): GTGGCCGATGTTTCGCATCGGCGTACGACT
    #   Round2 linker RC (22bp): ATCCACGTGCTTGAGACTGTGG
    "parse":         "1{r:}2{u[10]b[8]x[30]b[8]x[22]b[8]x:}",
    # Simple plate/well-based with UMI
    "celseq":        "1{b[8]u[4]x:}2{r:}",
    "celseq2":       "1{b[6]u[6]x:}2{r:}",
    "marsseq":       "1{b[7]u[8]x:}2{r:}",
    "quartzseq":     "1{b[15]u[8]x:}2{r:}",
    "strtseq":       "1{b[6]u[5]x:}2{r:}",
    "icell8":        "1{b[11]u[14]x:}2{r:}",
    # SureCell 3' WTA: R1(68bp) = BC1(6) + spacer1(15) + BC2(6) + spacer2(15) + BC3(6) + ACG(3) + UMI(8) + dT
    # Composite 18bp barcode (6+6+6) from split-pool synthesis.
    "surecell":      "1{b[6]x[15]b[6]x[15]b[6]x[3]u[8]x:}2{r:}",  # 59bp R1 content
    "scopeseq":      "1{b[10]u[10]x:}2{r:}",
    "microwell":     "1{b[18]u[6]x:}2{r:}",
    # Plate-based / full-length
    "smartseq2":     "smartseq",
    "smartseq3":     "smartseq",
    "plate_based":   "smartseq",
    # Suspect / unknown
    "10x_suspect":   "10xv3",
    "unknown":       None,
    "unknown_sc":    None,
    "snRNA_unknown": None,
    # CITE-seq GEX libraries use 10x chemistry; ADT libraries will fail
    # detection (wrong read lengths) or QC (low genes) — both correct.
    "citeseq":       "10xv3",
}

# Protocols that are NOT transcriptomic (always exclude)
NON_RNA_PROTOCOLS: Set[str] = {
    "scATAC", "10x_atac", "chipseq", "methylation", "hi_c",
    "dnase_seq", "mnase_seq", "mirna_seq", "rip_seq",
    "visium", "slideseq",
}

# Protocols with inherently lower gene detection per cell.
# These use shorter barcodes/UMI or lower-efficiency capture chemistry,
# resulting in fewer detected genes per cell than 10x Chromium.
# QC uses a relaxed min_genes_per_cell threshold for these.
LOW_SENSITIVITY_PROTOCOLS: Set[str] = {
    "seqwell", "dropseq", "microwell", "surecell", "celseq", "celseq2",
    "marsseq", "strtseq", "icell8", "scopeseq", "ddseq", "indrop",
}

# library_strategy values we accept
ACCEPTED_STRATEGIES: Set[str] = {"RNA-Seq", "OTHER"}

# Builtin chemistry names (simpleaf auto-fetches whitelists)
BUILTIN_CHEMISTRIES: Set[str] = {"10xv2", "10xv3", "10xv4-3p"}

# Protocol detection thresholds (from FASTQ R1 inspection)
DROPLET_R1_MAX = 50      # R1 > 50bp → probably full-length / plate-based
R1_10XV2 = 26
R1_10XV3 = 28
R1_10XV4 = 28
R1_DROPSEQ = 20


def get_chemistry(protocol: str) -> Optional[str]:
    """Get simpleaf chemistry string for a protocol.
    
    Args:
        protocol: Protocol name from GEO catalog
        
    Returns:
        Chemistry string or None if unknown/unsupported
    """
    return PROTOCOL_CHEMISTRY.get(protocol.lower())


def is_droplet_protocol(protocol: str) -> bool:
    """Check if protocol is droplet-based (vs plate-based)."""
    chem = get_chemistry(protocol)
    return chem is not None and chem != "smartseq"


def is_builtin_chemistry(chemistry: str) -> bool:
    """Check if chemistry has builtin simpleaf support."""
    return chemistry in BUILTIN_CHEMISTRIES
