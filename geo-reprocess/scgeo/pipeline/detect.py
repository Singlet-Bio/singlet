"""
Protocol detection module for single-cell RNA-seq samples.

Detects sequencing protocol (10x, Smart-seq, Drop-seq, etc.) from FASTQ files.
Uses FASTQ header inspection to determine:
- Read lengths (R1 and R2)
- Protocol type (droplet vs plate-based)
- Chemistry string for quantification

Also provides pre-download FASTQ peek (HTTP Range request) to classify
unknown_sc samples before committing to a full download.
"""
import gzip
import logging
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set, Tuple
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


def _fgdl_min_length(geometry: str, read_num: int) -> int:
    """Compute minimum read length required by an FGDL geometry string for a given read.

    FGDL format: ``1{...}2{...}`` where 1{} and 2{} describe read 1 and read 2.
    Inside braces, elements are: b[N] (barcode), u[N] (UMI), f[SEQ] (fixed),
    x: (discard rest), r: (cDNA reads). Variable-length: b[N-M] uses the minimum N.

    Args:
        geometry: FGDL chemistry string (e.g., "1{b[12]u[8]x:}2{r:}")
        read_num: Which read to compute for (1 or 2)

    Returns:
        Minimum bp required for the specified read, or 0 if not parseable.
    """
    import re
    # Extract the content of the target read's braces
    pattern = rf'{read_num}\{{([^}}]+)\}}'
    match = re.search(pattern, geometry)
    if not match:
        return 0
    content = match.group(1)

    # If the read is just r: (cDNA) or x: (discard), any length works
    if content.strip() in ("r:", "x:"):
        return 0

    total = 0
    # Match all length-consuming elements (b=barcode, u=UMI, f=fixed, x=skip)
    for m in re.finditer(r'[bufx]\[([^\]]+)\]', content):
        val = m.group(1)
        if m.group(0).startswith('f['):
            # Fixed sequence — length is the sequence length
            total += len(val)
        elif '-' in val:
            # Variable length b[N-M] — use minimum N
            low = int(val.split('-')[0])
            total += low
        else:
            total += int(val)
    return total


def _normalize_chemistry(chemistry: str, r1_len: int = 0, r2_len: int = 0, k: int = 31) -> str:
    """Normalize FGDL chemistry strings for simpleaf compatibility.

    Applies two corrections:
    1. Range syntax: ``b[9-10]`` → ``b[10]`` (simpleaf ≤0.19.x can't parse ranges)
    2. Short cDNA fix: if chemistry assigns cDNA to a read shorter than *k*,
       and the OTHER read has enough room after barcodes/UMI, rewrites the
       chemistry so cDNA comes from the longer read.

    Args:
        chemistry: FGDL chemistry string (e.g., ``1{b[9-10]u[8]x:}2{r:}``)
        r1_len: R1 read length (0 = unknown, skip cDNA length check)
        r2_len: R2 read length (0 = unknown / single-end)
        k: piscem k-mer size (default 31)

    Returns:
        Corrected chemistry string (or the original if no fixes needed).
    """
    import re

    if not chemistry or chemistry in ("10xv2", "10xv3", "10xv4-3p", "smartseq"):
        return chemistry

    # ── Fix 1: Replace range notation b[N-M] → b[M] ──
    def _range_to_max(m):
        prefix = m.group(1)   # 'b' or 'u'
        low, high = m.group(2), m.group(3)
        return f"{prefix}[{high}]"

    chemistry = re.sub(r'([bu])\[(\d+)-(\d+)\]', _range_to_max, chemistry)

    # ── Fix 2: Short cDNA rewrite ──
    # Only applies when both read lengths are known
    if r1_len == 0 or r2_len == 0:
        return chemistry

    # Identify which read holds cDNA (contains 'r:')
    m1 = re.search(r'1\{([^}]+)\}', chemistry)
    m2 = re.search(r'2\{([^}]+)\}', chemistry)
    if not m1 or not m2:
        return chemistry

    r1_content, r2_content = m1.group(1), m2.group(1)
    r1_has_cdna = 'r:' in r1_content or 'r[' in r1_content
    r2_has_cdna = 'r:' in r2_content or 'r[' in r2_content

    if r2_has_cdna and not r1_has_cdna and r2_len < k:
        # cDNA assigned to R2 but R2 is too short for k-mer mapping
        # Check if R1 can hold both barcodes AND cDNA
        bc_umi_len = _fgdl_min_length(chemistry, 1)
        r1_cdna_avail = r1_len - bc_umi_len
        if r1_cdna_avail >= k:
            # Rewrite: move cDNA to R1, discard R2
            new_r1 = r1_content.replace('x:', 'r:')
            new_r2 = 'x:'
            chemistry = f"1{{{new_r1}}}2{{{new_r2}}}"
            logger.info(
                f"Rewrote chemistry: R2={r2_len}bp < k={k}, using R1 cDNA "
                f"({r1_cdna_avail}bp available after barcode/UMI)"
            )

    elif r1_has_cdna and not r2_has_cdna and r1_len < k:
        # Mirror case: cDNA in R1 but R1 too short
        bc_umi_len = _fgdl_min_length(chemistry, 2)
        r2_cdna_avail = r2_len - bc_umi_len
        if r2_cdna_avail >= k:
            new_r2 = r2_content.replace('x:', 'r:')
            new_r1 = 'x:'
            chemistry = f"1{{{new_r1}}}2{{{new_r2}}}"
            logger.info(
                f"Rewrote chemistry: R1={r1_len}bp < k={k}, using R2 cDNA "
                f"({r2_cdna_avail}bp available after barcode/UMI)"
            )

    return chemistry


# ── Module-level barcode whitelist cache ──
_BARCODE_WHITELIST: Optional[Set[str]] = None


@dataclass
class ProtocolDetection:
    """Result of protocol detection from FASTQ inspection.
    
    Attributes:
        protocol: Detected protocol name ("10xv3", "smartseq2", "dropseq", etc.)
        mode: Processing mode ("droplet" or "smartseq")
        confidence: Detection confidence ("high", "medium", "low")
        reason: Human-readable explanation
        r1_len: R1 read length
        r2_len: R2 read length (0 for single-end)
        chemistry: simpleaf chemistry string (for droplet protocols)
        reads_swapped: True when physical R1/R2 files should be swapped for simpleaf
    """
    protocol: str
    mode: str  # "droplet" or "smartseq"
    confidence: str  # "high", "medium", "low"
    reason: str
    r1_len: int = 0
    r2_len: int = 0
    chemistry: Optional[str] = None
    reads_swapped: bool = False


def detect_read_length(fastq_path: Path, num_reads: int = 100) -> Tuple[int, int]:
    """Determine read length from FASTQ file.
    
    Args:
        fastq_path: Path to FASTQ file (.fastq or .fastq.gz)
        num_reads: Number of reads to sample (default: 100)
        
    Returns:
        (median_length, n_reads_sampled)
    """
    lengths = []
    
    try:
        # Handle gzipped or plain FASTQ
        if str(fastq_path).endswith('.gz'):
            fh = gzip.open(fastq_path, 'rt')
        else:
            fh = open(fastq_path, 'r')
        
        try:
            line_count = 0
            for line in fh:
                if line_count % 4 == 1:  # Sequence line
                    lengths.append(len(line.strip()))
                    if len(lengths) >= num_reads:
                        break
                line_count += 1
        finally:
            fh.close()
        
        if not lengths:
            return 0, 0
        
        # Return median length
        lengths.sort()
        median_len = lengths[len(lengths) // 2]
        return median_len, len(lengths)
        
    except Exception as e:
        logger.error(f"Failed to detect read length from {fastq_path}: {e}")
        return 0, 0


def detect_file_length(fastq_path: Path) -> int:
    """Quick read length detection (first read only).
    
    Args:
        fastq_path: Path to FASTQ file
        
    Returns:
        Read length (0 if failed)
    """
    try:
        if str(fastq_path).endswith('.gz'):
            fh = gzip.open(fastq_path, 'rt')
        else:
            fh = open(fastq_path, 'r')
        
        try:
            # Skip first line (header)
            fh.readline()
            # Read sequence line
            seq = fh.readline().strip()
            return len(seq)
        finally:
            fh.close()
    except Exception as e:
        logger.error(f"Failed to read {fastq_path}: {e}")
        return 0


def _load_barcode_whitelist() -> Set[str]:
    """Load and cache combined 10x v2+v3 barcode whitelist from ALEVIN_FRY_HOME.

    The whitelist files are stored as plain text (one barcode per line) in
    $ALEVIN_FRY_HOME/plist/.  We load all files found there, extracting the
    first 16 bases of each valid DNA barcode, and cache the result as a
    module-level set for reuse across samples.

    Returns:
        Set of 16-bp barcode prefixes.
    """
    global _BARCODE_WHITELIST
    if _BARCODE_WHITELIST is not None:
        return _BARCODE_WHITELIST

    af_home = os.environ.get("ALEVIN_FRY_HOME", os.path.expanduser("~/.alevin_fry"))
    plist_dir = Path(af_home) / "plist"

    barcodes: set = set()
    if plist_dir.exists():
        valid_bases = set("ACGTNacgtn")
        for wl_file in plist_dir.iterdir():
            if not wl_file.is_file():
                continue
            try:
                with open(wl_file) as f:
                    for line in f:
                        bc = line.strip()
                        if len(bc) >= 12 and all(c in valid_bases for c in bc[:16]):
                            barcodes.add(bc[:16].upper())
            except Exception as e:
                logger.warning(f"Failed to load whitelist {wl_file}: {e}")

    logger.info(f"Loaded {len(barcodes):,} barcode prefixes from {plist_dir}")
    _BARCODE_WHITELIST = barcodes
    return barcodes


def _check_barcode_fraction(
    fastq_path: Path,
    whitelist: Set[str],
    n_reads: int = 200,
    bc_len: int = 16,
) -> float:
    """Check what fraction of reads start with a known 10x barcode.

    Samples the first *n_reads* sequences from *fastq_path* and checks
    whether the leading *bc_len* bases appear in *whitelist*.

    Returns:
        Fraction of reads matching (0.0 – 1.0).
    """
    matches = 0
    total = 0
    try:
        opener = gzip.open if str(fastq_path).endswith('.gz') else open
        with opener(fastq_path, 'rt') as fh:
            line_count = 0
            for line in fh:
                if line_count % 4 == 1:  # sequence line
                    seq = line.strip()[:bc_len].upper()
                    if len(seq) >= bc_len and seq in whitelist:
                        matches += 1
                    total += 1
                    if total >= n_reads:
                        break
                line_count += 1
    except Exception as e:
        logger.debug(f"Barcode check failed for {fastq_path}: {e}")
    return matches / max(total, 1)


def infer_protocol(
    r1_len: int,
    r2_len: int,
    catalog_hint: Optional[str] = None,
    config = None,
    chemistry_hint: Optional[str] = None,
    catalog_confidence: Optional[str] = None,
) -> ProtocolDetection:
    """Infer scRNA-seq protocol from read lengths and optional catalog metadata.
    
    When a catalog_hint is provided (e.g., from the catalog's protocol_inferred
    column), it takes priority over read-length heuristics — only falling back
    to heuristics if the hint is None, empty, or "unknown".  This resolves the
    major class of failures where ambiguous R1 lengths (e.g., 25bp) were
    misclassified via heuristics alone.
    
    Args:
        r1_len: R1 read length
        r2_len: R2 read length (0 for single-end)
        catalog_hint: Protocol hint from GEO metadata (e.g., "10xv3", "dropseq")
        config: Configuration object with protocol detection thresholds
        chemistry_hint: Per-sample chemistry from catalog (overrides PROTOCOL_CHEMISTRY)
        catalog_confidence: Confidence from the catalog ("high", "medium", "low").
            When provided, the output confidence is capped at this level.
        
    Returns:
        ProtocolDetection with protocol, mode, and confidence
    """
    from scgeo.config.protocols import get_chemistry, is_droplet_protocol
    
    # ── Stage 1: Try catalog hint first ──
    if catalog_hint and catalog_hint.lower() not in ("unknown", "unknown_sc", ""):
        hint = catalog_hint.lower().strip()
        # Use per-sample chemistry_hint if provided, else fall back to PROTOCOL_CHEMISTRY
        if chemistry_hint:
            chemistry = chemistry_hint
        else:
            chemistry = get_chemistry(hint)
        
        if chemistry is not None:
            # Always convert f[SEQ] → x[len(SEQ)] (positional skip) because
            # simpleaf strips f[] elements before passing geometry to piscem,
            # which causes barcode extraction at wrong positions.
            # PROTOCOL_CHEMISTRY already uses x[] but chemistry_hint from
            # catalogs may still contain f[] sequences.
            import re as _re
            if _re.search(r'f\[', chemistry):
                chemistry = _re.sub(r'f\[([^\]]+)\]',
                                    lambda m: f'x[{len(m.group(1))}]',
                                    chemistry)
                logger.info(f"Converted f[] anchors to x[] skips: {chemistry}")
            mode = "smartseq" if chemistry == "smartseq" else "droplet"

            # Guard: droplet protocol but data is single-end (no R2)
            # This catches GEO misannotations and datasets that only uploaded R1.
            if mode == "droplet" and r2_len == 0:
                logger.warning(
                    f"Catalog hint '{hint}' is a droplet protocol but data is single-end "
                    f"(R1={r1_len}bp, R2=0). Skipping catalog hint — falling through to heuristics."
                )
                # Fall through to Stage 2 heuristics below
            else:
                # Normalize chemistry for simpleaf compatibility:
                # - Fix range notation (b[9-10] → b[10])
                # - Move cDNA to longer read when short read < k-mer size
                if mode == "droplet":
                    chemistry = _normalize_chemistry(chemistry, r1_len, r2_len)

                # Validate read lengths against chemistry geometry.
                swap_reads = False
                confidence = "high"  # Default; may be downgraded by validation below
                if mode == "droplet" and chemistry in ("10xv2", "10xv3", "10xv4-3p"):
                    # Builtin 10x chemistries expect R1=26-28bp (barcode+UMI).
                    expected_r1 = {"10xv2": 26, "10xv3": 28, "10xv4-3p": 28}[chemistry]
                    
                    # R1 too short for 10x barcode extraction (< 20bp).
                    # R1 must be ≥ 24bp (16bp barcode + at least some UMI).
                    # R1=10bp or R1=20bp is clearly NOT 10x data.
                    if r1_len > 0 and r1_len < 24 and r2_len > 0:
                        logger.warning(
                            f"10x {chemistry}: R1={r1_len}bp is too short for "
                            f"barcode extraction (need ≥24bp). Catalog misclassification."
                        )
                        confidence = "low"
                    # R1 is non-standard length (31-50bp): not matching any 10x chemistry.
                    # 10xv2=26, 10xv3=28, 10xv4=28. 31-50bp doesn't fit any.
                    # Only flag if R2 is also short (ruling out "long R1 = barcode + cDNA")
                    elif 31 <= r1_len <= 50 and r2_len > 50:
                        logger.warning(
                            f"10x {chemistry}: R1={r1_len}bp is non-standard for 10x "
                            f"(expected ~{expected_r1}bp). May be a different protocol."
                        )
                        # Don't downgrade — could be a valid sample with extra bases
                    
                    # R1 length mismatch: catalog says 10xv3 but R1=26bp → 10xv2
                    elif chemistry == "10xv3" and r1_len == 26 and r2_len > 50:
                        logger.info(
                            f"10x chemistry correction: catalog says 10xv3 (28bp R1) "
                            f"but R1={r1_len}bp matches 10xv2 (26bp = 16bp BC + 10bp UMI). "
                            f"Downgrading chemistry to 10xv2."
                        )
                        chemistry = "10xv2"
                    # Reverse: catalog says 10xv2 but R1=28bp → 10xv3
                    elif chemistry == "10xv2" and r1_len == 28 and r2_len > 50:
                        logger.info(
                            f"10x chemistry correction: catalog says 10xv2 (26bp R1) "
                            f"but R1={r1_len}bp matches 10xv3 (28bp = 16bp BC + 12bp UMI). "
                            f"Upgrading chemistry to 10xv3."
                        )
                        chemistry = "10xv3"
                    
                    if r1_len > 50 and 24 <= r2_len <= 30:
                        # R1 is too long, R2 matches barcode length → reads are swapped
                        swap_reads = True
                        logger.info(
                            f"10x {chemistry}: R1={r1_len}bp too long for barcode read, "
                            f"R2={r2_len}bp matches barcode length — will swap R1/R2"
                        )
                    elif r1_len > 50 and r2_len > 50:
                        logger.info(
                            f"10x {chemistry}: both reads long (R1={r1_len}bp, R2={r2_len}bp), "
                            f"will check barcode whitelist to determine orientation"
                        )
                    elif 0 < r2_len < 31:
                        # R2 is too short for piscem k-mer mapping (k=31).
                        # Standard 10x: R1=barcode, R2=biological. If R2 < 31bp,
                        # mapping rate will be 0% — guaranteed failure.
                        logger.warning(
                            f"10x {chemistry}: R2={r2_len}bp is shorter than piscem k-mer size (31bp). "
                            f"Mapping will fail — reads are too truncated."
                        )
                        confidence = "low"

                elif mode == "droplet" and chemistry not in ("10xv2", "10xv3", "10xv4-3p"):
                    r1_need = _fgdl_min_length(chemistry, 1)
                    r2_need = _fgdl_min_length(chemistry, 2)
                    r1_ok = (r1_need == 0 or r1_len >= r1_need)
                    r2_ok = (r2_need == 0 or r2_len >= r2_need)

                    if not r1_ok or not r2_ok:
                        # Check if swapping physical files would fix the issue
                        r1_ok_swap = (r1_need == 0 or r2_len >= r1_need)
                        r2_ok_swap = (r2_need == 0 or r1_len >= r2_need)
                        if r1_ok_swap and r2_ok_swap:
                            swap_reads = True
                            logger.info(
                                f"R1={r1_len}bp too short for chemistry (needs {r1_need}bp), "
                                f"will swap R1/R2 files for simpleaf"
                            )
                            # After swap, the cDNA read is the shorter physical read.
                            # piscem requires reads >= k (31bp) for mapping.
                            # Identify which read becomes cDNA after swap.
                            import re as _re_cdna
                            _m1 = _re_cdna.search(r'1\{([^}]+)\}', chemistry)
                            _m2 = _re_cdna.search(r'2\{([^}]+)\}', chemistry)
                            if _m1 and _m2:
                                _r1c = _m1.group(1)
                                _r2c = _m2.group(1)
                                # After swap: physical R2→logical R1, physical R1→logical R2
                                # cDNA is in whichever logical read has 'r:'
                                _cdna_phys_len = r1_len if ('r:' in _r2c or 'r[' in _r2c) else r2_len
                                if _cdna_phys_len > 0 and _cdna_phys_len < 31:
                                    logger.warning(
                                        f"After swap, cDNA read is only {_cdna_phys_len}bp "
                                        f"(< piscem k=31). Mapping will fail."
                                    )
                                    confidence = "low"
                        else:
                            logger.warning(
                                f"Read lengths R1={r1_len}bp/R2={r2_len}bp incompatible with "
                                f"chemistry {chemistry} (needs R1≥{r1_need}bp, R2≥{r2_need}bp)"
                            )
                            confidence = "low"
                    else:
                        # FGDL structural requirements met — but still check
                        # that the cDNA read is long enough for piscem (k=31).
                        import re as _re_cdna2
                        _m1 = _re_cdna2.search(r'1\{([^}]+)\}', chemistry)
                        _m2 = _re_cdna2.search(r'2\{([^}]+)\}', chemistry)
                        if _m1 and _m2:
                            _r1c = _m1.group(1)
                            _r2c = _m2.group(1)
                            _cdna_len = r2_len if ('r:' in _r2c or 'r[' in _r2c) else r1_len
                            if 0 < _cdna_len < 31:
                                logger.warning(
                                    f"cDNA read is only {_cdna_len}bp (< piscem k=31). "
                                    f"Mapping will fail."
                                )
                                confidence = "low"

                # Cap confidence at the catalog's own confidence level.
                # E.g. catalog says protocol_confidence=low → don't upgrade to "high".
                # But also respect validation downgrades (e.g. R1 too short → "low").
                _levels = {"low": 0, "medium": 1, "high": 2}
                if catalog_confidence and catalog_confidence in ("low", "medium"):
                    if _levels.get(catalog_confidence, 2) < _levels.get(confidence, 2):
                        confidence = catalog_confidence
                    if catalog_confidence == "low":
                        logger.info(
                            f"Catalog hint '{hint}' has low confidence — "
                            f"preserving low confidence for downstream gating"
                        )

                return ProtocolDetection(
                    protocol=hint,
                    mode=mode,
                    confidence=confidence,
                    reason=f"Catalog protocol='{hint}', chemistry='{chemistry}' (R1={r1_len}bp, R2={r2_len}bp)",
                    r1_len=r1_len,
                    r2_len=r2_len,
                    chemistry=chemistry if mode == "droplet" else None,
                    reads_swapped=swap_reads,
                )
    
    # ── Stage 2: Read-length heuristics (fallback) ──
    # Apply config thresholds if provided
    droplet_max = config.detection.droplet_r1_max if config else 75
    long_threshold = config.detection.rlen_long_threshold if config else 85
    
    # Single-end likely means Smart-seq
    if r2_len == 0:
        return ProtocolDetection(
            protocol="smartseq2",
            mode="smartseq",
            confidence="high",
            reason="Single-end data (Smart-seq)",
            r1_len=r1_len,
            r2_len=0,
            chemistry=None
        )
    
    # Short R1 (24-30bp) → likely 10x Chromium or similar droplet
    # 10xv2: 16bp BC + 10bp UMI = 26bp
    # 10xv3: 16bp BC + 12bp UMI = 28bp
    # Applies regardless of R2 length (R2 can be short if library was sequenced with few cycles)
    if 24 <= r1_len <= 30 and r2_len > 0:
        # Determine 10x chemistry version from R1 length
        if r1_len == 28:
            chemistry = "10xv3"
        elif r1_len in (26, 27):
            chemistry = "10xv2"
        else:
            chemistry = "10xv2"  # Default for 24-30 range
        
        conf = "high" if r2_len > long_threshold else "medium"
        return ProtocolDetection(
            protocol=chemistry,
            mode="droplet",
            confidence=conf,
            reason=f"R1={r1_len}bp (10x barcode), R2={r2_len}bp",
            r1_len=r1_len,
            r2_len=r2_len,
            chemistry=chemistry
        )
    
    # Swapped reads — R2 is the barcode read (24-30bp), R1 is cDNA
    if 24 <= r2_len <= 30 and r1_len > 30:
        if r2_len == 28:
            chemistry = "1{r:}2{b[16]u[12]x:}"  # 10xv3, barcodes in R2
            protocol = "10xv3"
        elif r2_len in (26, 27):
            chemistry = "1{r:}2{b[16]u[10]x:}"  # 10xv2, barcodes in R2
            protocol = "10xv2"
        else:
            chemistry = "1{r:}2{b[16]u[10]x:}"
            protocol = "10xv2"

        return ProtocolDetection(
            protocol=protocol,
            mode="droplet",
            confidence="medium",
            reason=f"R1={r1_len}bp (cDNA), R2={r2_len}bp (barcode) — swapped orientation",
            r1_len=r1_len,
            r2_len=r2_len,
            chemistry=chemistry,
            reads_swapped=True,
        )

    # Swapped reads — R2 has truncated barcode/UMI (16-23bp or 31-50bp)
    # R2=16bp: 10x barcode only (no UMI), R2=20-23bp: partial UMI
    # R2=31-50bp: Drop-seq or other droplet barcodes in R2
    if r1_len > 50 and 16 <= r2_len <= 50:
        if r2_len <= 23:
            # Short R2 in barcode range — likely truncated 10x
            chemistry = "1{r:}2{b[16]u[12]x:}" if r2_len >= 20 else "1{r:}2{b[16]x:}"
            return ProtocolDetection(
                protocol="10xv3",
                mode="droplet",
                confidence="low",
                reason=f"R1={r1_len}bp (cDNA), R2={r2_len}bp (truncated barcode) — swapped",
                r1_len=r1_len,
                r2_len=r2_len,
                chemistry=chemistry,
                reads_swapped=True,
            )
        else:
            # R2=31-50bp: could be Drop-seq barcodes in R2
            return ProtocolDetection(
                protocol="dropseq",
                mode="droplet",
                confidence="low",
                reason=f"R1={r1_len}bp (cDNA), R2={r2_len}bp (barcode) — swapped droplet",
                r1_len=r1_len,
                r2_len=r2_len,
                chemistry="1{r:}2{b[12]u[8]x:}",
                reads_swapped=True,
            )

    # Short R1 (31-75bp) + paired-end → could be Drop-seq or ambiguous.
    # For R1 31-50bp, the barcode structure doesn't match 10x but is short enough
    # for a non-10x droplet protocol. Use "medium" confidence.
    # For R1 51-75bp, this is ambiguous — could be droplet or degraded PE data.
    # Use "low" confidence to trigger barcode whitelist check downstream.
    if r1_len <= droplet_max and r2_len > 0:
        if r1_len <= 50:
            return ProtocolDetection(
                protocol="dropseq",
                mode="droplet",
                confidence="medium",
                reason=f"R1={r1_len}bp, R2={r2_len}bp (likely droplet barcode)",
                r1_len=r1_len,
                r2_len=r2_len,
                chemistry="dropseq"
            )
        else:
            # 51-75bp R1: ambiguous zone — could be truncated PE or droplet.
            # Return low confidence so barcode whitelist matching is triggered.
            return ProtocolDetection(
                protocol="unknown",
                mode="unknown",
                confidence="low",
                reason=f"R1={r1_len}bp, R2={r2_len}bp (ambiguous — short PE or non-standard droplet)",
                r1_len=r1_len,
                r2_len=r2_len,
                chemistry=None
            )
    
    # Both reads long → Smart-seq
    if r1_len > long_threshold and r2_len > long_threshold:
        return ProtocolDetection(
            protocol="smartseq2",
            mode="smartseq",
            confidence="high",
            reason=f"R1={r1_len}bp, R2={r2_len}bp (full-length paired-end)",
            r1_len=r1_len,
            r2_len=r2_len,
            chemistry=None
        )
    
    # One read long + one read short (76-85bp) — ambiguous zone
    # These are commonly R1>85 + R2=76-85 or vice versa.
    # Return low confidence to trigger whitelist matching.
    return ProtocolDetection(
        protocol="unknown",
        mode="unknown",
        confidence="low",
        reason=f"R1={r1_len}bp, R2={r2_len}bp (ambiguous lengths)",
        r1_len=r1_len,
        r2_len=r2_len,
        chemistry=None
    )


def detect_protocol_from_files(
    r1_path: Path,
    r2_path: Optional[Path],
    catalog_hint: Optional[str] = None,
    chemistry_hint: Optional[str] = None,
    config = None,
    catalog_confidence: Optional[str] = None,
) -> ProtocolDetection:
    """Detect protocol by inspecting FASTQ files.
    
    Args:
        r1_path: R1 FASTQ file path
        r2_path: R2 FASTQ file path (None for single-end)
        catalog_hint: Protocol hint from metadata catalog
        chemistry_hint: Per-sample chemistry from catalog (overrides PROTOCOL_CHEMISTRY)
        config: Configuration object
        catalog_confidence: Confidence from the catalog ("high", "medium", "low").
            When provided, the output confidence is capped at this level.
        
    Returns:
        ProtocolDetection with protocol type and confidence
    """
    # Detect read lengths
    r1_len, n_reads = detect_read_length(r1_path, num_reads=100)
    
    if r1_len == 0:
        return ProtocolDetection(
            protocol="unknown",
            mode="unknown",
            confidence="low",
            reason="Failed to read R1 file",
            r1_len=0,
            r2_len=0,
            chemistry=None
        )
    
    r2_len = 0
    if r2_path:
        r2_len, _ = detect_read_length(r2_path, num_reads=100)
    
    # Infer protocol from read lengths
    detection = infer_protocol(r1_len, r2_len, catalog_hint, config, chemistry_hint=chemistry_hint, catalog_confidence=catalog_confidence)

    # ── Barcode orientation check for 10x with both reads long ──
    # When catalog says 10x but BOTH R1 & R2 are >50bp (e.g., 150×150), we can't
    # tell from lengths alone which read has barcodes. Use whitelist matching to
    # determine orientation and set reads_swapped accordingly.
    if (detection.chemistry in ("10xv2", "10xv3", "10xv4-3p")
            and r1_len > 50 and r2_len > 50
            and r2_path
            and not detection.reads_swapped):
        try:
            whitelist = _load_barcode_whitelist()
            if whitelist:
                r1_frac = _check_barcode_fraction(r1_path, whitelist)
                r2_frac = _check_barcode_fraction(r2_path, whitelist)
                logger.info(
                    f"10x barcode orientation check: R1={r1_frac:.1%}, R2={r2_frac:.1%}"
                )
                BARCODE_THRESHOLD = 0.30
                if r2_frac >= BARCODE_THRESHOLD and r2_frac > r1_frac:
                    # Barcodes are in R2, not R1 — need to swap
                    detection = ProtocolDetection(
                        protocol=detection.protocol,
                        mode="droplet",
                        confidence="high",
                        reason=(
                            f"Catalog={catalog_hint}, R1={r1_len}bp R2={r2_len}bp; "
                            f"barcode whitelist R1={r1_frac:.0%} R2={r2_frac:.0%} — SWAPPED"
                        ),
                        r1_len=r1_len,
                        r2_len=r2_len,
                        chemistry=detection.chemistry,
                        reads_swapped=True,
                    )
                elif r1_frac >= BARCODE_THRESHOLD:
                    # Barcodes are in R1 (standard orientation), no swap needed
                    logger.info(
                        f"10x barcodes confirmed in R1 ({r1_frac:.0%}), standard orientation"
                    )
                else:
                    # Neither read has good barcode match — this is NOT 10x data.
                    # Downgrade to "low" confidence so the pipeline rejects it.
                    logger.warning(
                        f"10x {detection.chemistry}: neither read matches barcode whitelist "
                        f"(R1={r1_frac:.0%}, R2={r2_frac:.0%}). Downgrading to low confidence."
                    )
                    detection = ProtocolDetection(
                        protocol=detection.protocol,
                        mode="droplet",
                        confidence="low",
                        reason=(
                            f"Catalog={catalog_hint} but 0% barcode whitelist match "
                            f"(R1={r1_frac:.0%}, R2={r2_frac:.0%}) — not 10x data"
                        ),
                        r1_len=r1_len,
                        r2_len=r2_len,
                        chemistry=detection.chemistry,
                        reads_swapped=False,
                    )
        except Exception as e:
            logger.warning(f"Barcode orientation check failed: {e}")

    # ── Barcode whitelist fallback for ambiguous cases ──
    # When heuristics can't determine the protocol (e.g. R1=101, R2=101),
    # check whether R1 or R2 starts with known 10x barcodes.
    if detection.confidence == "low" and r2_path and r1_len > 0 and r2_len > 0:
        try:
            whitelist = _load_barcode_whitelist()
            if whitelist:
                r1_frac = _check_barcode_fraction(r1_path, whitelist)
                r2_frac = _check_barcode_fraction(r2_path, whitelist)
                logger.info(
                    f"Whitelist matching: R1={r1_frac:.1%}, R2={r2_frac:.1%}"
                )

                BARCODE_THRESHOLD = 0.30  # ≥30 % match → confident barcodes

                if r1_frac >= BARCODE_THRESHOLD and r1_frac >= r2_frac:
                    # R1 contains barcodes → R2 is cDNA.
                    # Don't promote if cDNA read is too short for piscem.
                    if 0 < r2_len < 31:
                        logger.warning(
                            f"Barcode match confirmed (R1={r1_frac:.0%}) but "
                            f"R2={r2_len}bp too short for piscem (k=31). "
                            f"Keeping low confidence."
                        )
                    else:
                        detection = ProtocolDetection(
                            protocol="10xv3",
                            mode="droplet",
                            confidence="medium",
                            reason=(
                                f"R1={r1_len}bp, R2={r2_len}bp; "
                                f"barcode whitelist match R1={r1_frac:.0%}"
                            ),
                            r1_len=r1_len,
                            r2_len=r2_len,
                            chemistry="10xv3",
                        )
                elif r2_frac >= BARCODE_THRESHOLD and r2_frac > r1_frac:
                    # R2 contains barcodes → R1 is cDNA (swapped).
                    # Don't promote if cDNA read is too short for piscem.
                    if 0 < r1_len < 31:
                        logger.warning(
                            f"Barcode match confirmed (R2={r2_frac:.0%}) but "
                            f"R1={r1_len}bp too short for piscem (k=31). "
                            f"Keeping low confidence."
                        )
                    else:
                        detection = ProtocolDetection(
                            protocol="10xv3",
                            mode="droplet",
                            confidence="medium",
                            reason=(
                                f"R1={r1_len}bp, R2={r2_len}bp; "
                                f"barcode whitelist match R2={r2_frac:.0%} (SWAPPED)"
                            ),
                            r1_len=r1_len,
                            r2_len=r2_len,
                            chemistry="1{r:}2{b[16]u[12]x:}",
                        )
                elif r1_len >= 86 and r2_len >= 86:
                    # Both reads long, no barcode match → likely Smart-seq
                    detection = ProtocolDetection(
                        protocol="smartseq2",
                        mode="smartseq",
                        confidence="medium",
                        reason=(
                            f"R1={r1_len}bp, R2={r2_len}bp; "
                            f"no barcode match, likely Smart-seq"
                        ),
                        r1_len=r1_len,
                        r2_len=r2_len,
                        chemistry=None,
                    )
                elif r1_len >= 51 and r2_len >= 51:
                    # Both reads medium-length (51-85bp), no barcode match.
                    # Commonly truncated Smart-seq PE data (R1=R2=51/75/76/80).
                    detection = ProtocolDetection(
                        protocol="smartseq2",
                        mode="smartseq",
                        confidence="medium",
                        reason=(
                            f"R1={r1_len}bp, R2={r2_len}bp; "
                            f"no barcode match (R1={r1_frac:.0%}, R2={r2_frac:.0%}), "
                            f"likely truncated Smart-seq PE"
                        ),
                        r1_len=r1_len,
                        r2_len=r2_len,
                        chemistry=None,
                    )
                # else: stays as "low" confidence → pipeline will skip/fail
        except Exception as e:
            logger.warning(f"Barcode whitelist matching failed: {e}")

    logger.info(f"Detected: {detection.protocol} ({detection.confidence} confidence) - {detection.reason}")

    return detection


# ── Pre-download FASTQ peek ──

# Maximum barcode+UMI read length for any known droplet protocol.
# inDrop v3 is the widest at ~50bp.  Anything above this in BOTH reads
# is definitively non-droplet (plate-based / Smart-seq).
_PEEK_BARCODE_MAX_BP = 50

# Compressed bytes to fetch via HTTP Range request.  16 KB of gzipped FASTQ
# decompresses to ~1000 lines (≈250 reads) — more than enough.
_PEEK_BYTES = 16384


def peek_fastq_read_length(
    url: str,
    num_reads: int = 10,
    timeout: int = 15,
) -> Optional[int]:
    """Fetch the first bytes of a gzipped FASTQ via HTTP Range and return the
    median read length, or *None* on any failure.

    The ENA FTP URLs (``ftp://ftp.sra.ebi.ac.uk/...``) are transparently
    converted to their HTTP mirror so that Range requests work.

    Args:
        url: ENA FASTQ URL (FTP or HTTP).
        num_reads: Number of reads to sample from the decompressed header.
        timeout: HTTP timeout in seconds.

    Returns:
        Median read length (int) or None if the peek fails.
    """
    http_url = url.replace("ftp://", "http://")
    req = Request(http_url)
    req.add_header("Range", f"bytes=0-{_PEEK_BYTES - 1}")

    try:
        resp = urlopen(req, timeout=timeout)
        compressed = resp.read()
    except Exception as exc:
        logger.debug(f"FASTQ peek failed for {url}: {exc}")
        return None

    try:
        decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)  # auto gzip
        text = decompressor.decompress(compressed).decode("ascii", errors="ignore")
    except Exception as exc:
        logger.debug(f"FASTQ peek decompress failed for {url}: {exc}")
        return None

    lengths: list[int] = []
    for i, line in enumerate(text.split("\n")):
        if i % 4 == 1 and line.strip():
            lengths.append(len(line.strip()))
            if len(lengths) >= num_reads:
                break

    if not lengths:
        return None

    lengths.sort()
    return lengths[len(lengths) // 2]


def peek_protocol(
    ena_r1_url: Optional[str],
    ena_r2_url: Optional[str] = None,
    timeout: int = 15,
) -> Optional[ProtocolDetection]:
    """Pre-download protocol check via FASTQ header peek.

    Fetches the first 16 KB of R1 (and R2 if available) from ENA's HTTP
    mirror, decompresses the gzip header, and checks actual read lengths.

    Only returns a :class:`ProtocolDetection` for **high-confidence** cases:

    - R1 > 50 bp **and** R2 > 50 bp → plate-based (smartseq) → **skip**
    - R1 > 50 bp and no R2 URL → single-end plate-based → **skip**

    All other cases (peek failure, ambiguous lengths, short R1, etc.) return
    ``None``, causing the caller to fall through to the normal
    download-then-detect pipeline.

    Args:
        ena_r1_url: ENA R1 FASTQ URL.
        ena_r2_url: ENA R2 FASTQ URL (may be None / empty).
        timeout: HTTP timeout in seconds.

    Returns:
        :class:`ProtocolDetection` with ``mode="smartseq"`` to skip, or
        ``None`` to fall through.
    """
    if not ena_r1_url:
        return None

    r1_len = peek_fastq_read_length(ena_r1_url, timeout=timeout)
    if r1_len is None:
        logger.debug("FASTQ peek: R1 peek failed — falling through")
        return None

    # Short R1 → barcode read → definitely *not* smartseq → proceed normally
    if r1_len <= _PEEK_BARCODE_MAX_BP:
        logger.info(
            f"FASTQ peek: R1={r1_len}bp (≤{_PEEK_BARCODE_MAX_BP}bp) — "
            f"likely droplet barcode, proceeding with full download"
        )
        return None

    # R1 is long — check R2
    has_r2_url = ena_r2_url and isinstance(ena_r2_url, str) and len(ena_r2_url) > 5
    if has_r2_url:
        r2_len = peek_fastq_read_length(ena_r2_url, timeout=timeout)
        if r2_len is None:
            # Can't verify R2 — be conservative, proceed
            logger.debug("FASTQ peek: R2 peek failed — falling through")
            return None

        if r2_len <= _PEEK_BARCODE_MAX_BP:
            # R1 long + R2 short → swapped droplet orientation → proceed
            logger.info(
                f"FASTQ peek: R1={r1_len}bp, R2={r2_len}bp — "
                f"likely swapped-orientation droplet, proceeding"
            )
            return None

        # Both reads long → plate-based / Smart-seq
        logger.info(
            f"FASTQ peek: R1={r1_len}bp, R2={r2_len}bp — "
            f"both reads long, classifying as Smart-seq (pre-download skip)"
        )
        return ProtocolDetection(
            protocol="smartseq2",
            mode="smartseq",
            confidence="high",
            reason=(
                f"FASTQ peek: R1={r1_len}bp, R2={r2_len}bp "
                f"(both >{_PEEK_BARCODE_MAX_BP}bp — full-length reads)"
            ),
            r1_len=r1_len,
            r2_len=r2_len,
            chemistry=None,
        )

    # R1 long + no R2 URL → single-end plate-based
    logger.info(
        f"FASTQ peek: R1={r1_len}bp, no R2 URL — "
        f"single-end full-length, classifying as Smart-seq (pre-download skip)"
    )
    return ProtocolDetection(
        protocol="smartseq2",
        mode="smartseq",
        confidence="high",
        reason=f"FASTQ peek: R1={r1_len}bp, no R2 (single-end full-length reads)",
        r1_len=r1_len,
        r2_len=0,
        chemistry=None,
    )


def get_chemistry_string(protocol: str, r1_len: int, r2_len: int, config = None) -> Optional[str]:
    """Get simpleaf chemistry string for a protocol.
    
    Args:
        protocol: Protocol name ("10xv2", "10xv3", "dropseq", etc.)
        r1_len: R1 read length
        r2_len: R2 read length
        config: Configuration object with protocol chemistry mappings
        
    Returns:
        Chemistry string for simpleaf (e.g., "10xv3", "1{b[16]u[12]x:}2{r:}")
        None if protocol doesn't require chemistry string
    """
    # Use config mappings if available (includes all protocols)
    if config and hasattr(config, 'protocol_chemistry'):
        return config.protocol_chemistry.get(protocol)
    
    # Fallback: use PROTOCOL_CHEMISTRY directly (complete mapping)
    from scgeo.config.protocols import PROTOCOL_CHEMISTRY
    return PROTOCOL_CHEMISTRY.get(protocol)
