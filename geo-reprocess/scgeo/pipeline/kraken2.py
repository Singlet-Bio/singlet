"""
Kraken2 non-host classification module for single-cell RNA-seq.

Classifies cDNA reads (R2) against a reference database (e.g., PlusPF)
to detect non-host organisms at the per-cell (barcode) level.

Produces:
- kraken2_report.txt: Standard kraken2 classification report
- kraken2_cell_taxa.parquet: Per-barcode per-taxon UMI counts
- kraken2_summary.json: Aggregate stats with top taxa
"""
import gzip
import json
import logging
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Kraken2Result:
    """Result of kraken2 non-host classification.
    
    Attributes:
        success: Whether classification succeeded
        total_reads: Total reads processed
        classified_reads: Reads classified by kraken2
        nonhost_reads: Classified reads not matching host
        frac_classified: Fraction of reads classified
        frac_nonhost: Fraction of reads classified as non-host
        cells_with_nonhost: Number of cells with non-host UMIs
        total_nonhost_umis: Total non-host UMI count
        time_s: Classification time in seconds
        error: Error message if failed
        top_taxa: Top taxa list (dicts with taxon_id, name, reads, pct)
    """
    success: bool = False
    total_reads: int = 0
    classified_reads: int = 0
    nonhost_reads: int = 0
    frac_classified: float = 0.0
    frac_nonhost: float = 0.0
    cells_with_nonhost: int = 0
    total_nonhost_umis: int = 0
    time_s: float = 0.0
    error: str = ""
    top_taxa: list = field(default_factory=list)


def _parse_barcode_umi_lens(chemistry: str) -> Tuple[int, int]:
    """Extract barcode and UMI lengths from chemistry string.
    
    Args:
        chemistry: simpleaf chemistry string (e.g., "10xv3", "1{b[12]u[8]x:}2{r:}")
        
    Returns:
        (barcode_len, umi_len)
    """
    # Builtin chemistries
    builtin = {
        "10xv2": (16, 10),
        "10xv3": (16, 12),
        "10xv4-3p": (16, 12),
    }
    if chemistry in builtin:
        return builtin[chemistry]
    
    # Parse geometry string: 1{b[BC]u[UMI]x:}2{r:}
    import re
    bc_match = re.search(r'b\[(\d+)(?:-\d+)?\]', chemistry)
    umi_match = re.search(r'u\[(\d+)\]', chemistry)
    
    if bc_match and umi_match:
        # Handle multi-part barcodes (e.g., splitseq: b[8]b[8]b[8])
        bc_parts = re.findall(r'b\[(\d+)(?:-\d+)?\]', chemistry)
        bc_len = sum(int(x) for x in bc_parts)
        umi_len = int(umi_match.group(1))
        return bc_len, umi_len
    
    # Default fallback (10xv3)
    logger.warning(f"Could not parse chemistry '{chemistry}', defaulting to (16, 12)")
    return 16, 12


def _parse_kraken_report(
    report_path: Path,
    host_taxon_id: int,
    max_taxa: int = 20,
) -> List[Dict]:
    """Parse kraken2 report for top non-host taxa.
    
    Args:
        report_path: Path to kraken2 report.txt
        host_taxon_id: Host taxon ID to exclude
        max_taxa: Maximum number of taxa to return
        
    Returns:
        List of dicts with taxon_id, name, reads_clade, reads_taxon, pct
    """
    taxa = []
    host_clade_ids = set()
    
    try:
        with open(report_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 6:
                    continue
                pct = float(parts[0].strip())
                reads_clade = int(parts[1].strip())
                reads_taxon = int(parts[2].strip())
                rank = parts[3].strip()
                taxon_id = int(parts[4].strip())
                name = parts[5].strip()
                
                # Skip host and its descendants
                if taxon_id == host_taxon_id:
                    host_clade_ids.add(taxon_id)
                    continue
                
                # Only report species (S) and strain (S1/S2) level
                if rank in ("S", "S1", "S2") and taxon_id not in host_clade_ids:
                    taxa.append({
                        "taxon_id": taxon_id,
                        "name": name,
                        "reads_clade": reads_clade,
                        "reads_taxon": reads_taxon,
                        "pct": round(pct, 2),
                    })
        
        # Sort by clade reads and return top N
        taxa.sort(key=lambda x: x["reads_clade"], reverse=True)
        return taxa[:max_taxa]
        
    except Exception as e:
        logger.warning(f"Failed to parse kraken2 report: {e}")
        return []


def _load_valid_barcodes(quant_dir: Path) -> Set[str]:
    """Load valid cell barcodes from alevin-fry output.
    
    Args:
        quant_dir: simpleaf output directory (contains af_quant/)
        
    Returns:
        Set of valid barcode strings
    """
    # Try standard alevin-fry location
    barcodes_file = quant_dir / "af_quant" / "alevin" / "quants_mat_rows.txt"
    if not barcodes_file.exists():
        # Try alternate path
        barcodes_file = quant_dir / "alevin" / "quants_mat_rows.txt"
    
    if not barcodes_file.exists():
        logger.warning(f"Barcodes file not found: {barcodes_file}")
        return set()
    
    barcodes = set()
    with open(barcodes_file) as f:
        for line in f:
            bc = line.strip()
            if bc:
                barcodes.add(bc)
    
    logger.info(f"Loaded {len(barcodes)} valid barcodes")
    return barcodes


def _r1_sequence_reader(r1_paths: List[Path]):
    """Generator yielding the sequence line for each read across R1 FASTQ files.
    
    Reads R1 files in order, yielding one sequence string per FASTQ record.
    Used for lockstep reading alongside kraken2 stdout.
    """
    for r1_path in r1_paths:
        open_fn = gzip.open if str(r1_path).endswith('.gz') else open
        mode = 'rt' if str(r1_path).endswith('.gz') else 'r'
        with open_fn(r1_path, mode) as fh:
            while True:
                header = fh.readline()
                if not header:
                    break
                seq = fh.readline().strip()
                fh.readline()  # +
                fh.readline()  # quality
                yield seq


def classify_nonhost(
    r1_paths: List[Path],
    r2_paths: List[Path],
    chemistry: str,
    host_taxon_id: int,
    quant_dir: Path,
    output_dir: Path,
    config,
) -> Kraken2Result:
    """Classify non-host reads at per-cell level using kraken2.
    
    Pipeline:
    1. Run kraken2 on R2 (cDNA) reads, streaming output
    2. Read R1 in lockstep to extract barcode + UMI for classified reads
    3. Filter to valid cell barcodes (from alevin-fry)
    4. Deduplicate by (barcode, taxon_id, UMI)
    5. Write per-cell taxa counts
    
    Args:
        r1_paths: R1 FASTQ files (barcode reads)
        r2_paths: R2 FASTQ files (cDNA reads)
        chemistry: Chemistry string (for barcode/UMI length parsing)
        host_taxon_id: NCBI taxonomy ID for host organism
        quant_dir: simpleaf output directory (for valid barcodes)
        output_dir: Output directory for kraken2 results
        config: Configuration object with Kraken2Config
        
    Returns:
        Kraken2Result with classification statistics
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not config.kraken2.enabled:
        return Kraken2Result(success=True, time_s=0, error="kraken2 disabled")
    
    db_path = config.kraken2.db
    if not db_path or not Path(db_path).exists():
        return Kraken2Result(
            success=False,
            error=f"Kraken2 database not found: {db_path}",
            time_s=time.time() - t0,
        )
    
    # Parse barcode/UMI lengths from chemistry
    bc_len, umi_len = _parse_barcode_umi_lens(chemistry)
    logger.info(f"Kraken2: chemistry={chemistry}, bc_len={bc_len}, umi_len={umi_len}")
    
    # Load valid barcodes from quantification output
    valid_barcodes = _load_valid_barcodes(quant_dir)
    if not valid_barcodes:
        return Kraken2Result(
            success=False,
            error="No valid barcodes found from quantification output",
            time_s=time.time() - t0,
        )
    
    report_path = output_dir / "kraken2_report.txt"
    
    kraken2_cmd = [
        "kraken2",
        "--db", str(db_path),
        "--threads", str(config.kraken2.threads),
        "--confidence", str(config.kraken2.confidence),
        "--report", str(report_path),
        "--output", "/dev/stdout",  # Stream classifications to stdout
        "--gzip-compressed",
    ]
    if getattr(config.kraken2, "memory_mapping", True):
        kraken2_cmd.insert(3, "--memory-mapping")  # mmap instead of full DB load
    kraken2_cmd.extend(str(p) for p in r2_paths)
    
    logger.info(f"Running kraken2 on {len(r2_paths)} R2 files...")
    
    try:
        # Run kraken2 and capture stdout for line-by-line processing
        kraken_proc = subprocess.Popen(
            kraken2_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        
        # Lockstep R1 reading: advance R1 in sync with kraken2 stdout
        # so we extract barcode+UMI without a second pass over R1.
        # Kraken2 outputs exactly one line per input read, in order.
        r1_iter = _r1_sequence_reader(r1_paths)
        cell_taxa_umis = defaultdict(set)  # barcode -> set of (taxon_id, umi)
        total_reads = 0
        classified_count = 0
        nonhost_count = 0
        
        for line in kraken_proc.stdout:
            total_reads += 1
            
            # Advance R1 in lockstep — one R1 record per kraken2 output line
            try:
                r1_seq = next(r1_iter)
            except StopIteration:
                # R1 exhausted before kraken2 output (file mismatch)
                logger.warning(f"R1 exhausted at read {total_reads}, continuing without barcode extraction")
                r1_seq = None
            
            parts = line.split("\t", 3)  # Only need first 3 fields
            if len(parts) < 3:
                continue
            
            if parts[0] == "C":  # Classified
                classified_count += 1
                taxon_id = int(parts[2])
                if taxon_id != host_taxon_id:
                    nonhost_count += 1
                    if r1_seq is not None:
                        barcode = r1_seq[:bc_len]
                        if barcode in valid_barcodes:
                            umi = r1_seq[bc_len:bc_len + umi_len]
                            cell_taxa_umis[barcode].add((taxon_id, umi))
        
        # Wait for kraken2 to finish
        _, stderr = kraken_proc.communicate(timeout=config.kraken2.timeout)
        
        if kraken_proc.returncode != 0:
            return Kraken2Result(
                success=False,
                error=f"kraken2 failed (exit {kraken_proc.returncode}): {stderr[:500]}",
                time_s=time.time() - t0,
            )
        
        logger.info(f"Kraken2: {total_reads} reads, {classified_count} classified, "
                     f"{nonhost_count} non-host")
        
        if nonhost_count == 0:
            # Write empty results
            _write_empty_results(output_dir, total_reads, classified_count, 
                                 host_taxon_id, time.time() - t0)
            return Kraken2Result(
                success=True,
                total_reads=total_reads,
                classified_reads=classified_count,
                nonhost_reads=0,
                frac_classified=classified_count / max(total_reads, 1),
                time_s=time.time() - t0,
            )
        
        # Compute per-cell per-taxon deduplicated UMI counts
        cell_taxa_counts = defaultdict(lambda: defaultdict(int))
        total_nonhost_umis = 0
        
        for barcode, taxa_umi_set in cell_taxa_umis.items():
            for taxon_id, umi in taxa_umi_set:
                cell_taxa_counts[barcode][taxon_id] += 1
                total_nonhost_umis += 1
        
        cells_with_nonhost = len(cell_taxa_counts)
        
        logger.info(f"Kraken2: {cells_with_nonhost} cells with non-host, "
                     f"{total_nonhost_umis} non-host UMIs")
        
        # Write kraken2_cell_taxa.parquet
        _write_cell_taxa_parquet(cell_taxa_counts, output_dir)
        
        # Parse report for top taxa
        top_taxa = _parse_kraken_report(report_path, host_taxon_id)
        
        # Write summary JSON
        elapsed = time.time() - t0
        summary = {
            "kraken2_version": _get_kraken2_version(),
            "total_reads": total_reads,
            "classified_reads": classified_count,
            "nonhost_classified_reads": nonhost_count,
            "frac_classified": round(classified_count / max(total_reads, 1), 6),
            "frac_nonhost": round(nonhost_count / max(total_reads, 1), 6),
            "cells_with_nonhost": cells_with_nonhost,
            "total_nonhost_umis": total_nonhost_umis,
            "cells_infected": sum(
                1 for bc_taxa in cell_taxa_counts.values()
                if sum(bc_taxa.values()) >= config.kraken2.min_nonhost_umis
            ),
            "barcode_len": bc_len,
            "umi_len": umi_len,
            "host_taxon_id": host_taxon_id,
            "kraken2_time_s": round(elapsed, 1),
            "top_taxa": top_taxa,
        }
        
        summary_path = output_dir / "kraken2_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        return Kraken2Result(
            success=True,
            total_reads=total_reads,
            classified_reads=classified_count,
            nonhost_reads=nonhost_count,
            frac_classified=round(classified_count / max(total_reads, 1), 6),
            frac_nonhost=round(nonhost_count / max(total_reads, 1), 6),
            cells_with_nonhost=cells_with_nonhost,
            total_nonhost_umis=total_nonhost_umis,
            time_s=elapsed,
            top_taxa=top_taxa,
        )
        
    except subprocess.TimeoutExpired:
        if 'kraken_proc' in dir():
            kraken_proc.kill()
        return Kraken2Result(
            success=False,
            error=f"kraken2 timed out after {config.kraken2.timeout}s",
            time_s=time.time() - t0,
        )
    except Exception as e:
        return Kraken2Result(
            success=False,
            error=f"kraken2 exception: {e}",
            time_s=time.time() - t0,
        )


def _write_cell_taxa_parquet(
    cell_taxa_counts: Dict[str, Dict[int, int]],
    output_dir: Path,
) -> None:
    """Write per-cell per-taxon UMI counts to parquet.
    
    Args:
        cell_taxa_counts: {barcode: {taxon_id: umi_count}}
        output_dir: Output directory
    """
    try:
        import pandas as pd
        
        rows = []
        for barcode, taxa in cell_taxa_counts.items():
            for taxon_id, count in taxa.items():
                rows.append({
                    "barcode": barcode,
                    "taxon_id": taxon_id,
                    "umi_count": count,
                })
        
        if rows:
            df = pd.DataFrame(rows)
            df.to_parquet(output_dir / "kraken2_cell_taxa.parquet", index=False)
            logger.info(f"Wrote {len(df)} cell-taxa records to parquet")
        else:
            # Write empty parquet
            df = pd.DataFrame(columns=["barcode", "taxon_id", "umi_count"])
            df.to_parquet(output_dir / "kraken2_cell_taxa.parquet", index=False)
            
    except Exception as e:
        logger.warning(f"Failed to write cell taxa parquet: {e}")


def _write_empty_results(
    output_dir: Path,
    total_reads: int,
    classified_reads: int,
    host_taxon_id: int,
    elapsed: float,
) -> None:
    """Write empty kraken2 results when no non-host reads found."""
    summary = {
        "kraken2_version": _get_kraken2_version(),
        "total_reads": total_reads,
        "classified_reads": classified_reads,
        "nonhost_classified_reads": 0,
        "frac_classified": round(classified_reads / max(total_reads, 1), 6),
        "frac_nonhost": 0.0,
        "cells_with_nonhost": 0,
        "total_nonhost_umis": 0,
        "cells_infected": 0,
        "host_taxon_id": host_taxon_id,
        "kraken2_time_s": round(elapsed, 1),
        "top_taxa": [],
    }
    
    with open(output_dir / "kraken2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    try:
        import pandas as pd
        df = pd.DataFrame(columns=["barcode", "taxon_id", "umi_count"])
        df.to_parquet(output_dir / "kraken2_cell_taxa.parquet", index=False)
    except Exception:
        pass


def _get_kraken2_version() -> str:
    """Get kraken2 version string."""
    try:
        result = subprocess.run(
            ["kraken2", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.split("\n"):
            if "version" in line.lower():
                return line.strip().split()[-1]
        return "unknown"
    except Exception:
        return "unknown"
