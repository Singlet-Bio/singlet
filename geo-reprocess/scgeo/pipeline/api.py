"""
High-level API for sc-geo pipeline.

Provides user-friendly functions for processing single-cell samples:
- process_sample: Single GSM processing
- process_gse: Batch GSE processing
- process_samples: Parallel batch processing
"""
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, List

from scgeo.config import get_config
from scgeo.pipeline.download import download_sample, DownloadResult, compute_download_timeout
from scgeo.pipeline.detect import detect_protocol_from_files, ProtocolDetection
from scgeo.pipeline.quantify import quantify_simpleaf, QuantResult
from scgeo.pipeline.qc import compute_qc_simpleaf, QCMetrics
from scgeo.pipeline.kraken2 import classify_nonhost, Kraken2Result

# Lazy imports for .1pz export (singlepress, scipy.io)
# These are only needed in _export_counts_matrix() and deferred to avoid
# hard dependency on singlepress at import time.
import pandas as pd

logger = logging.getLogger(__name__)

PIPELINE_VERSION = "0.2.0"


def _deinterleave_fastq(r1_path: Path, gsm_id: str) -> tuple:
    """Deinterleave a single interleaved FASTQ into separate R1/R2 files.

    Some protocols (Bio-Rad SureCell, etc.) store paired reads interleaved
    in a single FASTQ file. This splits odd reads into R1 and even reads
    into R2.

    Args:
        r1_path: Path to the interleaved FASTQ file
        gsm_id: Sample ID for logging

    Returns:
        (new_r1_path, new_r2_path) if deinterleaving succeeded,
        (None, None) if the file does not appear to be interleaved.
    """
    import gzip

    # Quick check: peek at first 8 reads to see if alternating /1 and /2,
    # paired headers, or alternating read lengths (SureCell/Bio-Rad).
    try:
        opener = gzip.open if str(r1_path).endswith('.gz') else open
        with opener(r1_path, 'rt') as fh:
            headers = []
            sequences = []
            for i, line in enumerate(fh):
                if i % 4 == 0:  # header line
                    headers.append(line.strip())
                elif i % 4 == 1:  # sequence line
                    sequences.append(line.strip())
                if len(headers) >= 8:
                    break

        if len(headers) < 4:
            return None, None

        # Check for /1 /2 suffix pattern
        has_suffix_pattern = all(
            h.endswith('/1') or h.endswith('/2') for h in headers[:4]
        )
        # Check for paired headers (consecutive reads have same base name)
        has_pair_pattern = False
        if len(headers) >= 4:
            # Strip any /1 /2 and trailing space+comment
            def _base_name(h):
                name = h.split()[0]  # take first field
                if name.endswith('/1') or name.endswith('/2'):
                    name = name[:-2]
                return name
            # Pairs should share base names: reads 0&1, 2&3, etc.
            has_pair_pattern = (
                _base_name(headers[0]) == _base_name(headers[1])
                and _base_name(headers[2]) == _base_name(headers[3])
                and _base_name(headers[0]) != _base_name(headers[2])
            )

        # Check for alternating read lengths (e.g., SureCell: 68bp barcode + 75bp cDNA)
        # If odd and even reads have consistently different lengths, it's interleaved.
        has_length_pattern = False
        if len(sequences) >= 6:
            odd_lens = [len(sequences[i]) for i in range(0, len(sequences), 2)]
            even_lens = [len(sequences[i]) for i in range(1, len(sequences), 2)]
            # Odd reads all same length, even reads all same length, lengths differ
            if (len(set(odd_lens)) == 1 and len(set(even_lens)) == 1
                    and odd_lens[0] != even_lens[0]):
                has_length_pattern = True
                logger.info(
                    f"[{gsm_id}] Detected alternating read lengths: "
                    f"odd={odd_lens[0]}bp, even={even_lens[0]}bp"
                )

        if not (has_suffix_pattern or has_pair_pattern or has_length_pattern):
            logger.info(f"[{gsm_id}] FASTQ does not appear to be interleaved")
            return None, None

    except Exception as e:
        logger.warning(f"[{gsm_id}] Failed to check interleave pattern: {e}")
        return None, None

    # Deinterleave: odd records → R1, even records → R2
    logger.info(f"[{gsm_id}] Deinterleaving single FASTQ into R1/R2")
    parent = r1_path.parent
    suffix = ".fastq.gz"
    new_r1 = parent / f"{gsm_id}_deint_R1{suffix}"
    new_r2 = parent / f"{gsm_id}_deint_R2{suffix}"

    try:
        opener = gzip.open if str(r1_path).endswith('.gz') else open
        with opener(r1_path, 'rt') as fh_in, \
             gzip.open(new_r1, 'wt', compresslevel=1) as fh_r1, \
             gzip.open(new_r2, 'wt', compresslevel=1) as fh_r2:
            record_idx = 0
            line_in_record = 0
            for line in fh_in:
                target = fh_r1 if record_idx % 2 == 0 else fh_r2
                target.write(line)
                line_in_record += 1
                if line_in_record == 4:
                    line_in_record = 0
                    record_idx += 1

        n_records = record_idx
        logger.info(
            f"[{gsm_id}] Deinterleaved {n_records} records → "
            f"{n_records // 2} R1 + {n_records // 2} R2"
        )
        return new_r1, new_r2

    except Exception as e:
        logger.error(f"[{gsm_id}] Deinterleave failed: {e}")
        # Clean up partial files
        for p in (new_r1, new_r2):
            try:
                p.unlink()
            except Exception:
                pass
        return None, None


def _cleanup_sample_files(gsm_id: str, download_dir: Path, quant_dir: Path, config):
    """Remove downloaded FASTQs and intermediate files for a sample.

    Called after processing (success, failure, or skip) to free disk space.
    Scans the download directory on disk rather than relying on download_result
    paths, so it catches partial/orphaned files too.
    Safe to call even if no files exist.
    """
    if not config.cleanup.after_qc:
        return
    import shutil
    logger.info(f"[{gsm_id}] Cleaning up downloads and intermediates")
    # Remove all FASTQs from the sample's download directory
    dl_sample_dir = download_dir / gsm_id
    if dl_sample_dir.exists():
        # Match all FASTQ variants: compressed (.fastq.gz, .fq.gz) and uncompressed
        # (.fastq, .fq) produced when pigz compression fails after fasterq-dump,
        # plus SRA prefetch files (.sra, .sra.cache) and split-3 files (*_1.*, *_2.*).
        fq_patterns = (
            "*.fastq.gz", "*.fq.gz",
            "*.fastq", "*.fq",
            "*.sra", "*.sra.cache",
            ".dl_segments_*",
        )
        for pattern in fq_patterns:
            for fq in list(dl_sample_dir.glob(pattern)):
                try:
                    fq.unlink()
                except Exception as e:
                    logger.warning(f"[{gsm_id}] Failed to remove {fq.name}: {e}")
        # Remove directory if empty
        try:
            if not any(dl_sample_dir.iterdir()):
                dl_sample_dir.rmdir()
        except Exception as e:
            logger.warning(f"[{gsm_id}] Failed to remove download dir: {e}")
    # Remove simpleaf intermediate directories (af_map/, af_quant/)
    if config.cleanup.intermediates:
        for subdir in ("af_map", "af_quant"):
            intermediate = quant_dir / subdir
            if intermediate.exists():
                try:
                    shutil.rmtree(intermediate)
                    logger.info(f"[{gsm_id}] Removed {subdir}/")
                except Exception as e:
                    logger.warning(f"[{gsm_id}] Failed to remove {subdir}/: {e}")


@dataclass
class SampleResult:
    """Complete processing result for a single sample.
    
    Attributes:
        gsm_id: GSM accession
        gse_id: GSE series accession
        organism: Organism name
        status: Overall status ("success", "failed", "qc_warn")
        download: Download result
        detection: Protocol detection result
        quantification: Quantification result
        qc: QC metrics
        error: Error message if failed
        total_time_s: Total processing time
    """
    gsm_id: str
    gse_id: str
    organism: str
    status: str = "pending"
    download: Optional[DownloadResult] = None
    detection: Optional[ProtocolDetection] = None
    quantification: Optional[QuantResult] = None
    qc: Optional[QCMetrics] = None
    kraken2: Optional[Kraken2Result] = None
    error: str = ""
    fail_stage: str = ""  # download, detect, quant, qc, export
    total_time_s: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary for CSV export."""
        result = asdict(self)
        result["fail_stage"] = self.fail_stage
        # Flatten nested dataclasses — always include available metrics
        if self.download:
            result["download_success"] = self.download.success
            result["download_method"] = self.download.method
            result["download_time_s"] = self.download.time_s
        if self.detection:
            result["protocol"] = self.detection.protocol
            result["mode"] = self.detection.mode
            result["r1_len"] = self.detection.r1_len
            result["r2_len"] = self.detection.r2_len
        if self.quantification:
            result["quant_tool"] = self.quantification.tool
            result["mapping_rate"] = self.quantification.mapping_rate
            result["n_reads"] = self.quantification.n_reads
            result["quant_time_s"] = self.quantification.time_s
            result["permit_strategy"] = self.quantification.permit_strategy
        if self.qc:
            result["n_cells"] = self.qc.n_cells
            result["n_genes"] = self.qc.n_genes
            result["median_genes"] = self.qc.median_genes_per_cell
            result["median_counts"] = self.qc.median_counts_per_cell
            result["qc_status"] = self.qc.qc_status
        if self.kraken2:
            result["kraken2_success"] = self.kraken2.success
            result["kraken2_frac_nonhost"] = self.kraken2.frac_nonhost
            result["kraken2_cells_with_nonhost"] = self.kraken2.cells_with_nonhost
        return result


def process_sample(
    gsm_id: str,
    gse_id: str,
    organism: str,
    ena_r1_url: Optional[str] = None,
    ena_r2_url: Optional[str] = None,
    srr_accession: Optional[str] = None,
    protocol_hint: Optional[str] = None,
    chemistry_hint: Optional[str] = None,
    output_base: Optional[Path] = None,
    config = None,
    read_count: int = 0,
    catalog_confidence: Optional[str] = None,
) -> SampleResult:
    """Process a single sample through complete pipeline.
    
    Pipeline stages:
    1. Download FASTQ files (ENA → SRA fallback)
    2. Detect protocol from FASTQ inspection (catalog hint preferred)
    3. Quantify with simpleaf
    4. Compute QC metrics
    
    Args:
        gsm_id: GSM accession
        gse_id: GSE series accession
        organism: Organism name (e.g., "human", "mouse")
        ena_r1_url: ENA URL for R1 FASTQ
        ena_r2_url: ENA URL for R2 FASTQ (optional)
        srr_accession: SRR accession for SRA fallback
        protocol_hint: Protocol from catalog metadata (e.g., "10xv3", "dropseq").
            When provided, overrides read-length heuristics for chemistry detection.
        chemistry_hint: simpleaf chemistry string from catalog (e.g., FGDL geometry).
            When provided, overrides PROTOCOL_CHEMISTRY lookup in detection.
        output_base: Base output directory (default: from config)
        config: Configuration object (default: get_config())
        
    Returns:
        SampleResult with complete processing information
    """
    import time
    import json as _json
    t0 = time.time()
    
    # Get config
    if config is None:
        config = get_config()
    
    # Set output directory
    if output_base is None:
        output_base = config.paths.pipeline_dir
    
    download_dir = output_base / "downloads" / gse_id
    quant_dir = output_base / "quant" / gse_id / gsm_id
    
    # ── Resume check: skip if already completed, QC-failed, or permanently skipped ──
    # Note: "failed" samples are NOT skipped — they are retried with the latest
    # pipeline code (improved detection, retries, etc.).
    # When a protocol_hint is provided and the previous run was skipped due to
    # protocol detection (e.g., misclassified as smartseq2), we retry since
    # the catalog hint may correct misclassification.
    manifest_path = quant_dir / "sample_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                manifest = _json.load(f)
            prev_qc = manifest.get("qc_status", "")
            prev_status = manifest.get("status", "")
            prev_protocol = manifest.get("protocol_detected", "")
            prev_chemistry = manifest.get("chemistry_used", "")
            prev_error = manifest.get("error", "")
            # Retry skipped samples if catalog hint disagrees with detected protocol
            if prev_status == "skipped" and protocol_hint and prev_protocol != protocol_hint:
                logger.info(
                    f"[{gsm_id}] RETRY: previously skipped ({prev_protocol}) "
                    f"but catalog hint is {protocol_hint}"
                )
            # Retry failed samples when catalog hint would change the chemistry
            # (e.g., previous run guessed 10xv2 but catalog says smartseq2 or dropseq)
            elif prev_status == "failed" and protocol_hint:
                hint_lower = protocol_hint.lower().strip()
                if hint_lower not in ("unknown", "unknown_sc", ""):
                    from scgeo.config.protocols import get_chemistry, is_droplet_protocol
                    hint_chem = get_chemistry(hint_lower)
                    hint_is_plate = hint_chem == "smartseq" or not is_droplet_protocol(hint_lower)
                    prev_is_permit_fail = "permit-list" in prev_error.lower()
                    # If catalog says plate-based but we tried droplet → skip immediately
                    if hint_is_plate:
                        logger.info(
                            f"[{gsm_id}] RECLASSIFY: catalog says '{hint_lower}' (plate-based), "
                            f"previously ran as {prev_chemistry} — marking as skipped"
                        )
                        # Overwrite manifest to correct the record
                        manifest["status"] = "skipped"
                        manifest["protocol_catalog"] = hint_lower
                        manifest["error"] = (
                            f"Reclassified: catalog protocol='{hint_lower}' is plate-based. "
                            f"Previous detection ({prev_chemistry}) was incorrect."
                        )
                        with open(manifest_path, "w") as f:
                            _json.dump(manifest, f, indent=2)
                        return SampleResult(
                            gsm_id=gsm_id,
                            gse_id=gse_id,
                            organism=organism,
                            status="skipped",
                            error=manifest["error"],
                            total_time_s=time.time() - t0,
                        )
                    # If catalog suggests a different droplet chemistry → retry
                    elif hint_chem and hint_chem != prev_chemistry:
                        logger.info(
                            f"[{gsm_id}] RETRY: catalog hint '{hint_lower}' (chem={hint_chem}) "
                            f"differs from previous {prev_chemistry}"
                        )
                    elif prev_is_permit_fail:
                        logger.info(
                            f"[{gsm_id}] RETRY: permit-list failure, retrying with improved strategy"
                        )
                    else:
                        logger.info(f"[{gsm_id}] RETRY: previously failed — {prev_error}")
                else:
                    logger.info(f"[{gsm_id}] RETRY: previously failed — {prev_error}")
            elif prev_qc in ("qc_pass", "qc_warn", "qc_fail") or prev_status == "skipped":
                skip_reason = prev_qc or prev_status
                logger.info(f"[{gsm_id}] SKIP: already processed (status={skip_reason})")
                return SampleResult(
                    gsm_id=gsm_id,
                    gse_id=gse_id,
                    organism=organism,
                    status="skipped",
                    error=f"Already processed: {skip_reason}",
                    total_time_s=time.time() - t0,
                )
            elif prev_status == "failed":
                logger.info(f"[{gsm_id}] RETRY: previously failed — {manifest.get('error', 'unknown')}")
        except Exception:
            pass  # Corrupt manifest, reprocess
    
    result = SampleResult(
        gsm_id=gsm_id,
        gse_id=gse_id,
        organism=organism
    )

    # ── B-G8-5: None SRR accession catalog filter ──
    # If no download path exists (no ENA URL and no SRR accession), emit a
    # terminal download_fail immediately rather than letting download_sample
    # attempt and fail later.  This is the B-X-2 catalog filter.
    _has_ena = bool(ena_r1_url and str(ena_r1_url).strip())
    _has_srr = bool(srr_accession and str(srr_accession).strip()
                    and str(srr_accession).strip().lower() != "none")
    if not _has_ena and not _has_srr:
        result.status = "failed"
        result.fail_stage = "download"
        result.error = "missing_sra_accession"
        logger.warning(
            f"[{gsm_id}] B-G8-5: no ENA URL and no SRR accession — "
            f"status=download_fail failure_reason=missing_sra_accession"
        )
        _write_sample_manifest(result, None, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result

    # ── Early skip for known plate-based protocols (before download) ──
    if protocol_hint:
        from scgeo.config.protocols import get_chemistry
        _hint_chem = get_chemistry(protocol_hint.lower().strip())
        if _hint_chem == "smartseq":
            result.status = "skipped"
            result.error = (
                f"Protocol '{protocol_hint}' is plate-based (smartseq) — "
                f"skipping before download"
            )
            logger.info(f"[{gsm_id}] SKIP: {result.error}")
            _write_sample_manifest(result, None, quant_dir, config)
            result.total_time_s = time.time() - t0
            return result
    
    # ── FASTQ peek: pre-download classification for unknown protocols ──
    # For unknown_sc / unknown samples, peek at the first 16 KB of R1 (and R2)
    # via HTTP Range request to check actual read lengths.  If both reads are
    # full-length (>50 bp), this is plate-based/Smart-seq → skip before download.
    # Saves the full FASTQ download (~30 s median) for the ~85% of unknown_sc
    # samples that turn out to be smartseq.
    if (protocol_hint
            and protocol_hint.lower().strip() in ("unknown_sc", "unknown")
            and ena_r1_url):
        from scgeo.pipeline.detect import peek_protocol
        peek_det = peek_protocol(ena_r1_url, ena_r2_url)
        if peek_det and peek_det.mode == "smartseq":
            result.status = "skipped"
            result.detection = peek_det
            result.error = (
                f"Pre-download FASTQ peek: {peek_det.reason} — "
                f"skipping before download"
            )
            logger.info(f"[{gsm_id}] SKIP (peek): {result.error}")
            _write_sample_manifest(result, peek_det, quant_dir, config)
            result.total_time_s = time.time() - t0
            return result
        elif peek_det is None:
            logger.info(
                f"[{gsm_id}] FASTQ peek inconclusive — proceeding with full download"
            )
    
    # Stage 1: Download
    logger.info(f"[{gsm_id}] Stage 1: Downloading FASTQ files")
    sample_max_time = compute_download_timeout(read_count)
    download_result = download_sample(
        gsm_id=gsm_id,
        ena_r1_url=ena_r1_url,
        ena_r2_url=ena_r2_url,
        srr_accession=srr_accession,
        output_dir=download_dir,
        config=config,
        sample_max_time=sample_max_time,
    )
    result.download = download_result
    
    if not download_result.success:
        result.status = "failed"
        result.fail_stage = "download"
        result.error = f"Download failed: {download_result.error}"
        _write_sample_manifest(result, None, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # ── Deinterleave single-file paired reads (SureCell, etc.) ──
    # Some droplet protocols have interleaved R1/R2 in a single FASTQ
    if (download_result.r1_paths
            and not download_result.r2_paths
            and protocol_hint):
        from scgeo.config.protocols import is_droplet_protocol
        if is_droplet_protocol(protocol_hint):
            logger.info(
                f"[{gsm_id}] No R2 but droplet protocol '{protocol_hint}' — "
                f"checking for interleaved FASTQ"
            )
            new_r1, new_r2 = _deinterleave_fastq(
                download_result.r1_paths[0], gsm_id
            )
            if new_r1 and new_r2:
                download_result.r1_paths = [new_r1]
                download_result.r2_paths = [new_r2]
    
    # Stage 2: Protocol detection
    logger.info(f"[{gsm_id}] Stage 2: Detecting protocol")
    r1_path = download_result.r1_paths[0] if download_result.r1_paths else None
    r2_path = download_result.r2_paths[0] if download_result.r2_paths else None
    
    if not r1_path:
        result.status = "failed"
        result.fail_stage = "download"
        result.error = "No R1 file found after download"
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    detection = detect_protocol_from_files(
        r1_path=r1_path,
        r2_path=r2_path,
        catalog_hint=protocol_hint,
        chemistry_hint=chemistry_hint,
        config=config,
        catalog_confidence=catalog_confidence,
    )
    result.detection = detection
    
    if detection.confidence == "low":
        result.status = "failed"
        result.fail_stage = "detect"
        result.error = f"Protocol detection failed: {detection.reason}"
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # Stage 3: Quantification
    logger.info(f"[{gsm_id}] Stage 3: Quantifying ({detection.mode} mode)")
    
    if detection.mode == "droplet":
        if not detection.chemistry:
            result.status = "failed"
            result.fail_stage = "detect"
            result.error = "No chemistry string for droplet protocol"
            _write_sample_manifest(result, detection, quant_dir, config)
            _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
            result.total_time_s = time.time() - t0
            return result
        
        # Check if quant already completed (af_quant/alevin/quants_mat.mtx exists)
        mtx_path = quant_dir / "af_quant" / "alevin" / "quants_mat.mtx"
        if mtx_path.exists() and mtx_path.stat().st_size > 0:
            logger.info(f"[{gsm_id}] Reusing existing quantification output")
            quant_result = QuantResult(success=True, tool="simpleaf", mapping_rate=0.0, time_s=0.0)
        else:
            # Swap R1/R2 files if detection flagged a read orientation mismatch
            q_r1 = download_result.r1_paths
            q_r2 = download_result.r2_paths
            if detection.reads_swapped:
                logger.info(f"[{gsm_id}] Swapping R1/R2 files (reads_swapped=True)")
                q_r1, q_r2 = q_r2, q_r1

            quant_result = quantify_simpleaf(
            r1_paths=q_r1,
            r2_paths=q_r2,
            chemistry=detection.chemistry,
            organism=organism,
            output_dir=quant_dir,
            config=config
        )
    elif detection.mode == "smartseq":
        result.status = "skipped"
        result.error = "Smart-seq (plate-based) protocol — skipping (droplet-only pipeline)"
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    elif detection.mode == "unknown":
        result.status = "skipped"
        result.error = f"Unknown protocol mode — skipping"
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    else:
        result.status = "failed"
        result.fail_stage = "detect"
        result.error = f"Unknown mode: {detection.mode}"
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    result.quantification = quant_result
    
    if not quant_result.success:
        result.status = "failed"
        result.fail_stage = "quant"
        result.error = f"Quantification failed: {quant_result.error}"
        logger.error(f"[{gsm_id}] Quantification failed: {quant_result.error}")
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # Early-exit: 0% mapping means reads have no transcriptome content
    # (e.g. ADT/HTO/VDJ libraries misclassified as GEX). Skip expensive
    # QC matrix read since permit-list will be empty or garbage.
    if quant_result.mapping_rate == 0.0 and quant_result.time_s > 0:
        result.status = "failed"
        result.fail_stage = "quant"
        result.error = "Zero mapping rate — reads have no transcriptome content (likely non-GEX library: ADT/HTO/VDJ)"
        logger.warning(f"[{gsm_id}] 0% mapping rate — aborting (non-GEX library)")
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # Stage 4: QC metrics
    logger.info(f"[{gsm_id}] Stage 4: Computing QC metrics")
    
    qc_metrics = compute_qc_simpleaf(quant_dir, config, protocol=detection.protocol)
    
    result.qc = qc_metrics
    
    # Determine final status based on QC
    if qc_metrics.qc_status == "pass":
        result.status = "success"
    elif qc_metrics.qc_status == "warn":
        result.status = "qc_warn"
    else:
        result.status = "failed"
        result.fail_stage = "qc"
        result.error = f"QC failed: {', '.join(qc_metrics.fail_reasons)}"
        logger.warning(f"[{gsm_id}] QC failed: {', '.join(qc_metrics.fail_reasons)}")
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # Stage 5: Kraken2 non-host classification (only for QC-pass/warn droplet)
    if config.kraken2.enabled and detection.mode == "droplet":
        logger.info(f"[{gsm_id}] Stage 5: Kraken2 non-host classification")
        
        # Use swapped paths if detection flagged a read orientation mismatch
        kr_r1 = download_result.r1_paths
        kr_r2 = download_result.r2_paths
        if detection.reads_swapped:
            kr_r1, kr_r2 = kr_r2, kr_r1
        
        # Resolve host taxon ID
        host_taxon_id = config.get_taxon_id(organism)
        if host_taxon_id is None:
            logger.warning(f"[{gsm_id}] No taxon ID for {organism}, skipping kraken2")
        else:
            try:
                kraken2_result = classify_nonhost(
                    r1_paths=kr_r1,
                    r2_paths=kr_r2,
                    chemistry=detection.chemistry,
                    host_taxon_id=host_taxon_id,
                    quant_dir=quant_dir,
                    output_dir=quant_dir,
                    config=config,
                )
                result.kraken2 = kraken2_result
                if not kraken2_result.success:
                    logger.warning(f"[{gsm_id}] Kraken2 failed (non-fatal): {kraken2_result.error}")
            except Exception as e:
                logger.warning(f"[{gsm_id}] Kraken2 exception (non-fatal): {e}")
    
    # Stage 6: Write sample manifest
    logger.info(f"[{gsm_id}] Stage 6: Writing sample manifest")
    _write_sample_manifest(result, detection, quant_dir, config)
    
    # Cleanup downloads and intermediates to save disk
    _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
    
    result.total_time_s = time.time() - t0
    logger.info(f"[{gsm_id}] Complete: {result.status} ({result.total_time_s:.1f}s)")
    
    return result


def _process_after_download(
    gsm_id: str,
    gse_id: str,
    organism: str,
    download_result: DownloadResult,
    output_base: Path,
    config,
    t0: float,
    protocol_hint: Optional[str] = None,
    chemistry_hint: Optional[str] = None,
    catalog_confidence: Optional[str] = None,
) -> SampleResult:
    """Process a sample through stages 2-6 given a completed download.
    
    Used by prefetch pipeline. Runs detect → quant → QC → kraken2 → manifest → cleanup.
    """
    import time
    
    quant_dir = output_base / "quant" / gse_id / gsm_id
    download_dir = output_base / "downloads" / gse_id
    
    result = SampleResult(gsm_id=gsm_id, gse_id=gse_id, organism=organism)
    result.download = download_result
    
    if not download_result.success:
        result.status = "failed"
        result.fail_stage = "download"
        result.error = f"Download failed: {download_result.error}"
        _write_sample_manifest(result, None, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # ── Deinterleave single-file paired reads (SureCell, etc.) ──
    if (download_result.r1_paths
            and not download_result.r2_paths
            and protocol_hint):
        from scgeo.config.protocols import is_droplet_protocol
        if is_droplet_protocol(protocol_hint):
            logger.info(
                f"[{gsm_id}] No R2 but droplet protocol '{protocol_hint}' — "
                f"checking for interleaved FASTQ"
            )
            new_r1, new_r2 = _deinterleave_fastq(
                download_result.r1_paths[0], gsm_id
            )
            if new_r1 and new_r2:
                download_result.r1_paths = [new_r1]
                download_result.r2_paths = [new_r2]
    
    # Stage 2: Protocol detection
    logger.info(f"[{gsm_id}] Stage 2: Detecting protocol")
    r1_path = download_result.r1_paths[0] if download_result.r1_paths else None
    r2_path = download_result.r2_paths[0] if download_result.r2_paths else None
    
    if not r1_path:
        result.status = "failed"
        result.fail_stage = "download"
        result.error = "No R1 file found after download"
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    detection = detect_protocol_from_files(r1_path=r1_path, r2_path=r2_path, catalog_hint=protocol_hint, chemistry_hint=chemistry_hint, config=config, catalog_confidence=catalog_confidence)
    result.detection = detection
    
    if detection.confidence == "low":
        result.status = "failed"
        result.fail_stage = "detect"
        result.error = f"Protocol detection failed: {detection.reason}"
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # Stage 3: Quantification
    logger.info(f"[{gsm_id}] Stage 3: Quantifying ({detection.mode} mode)")
    
    if detection.mode == "droplet":
        if not detection.chemistry:
            result.status = "failed"
            result.fail_stage = "detect"
            result.error = "No chemistry string for droplet protocol"
            _write_sample_manifest(result, detection, quant_dir, config)
            _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
            result.total_time_s = time.time() - t0
            return result
        
        # Check if quant already completed (af_quant/alevin/quants_mat.mtx exists)
        mtx_path = quant_dir / "af_quant" / "alevin" / "quants_mat.mtx"
        if mtx_path.exists() and mtx_path.stat().st_size > 0:
            logger.info(f"[{gsm_id}] Reusing existing quantification output")
            quant_result = QuantResult(success=True, tool="simpleaf", mapping_rate=0.0, time_s=0.0)
        else:
            # Swap R1/R2 files if detection flagged a read orientation mismatch
            q_r1 = download_result.r1_paths
            q_r2 = download_result.r2_paths
            if detection.reads_swapped:
                logger.info(f"[{gsm_id}] Swapping R1/R2 files (reads_swapped=True)")
                q_r1, q_r2 = q_r2, q_r1

            quant_result = quantify_simpleaf(
                r1_paths=q_r1,
                r2_paths=q_r2,
                chemistry=detection.chemistry,
                organism=organism,
                output_dir=quant_dir,
                config=config
            )
    elif detection.mode == "smartseq":
        result.status = "skipped"
        result.error = "Smart-seq (plate-based) protocol — skipping (droplet-only pipeline)"
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    elif detection.mode == "unknown":
        result.status = "skipped"
        result.error = f"Unknown protocol mode — skipping"
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    else:
        result.status = "failed"
        result.fail_stage = "detect"
        result.error = f"Unknown mode: {detection.mode}"
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    result.quantification = quant_result
    
    if not quant_result.success:
        result.status = "failed"
        result.fail_stage = "quant"
        result.error = f"Quantification failed: {quant_result.error}"
        logger.error(f"[{gsm_id}] Quantification failed: {quant_result.error}")
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # Early-exit: 0% mapping means reads have no transcriptome content
    if quant_result.mapping_rate == 0.0 and quant_result.time_s > 0:
        result.status = "failed"
        result.fail_stage = "quant"
        result.error = "Zero mapping rate — reads have no transcriptome content (likely non-GEX library: ADT/HTO/VDJ)"
        logger.warning(f"[{gsm_id}] 0% mapping rate — aborting (non-GEX library)")
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # Stage 4: QC metrics
    logger.info(f"[{gsm_id}] Stage 4: Computing QC metrics")
    qc_metrics = compute_qc_simpleaf(quant_dir, config, protocol=detection.protocol)
    result.qc = qc_metrics
    
    if qc_metrics.qc_status == "pass":
        result.status = "success"
    elif qc_metrics.qc_status == "warn":
        result.status = "qc_warn"
    else:
        result.status = "failed"
        result.fail_stage = "qc"
        result.error = f"QC failed: {', '.join(qc_metrics.fail_reasons)}"
        logger.warning(f"[{gsm_id}] QC failed: {', '.join(qc_metrics.fail_reasons)}")
        _write_sample_manifest(result, detection, quant_dir, config)
        _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
        result.total_time_s = time.time() - t0
        return result
    
    # Stage 4.5: Export counts matrix to .1pz format (only for QC pass/warn)
    logger.info(f"[{gsm_id}] Stage 4.5: Exporting matrix")
    export_success = _export_counts_matrix(gsm_id, quant_dir)
    if not export_success:
        logger.warning(f"[{gsm_id}] Failed to export matrix (non-fatal)")
    
    # Stage 5: Kraken2 (only for QC-pass/warn droplet)
    if config.kraken2.enabled and detection.mode == "droplet":
        logger.info(f"[{gsm_id}] Stage 5: Kraken2 non-host classification")
        
        # Use swapped paths if detection flagged a read orientation mismatch
        kr_r1 = download_result.r1_paths
        kr_r2 = download_result.r2_paths
        if detection.reads_swapped:
            kr_r1, kr_r2 = kr_r2, kr_r1
        
        host_taxon_id = config.get_taxon_id(organism)
        if host_taxon_id is None:
            logger.warning(f"[{gsm_id}] No taxon ID for {organism}, skipping kraken2")
        else:
            try:
                kraken2_result = classify_nonhost(
                    r1_paths=kr_r1,
                    r2_paths=kr_r2,
                    chemistry=detection.chemistry,
                    host_taxon_id=host_taxon_id,
                    quant_dir=quant_dir,
                    output_dir=quant_dir,
                    config=config,
                )
                result.kraken2 = kraken2_result
                if not kraken2_result.success:
                    logger.warning(f"[{gsm_id}] Kraken2 failed (non-fatal): {kraken2_result.error}")
            except Exception as e:
                logger.warning(f"[{gsm_id}] Kraken2 exception (non-fatal): {e}")
    
    # Stage 6: Write manifest
    logger.info(f"[{gsm_id}] Stage 6: Writing sample manifest")
    _write_sample_manifest(result, detection, quant_dir, config)
    
    # Cleanup downloads and intermediates to save disk
    _cleanup_sample_files(gsm_id, download_dir, quant_dir, config)
    
    result.total_time_s = time.time() - t0
    logger.info(f"[{gsm_id}] Complete: {result.status} ({result.total_time_s:.1f}s)")
    return result


def _export_counts_matrix(
    gsm_id: str,
    quant_dir: Path,
) -> bool:
    """Export simpleaf quantification to .1pz format with metadata.
    
    Reads af_quant/alevin output and writes:
    - counts.1pz (VOCSC compressed matrix)
    - cell_metadata.parquet (cell barcodes + any metadata)
    - feature_metadata.parquet (gene names/IDs)
    
    Returns:
        True if export succeeded, False otherwise
    """
    try:
        import scipy.io as sio
        try:
            import singlepress as sprs
            if not hasattr(sprs, 'write_1pz'):
                # Editable install: C++ extension may have failed to load on first import;
                # purge the cached module and retry once.
                import importlib, sys
                sys.modules.pop('singlepress', None)
                sys.modules.pop('singlepress._pz_codec', None)
                import singlepress as sprs
        except ImportError:
            logger.error(f"[{gsm_id}] singlepress not available, skipping .1pz export")
            return False

        af_quant_dir = quant_dir / "af_quant" / "alevin"
        
        # Read count matrix
        mtx_path = af_quant_dir / "quants_mat.mtx"
        if not mtx_path.exists():
            logger.warning(f"[{gsm_id}] No quants_mat.mtx found at {mtx_path}")
            return False
        
        logger.info(f"[{gsm_id}] Exporting matrix to .1pz format")
        matrix = sio.mmread(str(mtx_path))
        
        # Read cell barcodes (rows in quants_mat.mtx)
        barcodes_path = af_quant_dir / "quants_mat_rows.txt"
        if barcodes_path.exists():
            with open(barcodes_path) as f:
                cell_ids = [line.strip() for line in f]
        else:
            cell_ids = [f"cell_{i}" for i in range(matrix.shape[0])]
        
        # Read gene names (cols in quants_mat.mtx)
        genes_path = af_quant_dir / "quants_mat_cols.txt"
        if genes_path.exists():
            with open(genes_path) as f:
                gene_names = [line.strip() for line in f]
        else:
            gene_names = [f"gene_{i}" for i in range(matrix.shape[1])]
        
        # Convert to CSC (genes x cells) for compression
        # quants_mat.mtx is cells x genes, need to transpose
        import scipy.sparse as sp
        matrix_csc = sp.csc_matrix(matrix.T)  # Now genes x cells
        
        logger.info(f"[{gsm_id}]   Matrix: {matrix_csc.shape[0]} genes x {matrix_csc.shape[1]} cells, nnz={matrix_csc.nnz:,}")
        
        # Write counts.1pz (VOCSC + zstd-3)
        pz_path = quant_dir / "counts.1pz"
        stats = sprs.write_1pz(str(pz_path), matrix_csc)
        
        pz_size_mb = pz_path.stat().st_size / 1024**2
        logger.info(f"[{gsm_id}]   Written: {pz_size_mb:.2f} MB (ratio: {stats['ratio']:.1f}x)")
        
        # Write cell_metadata.parquet
        cell_df = pd.DataFrame({"barcode": cell_ids})
        cell_meta_path = quant_dir / "cell_metadata.parquet"
        cell_df.to_parquet(str(cell_meta_path), compression="zstd", index=False)
        logger.info(f"[{gsm_id}]   Written: {cell_meta_path.name}")
        
        # Write feature_metadata.parquet
        gene_df = pd.DataFrame({"gene_name": gene_names})
        feat_meta_path = quant_dir / "feature_metadata.parquet"
        gene_df.to_parquet(str(feat_meta_path), compression="zstd", index=False)
        logger.info(f"[{gsm_id}]   Written: {feat_meta_path.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"[{gsm_id}] Failed to export matrix: {e}")
        return False


def _write_sample_manifest(
    result: SampleResult,
    detection: ProtocolDetection,
    quant_dir: Path,
    config,
) -> None:
    """Write sample_manifest.json with all processing metadata."""
    import json as _json
    import time as _time
    
    manifest = {
        "gsm_id": result.gsm_id,
        "gse_id": result.gse_id,
        "organism": result.organism,
        "taxon_id": config.get_taxon_id(result.organism),
        "protocol_detected": detection.protocol if detection else "",
        "protocol_confidence": detection.confidence if detection else "",
        "chemistry_used": detection.chemistry if detection else "",
        "mode": detection.mode if detection else "",
        "reads_swapped": detection.reads_swapped if detection else False,
    }
    
    if result.qc:
        manifest.update({
            "n_cells": result.qc.n_cells,
            "n_genes": result.qc.n_genes,
            "median_genes_per_cell": result.qc.median_genes_per_cell,
            "median_counts_per_cell": result.qc.median_counts_per_cell,
            "mapping_rate": result.qc.mapping_rate,
            "frac_unspliced": result.qc.frac_unspliced,
            "qc_status": f"qc_{result.qc.qc_status}",  # Store as qc_pass/qc_warn/qc_fail
        })
    
    if result.kraken2 and result.kraken2.success:
        manifest.update({
            "kraken2_frac_nonhost": result.kraken2.frac_nonhost,
            "kraken2_cells_with_nonhost": result.kraken2.cells_with_nonhost,
            "kraken2_total_nonhost_umis": result.kraken2.total_nonhost_umis,
        })
    
    manifest["pipeline_version"] = PIPELINE_VERSION
    manifest["pipeline_mtime"] = _time.time()
    manifest["status"] = result.status  # success/skipped/failed
    if result.error:
        manifest["error"] = result.error
    
    quant_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = quant_dir / "sample_manifest.json"
    with open(manifest_path, "w") as f:
        _json.dump(manifest, f, indent=2)


def process_gse(
    gse_id: str,
    samples: List[dict],
    output_base: Optional[Path] = None,
    config = None
) -> List[SampleResult]:
    """Process all samples in a GSE series.
    
    Args:
        gse_id: GSE series accession
        samples: List of sample dictionaries with keys:
            - gsm_id
            - organism
            - ena_r1_url (optional)
            - ena_r2_url (optional)
            - srr_accession (optional)
        output_base: Base output directory
        config: Configuration object
        
    Returns:
        List of SampleResult objects
    """
    if config is None:
        config = get_config()
    
    logger.info(f"Processing GSE {gse_id} with {len(samples)} samples")
    
    results = []
    for sample in samples:
        result = process_sample(
            gsm_id=sample["gsm_id"],
            gse_id=gse_id,
            organism=sample["organism"],
            ena_r1_url=sample.get("ena_r1_url"),
            ena_r2_url=sample.get("ena_r2_url"),
            srr_accession=sample.get("srr_accession"),
            protocol_hint=sample.get("protocol_hint"),
            output_base=output_base,
            config=config,
            catalog_confidence=sample.get("catalog_confidence"),
        )
        results.append(result)
    
    n_success = sum(1 for r in results if r.status == "success")
    n_warn = sum(1 for r in results if r.status == "qc_warn")
    n_fail = sum(1 for r in results if r.status == "failed")
    
    logger.info(f"GSE {gse_id} complete: {n_success} success, {n_warn} warnings, {n_fail} failed")
    
    return results


def process_samples(
    samples: List[dict],
    output_base: Optional[Path] = None,
    config = None,
    n_workers: int = 1
) -> List[SampleResult]:
    """Process multiple samples with download prefetching.
    
    Uses a background thread to download sample N+1 while the main thread
    processes sample N (detect → quant → QC → kraken2), overlapping I/O
    with CPU work for ~2x throughput.
    
    Args:
        samples: List of sample dictionaries with keys:
            - gsm_id, gse_id, organism
            - ena_r1_url, ena_r2_url (optional)
            - srr_accession (optional)
        output_base: Base output directory
        config: Configuration object
        n_workers: Ignored (kept for API compat); prefetch always used
        
    Returns:
        List of SampleResult objects
    """
    import time
    import json as _json
    from concurrent.futures import ThreadPoolExecutor, Future
    
    if config is None:
        config = get_config()
    
    if output_base is None:
        output_base = config.paths.pipeline_dir
    
    logger.info(f"Processing {len(samples)} samples with download prefetching")
    
    results = []
    prefetch_future: Optional[Future] = None
    prefetch_sample: Optional[dict] = None

    # ── SmartSeq GSE-level early skip ──
    # Track smartseq detections per GSE. If all processed samples from a GSE
    # are smartseq (minimum 3), skip remaining samples from that GSE without
    # downloading — the entire experiment is likely plate-based.
    _SMARTSEQ_SKIP_MIN = 3  # Minimum smartseq detections before skipping rest of GSE
    gse_smartseq_count: dict = {}   # gse_id → count of smartseq detections
    gse_total_count: dict = {}      # gse_id → total samples processed (excl. skips)

    # ── GSE-level QC failure tracking ──
    # If every processed sample from a GSE fails QC (minimum threshold), skip
    # remaining samples — the dataset is fundamentally incompatible.
    _GSE_FAIL_SKIP_MIN = 5  # Minimum failures before adaptive skip
    gse_fail_count: dict = {}       # gse_id → count of QC/quant failures
    gse_success_count: dict = {}    # gse_id → count of successes (incl. qc_warn)
    
    def _download_one(sample: dict) -> DownloadResult:
        """Download a single sample's FASTQs (runs in background thread)."""
        download_dir = output_base / "downloads" / sample["gse_id"]
        # Size-scaled timeout: use read_count as file-size proxy
        read_count = 0
        try:
            read_count = int(sample.get("read_count") or 0)
        except (ValueError, TypeError):
            pass
        sample_max_time = compute_download_timeout(read_count)
        return download_sample(
            gsm_id=sample["gsm_id"],
            ena_r1_url=sample.get("ena_r1_url"),
            ena_r2_url=sample.get("ena_r2_url"),
            srr_accession=sample.get("srr_accession"),
            output_dir=download_dir,
            config=config,
            sample_max_time=sample_max_time,
        )
    
    def _should_skip(sample: dict) -> Optional[SampleResult]:
        """Check if sample already has a manifest (resume check).

        Skips samples with QC results or explicit skip status.
        Previously *failed* samples are retried (returns None).
        When a protocol_hint disagrees with a previous detection,
        the sample is retried.  If the catalog says plate-based but the
        previous run used a droplet chemistry, the manifest is reclassified
        in-place and skipped immediately (no re-download needed).
        """
        gsm_id = sample["gsm_id"]
        quant_dir = output_base / "quant" / sample["gse_id"] / gsm_id
        manifest_path = quant_dir / "sample_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path) as f:
                    manifest = _json.load(f)
                prev_qc = manifest.get("qc_status", "")
                prev_status = manifest.get("status", "")
                prev_protocol = manifest.get("protocol_detected", "")
                prev_chemistry = manifest.get("chemistry_used", "")
                prev_error = manifest.get("error", "")
                hint = sample.get("protocol_hint", "")
                # Retry skipped samples if catalog hint disagrees
                if prev_status == "skipped" and hint and prev_protocol != hint:
                    logger.info(
                        f"[{gsm_id}] RETRY: previously skipped ({prev_protocol}) "
                        f"but catalog hint is {hint}"
                    )
                    return None
                # For failed samples with a catalog hint that suggests different chemistry
                if prev_status == "failed" and hint:
                    hint_lower = hint.lower().strip()
                    if hint_lower not in ("unknown", "unknown_sc", ""):
                        from scgeo.config.protocols import get_chemistry, is_droplet_protocol
                        hint_chem = get_chemistry(hint_lower)
                        hint_is_plate = hint_chem == "smartseq" or not is_droplet_protocol(hint_lower)
                        if hint_is_plate:
                            logger.info(
                                f"[{gsm_id}] RECLASSIFY: catalog='{hint_lower}' (plate-based), "
                                f"was {prev_chemistry}"
                            )
                            manifest["status"] = "skipped"
                            manifest["protocol_catalog"] = hint_lower
                            manifest["error"] = (
                                f"Reclassified: catalog protocol='{hint_lower}' is plate-based. "
                                f"Previous detection ({prev_chemistry}) was incorrect."
                            )
                            with open(manifest_path, "w") as f:
                                _json.dump(manifest, f, indent=2)
                            return SampleResult(
                                gsm_id=gsm_id,
                                gse_id=sample["gse_id"],
                                organism=sample.get("organism", ""),
                                status="skipped",
                                error=manifest["error"],
                                total_time_s=0.0,
                            )
                    # Different droplet chemistry or permit-list fail → retry
                    return None
                if prev_qc in ("qc_pass", "qc_warn", "qc_fail") or prev_status == "skipped":
                    skip_reason = prev_qc or prev_status
                    logger.info(f"[{gsm_id}] SKIP: already processed (status={skip_reason})")
                    return SampleResult(
                        gsm_id=gsm_id,
                        gse_id=sample["gse_id"],
                        organism=sample.get("organism", ""),
                        status="skipped",
                        error=f"Already processed: {skip_reason}",
                        total_time_s=0.0,
                    )
                if prev_status == "failed":
                    logger.info(f"[{gsm_id}] RETRY: previously failed — {prev_error}")
            except Exception:
                pass
        return None
    
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    
    # ── SIGTERM handler: clean up current sample's downloads on job kill ──
    # SLURM sends SIGTERM ~30s before SIGKILL. Without this handler, partial
    # downloads from the in-progress sample would be orphaned on disk.
    import signal
    _current_gsm = [None]        # mutable container for closure
    _current_gse = [None]
    _prev_sigterm = signal.getsignal(signal.SIGTERM)

    def _sigterm_cleanup(signum, frame):
        gsm, gse = _current_gsm[0], _current_gse[0]
        if gsm and gse:
            dl_dir = output_base / "downloads" / gse
            q_dir = output_base / "quant" / gse / gsm
            logger.warning(f"[{gsm}] SIGTERM received — cleaning up downloads before exit")
            try:
                _cleanup_sample_files(gsm, dl_dir, q_dir, config)
            except Exception:
                pass
        # Re-raise so the process actually exits
        if callable(_prev_sigterm):
            _prev_sigterm(signum, frame)
        else:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _sigterm_cleanup)

    try:
        for i, sample in enumerate(samples):
            t0 = time.time()
            gsm_id = sample["gsm_id"]
            gse_id = sample["gse_id"]
            organism = sample.get("organism", "")
            _current_gsm[0] = gsm_id
            _current_gse[0] = gse_id
            
            # ── Resume check ──
            skip_result = _should_skip(sample)
            if skip_result:
                results.append(skip_result)
                continue
            
            # ── SmartSeq GSE-level early skip ──
            # If all prior samples from this GSE were smartseq (≥ threshold),
            # skip without downloading — the whole experiment is plate-based.
            ss_count = gse_smartseq_count.get(gse_id, 0)
            total_count = gse_total_count.get(gse_id, 0)
            if (ss_count >= _SMARTSEQ_SKIP_MIN
                    and total_count > 0
                    and ss_count == total_count):
                logger.info(
                    f"[{gsm_id}] GSE-SKIP: {gse_id} has {ss_count}/{total_count} "
                    f"smartseq detections — skipping rest of GSE"
                )
                skip_r = SampleResult(
                    gsm_id=gsm_id,
                    gse_id=gse_id,
                    organism=organism,
                    status="skipped",
                    error=f"GSE-level smartseq skip: {ss_count}/{total_count} samples were smartseq",
                    total_time_s=0.0,
                )
                results.append(skip_r)
                # Write manifest so this doesn't get re-downloaded on retry
                quant_dir = output_base / "quant" / gse_id / gsm_id
                quant_dir.mkdir(parents=True, exist_ok=True)
                manifest_path = quant_dir / "sample_manifest.json"
                if not manifest_path.exists():
                    import json as _manifest_json
                    with open(manifest_path, "w") as f:
                        _manifest_json.dump({
                            "gsm_id": gsm_id,
                            "gse_id": gse_id,
                            "organism": organism,
                            "status": "skipped",
                            "protocol_detected": "smartseq2",
                            "mode": "smartseq",
                            "error": f"GSE-level smartseq skip ({ss_count}/{total_count})",
                            "pipeline_version": PIPELINE_VERSION,
                        }, f, indent=2)
                continue
            
            # ── Protocol-level smartseq skip (before download) ──
            _hint = sample.get("protocol_hint", "")
            if _hint:
                from scgeo.config.protocols import get_chemistry as _get_chem
                if _get_chem(_hint.lower().strip()) == "smartseq":
                    skip_msg = (
                        f"Protocol '{_hint}' is plate-based (smartseq) — "
                        f"skipping before download"
                    )
                    logger.info(f"[{gsm_id}] SKIP: {skip_msg}")
                    skip_r = SampleResult(
                        gsm_id=gsm_id,
                        gse_id=gse_id,
                        organism=organism,
                        status="skipped",
                        error=skip_msg,
                        total_time_s=0.0,
                    )
                    results.append(skip_r)
                    quant_dir = output_base / "quant" / gse_id / gsm_id
                    _write_sample_manifest(skip_r, None, quant_dir, config)
                    # Track for GSE-level smartseq skip
                    gse_total_count[gse_id] = gse_total_count.get(gse_id, 0) + 1
                    gse_smartseq_count[gse_id] = gse_smartseq_count.get(gse_id, 0) + 1
                    continue

            # ── GSE-level QC failure skip ──
            # If all prior samples from this GSE failed QC/quant (≥ threshold,
            # 0 successes), skip remaining — the dataset is incompatible.
            fail_count = gse_fail_count.get(gse_id, 0)
            success_count = gse_success_count.get(gse_id, 0)
            if (fail_count >= _GSE_FAIL_SKIP_MIN
                    and success_count == 0):
                logger.info(
                    f"[{gsm_id}] GSE-FAIL-SKIP: {gse_id} has {fail_count} consecutive "
                    f"failures, 0 successes — skipping rest of GSE"
                )
                skip_r = SampleResult(
                    gsm_id=gsm_id,
                    gse_id=gse_id,
                    organism=organism,
                    status="skipped",
                    error=f"GSE-level failure skip: {fail_count} failures, 0 successes",
                    total_time_s=0.0,
                )
                results.append(skip_r)
                quant_dir = output_base / "quant" / gse_id / gsm_id
                _write_sample_manifest(skip_r, None, quant_dir, config)
                continue

            # ── FASTQ peek for unknown protocols (before download) ──
            # For unknown_sc / unknown samples with ENA URLs, peek at the first
            # 16 KB of R1 (and R2) to check actual read lengths.  If both reads
            # are full-length (>50 bp) this is plate-based → skip before the
            # expensive full download.
            if (_hint
                    and _hint.lower().strip() in ("unknown_sc", "unknown")
                    and sample.get("ena_r1_url")):
                from scgeo.pipeline.detect import peek_protocol
                peek_det = peek_protocol(
                    sample.get("ena_r1_url"),
                    sample.get("ena_r2_url"),
                )
                if peek_det and peek_det.mode == "smartseq":
                    skip_msg = (
                        f"Pre-download FASTQ peek: {peek_det.reason} — "
                        f"skipping before download"
                    )
                    logger.info(f"[{gsm_id}] SKIP (peek): {skip_msg}")
                    skip_r = SampleResult(
                        gsm_id=gsm_id,
                        gse_id=gse_id,
                        organism=organism,
                        status="skipped",
                        detection=peek_det,
                        error=skip_msg,
                        total_time_s=0.0,
                    )
                    results.append(skip_r)
                    quant_dir = output_base / "quant" / gse_id / gsm_id
                    _write_sample_manifest(skip_r, peek_det, quant_dir, config)
                    gse_total_count[gse_id] = gse_total_count.get(gse_id, 0) + 1
                    gse_smartseq_count[gse_id] = gse_smartseq_count.get(gse_id, 0) + 1
                    continue
            
            # ── Get download result: from prefetch or download now ──
            download_result = None
            if prefetch_future and prefetch_sample and prefetch_sample["gsm_id"] == gsm_id:
                # This sample was prefetched
                try:
                    download_result = prefetch_future.result(timeout=config.download.timeout + 60)
                except Exception as e:
                    logger.warning(f"[{gsm_id}] Prefetch failed: {e}, downloading synchronously")
                prefetch_future = None
                prefetch_sample = None
            
            if download_result is None:
                download_result = _download_one(sample)
            
            # ── Start prefetching next sample ──
            if i + 1 < len(samples):
                next_sample = samples[i + 1]
                next_skip = _should_skip(next_sample)
                # Also skip prefetching known smartseq samples
                next_hint = next_sample.get("protocol_hint", "")
                next_is_smartseq = False
                next_needs_peek = False
                if next_hint:
                    _hint_lower = next_hint.lower().strip()
                    from scgeo.config.protocols import get_chemistry as _gc
                    next_is_smartseq = (_gc(_hint_lower) == "smartseq")
                    # Don't prefetch unknown_sc — needs FASTQ peek first
                    next_needs_peek = _hint_lower in ("unknown_sc", "unknown")
                if next_skip is None and not next_is_smartseq and not next_needs_peek:
                    # Not already done — prefetch it
                    prefetch_sample = next_sample
                    prefetch_future = executor.submit(_download_one, next_sample)
                else:
                    prefetch_future = None
                    prefetch_sample = None
            
            # ── Process this sample through remaining stages ──
            result = _process_after_download(
                gsm_id=gsm_id,
                gse_id=gse_id,
                organism=organism,
                download_result=download_result,
                output_base=output_base,
                config=config,
                t0=t0,
                protocol_hint=sample.get("protocol_hint"),
                chemistry_hint=sample.get("chemistry_hint"),
                catalog_confidence=sample.get("catalog_confidence"),
            )
            results.append(result)

            # ── Track smartseq detections for GSE-level skip ──
            if result.status != "skipped" or "smartseq" in (result.error or "").lower():
                gse_total_count[gse_id] = gse_total_count.get(gse_id, 0) + 1
                if (result.detection and result.detection.mode == "smartseq") or \
                   (result.status == "skipped" and "smartseq" in (result.error or "").lower()):
                    gse_smartseq_count[gse_id] = gse_smartseq_count.get(gse_id, 0) + 1

            # ── Track QC failures for GSE-level failure skip ──
            if result.status in ("success", "qc_warn"):
                gse_success_count[gse_id] = gse_success_count.get(gse_id, 0) + 1
            elif result.status == "failed":
                gse_fail_count[gse_id] = gse_fail_count.get(gse_id, 0) + 1
    
    finally:
        # Cancel any pending prefetch
        if prefetch_future:
            prefetch_future.cancel()
        executor.shutdown(wait=False)
        # Restore previous SIGTERM handler
        signal.signal(signal.SIGTERM, _prev_sigterm or signal.SIG_DFL)
        _current_gsm[0] = None
        _current_gse[0] = None
    
    n_success = sum(1 for r in results if r.status == "success")
    n_warn = sum(1 for r in results if r.status == "qc_warn")
    n_skip = sum(1 for r in results if r.status == "skipped")
    n_fail = sum(1 for r in results if r.status == "failed")
    
    logger.info(f"Batch complete: {n_success} success, {n_warn} warnings, {n_skip} skipped, {n_fail} failed")
    
    return results
