"""
Quantification module for single-cell RNA-seq samples.

Provides wrappers for:
- simpleaf (droplet protocols: 10x, Drop-seq, inDrops, etc.)
"""
import json
import logging
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class QuantResult:
    """Result of quantification operation.
    
    Attributes:
        success: Whether quantification succeeded
        output_dir: Directory containing quantification outputs
        tool: Tool used ("simpleaf")
        time_s: Quantification time in seconds
        mapping_rate: Alignment/mapping rate (0-1)
        n_reads: Number of reads processed
        error: Error message if failed
        fail_category: Standardized failure category
    """
    success: bool = False
    output_dir: Optional[Path] = None
    tool: str = ""
    time_s: float = 0.0
    mapping_rate: float = 0.0
    n_reads: int = 0
    error: str = ""
    fail_category: str = ""
    permit_strategy: str = ""  # which permit-list strategy succeeded (unfiltered-pl, knee, expect-5000, expect-500)


def quantify_simpleaf(
    r1_paths: List[Path],
    r2_paths: List[Path],
    chemistry: str,
    organism: str,
    output_dir: Path,
    config
) -> QuantResult:
    """Quantify droplet scRNA-seq data with simpleaf.
    
    Uses simpleaf quant in USA mode (unspliced/spliced/ambiguous counts) with
    piscem mapper and alevin-fry quantifier.
    
    Args:
        r1_paths: List of R1 FASTQ files
        r2_paths: List of R2 FASTQ files
        chemistry: simpleaf chemistry string (e.g., "10xv3")
        organism: Organism name (e.g., "human")
        output_dir: Output directory for quantification
        config: Configuration object with index paths and settings
        
    Returns:
        QuantResult with success status and metrics
    """
    t0 = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Reject empty R2 early — droplet protocols require paired-end reads
    if not r2_paths:
        return QuantResult(
            success=False,
            error=f"No R2 files for droplet protocol ({chemistry}); sample may be single-end",
            fail_category="missing_r2",
            time_s=time.time() - t0
        )
    
    # simpleaf requires equal numbers of R1 and R2 files
    if len(r1_paths) != len(r2_paths):
        return QuantResult(
            success=False,
            error=f"R1/R2 file count mismatch: {len(r1_paths)} R1 vs {len(r2_paths)} R2",
            fail_category="file_mismatch",
            time_s=time.time() - t0
        )
    
    # Resolve organism to taxon ID and get index path
    taxon_id = config.get_taxon_id(organism)
    if taxon_id is None:
        return QuantResult(
            success=False,
            error=f"Organism not in species config: {organism}",
            fail_category="no_index",
            time_s=time.time() - t0
        )
    
    species_info = config.species_ref.get(taxon_id)
    if species_info is None:
        return QuantResult(
            success=False,
            error=f"No species info for taxon {taxon_id}",
            fail_category="no_index",
            time_s=time.time() - t0
        )
    
    # Use config.paths.get_index_path() which returns {index_dir}/{species}_splici
    index_base = config.paths.get_index_path(species_info["name"], species_info["assembly"])
    # Piscem index files are inside the index/ subdirectory
    index_path = index_base / "index"
    if not index_path.exists():
        # Fallback: check if piscem files are directly in the base dir
        if (index_base / "piscem_idx.ctab").exists():
            index_path = index_base
        else:
            return QuantResult(
                success=False,
                error=f"Piscem index not found at {index_path} or {index_base}",
                fail_category="no_index",
                time_s=time.time() - t0
            )
    
    # Respect SLURM CPU allocation if available
    import os
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    threads = int(slurm_cpus) if slurm_cpus else config.resources.simpleaf_threads

    # Ensure ALEVIN_FRY_HOME is set (simpleaf requires it)
    if not os.environ.get("ALEVIN_FRY_HOME"):
        af_home = config.paths.af_home if hasattr(config.paths, 'af_home') else config.paths.project_base / "af_home"
        os.environ["ALEVIN_FRY_HOME"] = str(af_home)
        logger.info(f"Set ALEVIN_FRY_HOME={af_home}")

    # Translate chemistry name to simpleaf-compatible string
    # simpleaf only recognizes: 10xv2, 10xv3, 10xv4-3p, visium*, 10x-atac*
    # For dropseq/indrops/celseq, we need geometry strings
    from scgeo.pipeline.detect import get_chemistry_string
    simpleaf_chemistry = get_chemistry_string(chemistry, 0, 0, config)
    if simpleaf_chemistry is None:
        simpleaf_chemistry = chemistry  # Pass through as-is
    
    logger.info(f"Chemistry: {chemistry} → simpleaf: {simpleaf_chemistry}")

    # Clean up any old outputs from a previous run (important for retries)
    import shutil as _shutil
    for subdir in ("af_map", "af_quant"):
        p = output_dir / subdir
        if p.exists():
            logger.info(f"Cleaning old {subdir}/ from previous run")
            _shutil.rmtree(p, ignore_errors=True)

    # Build base simpleaf command (without permit-list strategy)
    # Note: t2g mapping is automatically resolved from simpleaf_index.json in the index dir
    # Note: simpleaf has no explicit memory limit flag; -m is --t2g-map
    base_cmd = [
        "simpleaf",
        "quant",
        "-c", simpleaf_chemistry,
        "-1", ",".join(str(p) for p in r1_paths),
        "-2", ",".join(str(p) for p in r2_paths),
        "-i", str(index_path),
        "-o", str(output_dir),
        "-t", str(threads),
        "--use-piscem",  # Use piscem mapper
        "--resolution", "cr-like",  # Cell Ranger-like resolution
    ]

    # Strategy selection for permit-list generation:
    #   For 10x chemistries: use --unfiltered-pl (whitelist) as primary strategy.
    #     This matches Cell Ranger's approach and avoids the extremely expensive
    #     --knee-distance computation (observed: 10+ min wall time for 25M reads).
    #   For non-10x: use --knee as primary (no whitelist available).
    #   Fallback chain: --expect-cells 5000 → --expect-cells 500
    import shutil as _shutil_fb

    # Resolve 10x whitelist path
    _10x_wl_path = None
    if simpleaf_chemistry in ("10xv2", "10xv3", "10xv4-3p"):
        af_home = os.environ.get("ALEVIN_FRY_HOME", "")
        chem_json = Path(af_home) / "chemistries.json"
        if chem_json.exists():
            try:
                with open(chem_json) as _cf:
                    chem_info = json.load(_cf)
                plist_name = chem_info.get(simpleaf_chemistry, {}).get("plist_name", "")
                if plist_name:
                    _wl = Path(af_home) / "plist" / plist_name
                    if _wl.exists():
                        _10x_wl_path = _wl
            except Exception:
                pass

    if _10x_wl_path is not None:
        cmd = base_cmd + ["--unfiltered-pl", str(_10x_wl_path)]
        initial_strategy = "unfiltered-pl"
        logger.info(f"Using --unfiltered-pl for {simpleaf_chemistry} (whitelist: {_10x_wl_path.name})")
    else:
        cmd = base_cmd + ["--knee"]
        initial_strategy = "knee"
        logger.info(f"Using --knee (no whitelist for {simpleaf_chemistry})")

    permit_strategy = initial_strategy
    logger.info(f"Running simpleaf: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config.resources.simpleaf_timeout
        )
        
        if result.returncode != 0:
            stderr = result.stderr or ""
            # Retry with fallback strategies if permit-list generation failed
            if "generate-permit-list failed" in stderr or "permit" in stderr.lower():
                # Known upstream alevin-fry bug (exit status 25856) — no
                # permit-list strategy can work around it, so bail out immediately.
                if "25856" in stderr:
                    logger.warning(
                        "simpleaf generate-permit-list hit known upstream bug "
                        "(exit 25856) — skipping retries"
                    )
                else:
                    # Build fallback strategies
                    fallback_strategies = []

                    # If we started with --unfiltered-pl, try --knee as fallback
                    if _10x_wl_path is not None:
                        fallback_strategies.append(
                            (base_cmd + ["--knee"],
                             "--knee (whitelist fallback)")
                        )

                    # Progressively smaller expected cell counts
                    fallback_strategies.append(
                        (base_cmd + ["--expect-cells", "5000"],
                         "--expect-cells 5000")
                    )
                    fallback_strategies.append(
                        (base_cmd + ["--expect-cells", "500"],
                         "--expect-cells 500")
                    )
                    
                    for retry_cmd, strategy_name in fallback_strategies:
                        # Before cleaning up, check mapping rate from first attempt.
                        # If mapping rate is below QC threshold, retrying permit-list
                        # strategies won't help — sample will fail QC regardless.
                        _mi = output_dir / "af_map" / "map_info.json"
                        if _mi.exists():
                            try:
                                with open(_mi) as _mf:
                                    _minfo = json.load(_mf)
                                _nr = _minfo.get("num_reads", 0)
                                _nm = _minfo.get("num_mapped", 0)
                                if _nr > 0 and (_nm == 0 or _nm / _nr < config.qc.min_mapping_rate):
                                    _mr_pct = f"{_nm/_nr:.2%}" if _nr > 0 else "0%"
                                    logger.info(
                                        f"Mapping rate is {_mr_pct} (below {config.qc.min_mapping_rate:.0%} "
                                        f"threshold) — skipping permit-list retries"
                                    )
                                    break
                            except Exception:
                                pass

                        # Clean intermediate outputs from failed attempt
                        for subdir in ("af_map", "af_quant"):
                            p = output_dir / subdir
                            if p.exists():
                                _shutil_fb.rmtree(p, ignore_errors=True)
                        
                        logger.info(
                            f"Permit-list failed, retrying with {strategy_name}"
                        )
                        result = subprocess.run(
                            retry_cmd,
                            capture_output=True,
                            text=True,
                            timeout=config.resources.simpleaf_timeout
                        )
                        if result.returncode == 0:
                            permit_strategy = strategy_name
                            logger.info(f"Retry succeeded with {strategy_name}")
                            break
            
            if result.returncode != 0:
                # Try to extract mapping rate from failed attempt for diagnostics
                _fail_mr = 0.0
                _fail_mi = output_dir / "af_map" / "map_info.json"
                if _fail_mi.exists():
                    try:
                        with open(_fail_mi) as _mf:
                            _minfo = json.load(_mf)
                        _nr = _minfo.get("num_reads", 0)
                        _nm = _minfo.get("num_mapped", 0)
                        if _nr > 0:
                            _fail_mr = _nm / _nr
                    except Exception:
                        pass
                return QuantResult(
                    success=False,
                    error=f"simpleaf failed (exit {result.returncode}): {(result.stderr or '')[:500]}",
                    fail_category="simpleaf_crash",
                    time_s=time.time() - t0,
                    mapping_rate=_fail_mr,
                )
        
        # Parse metrics from simpleaf output files
        mapping_rate = 0.0
        n_reads = 0
        
        # piscem writes map_info.json (not meta_info.json as in older salmon-based pipelines)
        map_info_path = output_dir / "af_map" / "map_info.json"
        if map_info_path.exists():
            try:
                with open(map_info_path) as f:
                    map_info = json.load(f)
                    n_reads = map_info.get("num_reads", 0)
                    num_mapped = map_info.get("num_mapped", 0)
                    if n_reads > 0:
                        mapping_rate = num_mapped / n_reads
            except Exception as e:
                logger.warning(f"Failed to parse map_info.json: {e}")
        
        # Also extract cell count from quant.json
        quant_json_path = output_dir / "af_quant" / "quant.json"
        n_cells = 0
        if quant_json_path.exists():
            try:
                with open(quant_json_path) as f:
                    quant_info = json.load(f)
                    n_cells = quant_info.get("num_quantified_cells", 0)
            except Exception as e:
                logger.warning(f"Failed to parse quant.json: {e}")
        
        elapsed = time.time() - t0
        logger.info(f"simpleaf completed in {elapsed:.1f}s (mapping rate: {mapping_rate:.2%}, cells: {n_cells}, reads: {n_reads})")
        
        return QuantResult(
            success=True,
            output_dir=output_dir,
            tool="simpleaf",
            time_s=elapsed,
            mapping_rate=mapping_rate,
            n_reads=n_reads,
            permit_strategy=permit_strategy,
        )
        
    except subprocess.TimeoutExpired:
        return QuantResult(
            success=False,
            error=f"simpleaf timed out after {config.resources.simpleaf_timeout}s",
            fail_category="simpleaf_timeout",
            time_s=time.time() - t0
        )
    except Exception as e:
        return QuantResult(
            success=False,
            error=f"simpleaf exception: {e}",
            fail_category="simpleaf_exception",
            time_s=time.time() - t0
        )
