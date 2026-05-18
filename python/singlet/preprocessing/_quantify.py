# SPDX-License-Identifier: MIT
"""Quantification module — simpleaf with piscem mapper."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class QuantResult:
    """Result of quantification."""

    success: bool = False
    output_dir: Optional[Path] = None
    tool: str = ""
    time_s: float = 0.0
    mapping_rate: float = 0.0
    n_reads: int = 0
    n_cells: int = 0
    error: str = ""


def quantify(
    r1_paths: List[str | Path],
    r2_paths: List[str | Path],
    protocol: str,
    organism: str,
    output_dir: str | Path,
    *,
    index_dir: Optional[str | Path] = None,
    threads: Optional[int] = None,
    resolution: str = "cr-like",
    use_knee: bool = True,
) -> QuantResult:
    """Quantify scRNA-seq with simpleaf.

    Parameters
    ----------
    r1_paths : list of paths
        R1 FASTQ files.
    r2_paths : list of paths
        R2 FASTQ files.
    protocol : str
        Protocol/chemistry string (e.g. "10xv3").
    organism : str
        Organism name (e.g. "human", "mouse").
    output_dir : path
        Output directory.
    index_dir : path, optional
        Path to piscem index. Auto-resolved from organism if not provided.
    threads : int, optional
        Number of threads. Reads ``SLURM_CPUS_PER_TASK`` if not set.
    resolution : str
        Cell resolution strategy. Default ``"cr-like"``.
    use_knee : bool
        Use knee-point cell filtering.

    Returns
    -------
    QuantResult
    """
    t0 = time.time()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if threads is None:
        threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

    # Resolve index from organism
    if index_dir is None:
        from singlet.preprocessing._species import get_species_info, get_taxon_id

        try:
            txid = get_taxon_id(organism)
            info = get_species_info(txid)
            index_dir = info.get("index_path")
        except KeyError:
            return QuantResult(error=f"Unknown organism: {organism}", time_s=time.time() - t0)

    if index_dir is None:
        return QuantResult(error=f"No index for organism: {organism}", time_s=time.time() - t0)

    # Map protocol to chemistry
    from singlet.preprocessing._detect import get_chemistry_string

    chemistry = get_chemistry_string(protocol)
    if chemistry is None:
        return QuantResult(
            error=f"Unknown chemistry for protocol: {protocol}",
            time_s=time.time() - t0,
        )

    r1_str = ",".join(str(p) for p in r1_paths)
    r2_str = ",".join(str(p) for p in r2_paths)

    cmd = [
        "simpleaf",
        "quant",
        "--reads1",
        r1_str,
        "--reads2",
        r2_str,
        "-c",
        chemistry,
        "--use-piscem",
        "-i",
        str(index_dir),
        "-o",
        str(out),
        "-t",
        str(threads),
        "--resolution",
        resolution,
    ]
    if use_knee:
        cmd.append("--knee")

    try:
        subprocess.run(cmd, timeout=14400, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return QuantResult(error=f"simpleaf failed: {e}", time_s=time.time() - t0)

    # Parse results
    mapping_rate = 0.0
    n_reads = 0
    n_cells = 0

    map_info = out / "af_quant" / "alevin" / "map_info.json"
    if map_info.exists():
        with open(map_info) as f:
            mi = json.load(f)
        n_reads = mi.get("num_processed", 0)
        mapped = mi.get("num_mapped", 0)
        mapping_rate = mapped / n_reads if n_reads > 0 else 0.0

    quant_json = out / "af_quant" / "alevin" / "quant.json"
    if quant_json.exists():
        with open(quant_json) as f:
            qi = json.load(f)
        n_cells = qi.get("num_quantified_cells", 0)

    return QuantResult(
        success=True,
        output_dir=out,
        tool="simpleaf",
        time_s=time.time() - t0,
        mapping_rate=mapping_rate,
        n_reads=n_reads,
        n_cells=n_cells,
    )
