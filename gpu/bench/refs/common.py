"""
singlet-gpu/bench/refs/common.py
SPDX-License-Identifier: GPL-2.0-or-later

Shared helpers for Python SOTA reference benchmark scripts.

Each reference script (io_pz_loader_ref.py, lognorm_ref.py, etc.) imports
this module and calls run_benchmark() to time its SOTA operation and emit
a result line that the C++ bench driver can parse from the subprocess output.

Protocol (stdout lines parsed by the C++ driver):
  SOTA_WALL_SEC=<float>    — wall-clock seconds (median of TIMED_ITERS runs)
  SOTA_MEM_MB=<float>      — peak RSS (process-level; reported via tracemalloc)
  SOTA_CELLS=<int>         — number of cells processed

The C++ bench driver reads these three lines from the subprocess stdout and
populates BenchRow.sota_wall_sec / sota_mem_mb.
"""

import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Callable, Any

WARMUP_ITERS = 2   # fewer than GPU warmup — Python startup cost amortised
TIMED_ITERS  = 5

CANONICAL_SAMPLE_DIR = Path(
    "/mnt/projects/debruinz_project/singlify_pipeline/quant/scrna"
    "/GSE127/GSE127918/GSM4037629"
)


def time_function(fn: Callable, *args, **kwargs) -> tuple[float, float, Any]:
    """
    Run fn(*args, **kwargs) WARMUP_ITERS + TIMED_ITERS times.
    Returns (median_wall_sec, peak_mem_mb, last_result).

    Memory is measured via tracemalloc over the timed iterations only.
    WHY tracemalloc: it tracks Python-heap allocations and gives a
    per-run peak without requiring root or nvidia-smi. For numpy/scipy
    arrays (which live in C malloc, not Python heap), we also sample
    process RSS via /proc/self/status on Linux.
    """
    # Warmup
    for _ in range(WARMUP_ITERS):
        fn(*args, **kwargs)

    wall_times = []
    peak_mb_list = []
    last_result = None

    for _ in range(TIMED_ITERS):
        tracemalloc.start()
        t0 = time.perf_counter()
        last_result = fn(*args, **kwargs)
        t1 = time.perf_counter()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        wall_times.append(t1 - t0)
        peak_mb_list.append(peak / (1024 * 1024))

    wall_times.sort()
    n = len(wall_times)
    if n % 2 == 1:
        median_wall = wall_times[n // 2]
    else:
        median_wall = 0.5 * (wall_times[n // 2 - 1] + wall_times[n // 2])

    # Also sample /proc RSS for C-heap allocations (numpy, scipy).
    rss_mb = _proc_rss_mb()
    peak_mb = max(max(peak_mb_list), rss_mb)

    return median_wall, peak_mb, last_result


def _proc_rss_mb() -> float:
    """Read VmRSS from /proc/self/status (Linux only; returns 0 elsewhere)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return 0.0


def emit_result(wall_sec: float, mem_mb: float, n_cells: int) -> None:
    """Print the three lines the C++ driver parses."""
    print(f"SOTA_WALL_SEC={wall_sec:.6f}", flush=True)
    print(f"SOTA_MEM_MB={mem_mb:.2f}", flush=True)
    print(f"SOTA_CELLS={n_cells}", flush=True)


def run_python_reference(script_path: str, extra_args: list[str] | None = None
                         ) -> tuple[float, float]:
    """
    Launch a reference script as a subprocess and parse its output.
    Returns (wall_sec, peak_mb).

    The script must print SOTA_WALL_SEC=..., SOTA_MEM_MB=..., SOTA_CELLS=...
    to stdout (in any order). Returns (-1, -1) if the script fails or is
    missing a required import.
    """
    cmd = [sys.executable, script_path] + (extra_args or [])
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,   # 10-minute timeout per reference run
        )
        if result.returncode != 0:
            print(f"[common.py] reference script failed (rc={result.returncode}): "
                  f"{script_path}", file=sys.stderr)
            print(result.stderr[-2000:], file=sys.stderr)
            return -1.0, -1.0

        wall_sec = -1.0
        mem_mb   = -1.0
        for line in result.stdout.splitlines():
            if line.startswith("SOTA_WALL_SEC="):
                wall_sec = float(line.split("=", 1)[1])
            elif line.startswith("SOTA_MEM_MB="):
                mem_mb = float(line.split("=", 1)[1])
        return wall_sec, mem_mb

    except subprocess.TimeoutExpired:
        print(f"[common.py] reference script timed out: {script_path}", file=sys.stderr)
        return -1.0, -1.0
    except Exception as e:
        print(f"[common.py] error running reference: {e}", file=sys.stderr)
        return -1.0, -1.0


def find_sample_dir(override: str | None = None) -> Path | None:
    """
    Return the canonical sample directory path if it exists, else None.
    Allows an optional override for testing.
    """
    p = Path(override) if override else CANONICAL_SAMPLE_DIR
    if p.exists():
        return p
    return None
