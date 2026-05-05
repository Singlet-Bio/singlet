#!/usr/bin/env python3
"""
Generate manuscript figures from benchmark_results_all.json.

Figures produced:
  - fig_sort_effect.pdf: Sort overhead vs nnz, SpMM invariance
  - fig_io_compute.pdf: I/O-compute ratio across latent dims
  - fig_autoencoder.pdf: Load vs compute fraction, cells/s
  - fig_worker_scaling.pdf: Throughput vs num_workers

Usage:
    python generate_benchmark_figures.py [benchmark_results_all.json]
"""
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# --- Style ---
rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.dpi": 300,
})

COLORS = {
    "1pz": "#2196F3",
    "sorted": "#FF5722",
    "unsorted": "#4CAF50",
    "compute": "#E91E63",
    "io": "#9E9E9E",
    "soma": "#FF9800",
}


def load_results(path="benchmark_results_all.json"):
    with open(path) as f:
        return json.load(f)


def fig_sort_effect(data, outpath="fig_sort_effect.pdf"):
    """Sort overhead vs nnz, plus SpMM timing bars."""
    results = data["sort_effect"]
    if not results:
        print("  No sort_effect data, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    # Panel A: Sort overhead vs nnz
    nnz = [r["nnz"] for r in results]
    overhead_ms = [r["sort_overhead_ms"] for r in results]
    overhead_pct = [r["sort_overhead_pct"] for r in results]

    ax1.scatter(nnz, overhead_ms, c=COLORS["sorted"], s=50, zorder=5)
    for r in results:
        ax1.annotate(f"{r['sort_overhead_pct']:.0f}%",
                     (r["nnz"], r["sort_overhead_ms"]),
                     textcoords="offset points", xytext=(5, 5),
                     fontsize=7, color="gray")
    ax1.set_xlabel("Nonzeros")
    ax1.set_ylabel("Sort overhead (ms)")
    ax1.set_xscale("log")
    ax1.set_title("(a) Sort overhead scales with nnz")
    ax1.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)

    # Panel B: SpMM timing — sorted vs unsorted
    gses = [r.get("gse_id", f"ds{i}") for i, r in enumerate(results)]
    spmm_s = [r.get("spmm_sorted_ms", 0) for r in results]
    spmm_u = [r.get("spmm_unsorted_ms", 0) for r in results]

    x = np.arange(len(gses))
    w = 0.35
    ax2.bar(x - w/2, spmm_s, w, label="Sorted", color=COLORS["sorted"], alpha=0.8)
    ax2.bar(x + w/2, spmm_u, w, label="Unsorted", color=COLORS["unsorted"], alpha=0.8)
    ax2.set_xlabel("Dataset")
    ax2.set_ylabel("SpMM time (ms)")
    ax2.set_title("(b) cuSPARSE SpMM: identical timing")
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.replace("GSE", "") for g in gses], rotation=45, ha="right")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f"  Saved {outpath}")


def fig_io_compute(data, outpath="fig_io_compute.pdf"):
    """I/O-compute ratio across latent dimensions."""
    results = data.get("io_compute", {})
    dims = results.get("latent_dims", [])
    if not dims:
        print("  No io_compute data, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    latents = [d["latent"] for d in dims]
    compute_ms = [d["compute_median_ms"] for d in dims]
    io_ms = [d["io_median_ms"] for d in dims]
    ratios = [d["io_ratio"] for d in dims]
    gflops = [d["gflops_per_sec"] for d in dims]

    # Panel A: Stacked bar — I/O vs compute time
    x = np.arange(len(latents))
    ax1.bar(x, io_ms, label="I/O (CPU read)", color=COLORS["io"])
    ax1.bar(x, compute_ms, bottom=io_ms, label="Compute (GPU)", color=COLORS["compute"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(latents)
    ax1.set_xlabel("Latent dimension")
    ax1.set_ylabel("Time per step (ms)")
    ax1.set_title("(a) I/O vs compute by model size")
    ax1.legend()

    # Panel B: I/O ratio (fraction of time spent on I/O)
    ax2.plot(latents, ratios, "o-", color=COLORS["1pz"], linewidth=2, markersize=6)
    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.5)
    ax2.set_xlabel("Latent dimension")
    ax2.set_ylabel("I/O fraction of total time")
    ax2.set_title("(b) I/O fraction decreases with compute")
    ax2.set_ylim(0, 1)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(latents, gflops, "s--", color=COLORS["compute"], linewidth=1,
                  markersize=4, alpha=0.7)
    ax2_twin.set_ylabel("GPU throughput (GFLOP/s)", color=COLORS["compute"])

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f"  Saved {outpath}")


def fig_autoencoder(data, outpath="fig_autoencoder.pdf"):
    """Autoencoder training: load vs compute, cells/s."""
    results = data.get("autoencoder", [])
    if not results:
        print("  No autoencoder data, skipping")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    gses = [r.get("gse_id", f"ds{i}") for i, r in enumerate(results)]
    load_pct = [r["load_pct"] for r in results]
    compute_pct = [r["compute_pct"] for r in results]
    cells_s = [r.get("cells_per_sec", 0) for r in results]
    nnz = [r["nnz"] for r in results]

    # Panel A: Stacked horizontal bar — load vs compute
    x = np.arange(len(gses))
    ax1.barh(x, load_pct, label="Data loading", color=COLORS["io"])
    ax1.barh(x, compute_pct, left=load_pct, label="GPU compute", color=COLORS["compute"])
    ax1.set_yticks(x)
    ax1.set_yticklabels([g.replace("GSE", "") for g in gses])
    ax1.set_xlabel("% of training time")
    ax1.set_title("(a) Load vs compute fraction")
    ax1.legend(loc="lower right")
    ax1.set_xlim(0, 100)

    # Panel B: Cells/s vs nnz
    ax2.scatter(nnz, cells_s, c=COLORS["1pz"], s=60, zorder=5)
    ax2.set_xlabel("Nonzeros per dataset")
    ax2.set_ylabel("Cells/s")
    ax2.set_xscale("log")
    ax2.set_title("(b) Training throughput")
    for i, r in enumerate(results):
        ax2.annotate(gses[i].replace("GSE", ""),
                     (r["nnz"], r.get("cells_per_sec", 0)),
                     textcoords="offset points", xytext=(5, 5), fontsize=7)

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f"  Saved {outpath}")


def fig_worker_scaling(data, outpath="fig_worker_scaling.pdf"):
    """Throughput vs number of DataLoader workers."""
    results = data.get("worker_scaling", {})
    configs = results.get("workers", [])
    if not configs:
        print("  No worker_scaling data, skipping")
        return

    fig, ax = plt.subplots(figsize=(4, 3))

    workers = [c["n_workers"] for c in configs]
    cells_s = [c["cells_per_sec"] for c in configs]

    ax.plot(workers, cells_s, "o-", color=COLORS["1pz"], linewidth=2, markersize=8)
    for i, c in enumerate(configs):
        ax.annotate(f"{c['cells_per_sec']:.0f}",
                    (workers[i], cells_s[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    ax.set_xlabel("Number of DataLoader workers")
    ax.set_ylabel("Cells/s")
    ax.set_title("DataLoader worker scaling")
    ax.set_xticks(workers)

    plt.tight_layout()
    fig.savefig(outpath)
    plt.close()
    print(f"  Saved {outpath}")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results_all.json"
    print(f"Loading results from {path}")
    data = load_results(path)

    print(f"\nDevice: {data.get('device', 'N/A')}")
    print(f"PyTorch: {data.get('pytorch_version', 'N/A')}")
    print(f"Datasets: {len(data.get('sort_effect', []))}")

    print("\nGenerating figures...")
    fig_sort_effect(data)
    fig_io_compute(data)
    fig_autoencoder(data)
    fig_worker_scaling(data)

    print("\nDone!")


if __name__ == "__main__":
    main()
