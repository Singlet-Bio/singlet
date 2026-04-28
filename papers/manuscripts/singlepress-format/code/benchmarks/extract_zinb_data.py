#!/usr/bin/env python3
"""Extract ZINB parameters and value distributions from benchmark datasets.

Reads .1pz files, fits ZINB parameters, computes information-theoretic
metrics, and merges with benchmark_results_v3.json.

Outputs: zinb_data.csv (one row per dataset) and value_distributions.csv.
"""
import sys, json, os, glob
import numpy as np
from scipy import stats, optimize, sparse

# ── Locate singlepress ──────────────────────────────────────────
sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/singlepress")
import singlepress

QUANT_DIR = "/mnt/projects/debruinz_project/cellarium/pipeline/quant"

# ── Load benchmark results ──────────────────────────────────────
with open("benchmark_results_v3.json") as f:
    bench = json.load(f)

gse_ids = [d["gse_id"] for d in bench]

def find_1pz_files(gse_id):
    """Find .1pz files for a given GSE: merged or individual samples."""
    merged = os.path.join(QUANT_DIR, gse_id, "merged", "counts.1pz")
    if os.path.isfile(merged):
        return [merged]
    pattern = os.path.join(QUANT_DIR, gse_id, "GSM*", "counts.1pz")
    files = sorted(glob.glob(pattern))
    return files[:3]  # cap at 3 samples for speed


def fit_zinb(values, n_zeros, n_total):
    """Fit ZINB to a count vector (nonzero values + known zero count).

    Returns (pi, r, p, mean_nz, var_nz, max_val, frac_one, entropy_nz).
    pi = zero-inflation probability (structural zeros beyond NB zeros)
    r = NB dispersion (size), p = NB success probability
    """
    vals = np.asarray(values, dtype=np.float64)
    n_nz = len(vals)
    frac_one = np.sum(vals == 1) / n_nz if n_nz > 0 else 0
    mean_nz = vals.mean() if n_nz > 0 else 0
    var_nz = vals.var() if n_nz > 0 else 0
    max_val = int(vals.max()) if n_nz > 0 else 0

    # Entropy of nonzero values
    if n_nz > 0:
        vc = np.bincount(vals.astype(int))
        vc = vc[vc > 0]
        p_vals = vc / vc.sum()
        entropy_nz = -np.sum(p_vals * np.log2(p_vals))
    else:
        entropy_nz = 0

    # Overall mean including zeros
    overall_mean = (n_nz * mean_nz) / n_total if n_total > 0 else 0
    overall_var = n_nz * (var_nz + mean_nz**2) / n_total - overall_mean**2

    # Fit NB to nonzero values using method of moments on full vector
    # E[X] = r(1-p)/p, Var[X] = r(1-p)/p^2
    if overall_mean > 0 and overall_var > overall_mean:
        p_nb = overall_mean / overall_var
        r_nb = overall_mean * p_nb / (1 - p_nb)
        r_nb = max(r_nb, 0.01)
        p_nb = min(max(p_nb, 0.001), 0.999)
    else:
        r_nb = 1.0
        p_nb = 0.5

    # pi = observed_zeros/total - NB_zero_prob
    nb_zero_prob = stats.nbinom.pmf(0, r_nb, p_nb)
    pi = max(0, n_zeros / n_total - (1 - n_zeros / n_total) * 0) if n_total > 0 else 0
    # More precisely: P(Y=0) = pi + (1-pi)*NB(0|r,p)
    # observed_zero_frac = n_zeros / n_total
    # pi = (obs_zero - NB(0|r,p)) / (1 - NB(0|r,p))
    obs_zero_frac = n_zeros / n_total if n_total > 0 else 0
    if nb_zero_prob < obs_zero_frac:
        pi = (obs_zero_frac - nb_zero_prob) / (1 - nb_zero_prob)
    else:
        pi = 0.0

    return pi, r_nb, p_nb, mean_nz, var_nz, max_val, frac_one, entropy_nz


def get_value_distribution(mat, max_val=50):
    """Get histogram of nonzero values, capped at max_val."""
    data = mat.data if sparse.issparse(mat) else mat[mat > 0]
    data = data.astype(np.int64)
    hist = np.bincount(np.minimum(data, max_val))
    return hist


rows = []
val_dist_rows = []

for d in bench:
    gse = d["gse_id"]
    files = find_1pz_files(gse)

    if not files:
        print(f"  {gse}: no files found, using summary stats only", file=sys.stderr)
        # Use ratio and density from benchmark data, estimate ZINB from density
        density = d["density"]
        pi_est = 1 - density
        rows.append({
            "gse_id": gse,
            "species": d["species"],
            "protocol": d["protocol"],
            "ncols": d["ncols"],
            "nrows": d["nrows"],
            "nnz": d["nnz"],
            "density": density,
            "pi": pi_est,
            "r": 0.5,  # placeholder
            "p": 0.01,  # placeholder
            "mean_nz": -1,
            "var_nz": -1,
            "max_val": -1,
            "frac_one": -1,
            "entropy_nz": -1,
            "pz_bytes": d["pz_file_size"],
            "raw_bytes": d["raw_int32_bytes"],
            "ratio": d["formats"]["1pz"]["ratio_vs_int32"],
            "read_mbps": d["formats"]["1pz"]["read_MBps_int32"],
            "write_mbps": d["formats"]["1pz"]["write_MBps_int32"],
            "h5ad_bytes": d["formats"].get("h5ad_gzip", {}).get("size", -1),
            "h5_bytes": d["formats"].get("10x_h5", {}).get("size", -1),
            "npz_bytes": d["formats"].get("npz", {}).get("size", -1),
            "h5ad_read_mbps": d["formats"].get("h5ad_gzip", {}).get("read_MBps_int32", -1),
            "h5_read_mbps": d["formats"].get("10x_h5", {}).get("read_MBps_int32", -1),
            "zinb_source": "estimated",
        })
        continue

    # Read first available file
    print(f"  {gse}: reading {os.path.basename(os.path.dirname(files[0]))}/{os.path.basename(files[0])}", file=sys.stderr)
    try:
        mat = singlepress.read_1pz(files[0])
    except Exception as e:
        print(f"  {gse}: read error: {e}", file=sys.stderr)
        continue

    # Extract nonzero values
    nz_vals = mat.data.copy()
    n_total = mat.shape[0] * mat.shape[1]
    n_zeros = n_total - mat.nnz

    pi, r, p, mean_nz, var_nz, max_val, frac_one, entropy_nz = fit_zinb(
        nz_vals, n_zeros, n_total)

    # Value distribution for histogram
    hist = get_value_distribution(mat, max_val=100)
    for v, count in enumerate(hist):
        if count > 0 and v > 0:
            val_dist_rows.append({
                "gse_id": gse,
                "species": d["species"],
                "value": v,
                "count": int(count),
                "frac": count / mat.nnz,
            })

    density = d["density"]
    rows.append({
        "gse_id": gse,
        "species": d["species"],
        "protocol": d["protocol"],
        "ncols": d["ncols"],
        "nrows": d["nrows"],
        "nnz": d["nnz"],
        "density": density,
        "pi": pi,
        "r": r,
        "p": p,
        "mean_nz": mean_nz,
        "var_nz": var_nz,
        "max_val": max_val,
        "frac_one": frac_one,
        "entropy_nz": entropy_nz,
        "pz_bytes": d["pz_file_size"],
        "raw_bytes": d["raw_int32_bytes"],
        "ratio": d["formats"]["1pz"]["ratio_vs_int32"],
        "read_mbps": d["formats"]["1pz"]["read_MBps_int32"],
        "write_mbps": d["formats"]["1pz"]["write_MBps_int32"],
        "h5ad_bytes": d["formats"].get("h5ad_gzip", {}).get("size", -1),
        "h5_bytes": d["formats"].get("10x_h5", {}).get("size", -1),
        "npz_bytes": d["formats"].get("npz", {}).get("size", -1),
        "h5ad_read_mbps": d["formats"].get("h5ad_gzip", {}).get("read_MBps_int32", -1),
        "h5_read_mbps": d["formats"].get("10x_h5", {}).get("read_MBps_int32", -1),
        "zinb_source": "fitted",
    })

    # Free memory
    del mat, nz_vals

# Write CSV
import csv

print(f"\nWriting zinb_data.csv ({len(rows)} datasets)", file=sys.stderr)
with open("zinb_data.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Writing value_distributions.csv ({len(val_dist_rows)} rows)", file=sys.stderr)
if val_dist_rows:
    with open("value_distributions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=val_dist_rows[0].keys())
        w.writeheader()
        w.writerows(val_dist_rows)

print("Done.", file=sys.stderr)
