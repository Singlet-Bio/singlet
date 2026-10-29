"""Export Singlet's own chrM outputs of one run to npz so analyze.py can score them with numpy only.

  python singlet_export.py SINGLET_OUT_DIR OUT.npz      (needs singlet._pz: run inside the Singlet runtime)

Writes: counts_* (private/mt_allele_counts.1pz, read level: feature = pos*4 + base, x barcodes), rev_* (its reverse-strand
subset), het_* (public mt_heteroplasmy.1pz: the stored per-cell value at each public site), barcodes, het_sites.
"""
import sys

import numpy as np
from singlet._pz import read_1pz


def csc(path):
    r = read_1pz(path)
    return r


out_dir, out = sys.argv[1], sys.argv[2]
res = {}
for key, fn in (("counts", "private/mt_allele_counts.1pz"), ("rev", "private/mt_allele_counts_rev.1pz"),
                ("het", "mt_heteroplasmy.1pz")):
    try:
        r = csc(f"{out_dir}/{fn}")
    except Exception as e:  # noqa: BLE001
        print(f"skip {fn}: {e}", file=sys.stderr)
        continue
    res[f"{key}_indptr"] = np.asarray(r["indptr"], np.int64)
    res[f"{key}_indices"] = np.asarray(r["indices"], np.int64)
    res[f"{key}_data"] = np.asarray(r["data"])
    res[f"{key}_shape"] = np.array([r["m"], r["n"]])
    res[f"{key}_barcodes"] = np.array([str(x) for x in r["colnames"]])
    if key == "het":
        res["het_sites"] = np.array([str(x) for x in r["rownames"]])
        res["het_meta"] = np.array([f"{k}={v}" for k, v in (r.get("user_kv") or {}).items()])
    print(key, r["m"], r["n"], len(r["data"]), file=sys.stderr)
np.savez_compressed(out, **res)
