# SPDX-License-Identifier: MIT
"""Research Alpha QC check 7: load studies end to end through the public Python client from data.singlet.bio, timed.

For each study (fresh cache): open_study; to_anndata() (default: counts + spliced + unspliced, called cells); the cells table
with labels; a single-sample partial read (one sample, a cell filter, only the needed chunk runs); and an independent
cross-check: per-cell totals of X against the legacy exon + intron matrices of the same cells (legacy reader, via the guard).

    python qc_clients.py --out OUT.json --scratch DIR GSE [GSE ...]
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.environ.get("QC_SRC", ""), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import qc_http as H  # noqa: E402


def legacy_totals(gse, gsm, scratch):
    """{barcode: exon+intron UMIs} of the legacy matrices of one sample (legacy reader)."""
    from singlet import _pz

    st, hh, _ = H.request(f"{H.BASE}/data/{gse}/{gse}.singlet", method="HEAD")
    rz = H.RangeZip(f"data/{gse}/{gse}.singlet", int(hh["content-length"]))
    tot = None
    names = None
    for mem in ("exon_counts.1pz", "intron_counts.1pz"):
        nm = f"samples/{gsm}/{mem}"
        if nm not in rz.entries:
            continue
        p = rz.read(nm, dest=os.path.join(scratch, mem))
        r = _pz.read_1pz(p)
        ip = np.asarray(r["indptr"]).astype(np.int64)
        dt = np.asarray(r["data"]).astype(np.float64)
        cs = np.bincount(np.repeat(np.arange(len(ip) - 1), np.diff(ip)), weights=dt, minlength=len(ip) - 1)
        if names is None:
            names, tot = list(r["colnames"]), cs
        else:
            assert list(r["colnames"]) == names
            tot = tot + cs
        os.remove(p)
    return dict(zip(names, tot)) if names else {}


def one(gse, scratch):
    from singlet.s1 import client as C

    R = {"gse": gse}
    t = time.time()
    st = C.open_study(gse, force=True)
    R["open_s"] = round(time.time() - t, 2)
    gsms = list(st.samples)
    R["n_samples"] = len(gsms)
    R["n_called_manifest"] = sum(int(st.samples[g].raw.get("n_called") or 0) for g in gsms)
    t = time.time()
    ad = st.to_anndata()
    R["to_anndata_s"] = round(time.time() - t, 2)
    R["shape"] = list(ad.shape)
    R["layers"] = [repr(k) for k in ad.layers.keys()]
    R["obs_columns"] = list(ad.obs.columns)[:60]
    R["n_obs_equals_called"] = int(ad.n_obs) == R["n_called_manifest"]
    if "spliced" in ad.layers and "unspliced" in ad.layers:
        d = (ad.X - ad.layers["spliced"] - ad.layers["unspliced"])
        R["X_equals_spliced_plus_unspliced"] = bool(abs(d).sum() == 0)
    R["labels_in_obs"] = [c for c in ad.obs.columns if c.startswith(("cell_type", "sex_", "ancestry"))]
    R["called_ed_in_obs"] = "called_ed" in ad.obs.columns
    try:
        tr = st.transport.stats() if hasattr(st.transport, "stats") else None
        R["transport_stats"] = {k: getattr(tr, k) for k in ("requests", "bytes")} if tr is not None else None
    except Exception:  # noqa: BLE001
        pass
    # cells with labels
    t = time.time()
    cells = st.cells(where={"called": True}, layers=False, max_cells=None)
    R["cells_s"] = round(time.time() - t, 2)
    R["cells_shape"] = list(cells.shape)
    R["cells_label_cols"] = [c for c in cells.obs.columns if c.startswith(("cell_type", "sex_", "ancestry"))]
    if "sex_chrom" in cells.obs:
        R["cells_sex_counts"] = cells.obs["sex_chrom"].astype(str).value_counts().to_dict()
    # single-sample partial read: largest sample, cells with many UMIs, first 200
    g = max(gsms, key=lambda x: int(st.samples[x].raw.get("n_called") or 0))
    t = time.time()
    part = st.cells(where="n_umi_exon > 500", samples=[g], max_cells=200, layers=True)
    R["partial"] = {"gsm": g, "shape": list(part.shape), "s": round(time.time() - t, 2), "layers": [repr(k) for k in part.layers.keys()]}
    t = time.time()
    one_s = st.to_anndata(samples=[g])
    R["one_sample"] = {"gsm": g, "shape": list(one_s.shape), "s": round(time.time() - t, 2)}
    # independent cross-check against the legacy matrices of that sample
    try:
        t = time.time()
        lt = legacy_totals(gse, g, scratch)
        bc = []
        for x in one_s.obs_names:
            x = str(x)
            for pre in (g + "_", g + ":", g + "-"):
                if x.startswith(pre) and x not in lt:
                    x = x[len(pre):]
            bc.append(x)
        R["legacy_crosscheck_obs_name_example"] = str(one_s.obs_names[0]) if one_s.n_obs else None
        x_tot = np.asarray(one_s.X.sum(axis=1)).ravel()
        lv = np.array([lt.get(b, np.nan) for b in bc])
        R["legacy_crosscheck"] = {"n": int(len(bc)), "matched_barcodes": int(np.isfinite(lv).sum()),
                                  "totals_equal": int(np.sum(np.isfinite(lv) & (lv == x_tot))),
                                  "sum_X": float(x_tot.sum()), "sum_legacy": float(np.nansum(lv)),
                                  "s": round(time.time() - t, 2)}
    except Exception as e:  # noqa: BLE001
        R["legacy_crosscheck"] = {"error": repr(e)[:300]}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("gses", nargs="+")
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    os.environ.setdefault("SINGLET_S1_CACHE", os.path.join(a.scratch, "cache"))
    out = []
    for g in a.gses:
        try:
            r = one(g, a.scratch)
        except Exception:  # noqa: BLE001
            r = {"gse": g, "error": traceback.format_exc()[-2000:]}
        print(json.dumps(r, default=str)[:3000], flush=True)
        out.append(r)
    json.dump(out, open(a.out, "w"), default=str, indent=1)


if __name__ == "__main__":
    main()
