# SPDX-License-Identifier: MIT
"""Collect per-library results into per-GEX-sample ADT/HTO members, summaries and the upload plan (nothing is uploaded).

    python -P bench/adt/assemble.py --data $WORK/data/adt --inventory adt_hto_inventory.tsv --out $WORK/state/adt/assembled

For each GEX sample with at least one processed library:

* libraries whose QC verdict is ``pass`` or ``warn`` are merged: the panel is the union of their rows; a feature present
  in two libraries (e.g. hashtags amplified in both an ADT and an HTO library) is taken from the library where it has
  the most UMIs, and the choice is recorded;
* ``v2/x/<GSM>/adt.<d16>.s2`` (sidecar object with ``adt_counts.1pz``, ``hto_counts.1pz``, ``adt_panel.json``) is
  written under ``<out>/objects/`` together with its manifest fragment ``samples[i].extra_objects.adt``;
* failing libraries are listed in ``excluded.tsv`` with every failed rule.

Outputs (small, safe to pull back): ``libraries.tsv`` (one row per library: status, verdict, QC metrics),
``samples.tsv`` (one row per GEX sample), ``excluded.tsv``, ``upload_plan.tsv`` (key, local path, bytes, sha256),
``summary.json``.
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

from singlet.adt.qc import HTO_MIN_SHARE

QC_COLS = ["verdict", "failed", "warned", "n_called", "n_features", "n_antibodies", "n_isotypes", "n_hashtags", "reads",
           "frac_valid_barcode", "frac_feature", "frac_usable_in_cells", "saturation", "median_umi_cells",
           "median_umi_empty", "cell_empty_ratio", "frac_cells_with_adt", "isotype_share_median", "panel_detected",
           "panel_unexplained", "rna_concordance", "rna_concordance_null", "rna_pairs", "rna_canonical",
           "hto_singlet_share", "hto_doublet_share", "hto_negative_share", "barcode_overlap_head"]


def load_library(d):
    st = json.load(open(os.path.join(d, "status.json")))
    row = {k: st.get(k) for k in ("gsm", "gse", "lib_class", "status", "seconds", "count_seconds", "panel_source",
                                  "offset", "panel_reason", "error", "n_features", "host", "job")}
    ch = st.get("chemistry") or {}
    row["chemistry"] = ch.get("name") if isinstance(ch, dict) else ch
    row["translation"] = (st.get("translation") or {}).get("use")
    row["gex_gsm"] = (st.get("gex") or {}).get("gsm")
    row["download_bytes"] = sum(r.get("bytes", 0) for r in st.get("runs", []) or [])
    row["download_seconds"] = sum(r.get("seconds", 0) for r in st.get("runs", []) or [])
    row["download_source"] = ",".join(sorted({r.get("source", "") for r in st.get("runs", []) or []}))
    q = os.path.join(d, "qc.json")
    if os.path.exists(q):
        qc = json.load(open(q))
        for k in QC_COLS:
            v = qc.get(k)
            row[k] = json.dumps(v) if isinstance(v, (list, dict)) else v
        row["molecules"] = qc.get("stats", {}).get("molecules")
    row["npz_bytes"] = os.path.getsize(os.path.join(d, "counts.npz")) if os.path.exists(os.path.join(d, "counts.npz")) else 0
    return row


def merge_sample(libs, gex_barcodes_n):
    """libs: list of (dir, qc). -> (matrix csc, panel df, provenance)."""
    from scipy import sparse

    from singlet.adt import count as C

    rows, mats, prov = [], [], []
    for d, qc in libs:
        M, feats, ambient, _, meta = C.load_npz(os.path.join(d, "counts.npz"))
        pan = pd.read_csv(os.path.join(d, "panel.tsv"), sep="\t", dtype=str, keep_default_na=False)
        tot = np.asarray(M.sum(axis=1)).ravel()
        for i, p in pan.iterrows():
            rows.append({**p.to_dict(), "_lib": qc["gsm"], "_umis": int(tot[i]), "_ambient": int(ambient[i]),
                         "mi_": len(mats), "row_": i})
        mats.append(M.tocsr())
        prov.append({"gsm": qc["gsm"], "lib_class": qc["lib_class"], "offset": meta.get("offset"),
                     "chemistry": meta.get("chemistry"), "translation": meta.get("translation"),
                     "verdict": qc["verdict"]})
    df = pd.DataFrame(rows)
    df = df.sort_values("_umis", ascending=False).drop_duplicates("sequence", keep="first")
    dropped = len(rows) - len(df)
    df = df.sort_values(["mi_", "row_"]).reset_index(drop=True)
    if df["name"].duplicated().any():
        dup = df["name"].duplicated(keep=False)
        df.loc[dup, "name"] = df.loc[dup, "name"] + " (" + df.loc[dup, "_lib"] + ")"
    M = sparse.vstack([mats[r.mi_][int(r.row_)] for r in df.itertuples()]).tocsc()
    return M, df, {"libraries": prov, "duplicate_features_dropped": dropped}


def hto_sidecar(M, pan, hrows, gx, out):
    import pyarrow.parquet as pq

    from singlet.adt import hto as H
    from singlet.adt import normalize as N

    totals = np.asarray(M.sum(axis=0)).ravel()
    called = np.asarray(gx.called, dtype=bool)
    cells = np.flatnonzero(called)
    emp = np.flatnonzero(N.empty_droplets(totals, called))
    X = M[hrows].tocsc()
    amb = np.asarray(X[:, emp].sum(axis=1)).ravel()
    call, lt, _ = H.hashed_drops(X[:, cells].toarray(), amb)
    t, names = H.feature_sidecar(gx.gsm, cells, call, lt, list(pan.name.iloc[hrows]))
    import io

    bio = io.BytesIO()
    pq.write_table(t, bio)
    data = bio.getvalue()
    d16 = hashlib.sha256(data).digest()[:16].hex()
    key = f"v2/cf/{gx.gsm}/hto.{d16}.parquet"
    path = os.path.join(out, "objects", key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return {"key": key, "path": path, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
            "kind": "feature_sidecar", "singlets": int((call >= 0).sum()), "doublets": int((call == -2).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-objects", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dirs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.data, "GSE*", "GSM*", "status.json")))
    lib = pd.DataFrame([load_library(d) for d in dirs])
    lib.to_csv(os.path.join(a.out, "libraries.tsv"), sep="\t", index=False)
    inv = pd.read_csv(a.inventory, sep="\t", dtype={"gex_gsm": str})
    summary = {"libraries": int(len(lib)), "status": lib.status.value_counts().to_dict() if len(lib) else {}}
    if "verdict" in lib:
        summary["verdict"] = lib.verdict.fillna("none").value_counts().to_dict()
    exc = lib[(lib.status != "ok") | (lib.get("verdict", pd.Series(index=lib.index)) == "fail")]
    exc[["gse", "gsm", "lib_class", "status", "verdict", "failed", "panel_reason", "error"]
        if "verdict" in lib else ["gse", "gsm", "lib_class", "status", "panel_reason", "error"]].to_csv(
        os.path.join(a.out, "excluded.tsv"), sep="\t", index=False)
    plan, samples = [], []
    if a.build_objects and "verdict" in lib:
        from singlet.adt import gex as G
        from singlet.adt import store as ST

        good = lib[(lib.status == "ok") & lib.verdict.isin(["pass", "warn"])]
        for (gse, gex_gsm), grp in good.groupby(["gse", "gex_gsm"]):
            libs = [(os.path.join(a.data, gse, r.gsm), r._asdict()) for r in grp.itertuples(index=False)]
            gx = G.read_legacy(gse, gex_gsm, with_counts=False)
            M, pan, prov = merge_sample(libs, len(gx.barcodes))
            if M.shape[1] != len(gx.barcodes):
                raise RuntimeError(f"{gex_gsm}: matrix has {M.shape[1]} columns, cells table {len(gx.barcodes)}")
            lib_summary = {"verdicts": dict(zip(grp.gsm, grp.verdict)),
                           "failed": [], "warned": sorted({w for x in grp.warned for w in json.loads(x or "[]")})}
            mem = ST.build_members(M, pan, gx.barcodes, {**lib_summary, "verdict": "warn" if lib_summary["warned"] else "pass"},
                                   pan["_ambient"].astype(int).values, prov)
            buf = ST.build_sidecar(gex_gsm, mem)
            d16 = hashlib.sha256(buf).digest()[:16].hex()
            key = f"v2/x/{gex_gsm}/adt.{d16}.s2"
            path = os.path.join(a.out, "objects", key)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                f.write(buf)
            from singlet.v2.container import S2Reader

            r = S2Reader(buf)
            frag = {"key": key, "bytes": len(buf), "sha256": hashlib.sha256(buf).hexdigest(),
                    "members": {n: {"kind": m.kind, "off": m.off, "len": m.len, "digest": m.digest.hex()}
                                for n, m in r.members.items()}}
            with open(path + ".manifest.json", "w") as f:
                json.dump({"gse": gse, "gsm": gex_gsm, "modalities": {"adt": frag}}, f, indent=1)
            plan.append({"gse": gse, "gsm": gex_gsm, "key": key, "path": path, "bytes": len(buf), "sha256": frag["sha256"]})
            # per-cell hashtag calls on the merged matrix -> feature sidecar v2/cf/<GSM>/hto.<d16>.parquet
            hrows = np.flatnonzero(pan.kind.values == "hashtag")
            called_cols = np.flatnonzero(np.asarray(gx.called, dtype=bool))
            hshare = float(M[hrows][:, called_cols].sum() / max(M[:, called_cols].sum(), 1)) if len(hrows) else 0.0
            calls_ok = not any(json.load(open(os.path.join(a.data, gse, g, "qc.json"))).get("hto_failed")
                               for g in grp.gsm)
            if len(hrows) >= 2 and hshare >= HTO_MIN_SHARE and calls_ok:  # a hashtag pool whose calls passed QC
                hp = hto_sidecar(M, pan, hrows, gx, a.out)
                if hp:
                    plan.append({"gse": gse, "gsm": gex_gsm, **hp})
            samples.append({"gse": gse, "gex_gsm": gex_gsm, "libraries": ",".join(grp.gsm), "n_columns": M.shape[1],
                            "n_features": M.shape[0], "nnz": int(M.nnz), "object_bytes": len(buf),
                            "adt_rows": int((pan.kind != "hashtag").sum()), "hto_rows": int((pan.kind == "hashtag").sum())})
    pd.DataFrame(plan).to_csv(os.path.join(a.out, "upload_plan.tsv"), sep="\t", index=False)
    pd.DataFrame(samples).to_csv(os.path.join(a.out, "samples.tsv"), sep="\t", index=False)
    if samples:
        summary["objects"] = len(samples)
        summary["object_bytes"] = int(sum(s["object_bytes"] for s in samples))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    sys.exit(main())
