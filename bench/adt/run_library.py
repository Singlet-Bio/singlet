# SPDX-License-Identifier: MIT
"""Process one ADT/HTO library end to end on Clipper: reads -> counts aligned to the GEX sample -> QC.

    python -P bench/adt/run_library.py --inventory adt_hto_inventory.tsv --gsm GSM... --out $WORK/data/adt \
        --scratch /tmp/$USER-$SLURM_JOB_ID [--threads 8] [--panels bench/adt/panels]

Writes ``<out>/<GSE>/<GSM>/``:

* ``status.json``: always; ``status`` is ``ok`` or why the library stopped (``no_gex_sample``, ``unsupported_chemistry``,
  ``no_reads``, ``panel_unresolved``, ``error``), timings and bytes.
* ``counts.npz``: features x GEX columns (``singlet.adt.count.save_npz``), ambient profile, outside-barcode totals.
* ``panel.tsv``: the panel (rows of the matrix), its source and catalog facts.
* ``qc.json``, ``concordance.tsv``, ``cells.parquet`` (per called cell), ``unmatched.tsv``.

Nothing is uploaded anywhere; FASTQs live in the job's scratch only.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

HEAD_READS = 500_000


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw)


def ena_runs(run_acc):
    import urllib.request

    url = ("https://www.ebi.ac.uk/ena/portal/api/filereport?accession=%s&result=read_run&format=json&fields="
           "run_accession,fastq_ftp,fastq_bytes,submitted_ftp,submitted_format,read_count" % run_acc)
    for i in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read().decode() or "[]")
        except Exception:
            time.sleep(2 ** i)
    return []


def first_len(path):
    from singlet.adt.count import _open

    with _open(path) as f:
        f.readline()
        return len(f.readline().rstrip())


def download_run(run, scratch, threads, log):
    """FASTQ files of one run in scratch: ENA fastq.gz over HTTPS when it has >= 2 files, else fasterq-dump."""
    d = os.path.join(scratch, run)
    os.makedirs(d, exist_ok=True)
    meta = ena_runs(run)
    urls = []
    if meta and meta[0].get("fastq_ftp"):
        urls = [u for u in meta[0]["fastq_ftp"].split(";") if u]
    files = []
    if len(urls) >= 2:
        for u in urls:
            dest = os.path.join(d, os.path.basename(u))
            sh(["curl", "-sSfL", "--retry", "8", "--retry-delay", "20", "--connect-timeout", "60", "-o", dest,
                "https://" + u])
            files.append(dest)
        log["source"] = "ena"
    else:
        sh(["prefetch", "--max-size", "u", "-O", d, run])
        sh(["fasterq-dump", "--split-files", "--include-technical", "-e", str(threads), "-t", d, "-O", d,
            os.path.join(d, run)])
        files = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".fastq"))
        log["source"] = "sra"
    log["bytes"] = int(sum(os.path.getsize(f) for f in files))
    return files


def assign_reads(files, r1_len):
    """(R1, R2) among a run's files by read length: R1 is the shortest file with length >= CB+UMI, R2 the longest."""
    lens = {f: first_len(f) for f in files}
    cand1 = [f for f in files if lens[f] >= r1_len]
    if len(cand1) < 2:
        return None, None, lens
    r1 = min(cand1, key=lambda f: (lens[f], f))
    r2 = max((f for f in cand1 if f != r1), key=lambda f: (lens[f], f))
    return r1, r2, lens


def process(row, a, out_dir, status):
    from singlet.adt import chemistry as CH
    from singlet.adt import count as C
    from singlet.adt import gex as G
    from singlet.adt import panel as P
    from singlet.adt import qc as Q
    from singlet.adt import seq as S

    t0 = time.time()
    gex_gsm = row["gex_gsm"] if isinstance(row["gex_gsm"], str) else ""
    if not gex_gsm or not row["gex_in_atlas"]:
        status["status"] = "no_gex_sample"
        return
    chem = CH.for_protocol(row["gex_protocol"])
    status["atlas_protocol"] = row["gex_protocol"]
    if not chem.supported:
        status["status"] = "unsupported_chemistry"
        return
    gx = G.read_legacy(row["gse"], gex_gsm)
    status["gex"] = {"gsm": gex_gsm, "n_columns": len(gx.barcodes), "n_called": int(gx.called.sum()),
                     "seconds": round(time.time() - t0, 1)}

    # reads
    r1s, r2s, runlog = [], [], []
    for run in [r for r in str(row["runs"]).split(";") if r and r != "nan"]:
        lg = {"run": run}
        t1 = time.time()
        files = download_run(run, a.scratch, a.threads, lg)
        r1, r2, lens = assign_reads(files, 26)
        lg.update({"lens": {os.path.basename(k): v for k, v in lens.items()}, "seconds": round(time.time() - t1, 1)})
        runlog.append(lg)
        if r1 is None:
            continue
        r1s.append(r1)
        r2s.append(r2)
    status["runs"] = runlog
    if not r1s:
        status["status"] = "no_reads"
        return

    # chemistry from the data: UMI length from the R1 length, whitelist from the GEX barcodes
    r1_len = first_len(r1s[0])
    chem = resolve_chemistry(gx, r1_len, chem, a.cache, status)
    # library geometry from the first reads
    head = []
    for i, p in enumerate(C.iter_pairs(r1s, r2s)):
        head.append(p)
        if i + 1 >= HEAD_READS:
            break
    cat = load_catalogs(a.geo_panels)
    refs = [os.path.join(a.panels, f"{x}.csv") for x in (row["gsm"], row["gse"])]
    study_ref = next((r for r in refs if os.path.exists(r)), None)
    if study_ref is None and a.geo_panels:
        study_ref = harvested_reference(a.geo_panels, row)
    ref = P.annotate_from_catalog(P.read_feature_reference(study_ref), cat) if study_ref else None
    status["panel_file"] = os.path.basename(study_ref) if study_ref else None
    seqs_for_offset = list(ref.sequence) if ref is not None else list(cat.sequence)
    bc_len = max(len(x) for x in seqs_for_offset)
    offs = P.detect_offset([b for _, b in head], seqs_for_offset)
    off = max(offs, key=offs.get)
    status["offsets"] = {k: round(v, 4) for k, v in offs.items() if v >= 0.005}
    status["offset"] = off
    wl_path = C_fetch(chem.whitelist, a.cache)
    gcodes = np.sort(S.encode_strings([b.split("-")[0] for b in gx.barcodes], chem.cb_len)[0])
    sel = [(x, y) for x, y in head if b"N" not in x[:chem.cb_len] and len(x) >= chem.cb_len]
    top_cb = S.encode_strings([x[:chem.cb_len].decode() for x, _ in sel], chem.cb_len)[0]
    table = None
    tr = {"direct": float((S.lookup(gcodes, top_cb) >= 0).mean()), "use": False}
    if chem.translation:
        table, tr = C.best_translation(top_cb, gcodes, C_fetch(chem.translation, a.cache))
    status["translation"] = tr
    status["barcode_overlap_head"] = max(tr.get("direct", 0), tr.get("translated", 0))
    if offs[off] < P.MIN_OFFSET_FRACTION and ref is None:
        status["status"] = "panel_unresolved"
        status["panel_reason"] = "no catalog barcode at any R2 offset"
        _save_unmatched_head(sel, off, bc_len, out_dir)
        return
    # panel
    tcb = C.translate(top_cb, table)[0] if table is not None else top_cb
    called_codes = np.sort(S.encode_strings([gx.barcodes[i].split("-")[0] for i in np.flatnonzero(gx.called)],
                                            chem.cb_len)[0])
    incell = S.lookup(called_codes, tcb) >= 0
    feats = [y[off:off + bc_len] for (x, y), ok in zip(sel, incell) if ok]
    if ref is not None:
        # the study's reference names the features; discovery against it measures completeness, and against the
        # catalog says what the unexplained sequences are
        as_cat = pd.DataFrame({"sequence": ref.sequence, "specificity": ref.name, "kind": ref.kind,
                               "barcode_id": ref.barcode_id, "ensembl": ref.ensembl, "clone": "", "reactivity": ""})
        disc = P.discover(feats, as_cat)
        cat_codes = set(cat.sequence)
        disc.notes["unexplained_in_catalog"] = [(q, n, q in cat_codes) for q, n in disc.notes["unexplained_top"]]
        panel = ref.assign(source="study")
        panel["reads_discovery"] = panel.sequence.map(dict(zip(disc.features.sequence,
                                                               disc.features.reads_discovery))).fillna(0).astype(int)
        status["panel_source"] = "study"
    else:
        disc = P.discover(feats, cat)
        if disc is None or disc.status == "unresolved":
            status["status"] = "panel_unresolved"
            status["panel_reason"] = "no catalog barcode with support in called cells"
            status["discovery"] = disc.notes if disc else None
            _save_unmatched_head(sel, off, bc_len, out_dir)
            return
        panel = disc.features
        status["panel_source"] = "catalog"
    status["discovery"] = {"status": disc.status, **disc.notes} if disc else None
    status["n_features"] = int(len(panel))

    # count
    t2 = time.time()
    wl = table[0] if table is not None else np.sort(S.encode_strings(C.load_barcode_list(wl_path), chem.cb_len)[0])
    lc = C.count_library(r1s, r2s, cb_len=chem.cb_len, umi_len=chem.umi_len, offset=off,
                         panel_sequences=list(panel.sequence), whitelist_codes=wl, gex_barcodes=gx.barcodes,
                         translation=table)
    status["count_seconds"] = round(time.time() - t2, 1)
    os.makedirs(out_dir, exist_ok=True)
    C.save_npz(os.path.join(out_dir, "counts.npz"), lc, {"gsm": row["gsm"], "gex_gsm": gex_gsm, "offset": off,
                                                        "chemistry": chem.name, "translation": table is not None})
    panel.to_csv(os.path.join(out_dir, "panel.tsv"), sep="\t", index=False)
    pd.DataFrame(lc.unmatched, columns=["sequence", "reads"]).to_csv(os.path.join(out_dir, "unmatched.tsv"), sep="\t",
                                                                     index=False)
    # QC
    t3 = time.time()
    res = Q.library_qc(lc.matrix, panel, gx.called, lc.stats, lc.ambient, gx.exon, gx.genes,
                       disc.notes if disc else None)
    lib = res["library"]
    lib["barcode_overlap_head"] = status["barcode_overlap_head"]
    with open(os.path.join(out_dir, "qc.json"), "w") as f:
        json.dump({"gsm": row["gsm"], "gse": row["gse"], "gex_gsm": gex_gsm, "lib_class": row["lib_class"],
                   "stats": lc.stats, **lib}, f, indent=1, default=float)
    if res["concordance"] is not None:
        res["concordance"].to_csv(os.path.join(out_dir, "concordance.tsv"), sep="\t", index=False)
    Q.cell_table(lc.matrix, panel, res["cells"], res["hto"]).to_parquet(os.path.join(out_dir, "cells.parquet"))
    status["qc_seconds"] = round(time.time() - t3, 1)
    status["verdict"] = lib["verdict"]
    status["failed"] = lib["failed"]
    status["status"] = "ok"


def load_catalogs(geo_dir):
    """Vendor catalog, plus the sequences named in harvested GEO references that the vendor list lacks
    (``source`` = ``geo``)."""
    from singlet.adt import panel as P

    cat = P.load_catalog()
    p = os.path.join(geo_dir, "geo_catalog.tsv") if geo_dir else None
    if not p or not os.path.exists(p):
        return cat
    g = pd.read_csv(p, sep="	", dtype=str, keep_default_na=False)
    g = g[g.sequence.str.len() == 15]
    g = g[~g.sequence.isin(set(cat.sequence))]
    if not len(g):
        return cat
    extra = pd.DataFrame({"sequence": g.sequence, "barcode_id": "", "formats": "", "specificity": g.name,
                          "clone": "", "reactivity": "", "ensembl": "", "kind": [P.kind_of(n) for n in g.name],
                          "aliases": g.aliases, "sources": "geo:" + g.studies})
    return pd.concat([cat, extra], ignore_index=True)


def harvested_reference(geo_dir, row):
    """The library's own reference among the harvested GEO files: a file of this GSM, else the largest file of the GSE."""
    import glob

    d = os.path.join(geo_dir, "files")
    for pat in (f"{row['gsm']}_*.csv", f"{row['gse']}_*.csv"):
        hits = sorted(glob.glob(os.path.join(d, pat)), key=lambda f: -os.path.getsize(f))
        if hits:
            return hits[0]
    return None


def resolve_chemistry(gx, r1_len, atlas_chem, cache, status):
    """10x barcode family from the GEX barcodes (share found in each whitelist) and UMI length from the ADT R1 length
    (26 = 16+10, 28 = 16+12; longer R1 falls back to the atlas protocol)."""
    from singlet.adt import chemistry as CH
    from singlet.adt import count as C
    from singlet.adt import seq as S

    gb = [b.split("-")[0] for b in gx.barcodes]
    rng = np.random.default_rng(0)
    probe = [gb[i] for i in rng.choice(len(gb), min(len(gb), 5000), replace=False)]
    pc, bad = S.encode_strings(probe, 16)
    share = {}
    for name in ("737K-august-2016.txt.gz", "3M-february-2018.txt.gz"):
        wl = np.sort(S.encode_strings(C.load_barcode_list(C_fetch(name, cache)), 16)[0])
        share[name] = float((S.lookup(wl, pc[bad == 0]) >= 0).mean())
    best = max(share, key=share.get)
    if r1_len in (26, 27):
        umi = 10
    elif 28 <= r1_len <= 31:
        umi = 12
    else:
        umi = atlas_chem.umi_len or (12 if best.startswith("3M") else 10)
    base = CH.CHEMISTRIES["10xv3" if best.startswith("3M") else "10xv2"]
    chem = CH.Chemistry(base.name if umi == base.umi_len else f"{base.name}-umi{umi}", 16, umi, best,
                        base.translation)
    status["chemistry"] = {"name": chem.name, "r1_len": r1_len, "umi_len": umi, "whitelist": best,
                           "gex_barcodes_in_whitelist": share, "atlas": atlas_chem.name}
    return chem


def _save_unmatched_head(sel, off, bc_len, out_dir):
    from collections import Counter

    os.makedirs(out_dir, exist_ok=True)
    c = Counter(y[off:off + bc_len] for _, y in sel)
    pd.DataFrame([(k.decode(), v) for k, v in c.most_common(200)], columns=["sequence", "reads"]).to_csv(
        os.path.join(out_dir, "unmatched.tsv"), sep="\t", index=False)


def C_fetch(name, cache):
    from singlet.adt.chemistry import WL_BASE

    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, name)
    if not os.path.exists(dest):
        tmp = dest + f".{os.getpid()}.part"
        sh(["curl", "-sSfL", "--retry", "5", "-o", tmp, WL_BASE + name])
        os.replace(tmp, dest)
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inventory", required=True)
    ap.add_argument("--gsm", required=True, help="GSM id, or @FILE with one GSM per line and --index")
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--cache", required=True, help="shared dir for whitelists")
    ap.add_argument("--panels", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "panels"))
    ap.add_argument("--geo-panels", default=None, help="harvest_panels.py output dir (optional)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    gsm = a.gsm
    if gsm.startswith("@"):
        gsm = [x.strip() for x in open(gsm[1:]) if x.strip()][a.index]
    inv = pd.read_csv(a.inventory, sep="\t", dtype={"gex_gsm": str})
    row = inv[inv.gsm == gsm].iloc[0].to_dict()
    out_dir = os.path.join(a.out, row["gse"], gsm)
    st_path = os.path.join(out_dir, "status.json")
    if os.path.exists(st_path) and not a.force:
        print("done already:", open(st_path).read()[:300])
        return 0
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(a.scratch, exist_ok=True)
    status = {"gsm": gsm, "gse": row["gse"], "lib_class": row["lib_class"], "host": os.uname().nodename,
              "job": os.environ.get("SLURM_JOB_ID"), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t0 = time.time()
    try:
        process(row, a, out_dir, status)
    except Exception as e:  # recorded, so the array keeps going and the failure is reported
        status["status"] = "error"
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()[-4000:]
    finally:
        status["seconds"] = round(time.time() - t0, 1)
        shutil.rmtree(a.scratch, ignore_errors=True)
        with open(st_path, "w") as f:
            json.dump(status, f, indent=1, default=str)
    print(json.dumps({k: v for k, v in status.items() if k != "traceback"}, default=str)[:2000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
