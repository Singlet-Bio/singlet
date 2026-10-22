#!/usr/bin/env python
"""w4_corpus: re-run the E4 derivability checks (read-only reuse of tools/e4_audit.py functions) on the w4 corpus samples.

    python -P tools/w4_corpus_e4.py [id ...]      # default: every sample directory, appends to results/w4_corpus_e4.tsv

One row per (sample, claim, item): n_compared, n_equal. Claims: colsums / ptr / perm blocks of every matrix member present,
psi (splice_psi.1pz vs rint(PSI) recomputed from sj_counts.1pz, python re-implementation only: no production-code comparison),
cell_calls (whole file regenerated from the called barcode set + exon column sums), summary (numeric fields of summary.json,
pileup_stats.json, provenance.json that are functions of the matrices). Never executes downloaded content.
"""
import os, sys, json, time, csv, traceback
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import e4_lib as L          # read-only
import e4_audit as A        # read-only (module-level code only defines functions)

SCR = r"C:\Users\zacha\AppData\Local\Temp\claude\C--Users-zacha-Projects-singlet-bio\8e8b9acd-9e9b-409a-a4ac-d2afe710d803\scratchpad\w4e4"
L.SCRATCH = SCR
CORP = os.path.join(BC, "corpus_w4_corpus")
OUT = os.environ.get("W4_E4_OUT", os.path.join(BC, "results", "w4_corpus_e4.tsv"))
KINDS = ["exon_counts", "intron_counts", "sj_counts", "splice_psi"]


def load_local(sid, pz):
    d = os.path.join(CORP, sid)
    S = dict(gsm=sid, raw={}, mat={}, pzinfo={})
    for fn in os.listdir(d):
        p = os.path.join(d, fn)
        if not os.path.isfile(p) or fn.endswith(".json") and fn in ("manifest.json", "study_meta.json", "sample.json"):
            continue
        S["raw"][fn] = open(p, "rb").read()
    for k in KINDS:
        fn = k + ".1pz"
        if fn in S["raw"]:
            h, o = L.parse_pz(S["raw"][fn])
            S["pzinfo"][k] = (h, o)
            S["mat"][k] = pz.read_1pz(os.path.join(d, fn)) if h["m"] > 0 and h["n"] > 0 else None
    return S


def main():
    os.makedirs(SCR, exist_ok=True)
    pz = L.load_pz()
    helper = L.build_helper()
    ids = sys.argv[1:] or sorted(d for d in os.listdir(CORP) if not d.startswith("_") and os.path.isdir(os.path.join(CORP, d)))
    new = not os.path.exists(OUT)
    f = open(OUT, "a", newline="")
    w = csv.writer(f, delimiter="\t", lineterminator="\n")
    if new:
        w.writerow(["id", "claim", "item", "n_compared", "n_equal", "note"])
    for sid in ids:
        t = time.time()
        try:
            S = load_local(sid, pz)
        except Exception as e:
            w.writerow([sid, "load", "all", 1, 0, repr(e)[:200]]); f.flush(); continue
        try:
            out = A.test_container_blocks(S, helper)
            for (kind, what), r in out.items():
                if what == "rowcounts_bytes":
                    w.writerow([sid, "rowcounts", kind, 1, 1, f"bytes={r}"]); continue
                w.writerow([sid, what, kind, 1, int(bool(r["raw_equal"])), ""])
                if what == "colsums":
                    w.writerow([sid, "colsums_frame_bytes_zstd155", kind, 1, int(bool(r["bytes_cpp155"])), ""])
        except Exception as e:
            w.writerow([sid, "blocks", "all", 1, 0, repr(e)[:200]])
        try:
            sj, psi = S["mat"].get("sj_counts"), S["mat"].get("splice_psi")
            if sj is not None and psi is not None:
                pr = A.test_psi(S, helper, full_ref=False)
                w.writerow([sid, "psi_structure", "splice_psi", 1, int(pr["struct_equal"]), f"events={pr['events']} nnz={pr['psi_nnz_stored']}"])
                w.writerow([sid, "psi_values_rint", "splice_psi", 1, int(pr["rint_values_equal"]), f"n_rint_diff={pr['n_rint_diff']}"])
                w.writerow([sid, "psi_values_trunc", "splice_psi", 1, int(pr["trunc_values_equal"]), f"n_trunc_diff={pr['n_trunc_diff']}"])
            elif sj is not None and psi is None:
                ptr, idx, pf, n_ev = A.psi_recompute(sj)
                w.writerow([sid, "psi_stub", "splice_psi", 1, int(len(pf) == 0), f"psi member empty/absent but sj implies {len(pf)} entries, events={n_ev}"])
            else:
                w.writerow([sid, "psi_skipped", "splice_psi", 0, 0, "no sj_counts matrix"])
        except Exception as e:
            w.writerow([sid, "psi", "all", 1, 0, repr(e)[:200]])
        try:
            ex, it = S["mat"].get("exon_counts"), S["mat"].get("intron_counts")
            if ex is not None and "cell_calls.tsv" in S["raw"]:
                r6, tot, called = A.test_cell_calls(sid, ex, S["raw"]["cell_calls.tsv"])
                w.writerow([sid, "cell_calls_file_regen", "cell_calls.tsv", 1, int(bool(r6.get("file_bytes_equal"))), f"rows={r6['n_rows']} all_TRUE={r6['all_is_cell_TRUE']} methods={r6['methods']} pvals={r6['pvalue_constant']}"])
                w.writerow([sid, "cell_calls_log10", "cell_calls.tsv", r6["n_rows"], r6["log10_text_equal"], ""])
                w.writerow([sid, "cell_calls_info", "called_set", 1, 1, json.dumps(r6["info"])])
                if it is not None and S["mat"].get("sj_counts") is not None and "summary.json" in S["raw"]:
                    res, psr, pvr, sm, ps, prov = A.test_summary(S, ex, it, tot, called, None)
                    for k, (stored, expd, ok) in res.items():
                        w.writerow([sid, "summary_field", k, 1, int(bool(ok)), f"stored={stored} expected={expd}"])
                    for k, (a, b) in psr.items():
                        w.writerow([sid, "pileup_stats_field", k, 1, int(a == b), f"stored={a} matrix={b}"])
                    for k, (a, b) in pvr.items():
                        w.writerow([sid, "provenance_field", k, 1, int(a == b), f"stored={a} matrix={b}"])
        except Exception as e:
            w.writerow([sid, "cell_calls_summary", "all", 1, 0, (repr(e) + traceback.format_exc()[-150:])[:300].replace("\n", " ")])
        f.flush()
        print(sid, f"{time.time() - t:.1f}s", flush=True)
    f.close()


if __name__ == "__main__":
    main()
