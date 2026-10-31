# SPDX-License-Identifier: MIT
"""Markdown table of one scored variant (scores/<set><tag>.json + runs/<lib>/metrics<tag>.json), optionally against a base tag
with the development criterion of PREREG.md.

    python -P table.py --runs RUNS --scores SCORES --set dev --tag _v1 --libs a:TCR,b:BCR [--base _base]
"""
from __future__ import annotations

import argparse
import json
import os


def load(a, tag):
    p = json.load(open(os.path.join(a.scores, f"{a.set}{tag}.json")))
    m = {}
    for it in a.libs.split(","):
        lib = it.split(":")[0]
        f = os.path.join(a.runs, lib, f"metrics{tag}.json")
        m[lib] = json.load(open(f)) if os.path.exists(f) else {}
    return p, m


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return float("nan")
        d = d[k]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--libs", required=True)
    ap.add_argument("--base", default=None)
    a = ap.parse_args()
    p, m = load(a, a.tag)
    b = load(a, a.base) if a.base and a.base != a.tag else None
    libs = [it.split(":")[0] for it in a.libs.split(",")]
    print(f"### {a.set}{a.tag}" + (f" (vs {a.base})" if b else ""))
    print("| library | cells | pairing Singlet [95% CI] | TRUST4 | diff | contig F1 S / T | ARI S / T | chain recovery S / T | V S / T | J S / T | wrong lineage S / CR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    fails = []
    for lib in libs:
        pl = p["per_library"][lib]
        s, t = m[lib].get("singlet", {}), m[lib].get("trust4", {})
        wl = p["libs"][lib].get("wrong_lineage_singlet")
        wc = p["libs"][lib].get("wrong_lineage_cellranger")
        row = (f"| {lib} | {pl['singlet']['n_cells']} | {pl['singlet']['accuracy']:.4f} [{pl['singlet']['ci95'][0]:.3f},{pl['singlet']['ci95'][1]:.3f}] | "
               f"{pl['trust4']['accuracy']:.4f} | {pl['diff_singlet_minus_trust4']['diff']:+.4f} | {g(s,'contig_f1'):.4f} / {g(t,'contig_f1'):.4f} | "
               f"{g(s,'clonotype_ari'):.4f} / {g(t,'clonotype_ari'):.4f} | {g(s,'cell_chain_recovery'):.4f} / {g(t,'cell_chain_recovery'):.4f} | "
               f"{g(s,'v_gene_agreement'):.3f} / {g(t,'v_gene_agreement'):.3f} | {g(s,'j_gene_agreement'):.3f} / {g(t,'j_gene_agreement'):.3f} | "
               + (f"{wl[0]}/{wl[1]} / {wc[0]}/{wc[1]} |" if wl else "- |"))
        if b:
            bp, bm = b
            d_pair = pl["singlet"]["accuracy"] - bp["per_library"][lib]["singlet"]["accuracy"]
            d_f1 = g(s, "contig_f1") - g(bm[lib].get("singlet", {}), "contig_f1")
            d_ari = g(s, "clonotype_ari") - g(bm[lib].get("singlet", {}), "clonotype_ari")
            d_wl = g(s, "wrong_lineage_method") - g(bm[lib].get("singlet", {}), "wrong_lineage_method")
            row += f" d pairing {d_pair:+.4f}, F1 {d_f1:+.4f}, ARI {d_ari:+.4f}" + (f", WL {d_wl:+.4f}" if d_wl == d_wl else "")
            for name, v, bad in (("pairing", d_pair, d_pair < -0.005), ("F1", d_f1, d_f1 < -0.005), ("ARI", d_ari, d_ari < -0.005),
                                 ("wrong lineage", d_wl, d_wl > 0.005)):
                if bad:
                    fails.append(f"{lib} {name} {v:+.4f}")
        print(row)
    print()
    print("| group | cells | Singlet [95% CI] | TRUST4 | diff [95% CI] |" + (" d vs base |" if b else ""))
    print("|---|---|---|---|---|" + ("---|" if b else ""))
    for grp, r in p["primary"].items():
        d = r["diff_singlet_minus_trust4"]
        line = (f"| {grp} | {r['singlet']['n_cells']} | {r['singlet']['accuracy']:.4f} [{r['singlet']['ci95'][0]:.3f},{r['singlet']['ci95'][1]:.3f}] | "
                f"{r['trust4']['accuracy']:.4f} | {d['diff']:+.4f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] |")
        if b:
            line += f" {r['singlet']['accuracy'] - b[0]['primary'][grp]['singlet']['accuracy']:+.4f} |"
        print(line)
    w = p["wrong_lineage_pooled"]
    print(f"\nwrong lineage pooled: Singlet {w['singlet']['wrong']}/{w['singlet']['known']} = {w['singlet']['rate']:.4f}, Cell Ranger "
          f"{w['cellranger']['wrong']}/{w['cellranger']['known']} = {w['cellranger']['rate']:.4f}, diff {w['singlet_minus_cellranger']:+.4f}")
    print(f"pass (held-out rules): {p['pass']}")
    if b:
        up = p["primary"]["all"]["singlet"]["accuracy"] - b[0]["primary"]["all"]["singlet"]["accuracy"]
        print(f"dev criterion: pooled pairing {up:+.4f} ({'improves' if up > 0 else 'does not improve'}); "
              f"per-library losses > 0.005: {fails or 'none'} -> {'KEEP' if up > 0 and not fails else 'REJECT'}")


if __name__ == "__main__":
    main()
