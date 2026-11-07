# SPDX-License-Identifier: MIT
"""Print primary.py outputs compactly:  python -P summarize.py a.json [b.json ...]"""
import json
import sys

for p in sys.argv[1:]:
    r = json.load(open(p))
    print("==", p)
    for g, v in r["primary"].items():
        s, t = v["singlet"], v["trust4"]
        d = v["diff_singlet_minus_trust4"]
        print(f"{g:4s} n={s['n_cells']:5d} singlet {s['accuracy']:.4f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}]  "
              f"trust4 {t['accuracy']:.4f} [{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}]  diff {d['diff']:+.4f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]")
    for l, v in r["per_library"].items():
        s, t, d = v["singlet"], v["trust4"], v["diff_singlet_minus_trust4"]
        print(f"  {l:18s} n={s['n_cells']:5d} singlet {s['accuracy']:.3f} [{s['ci95'][0]:.3f},{s['ci95'][1]:.3f}] "
              f"trust4 {t['accuracy']:.3f} [{t['ci95'][0]:.3f},{t['ci95'][1]:.3f}] diff {d['diff']:+.3f}")
    w = r["wrong_lineage_pooled"]
    print(f"wrong lineage: singlet {w['singlet']['rate']:.4f} ({w['singlet']['wrong']}/{w['singlet']['known']})  "
          f"cellranger {w['cellranger']['rate']:.4f} ({w['cellranger']['wrong']}/{w['cellranger']['known']})  "
          f"diff {w['singlet_minus_cellranger']:+.4f}")
    print("pass:", r["pass"])
