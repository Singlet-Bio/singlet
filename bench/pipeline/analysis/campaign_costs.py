"""Per-sample runtime and depth of the GRCh38 campaign, from every kept results/<GSM>/out.

    python -I campaign_costs.py <results_dir> <out.csv>
Reads pileup_stats.json (wall_time_s, total_reads, star_threads, mode) and provenance.json.
"""
import json
import sys
from pathlib import Path

res, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for d in res.iterdir():
    p = d / "out" / "pileup_stats.json"
    if not p.is_file():
        continue
    try:
        s = json.loads(p.read_text())
    except Exception:  # noqa: BLE001
        continue
    pv = {}
    q = d / "out" / "provenance.json"
    if q.is_file():
        try:
            pv = json.loads(q.read_text())
        except Exception:  # noqa: BLE001
            pass
    rows.append({
        "gsm": d.name, "total_reads": s.get("total_reads"), "wall_s": s.get("wall_time_s"),
        "pileup_s": s.get("pileup_time_s"), "threads": s.get("star_threads") or pv.get("threads"),
        "mode": s.get("mode"), "n_barcodes": s.get("n_barcodes"),
        "exon_hits": s.get("exon_hits"), "intron_hits": s.get("intron_hits"),
        "umi_unique": s.get("umi_unique"), "pipeline": pv.get("pipeline"),
        "directional": pv.get("umi_dedup_directional"), "version": s.get("version"),
    })
import csv  # noqa: E402

with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "samples")
