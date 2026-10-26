"""Write cohort.tsv plus the rows of cohort_additions.py to OUT.tsv (used to validate these results until the cohort
owner merges the rows): python bench/public/genotype/merged_cohort.py OUT.tsv"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
src = (HERE.parent / "cohort.tsv").read_text(encoding="utf-8").splitlines(keepends=True)
head = [l for l in src if l.startswith("#")]
rows = list(csv.DictReader((l for l in src if not l.startswith("#")), delimiter="\t"))
cols = list(rows[0].keys())
have = {r["id"] for r in rows}
R = []


def add(**k):
    k.setdefault("data_kind", "real")
    if k["id"] not in have:
        R.append({c: str(k.get(c, "")) for c in cols})


exec(compile((HERE / "cohort_additions.py").read_text(encoding="utf-8"), "cohort_additions.py", "exec"), {"add": add})
with open(sys.argv[1], "w", newline="", encoding="utf-8") as fh:
    fh.writelines(head)
    w = csv.DictWriter(fh, cols, delimiter="\t", lineterminator="\n")
    w.writeheader()
    w.writerows(rows + R)
print(len(rows), "+", len(R), "rows")
