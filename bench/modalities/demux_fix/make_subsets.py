"""Barcode lists of truth singlets from 1, 2, 4 and 6 of the Kang donors, to test auto-K on real data
with known K. Usage: make_subsets.py TRUTH.tsv.gz OUTDIR"""
import os
import sys
from common import load_truth

T = load_truth(sys.argv[1])
out = sys.argv[2]
os.makedirs(out, exist_ok=True)
sing = T[T["multiplets"] == "singlet"]
sets = {
    "d1": ["1244"],
    "d2": ["1244", "1256"],
    "d3": ["1244", "1256", "107"],
    "d4": ["1244", "1256", "1488", "1016"],
    "d5": ["1244", "1256", "1488", "1016", "1015"],
    "d6": ["1244", "1256", "1488", "1016", "1015", "101"],
    "d7": ["1244", "1256", "1488", "1016", "1015", "101", "107"],
}
for name, donors in sets.items():
    bc = sing.index[sing["ind"].isin(donors)]
    with open(os.path.join(out, name + ".txt"), "w") as f:
        f.write("\n".join(bc) + "\n")
    print(name, donors, len(bc))
