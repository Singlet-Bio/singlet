"""Is a FASTASSIGN=on run identical to the off run?  python txf_equiv.py <perf/<dataset>_txf dir> [off_r1 on_r1]

Checks: (1) STAR BAM record streams identical (every record's SAM text, in file order); (2) STAR Log.final.out
identical apart from the timing lines; (3) every matrix member identical (perf_equiv.compare); (4) STARsolo raw
outputs byte-identical. Prints one JSON summary.
"""

import filecmp
import hashlib
import json
import sys
from pathlib import Path

import pysam

sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/atlas-rerun")
import perf_equiv as pe  # noqa: E402

base = Path(sys.argv[1])
a, b = (base / sys.argv[2], base / sys.argv[3]) if len(sys.argv) > 3 else (base / "off_r1", base / "on_r1")
res = {"a": str(a), "b": str(b)}


def bam_digest(p: Path):
    h, n = hashlib.sha256(), 0
    with pysam.AlignmentFile(str(p), check_sq=False) as f:
        for r in f.fetch(until_eof=True):
            h.update(r.to_string().encode()); h.update(b"\n"); n += 1
    return n, h.hexdigest()


bams = [x / "star_Aligned.sortedByCoord.out.bam" for x in (a, b)]
if all(p.exists() for p in bams):
    (na, ha), (nb, hb) = bam_digest(bams[0]), bam_digest(bams[1])
    res["bam_records"] = [na, nb]
    res["bam_identical"] = ha == hb
else:
    res["bam_identical"] = "missing: " + " ".join(str(p) for p in bams if not p.exists())

TIMING = ("Started job on", "Started mapping on", "Finished on", "Mapping speed")
logs = []
for x in (a, b):
    logs.append([l for l in (x / "star_Log.final.out").read_text().splitlines() if not l.strip().startswith(TIMING)])
res["log_final_identical"] = logs[0] == logs[1]
if not res["log_final_identical"]:
    res["log_final_diff"] = [(p, q) for p, q in zip(*logs) if p != q][:10]

res["matrices"] = {}
for m in pe.MEMBERS:
    x, y = pe.load(a, m), pe.load(b, m)
    res["matrices"][m] = "missing" if x is None or y is None else pe.compare(x, y)["identical"]

solo_diff = []
for f in sorted((a / "star_Solo.out").rglob("*")):
    if f.is_file():
        g = b / f.relative_to(a)
        if not g.exists() or not filecmp.cmp(f, g, shallow=False):
            solo_diff.append(str(f.relative_to(a)))
res["solo_files_differing"] = solo_diff
res["all_identical"] = (res["bam_identical"] is True and res["log_final_identical"]
                        and all(v is True for v in res["matrices"].values() if v != "missing") and not solo_diff)
print(json.dumps(res, indent=1))
