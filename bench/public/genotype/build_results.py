"""Build bench/public/results/{donor-demux,donor-snps,mtdna}.json from pulled artifacts and validate them.

    python bench/public/genotype/build_results.py ARTIFACT_ROOT CODE_COMMIT [--cohort COHORT.tsv]
ARTIFACT_ROOT holds donor-demux/, donor-snps/, mtdna/ (copies of $WORK/bench-public/artifacts/<capability>/).
CODE_COMMIT is the pushed commit that contains this harness (pinned in code[].url).
The default cohort is bench/public/cohort.tsv plus genotype/cohort_additions.py (rows the cohort owner has not merged yet).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUB = HERE.parent
sys.path.insert(0, str(PUB)); sys.path.insert(0, str(PUB / "capabilities"))
import contract  # noqa: E402

root, commit = Path(sys.argv[1]), sys.argv[2]
cohort_path = sys.argv[sys.argv.index("--cohort") + 1] if "--cohort" in sys.argv else None
if cohort_path is None:
    cohort_path = str(Path(tempfile.gettempdir()) / "bench_public_cohort_merged.tsv")
    subprocess.run([sys.executable, str(HERE / "merged_cohort.py"), cohort_path], check=True)
subprocess.run([sys.executable, str(HERE / "make_provenance.py"), str(root), commit], check=True)
cohort = contract.load_cohort(cohort_path)
import donor_demux, donor_snps, mtdna  # noqa: E402

(PUB / "results").mkdir(exist_ok=True)
bad = 0
for cap, mod in (("donor-demux", donor_demux), ("donor-snps", donor_snps), ("mtdna", mtdna)):
    doc = mod.compute(root / cap)
    errs = contract.validate(doc, cohort)
    st = contract.capability_status(doc)
    (PUB / "results" / f"{cap}.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(cap, "OK" if not errs else "INVALID", st)
    for e in errs:
        print("  -", e)
    bad += bool(errs)
sys.exit(1 if bad else 0)
