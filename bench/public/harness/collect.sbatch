#!/bin/bash
#SBATCH --job-name=bpub-collect
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Turn the harness runs into contract documents:
#  1. resources of every run (provenance.py) -> $BP/work/resources.json;
#  2. for each capability: if bench/public/capabilities/<cap>_prepare.py exists, prepare(runs_dir, artifact_dir, resources)
#     builds the small per-cell artifacts from the runs; then <cap>.py:compute(artifact_dir) writes $BP/results/<cap>.json;
#  3. documents delivered by owners to $BP/results-incoming/<cap>.json are validated and copied;
#  4. index.json. Pull $BP/results/ back to bench/public/results/ (JSON only; artifacts stay until the R2 upload).
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
export PYTHONPATH=$SRC/python
in_runtime python $SRC/bench/public/harness/provenance.py $BP/runs $BP/work/resources.json
mkdir -p $BP/results-incoming
in_runtime python - <<EOF
import importlib.util, json, shutil, subprocess, sys
from pathlib import Path
src = Path("$SRC"); bp = Path("$BP")
sys.path.insert(0, str(src / "bench/public")); sys.path.insert(0, str(src / "bench/public/capabilities"))
import contract as C
cohort = C.load_cohort(src / "bench/public/cohort.tsv")
for cap in C.CAPABILITIES:
    mod = cap.replace("-", "_")
    art = bp / "artifacts" / cap
    prep = src / f"bench/public/capabilities/{mod}_prepare.py"
    if prep.exists():
        spec = importlib.util.spec_from_file_location(f"{mod}_prepare", prep); m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m); art.mkdir(parents=True, exist_ok=True)
        m.prepare(bp / "runs", art, json.loads((bp / "work/resources.json").read_text()))
    comp = src / f"bench/public/capabilities/{mod}.py"
    inc = bp / "results-incoming" / f"{cap}.json"
    out = bp / "results" / f"{cap}.json"
    if comp.exists() and (art / "provenance.json").exists():
        r = subprocess.run([sys.executable, str(src / "bench/public/run_capability.py"), cap, str(art), "--out", str(out)])
        print(cap, "computed" if r.returncode == 0 else "computed with validation errors")
    elif inc.exists():
        errs = C.validate(json.loads(inc.read_text()), cohort)
        shutil.copy(inc, out); print(cap, "incoming", "valid" if not errs else errs[:5])
idx = C.build_index(bp / "results")
(bp / "results" / "index.json").write_text(json.dumps(idx, indent=2) + "\n")
print("overall", idx["overall"], idx["n_pass"], "/", idx["n_capabilities"])
EOF
