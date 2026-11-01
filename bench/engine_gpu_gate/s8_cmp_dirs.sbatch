#!/bin/bash
#SBATCH --job-name=s8-cmpdirs
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: where do two output directories of the same sample first differ? A=$A_DIR, B=$B_DIR. STAR-level files first (Log.final.out numbers, SJ.out.tab, Unmapped sorted),
# then the engine's own: cr4_probe.json, provenance clip5p, pileup_stats.json, per-file cmp.
set -uo pipefail
A=${A_DIR:?}; B=${B_DIR:?}
echo "A=$A"; echo "B=$B"
echo "== Log.final.out (lines that differ, excluding times)"; diff <(grep -v -i -E 'time|speed' $A/star_Log.final.out) <(grep -v -i -E 'time|speed' $B/star_Log.final.out) && echo same
echo "== SJ.out.tab"; cmp $A/star_SJ.out.tab $B/star_SJ.out.tab && echo same
echo "== star_Solo.out"; ls $A/star_Solo.out $B/star_Solo.out 2>&1 | head -12
echo "== cr4_probe.json"; diff <(python3 -c "import json,sys;d=json.load(open('$A/cr4_probe.json'));print(json.dumps(d,indent=1,sort_keys=True))") <(python3 -c "import json,sys;d=json.load(open('$B/cr4_probe.json'));print(json.dumps(d,indent=1,sort_keys=True))") && echo same
echo "== provenance clip5p / cr4"; for d in $A $B; do python3 - <<PY
import json
p=json.load(open("$d/provenance.json"))
def find(o,keys,path=""):
    if isinstance(o,dict):
        for k,v in o.items():
            if k in keys: print("$d".split("/")[-3], path+"/"+k, json.dumps(v)[:300])
            find(v,keys,path+"/"+k)
find(p,{"clip5p","cr4","threads","engine","n_threads","star_threads"})
PY
done
echo "== pileup_stats.json"; diff <(python3 -m json.tool --sort-keys $A/pileup_stats.json) <(python3 -m json.tool --sort-keys $B/pileup_stats.json)
echo "== files that differ (cmp)"; for f in $(cd $A && ls); do [ -f $A/$f ] && { cmp -s $A/$f $B/$f || echo "DIFF $f"; }; done
