#!/bin/bash
# Atlas cell index build after a .s1 transcode run (docs/format-s1-cell-index.md), as its own Clipper job:
#   1. mirror the studies' .s1 objects from R2 to node-local scratch (s1/m, s1/s, s1/a, s1/ref; read-only);
#   2. scripts/s1_cidx_build.py --base <mirror> (labels from the manifests' annotations_default sidecars);
#   3. the index tree goes to $WORK/state/s1/<RUN>/cidx/ (server layout). Nothing is written to R2 or D1: uploading the index is
#      a separate, owner-run step (objects first, current.json last).
#
#   sbatch --export=ALL,COMMIT=<sha>,BRANCH=<branch>,GSE_LIST=<file>,RUN=<name> scripts/s1_cidx_build.sbatch
# Needs the build of COMMIT from s1_migrate.sbatch (MODE=build) and ~/.config/singlet-r2.env.
#SBATCH --job-name=stcidx
#SBATCH --account=debruinz
# The whole atlas (7,461 studies, 52,893 samples, 237M cells) mirrors 646 GB and peaks near 200 GB of RAM: 192G was OOM-killed once
# (788505) and used 201 GB the next time (788516). bigmem nodes have 1.5 TB and 3.3 TB of /tmp; a run takes about 3.5 h.
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=10:00:00
#SBATCH --no-requeue
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=${WORK:-/mnt/projects/debruinz_project/debruinz/singlet}
COMMIT=${COMMIT:?set COMMIT}
C10=${COMMIT:0:10}
SRC=$WORK/src/singlet-py/v2mig__$C10
VENV=${VENV:-$WORK/venvs/v2tools}
SIF=${SIF:-$WORK/images/python-3.12-bookworm.sif}
GSE_LIST=${GSE_LIST:?set GSE_LIST}
RUN=${RUN:-cidx-$C10}
OUT=$WORK/state/s1/$RUN
ANNOT=${ANNOT_VERSION:-a2026-10}
[ -e "$SRC/.v2build.ok" ] || { echo "no build of $C10: run s1_migrate.sbatch MODE=build first"; exit 2; }
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
rt() { apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$@"; }
SCRATCH=/tmp/${USER:-$(id -un)}-${SLURM_JOB_ID:-x}
mkdir -p "$SCRATCH/mirror" "$OUT"
trap 'rm -rf "$SCRATCH"' EXIT
export XDG_CACHE_HOME=$WORK/cache SINGLET_S1_CACHE=$SCRATCH/s1-cache PYTHONPATH=$SRC/python
set -a; . "$HOME/.config/singlet-r2.env"; set +a
echo "cidx run=$RUN commit=$C10 host=$(hostname) studies=$(grep -c . "$GSE_LIST")"
T0=$(date +%s)
rt "$VENV/bin/python" - "$GSE_LIST" "$SCRATCH/mirror" "$SRC" <<'PY'
import concurrent.futures as cf, json, os, re, sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[3] + "/scripts")
import s1_campaign as camp
lst, root = Path(sys.argv[1]), Path(sys.argv[2])
gses = [x.split("#")[0].strip() for x in lst.read_text().splitlines() if x.split("#")[0].strip()]
st = camp.S3Store()
keys, have = [], []
for g in gses:
    k = camp.manifest_key(g)
    if st.head("singlet-data", k) is None:
        print(f"{g}: no .s1 manifest, skipped", flush=True)
        continue
    have.append(g)
    m = json.loads(st.get_bytes("singlet-data", k))
    keys.append(k)
    for s in m["samples"]:
        keys.append(s["key"])
        for ent in (s.get("annotations") or {}).values():
            keys.append(ent["key"])
    for v in (m.get("packs") or {}).values():
        for x in (v if isinstance(v, list) else [v]):
            if isinstance(x, str) and re.fullmatch(r"[0-9a-f]{32}", x):
                keys.append(f"s1/ref/{x}.pack")
t, n = time.time(), 0
def get(k):
    p = root / k
    p.parent.mkdir(parents=True, exist_ok=True)
    h = st.head("singlet-data", k)
    if h is None:
        return 0
    st.download("singlet-data", k, p, size=h.size)
    return h.size
with cf.ThreadPoolExecutor(16) as ex:
    n = sum(ex.map(get, sorted(set(keys))))
(root.parent / "gses-in-s1.txt").write_text("\n".join(have) + "\n")
pol = os.environ.get("ALPHA_POLICY")
if pol:  # independent check of what is public: no withheld study, no excluded accession anywhere in the mirrored manifests
    P = json.load(open(pol))
    bad = set(P["excluded_samples"]) | set(P["withheld_studies"])
    pat = re.compile(r"\b(GS[EM]\d+)\b")
    hits = {}
    for g in have:
        txt = (root / camp.manifest_key(g)).read_text("utf-8")
        found = sorted(set(pat.findall(txt)) & bad)
        if found or g in bad:
            hits[g] = len(found) + (g in bad)
    print(f"policy scan of {len(have)} manifests: {len(hits)} with withheld/excluded accessions {sorted(hits)[:5]}", flush=True)
    if hits:
        sys.exit(9)
print(f"mirrored {len(set(keys))} objects, {n/1e9:.2f} GB in {time.time()-t:.0f}s", flush=True)
PY
T1=$(date +%s)
rt "$VENV/bin/python" -P "$SRC/scripts/s1_cidx_build.py" --base "$SCRATCH/mirror" --gse-file "$SCRATCH/gses-in-s1.txt" \
  --annot-version "$ANNOT" --threads "${SLURM_CPUS_PER_TASK:-16}" --out "$OUT/cidx" --d1-export "$OUT/cidx-d1" > "$OUT/cidx-report.json" || rc=$?
T2=$(date +%s)
echo "mirror $((T1-T0)) s, build $((T2-T1)) s, exit ${rc:-0}"
head -c 3000 "$OUT/cidx-report.json"; du -sh "$OUT/cidx" || true
exit ${rc:-0}
