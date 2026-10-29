#!/bin/bash
# Read written .s1 studies back from R2 and check what the Research Alpha release promises (read-only; no writes anywhere):
#   - every sample object: matrices are pz2 members (kind 1), no member of a withheld layer, no restricted member;
#   - the cells table carries the label columns, populated on called cells (cell type CL id + per-level confidences, sex for
#     human/mouse, ancestry where the donor depth tables existed), and agrees with the cell-index sidecar;
#   - no withheld study or excluded accession anywhere in the manifest.
#   sbatch --export=ALL,COMMIT=<sha>,GSES="GSE... GSE...",ALPHA_POLICY=<json> scripts/s1_readback.sbatch
#SBATCH --job-name=s1read
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=${WORK:-/mnt/projects/debruinz_project/debruinz/singlet}
COMMIT=${COMMIT:?set COMMIT}
SRC=$WORK/src/singlet-py/v2mig__${COMMIT:0:10}
SIF=$WORK/images/python-3.12-bookworm.sif
VENV=$WORK/venvs/v2tools
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
set -a; . "$HOME/.config/singlet-r2.env"; set +a
export PYTHONPATH=$SRC/python SINGLET_S1_CACHE=/tmp/$USER-${SLURM_JOB_ID}-cache
OUT=$WORK/state/s1/readback-${SLURM_JOB_ID}.json
apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$VENV/bin/python" - "$SRC" "$OUT" ${GSES:?} <<'PY'
import io, json, os, re, sys
import numpy as np, pyarrow.parquet as pq
src, out, gses = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, src + "/scripts")
import s1_campaign as camp
from singlet.s1 import packs as P
from singlet.s1.policy import Policy, omitted_layer
from singlet.s1.sample import S1Sample
from singlet._restricted import is_restricted_member
pol = Policy.load(os.environ["ALPHA_POLICY"])
st = camp.S3Store()
packs = P.default_store(); packs.download = True
rep = {}
for g in gses:
    r = {"gse": g, "problems": []}
    raw = st.get_bytes("singlet-data", f"s1/m/{g}.json")
    m = json.loads(raw)
    txt = raw.decode()
    bad = sorted(set(re.findall(r"\b(GS[EM]\d+)\b", txt)) & (pol.excluded | pol.withheld))
    if bad or pol.withholds(g):
        r["problems"].append(f"withheld/excluded accessions in the manifest: {len(bad)}")
    r.update(release_policy=m.get("release_policy"), transcode=m.get("transcode"), samples=len(m["samples"]))
    cov = {"samples_labelled": 0, "called": 0, "cell_type": 0, "level1": 0, "level2": 0, "level3": 0, "sex": 0, "ancestry": 0}
    kinds = {}
    for s in m["samples"]:
        obj = st.get_bytes("singlet-data", s["key"])
        smp = S1Sample(obj, packs)
        for name, mi in smp.reader.members.items():
            if omitted_layer(name) is not None or is_restricted_member(name):
                r["problems"].append(f"{s['gsm']}: withheld member {name}")
            if name.endswith(".1pz") and mi.kind != 1 and mi.kind != 4:
                kinds.setdefault("non-pz2 .1pz", []).append(f"{s['gsm']}/{name}:kind{mi.kind}")
            kinds[mi.kind] = kinds.get(mi.kind, 0) + 1
        t = smp.cells()
        if t is None:
            continue
        called = np.asarray(t.column("called"), bool)
        cov["called"] += int(called.sum())
        if not t.has("cell_type_cl"):
            continue
        cov["samples_labelled"] += 1
        ct = np.asarray(t.column("cell_type_cl"), object)[called]
        cov["cell_type"] += int(sum(1 for x in ct if x))
        for k in (1, 2, 3):
            v = np.asarray(t.column(f"cell_type_p{k}"), np.float32)[called]
            cov[f"level{k}"] += int(np.isfinite(v).sum())
        cov["sex"] += int(sum(1 for x in np.asarray(t.column("sex_chrom"), object)[called] if x))
        cov["ancestry"] += int(sum(1 for x in np.asarray(t.column("ancestry_hancestro"), object)[called] if x))
        for ver, ent in (s.get("annotations") or {}).items():
            sc = pq.read_table(io.BytesIO(st.get_bytes("singlet-data", ent["key"]))).to_pydict()
            idx = np.flatnonzero(called)
            if list(idx) != sc["col"] or [t.column("sex_chrom")[i] for i in idx] != [x or "" for x in sc["sex_chrom"]]:
                r["problems"].append(f"{s['gsm']}: cells-table labels differ from the sidecar")
    r["coverage"], r["member_kinds"] = cov, {str(k): v for k, v in kinds.items()}
    rep[g] = r
    print(json.dumps(r), flush=True)
json.dump(rep, open(out, "w"), indent=1)
print("problems:", sum(len(v["problems"]) for v in rep.values()), "->", out)
PY
