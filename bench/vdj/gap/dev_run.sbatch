#!/bin/bash
#SBATCH --job-name=vdj-gap-dev
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Registered V(D)J rule (docs/beta-gap-table.md, section 5): shorter-seed rescue of V, J and C calls. One whole node.
# For every development library the same code runs twice, rescue off (base) and on (resc: rescue_k=11, rescue_hits=4), on the same reads and
# threads; metrics against Cell Ranger's filtered contigs (compare_vdj.py) and recovery by flank mismatch (shm_recovery.py, BCR libraries).
#   LIBS (default below) are development libraries only. For the out-of-corpus check on chx10k_b run it with LIBS=chx10k_b once.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/vdjgap; BARE=$W/repos/singlet-py.git
BRANCH=${BRANCH:-fix/vdj-mismatch-tolerant-seeds}; LIBS=${LIBS:-"ctrl10k_b pbmc10k_b pbmc2_b pbmc10k_t pbmc2_t"}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/vdjgap/head || exit 2
C=$(git -C $BARE rev-parse refs/vdjgap/head); echo "commit $C"
T=$(mktemp -d /tmp/vdjgap.XXXX); trap 'rm -rf $T' EXIT
git -C $BARE archive $C python bench/vdj | tar -x -C $T
export PATH=$W/envs/vdjcrispr/bin:$PATH PYTHONPATH=$T/python HOME=$W/cache/condahome NUMBA_CACHE_DIR=$W/cache/numba-vdjgap-$$
mkdir -p $J/out
WL=$W/pipeline-benchmark/whitelists/737K-august-2016.txt; REF=$W/state/vdjc/ref/vdj_human.tsv.gz
chains() { case $1 in *_t) echo TCR ;; *) echo BCR ;; esac; }
run_one() {  # lib variant
  L=$1; V=$2; D=$W/data/vdjc/$L; O=$J/out/$L/$V; mkdir -p $O
  ARGS=""; [ $V = resc ] && ARGS="--param rescue_k=11 --param rescue_hits=4"
  python -P $T/bench/vdj/run_vdj.py --fastq-dir $D/fastqs --whitelist $WL --ref $REF --out $O --threads 8 $ARGS > $O/log.txt 2>&1
  R=$W/state/vdjc/runs/$L
  [ -s $R/gex_cells.txt ] || R=$J/out/$L
  python -P $T/bench/vdj/compare_vdj.py --reference $D/filtered_contig_annotations.csv --chains $(chains $L) --vdj-ref $REF \
    $( [ -s $R/gex_cells.txt ] && echo --gex-cells $R/gex_cells.txt ) --singlet $O/contigs.csv --singlet-cells $O/cells.csv \
    $( [ -s $R/gex_lineage.csv ] && echo --gex-labels $R/gex_lineage.csv ) --out $O/metrics.json >> $O/log.txt 2>&1
  echo "[vdj] $L $V done $(date -Is)"
}
export -f run_one chains; export W J T WL REF
for L in $LIBS; do for V in base resc; do echo "$L $V"; done; done | xargs -P 6 -L 1 bash -c 'run_one "$@"' _
for L in $LIBS; do
  case $L in *_t) continue ;; esac
  python -P $T/bench/vdj/gap/shm_recovery.py $W/data/vdjc/$L $J/out/$L/recovery.json $J/out/$L/base=base $J/out/$L/resc=resc
done
python - <<'PYEOF'
import json, os, glob
J = "/mnt/projects/debruinz_project/debruinz/singlet/jobs/vdjgap/out"
for lib in sorted(os.listdir(J)):
    for v in ("base", "resc"):
        p = f"{J}/{lib}/{v}/metrics.json"
        if not os.path.exists(p):
            continue
        s = json.load(open(p)).get("singlet", {})
        print(lib, v, {k: round(s[k], 4) for k in ("correct_pairing", "pairing_rate_method", "contig_f1", "contig_recovery", "cell_chain_recovery", "wrong_lineage_method", "v_gene_agreement", "j_gene_agreement") if k in s and s[k] is not None})
PYEOF
