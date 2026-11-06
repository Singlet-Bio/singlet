#!/bin/bash
#SBATCH --job-name=gbold-ncukern
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=01:00:00
#SBATCH --exclude=g052
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-kern, build b5: device duration of lane B's phase 3 kernel with each toggle, by Nsight Compute (durations only, one pass, serialized, no replay of data).
# Arms: base, bsort (cost sort before k_p3e), b64 (64-register k_p3lb), bsortb64. One solo pbmc1k_v3 run each at MEMGB (45) so the batches are those of a 2-slot node.
# Also prints sizeof of the members of Work for each lane (SINGLET_BOLD_WORKSIZES=1) from the first arm.
# Parameters (--export): BINDIR, EXPECT_COMMIT, MEMGB (45).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; EXPECT_COMMIT=${EXPECT_COMMIT:?}; SAMPLE=pbmc1k_v3; MEMGB=${MEMGB:-45}
COMMIT=$(cat $BINDIR/commit-gpu); case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: commit $COMMIT"; exit 2 ;; esac
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$(sha256sum $BIN | cut -d' ' -f1) commit=$COMMIT sample=$SAMPLE memgb=$MEMGB $(date -Is)"
R=$W/jobs/gpu-bold/results/ncukern-$SLURM_JOB_ID; mkdir -p $R
SCR=/tmp/$USER-gbnk-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=1 SINGLET_ENGINE_GPU_MEM_GB=$MEMGB
G=$REF/GRCh38-2024-A; NCU=/usr/local/cuda/bin/ncu
cp $PB/data/$SAMPLE/$SAMPLE.1fq $SCR/
for arm in base bsort b64 bsortb64; do
  case $arm in base) E="X_ARM=base SINGLET_BOLD_WORKSIZES=1" ;; bsort) E="SINGLET_BOLD_BSORT=1" ;; b64) E="SINGLET_BOLD_B64=1" ;; bsortb64) E="SINGLET_BOLD_BSORT=1 SINGLET_BOLD_B64=1" ;; esac
  rm -rf $SCR/out; mkdir -p $SCR/out
  ( cd $SCR && env $E timeout 900 $NCU --target-processes all --cache-control none --clock-control none --kernel-name 'regex:k_p3e|k_p3es|k_p3lb|k_p3lbs|k_cost|k_p3w|k_p3a' \
      --metrics gpu__time_duration.sum,launch__grid_size,launch__registers_per_thread --csv --log-file $R/$arm.csv \
      $BIN $SCR/$SAMPLE.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 14 --pipeline --output-format 1pz --engine gpu > $R/$arm.run.log 2>&1 ); echo "arm $arm exit $?"
  cp $SCR/out/star_Log.out $R/$arm.star_Log.out 2>/dev/null
  grep -a "gpu/bold" $SCR/out/star_Log.out | cut -c1-1500 | tee $R/$arm.marks
  grep -a -o "engine_gpu: steps (s):.*total [0-9.]*" $SCR/out/star_Log.out | grep -o "lane[ABM]\.kernels [0-9.]*" | tr '\n' ' ' | tee -a $R/$arm.marks; echo
done
R=$R python3 - <<'EOF' | tee $R/summary.txt
import csv, os, re, collections
R = os.environ["R"]
print("| arm | lane B phase 3 kernel | launches | device s | mean ms | registers | grid mean | k_cost lane B s |")
print("|---|---|---|---|---|---|---|---|")
res = {}
for arm in ("base", "bsort", "b64", "bsortb64"):
    rows = list(csv.DictReader(l for l in open(f"{R}/{arm}.csv", errors="replace") if l.startswith('"')))
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        nm = r["Kernel Name"]
        if "(int)8, (int)128, (int)128" not in nm and "Caps<160, 8, 128, 128" not in nm:
            continue   # lane B only
        short = nm.split("<")[0].replace("void ", "").replace("<unnamed>::", "")
        try: v = float(r["Metric Value"].replace(",", ""))
        except ValueError: continue
        d[short][r["Metric Name"]].append(v)
    for k, m in d.items():
        t = m.get("gpu__time_duration.sum", [])
        if not t: continue
        g = m.get("launch__grid_size", [0]); rg = m.get("launch__registers_per_thread", [0])
        print(f"| {arm} | {k} | {len(t)} | {sum(t)/1e9:.3f} | {sum(t)/len(t)/1e6:.2f} | {rg[0]:.0f} | {sum(g)/len(g):.0f} | |")
        res[(arm, k)] = sum(t) / 1e9
EOF
echo done $(date -Is)
