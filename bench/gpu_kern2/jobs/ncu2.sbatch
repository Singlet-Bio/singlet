#!/bin/bash
#SBATCH --job-name=gk2-ncu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# gpu/bold-kern2: device duration of lane B's phase 3 kernel (k_p3e; k_p3c under WSHRINK) by Nsight Compute, arms base, wshrink, bstream, both.
# One solo pbmc1k_v3 run each at MEMGB (45, the budget of a 2-slot node). Kernel names are plain (templates do not match in the ncu filter).
# Records per arm the batch size and per read bytes from the engine log (wshrink line), the kernel's grid and the device seconds.
# Parameters (--export): BINDIR, EXPECT_COMMIT, MEMGB (45). Node: set by the submitter (g050).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; BIN=$BINDIR/singlet-gpu; EXPECT_COMMIT=${EXPECT_COMMIT:?}; SAMPLE=pbmc1k_v3; MEMGB=${MEMGB:-45}
COMMIT=$(cat $BINDIR/commit-gpu); case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: commit $COMMIT"; exit 2 ;; esac
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$(sha256sum $BIN | cut -d' ' -f1) commit=$COMMIT sample=$SAMPLE memgb=$MEMGB $(date -Is)"
R=$W/jobs/gpu-kern2/results/ncukern-$SLURM_JOB_ID; mkdir -p $R
SCR=/tmp/$USER-gk2nk-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=1 SINGLET_ENGINE_GPU_MEM_GB=$MEMGB
G=$REF/GRCh38-2024-A; NCU=/usr/local/cuda/bin/ncu
cp $PB/data/$SAMPLE/$SAMPLE.1fq $SCR/
for arm in base wshrink bstream both; do
  case $arm in base) E="X_ARM=base" ;; wshrink) E="SINGLET_BOLD_WSHRINK=1" ;; bstream) E="SINGLET_BOLD_BSTREAM=1" ;; both) E="SINGLET_BOLD_WSHRINK=1 SINGLET_BOLD_BSTREAM=1" ;; esac
  rm -rf $SCR/out; mkdir -p $SCR/out
  ( cd $SCR && env $E timeout 1500 $NCU --target-processes all --cache-control none --clock-control none --kernel-name 'regex:k_p3e|k_p3c|k_wplan|k_wkey' \
      --metrics gpu__time_duration.sum,launch__grid_size,launch__registers_per_thread --csv --log-file $R/$arm.csv \
      $BIN $SCR/$SAMPLE.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 14 --pipeline --output-format 1pz --engine gpu > $R/$arm.run.log 2>&1 ); echo "arm $arm exit $?"
  cp $SCR/out/star_Log.out $R/$arm.star_Log.out 2>/dev/null
  grep -a -o "kern2 wshrink.*" $SCR/out/star_Log.out | cut -c1-1500 | tee $R/$arm.marks
  grep -a -o "chunk [0-9]*: A [^;]*;" $SCR/out/star_Log.out | tr '\n' ' ' | tee -a $R/$arm.marks; echo
done
R=$R python3 - <<'EOF' | tee $R/summary.txt
import csv, os, re, collections
R = os.environ["R"]
print("| arm | kernel | launches | device s | mean ms | grid mean | registers | batch B | per read bytes |")
print("|---|---|---|---|---|---|---|---|---|")
for arm in ("base", "wshrink", "bstream", "both"):
    try:
        rows = list(csv.DictReader(l for l in open(f"{R}/{arm}.csv", errors="replace") if l.startswith('"')))
    except OSError:
        continue
    marks = open(f"{R}/{arm}.marks", errors="replace").read() if os.path.exists(f"{R}/{arm}.marks") else ""
    m = re.findall(r"B \d+ reads, batches of (\d+) \(([\d.]+) GB of Work\)", marks)
    batch = m[0][0] if m else "NA"
    pr = re.findall(r"per read (\d+) bytes", marks)
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        short = r["Kernel Name"].split("<")[0].replace("void ", "").replace("<unnamed>::", "").strip()
        try: v = float(r["Metric Value"].replace(",", ""))
        except ValueError: continue
        d[short][r["Metric Name"]].append(v)
    for k, mm in d.items():
        t = mm.get("gpu__time_duration.sum", [])
        if not t: continue
        g = mm.get("launch__grid_size", [0]); rg = mm.get("launch__registers_per_thread", [0])
        print(f"| {arm} | {k} | {len(t)} | {sum(t)/1e9:.3f} | {sum(t)/len(t)/1e6:.2f} | {sum(g)/len(g):.0f} | {rg[0]:.0f} | {batch} | {pr[0] if pr else 'NA'} |")
EOF
echo done $(date -Is)
