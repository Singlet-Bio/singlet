#!/bin/bash
#SBATCH --job-name=donorfix_pools
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=190G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Multi-donor regression check of the genotype-distance merge: the four stored pooled pilot libraries (10-donor pool, hashed pool,
# two cell-line mixes), auto-K, base branch against fix branch; reports K and the agreement (ARI) of the two label sets.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/pbench/bin:$PATH; P=$W/envs/pbench
O=$W/jobs/donorfix/pools; mkdir -p $O
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
for b in release/v1.4.0 fix/donor-single-split; do
  n=$(echo $b | tr '/' '_')
  git clone -q --depth 1 --branch $b https://github.com/Singlet-Bio/singlet.git $S/$n || exit 1
  x86_64-conda-linux-gnu-g++ -std=gnu++17 -O3 -DNDEBUG -march=x86-64-v3 -pthread -I$S/$n/include -I$P/include $S/$n/bench/modalities/demux_fix/demux_harness.cpp -o $S/$n/harness -L$P/lib -Wl,-rpath,$P/lib -lzstd -lz || exit 1
  for g in GSM8085472 GSM5682562 GSM4987144 GSM4987148; do
    mkdir -p $O/$n/$g
    $S/$n/harness --1pz $W/pilot/$g/s1cpu_0ea48783/out $O/$n/$g 16 auto -1 > $O/$n/$g/log.txt 2>&1
    echo "$b $g: $(grep -E 'selected K|harness\] auto' $O/$n/$g/log.txt | tr '\n' ' ')"
  done
done
$W/envs/pbench/bin/python3 -I - <<PY
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
O="$O"
for g in "GSM8085472 GSM5682562 GSM4987144 GSM4987148".split():
    a=pd.read_csv(f"{O}/release_v1.4.0/{g}/auto/donor_assignments.tsv",sep="\t",dtype=str).set_index("cell")
    b=pd.read_csv(f"{O}/fix_donor-single-split/{g}/auto/donor_assignments.tsv",sep="\t",dtype=str).set_index("cell")
    j=a.join(b,lsuffix="_a",rsuffix="_b",how="inner"); lab=j[j.cell_status_a=="cell"]
    print(g,"cells",len(lab),"K base",lab.donor_id_a.nunique(),"K fix",lab.donor_id_b.nunique(),"ARI(base,fix)",round(ari(lab.donor_id_a,lab.donor_id_b),5))
PY
