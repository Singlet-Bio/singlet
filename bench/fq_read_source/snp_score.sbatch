#!/bin/bash
#SBATCH --job-name=fqrs-snpscore
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch [--dependency=afterok:<snp_run>] snp_score.sbatch
# Scores the donor genotypes of each variant of snp_run.sbatch, plus the published campaign run of the same lane
# (bench-public/runs/kang2018_b2_stim/singlet_auto, whose .1fq came from `encode` with the old 6/15/25/37 bins), with
# the donor-snps benchmark's own scorer (bench/public/genotype/lib/snp_concordance.py, truth = control lane).
# Writes counts only: $W/jobs/fq-read-source/snp/<variant>.tsv and the false-alt / concordance summary.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
L=$W/bench-public/code/public/genotype/lib; T=$W/bench-public/data/kang_truth; ORIG=$W/bench-public/runs
S=$W/jobs/fq-read-source/snp
mkdir -p $S/published/kang2018_b2_stim
ln -sfn $ORIG/kang2018_b2_stim/singlet_auto $S/published/kang2018_b2_stim/singlet_auto
for VAR in published none ilmn4; do
  R=$S/$VAR
  [ -e $R/kang2018_b2_stim/singlet_auto/private/donor ] || { echo "$VAR: no donor genotypes"; continue; }
  for d in kang2018_b2_ctrl kang2018_ctrl; do ln -sfn $ORIG/$d $R/$d; done
  # the scorer reads RUNS/../private for 1000G truth only; Kang truth comes from the control lane's cellsnp counts
  python -I $L/snp_concordance.py $R $T $S/$VAR.tsv kang2018_stim > $S/$VAR.log 2>&1 || { tail -20 $S/$VAR.log; continue; }
  python -I - $S/$VAR.tsv $VAR <<'PY'
import sys
import pandas as pd
d = pd.read_csv(sys.argv[1], sep="\t")
d = d[d.tool == "singlet"]
for st in ("all", "dp10"):
    c = d[d.stratum == st][[f"t{t}_c{u}" for t in range(3) for u in range(3)]].sum().values.reshape(3, 3)
    n = c.sum(); hom = c[0].sum(); nonref = c[1:].sum()
    print(f"{sys.argv[2]:10s} {st:5s} sites={n:9d} conc={c.trace()/n:.4f} nonref_conc={(c[1,1]+c[2,2])/nonref:.4f} "
          f"false_alt={(c[0,1]+c[0,2])/hom:.4f} donors={d[d.stratum==st].donor.nunique()}")
PY
done
