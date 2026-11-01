#!/bin/bash
#SBATCH --job-name=s8-pathdiff
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: what still differs between the stream and the sorted pileup of one build (BINDIR), molecule by molecule. Both runs write SINGLET_MOLECULE_DUMP; the dumps are joined on
# (barcode, gene, umi) and the rows that differ are described by their evidence bits and read counts. SAMPLE = fixture name (default SRR19421372).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
BINDIR=${BINDIR:?}; SAMPLE=${SAMPLE:-SRR19421372}; IN=$W/jobs/s9/fixtures/$SAMPLE.1fg2
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
O=$W/s8/pathdiff_$(cut -c1-8 $BINDIR/commit)_$SAMPLE; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
for m in stream sorted; do
  X=""; [ $m = sorted ] && X="--sorted-bam"; mkdir -p $SCR/$m/out
  ( cd $SCR/$m && SINGLET_MOLECULE_DUMP=$O/mol_$m.tsv.gz $BINDIR/singlet-gpu $IN --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/$m/out/ --threads 8 --pipeline --output-format 1pz $X > $O/$m.log 2>&1 )
  echo "$m rc=$?"
done
/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python -P - <<PYEOF
import pandas as pd
O = "$O"
d = {m: pd.read_csv(f"{O}/mol_{m}.tsv.gz", sep="\t", dtype={"barcode": str, "gene_id": str, "umi": str, "label": str}) for m in ("stream", "sorted")}
for m, x in d.items(): print(m, len(x), x.label.value_counts().to_dict())
k = ["barcode", "gene_id", "umi"]
j = d["stream"].merge(d["sorted"], on=k, how="outer", suffixes=("_st", "_so"), indicator=True)
print("only in stream", int((j._merge == "left_only").sum()), "only in sorted", int((j._merge == "right_only").sum()), "both", int((j._merge == "both").sum()))
b = j[j._merge == "both"]
for c in ("label", "flags", "diag", "reads"):
    n = int((b[c + "_st"] != b[c + "_so"]).sum()); print("both, differing", c, n)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(j[j._merge != "both"].head(30).to_string())
dd = b[(b.label_st != b.label_so) | (b.flags_st != b.flags_so) | (b.diag_st != b.diag_so) | (b.reads_st != b.reads_so)]
print(dd.head(30).to_string())
j[j._merge != "both"].to_csv(f"{O}/only_one_side.tsv.gz", sep="\t", index=False); dd.to_csv(f"{O}/differing.tsv.gz", sep="\t", index=False)
PYEOF
