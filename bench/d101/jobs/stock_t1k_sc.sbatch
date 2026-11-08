#!/bin/bash
#SBATCH --job-name=d101-t1ksc
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Stock T1K comparator on one Oetjen 3' library (row SLURM_ARRAY_TASK_ID of libs.tsv): every cDNA read of the library, decoded from
# its .1fq, through run-t1k as published (-u, --preset hla, the full IPD-IMGT/HLA 3.66.0 index), with no Singlet prefilter or prior.
# T1K's barcode options only add per-cell output; the sample genotype comes from the pooled reads either way.
# Output (owner-only): $W/jobs/d101/stockt1k/<sc_run>/t1k_genotype.tsv, t1k.log, time.txt
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; ENV=$W/envs/pbench
T1K=$REF/hla/t1k; BIN=${BIN:?singlet binary with decode}; LIBS=${LIBS:?}; T=${SLURM_CPUS_PER_TASK:-16}
ROW=$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1' $LIBS); [ -n "$ROW" ] || exit 0
read -r DONOR SC SCG BULK BULKG <<< "$ROW"
O=$W/jobs/d101/stockt1k/$SC; [ -s $O/t1k_genotype.tsv ] && { echo "done $SC"; exit 0; }
mkdir -p $W/jobs/d101/stockt1k $O; chmod 700 $W/jobs/d101/stockt1k
IN=$W/data/d101/sc/$SC.1fq; [ -s $IN ] || { echo "no $IN"; exit 4; }
SCR=/tmp/$USER-t1ksc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
echo "donor $DONOR $SC"
t0=$(date +%s)
$BIN decode $IN --r2 $SCR/cdna.fq --threads $T > $O/decode.log 2>&1 || { echo "decode failed"; tail -5 $O/decode.log; exit 5; }
t1=$(date +%s)
perl $T1K/bin/run-t1k -u $SCR/cdna.fq --preset hla -f $T1K/hlaidx/hlaidx_rna_seq.fa -t $T -o t1k --od $SCR/t1k > $O/t1k.log 2>&1
t2=$(date +%s)
cp $SCR/t1k/t1k_genotype.tsv $O/t1k_genotype.tsv || { echo "T1K failed for $SC"; tail -5 $O/t1k.log; exit 6; }
echo "decode_s $((t1-t0)) t1k_s $((t2-t1)) reads $(( $(wc -l < $SCR/cdna.fq) / 4 ))" | tee $O/time.txt
