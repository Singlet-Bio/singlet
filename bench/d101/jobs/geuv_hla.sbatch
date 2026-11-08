#!/bin/bash
#SBATCH --job-name=d101-geuv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# One Geuvadis sample (row SLURM_ARRAY_TASK_ID of $W/jobs/d101/geuv/list.tsv): T1K v1.0.10 as published (run-t1k -1 -2 --preset hla) and
# Singlet's HLA path (singlet hla-prefilter, then the same run-t1k on the kept pairs), same IPD-IMGT/HLA index ($REF/hla/t1k).
# Output: $W/jobs/d101/geuv/res/<sample>/{t1k,singlet}_genotype.tsv, counts.json, times. The FASTQs are deleted at the end.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; ENV=$W/envs/pbench
T1K=$REF/hla/t1k; BIN=${BIN:?singlet binary with hla-prefilter}; D=$W/jobs/d101/geuv; T=${SLURM_CPUS_PER_TASK:-8}
read -r S RUN U1 U2 POP <<< "$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n' $D/list.tsv)"
[ -n "${S:-}" ] || exit 0
KF=$W/jobs/d101/hlaprior/hlaidx_rna_seq.K2.fa   # the frequency-prior index (alleles recorded in >= 2 cells), see hla_prior_test.sh
O=$D/res/$S; [ -s $O/singlet_genotype.tsv ] && [ -s $O/t1k_genotype.tsv ] && { [ ! -s $KF ] || [ -s $O/singlet_k2_genotype.tsv ]; } && { echo "done $S"; exit 0; }; mkdir -p $O; chmod 700 $O
SCR=/tmp/$USER-geuv-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
# EBI drops long transfers mid-file, so resume (-C -) until the file is complete and passes gzip -t.
get() { local u=${1/ftp:\/\//https://} f=$SCR/$(basename $1) i
  for i in 1 2 3 4 5 6 7 8 9 10; do curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 -C - -o $f $u && gzip -t $f 2>/dev/null && return 0; sleep 20; done
  return 1; }
for u in $U1 $U2; do get $u || { echo "download failed: $u"; exit 5; }; done
F1=$SCR/$(basename $U1); F2=$SCR/$(basename $U2); ls -l $F1 $F2
t0=$(date +%s.%N)
perl $T1K/bin/run-t1k -1 $F1 -2 $F2 --preset hla -f $T1K/hlaidx/hlaidx_rna_seq.fa -t $T -o t1k --od $SCR/t1k > $O/t1k.log 2>&1
t1=$(date +%s.%N)
$BIN hla-prefilter --t1k $T1K -1 $F1 -2 $F2 -o $SCR/pf > $O/prefilter.json 2> $O/prefilter.log
t2=$(date +%s.%N)
perl $T1K/bin/run-t1k -1 $SCR/pf_1.fq -2 $SCR/pf_2.fq --preset hla -f $T1K/hlaidx/hlaidx_rna_seq.fa -t $T -o singlet --od $SCR/sg > $O/singlet.log 2>&1
t3=$(date +%s.%N)
cp $SCR/t1k/t1k_genotype.tsv $O/t1k_genotype.tsv && cp $SCR/sg/singlet_genotype.tsv $O/singlet_genotype.tsv || { echo "T1K failed for $S"; tail -5 $O/t1k.log $O/singlet.log; exit 6; }
printf '{"sample": "%s", "run": "%s", "population": "%s", "t1k_s": %s, "prefilter_s": %s, "singlet_t1k_s": %s, "t1k_candidates": %s, "singlet_candidates": %s, "prefilter": %s}\n' \
  $S $RUN $POP $(awk "BEGIN{print $t1-$t0}") $(awk "BEGIN{print $t2-$t1}") $(awk "BEGIN{print $t3-$t2}") \
  $(( $(wc -l < $SCR/t1k/t1k_candidate_1.fq) / 4 )) $(( $(wc -l < $SCR/sg/singlet_candidate_1.fq) / 4 )) "$(cat $O/prefilter.json)" > $O/counts.json
# Same prefiltered pairs on the frequency-prior index; the pairs are kept (owner-only, a few MB) so later index changes need no download.
if [ -s $KF ]; then
  perl $T1K/bin/run-t1k -1 $SCR/pf_1.fq -2 $SCR/pf_2.fq --preset hla -f $KF -t $T -o k2 --od $SCR/k2 > $O/singlet_k2.log 2>&1
  cp $SCR/k2/k2_genotype.tsv $O/singlet_k2_genotype.tsv || { echo "K2 T1K failed for $S"; tail -5 $O/singlet_k2.log; }
fi
gzip -1 -c $SCR/pf_1.fq > $O/pf_1.fq.gz; gzip -1 -c $SCR/pf_2.fq > $O/pf_2.fq.gz
cat $O/counts.json; cmp -s $O/t1k_genotype.tsv $O/singlet_genotype.tsv && echo "GENOTYPE IDENTICAL" || { echo "GENOTYPE DIFFERS"; diff $O/t1k_genotype.tsv $O/singlet_genotype.tsv | head; }
