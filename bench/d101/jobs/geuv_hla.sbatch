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
AE=$W/envs/arcashla   # arcasHLA comparator (arcas_env.sbatch), run on the same FASTQs when the env is built
O=$D/res/$S; [ -s $O/singlet_genotype.tsv ] && [ -s $O/t1k_genotype.tsv ] && { [ ! -s $KF ] || [ -s $O/singlet_k2_genotype.tsv ]; } \
  && { [ ! -f $AE/.ok ] || [ -s $O/arcashla.genotype.json ]; } && { echo "done $S"; exit 0; }; mkdir -p $O; chmod 700 $O
SCR=/tmp/$USER-geuv-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
# EBI drops long transfers mid-file, so resume (-C -) until the file is complete and passes gzip -t.
get() { local u=${1/ftp:\/\//https://} f=$SCR/$(basename $1) i
  for i in 1 2 3 4 5 6 7 8 9 10; do curl -fsSL --retry 5 --retry-all-errors --retry-delay 10 -C - -o $f $u && gzip -t $f 2>/dev/null && return 0; sleep 20; done
  return 1; }
# NCBI's S3 copy first (about 10x EBI from Clipper nodes, memory fastest-read-source): the run's .sra object in sra-pub-run-odp,
# fetched as 8 resumable byte ranges, checked against its Content-Length, then fasterq-dump --split-files. The submitted FASTQs are
# not in sra-pub-src. EBI's FASTQs are the fallback when the size does not match or the conversion fails.
size() { { curl -fsSIL "$1" 2>/dev/null || true; } | awk '{gsub(/[^[:print:]]/,"")} tolower($1)=="content-length:"{n=$2} END{print n+0}'; }
s3get() { local u=https://sra-pub-run-odp.s3.amazonaws.com/sra/$RUN/$RUN f=$SCR/$RUN.sra NE N=8 L i pids="" ok=1
  NE=$(size $u); [ "$NE" -gt 0 ] || return 1; L=$(( (NE + N - 1) / N ))
  part() { local lo=$(( $1 * L )) hi=$(( ($1 + 1) * L - 1 )) p=$SCR/sra.$1 t have try
    [ $hi -ge $NE ] && hi=$(( NE - 1 )); t=$(( hi - lo + 1 ))
    for try in $(seq 1 20); do have=$(stat -c %s $p 2>/dev/null || echo 0); [ $have -ge $t ] && return 0
      curl -fsSL --retry 3 --retry-all-errors --retry-delay 5 -r $(( lo + have ))-$hi $u >> $p || sleep 5; done; return 1; }
  for i in $(seq 0 $(( N - 1 ))); do part $i & pids="$pids $!"; done
  for p in $pids; do wait $p || ok=0; done; [ $ok = 1 ] || return 1
  cat $(for i in $(seq 0 $(( N - 1 ))); do echo $SCR/sra.$i; done) > $f; rm -f $SCR/sra.*
  [ "$(stat -c %s $f)" = "$NE" ] || { echo "S3 size mismatch for $RUN"; return 1; }
  fasterq-dump --split-files -e $T -t $SCR -O $SCR $f > $SCR/fqd.log 2>&1 || { tail -3 $SCR/fqd.log; return 1; }
  rm -f $f; [ -s $SCR/${RUN}_1.fastq ] && [ -s $SCR/${RUN}_2.fastq ]; }
td=$(date +%s)
if s3get; then F1=$SCR/${RUN}_1.fastq; F2=$SCR/${RUN}_2.fastq; SRCN=ncbi-s3-odp
else
  echo "S3 route failed for $RUN; EBI"; rm -f $SCR/${RUN}* $SCR/sra.*
  for u in $U1 $U2; do get $u || { echo "download failed: $u"; exit 5; }; done
  F1=$SCR/$(basename $U1); F2=$SCR/$(basename $U2); SRCN=ebi
fi
ls -l $F1 $F2; echo "reads from $SRCN in $(( $(date +%s) - td )) s"; echo "download $SRCN $(( $(date +%s) - td ))" >> $O/times.txt
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
# arcasHLA as published for FASTQ input: genotype pseudoaligns every pair to its HLA index (kallisto), no prefilter.
if [ -f $AE/.ok ]; then
  ta=$(date +%s.%N)
  PATH=$AE/bin:$PATH arcasHLA genotype -t $T -g A,B,C,DQB1,DRB1 -o $SCR/arcas $F1 $F2 > $O/arcashla.log 2>&1
  cp $SCR/arcas/*.genotype.json $O/arcashla.genotype.json || { echo "arcasHLA failed for $S"; tail -5 $O/arcashla.log; }
  echo "arcashla_s $(awk "BEGIN{print $(date +%s.%N)-$ta}")" >> $O/times.txt
fi
cat $O/counts.json; cmp -s $O/t1k_genotype.tsv $O/singlet_genotype.tsv && echo "GENOTYPE IDENTICAL" || { echo "GENOTYPE DIFFERS"; diff $O/t1k_genotype.tsv $O/singlet_genotype.tsv | head; }
