#!/bin/bash
#SBATCH --job-name=ar-dlbench
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch dl_bench.sbatch <mode> <SRR,SRR,...>
# Download paths from a Clipper compute node, into node-local /tmp, timed:
#   singlet      `singlet download SRR` streams SRA straight to .1fq (the campaign's path), 1 run
#   prefetch8    8 runs fetched in parallel with prefetch (aggregate bandwidth)
#   prefetch16   16 runs in parallel
#   ena          ENA fastq.gz of 1 run over HTTPS (both files, curl)
#   ena_head     first 25% of the same run's fastq.gz (curl | head): what a cap costs to fetch
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; export PATH=$ENV/bin:$PATH
export HOME=$W/cache/condahome
MODE=$1; RUNS=(${2//,/ })
O=$W/data/atlas-rerun/dlbench/$MODE; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cd $SCR
t0=$(date +%s.%N)
case $MODE in
  singlet)
    r=${RUNS[0]}
    /usr/bin/time -v -o $O/time.txt $W/pipeline-benchmark/bin/singlet-fixed download $r -o $SCR/$r.1fq > $O/log.txt 2>&1 || echo "exit $?" >> $O/log.txt
    ;;
  prefetch8|prefetch16)
    n=${MODE#prefetch}
    for r in "${RUNS[@]:0:$n}"; do prefetch --max-size u -O $SCR/sra $r > /dev/null 2>&1 & done; wait
    ;;
  ena|ena_head)
    r=${RUNS[0]}
    urls=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$r&result=read_run&fields=fastq_ftp,fastq_bytes&format=tsv" | tail -1 | cut -f2)
    for u in ${urls//;/ }; do
      if [ $MODE = ena ]; then curl -s -o $SCR/$(basename $u) "https://$u" &
      else
        bytes=$(curl -sI "https://$u" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
        curl -s "https://$u" | head -c $((bytes / 4)) > $SCR/$(basename $u) &
      fi
    done; wait
    ;;
esac
t1=$(date +%s.%N)
bytes=$(du -sb $SCR | cut -f1)
python3 - "$t0" "$t1" "$bytes" <<'PY' | tee $O/result.txt
import sys; t0, t1, b = map(float, sys.argv[1:])
print(f"seconds={t1-t0:.0f} bytes={b:.0f} MB_per_s={b/(t1-t0)/1e6:.1f}")
PY
