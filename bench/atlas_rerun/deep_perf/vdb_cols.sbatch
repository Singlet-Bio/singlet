#!/bin/bash
#SBATCH --job-name=dp-vdbcol
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch vdb_cols.sbatch <run>
# Peak RSS and time of reading 10 rows of each SEQUENCE column remotely (vdb-dump), to find the column whose remote
# read pulls gigabytes into RAM.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
RUN=$1
which vdb-dump || ls $W/envs/pbench/bin | grep -i vdb
vdb-dump --info $RUN 2>&1 | head -30
for C in READ_LEN READ_START READ QUALITY NAME; do
  /usr/bin/time -f "$C rows 100001-100010: %e s, maxrss %M KB" vdb-dump -T SEQUENCE -R 100001-100010 -C $C $RUN > /dev/null
done
/usr/bin/time -f "QUALITY rows 5000001-5000010: %e s, maxrss %M KB" vdb-dump -T SEQUENCE -R 5000001-5000010 -C QUALITY $RUN > /dev/null
vdb-dump -T SEQUENCE -R 1-2 $RUN 2>&1 | cut -c1-200 | head -30
