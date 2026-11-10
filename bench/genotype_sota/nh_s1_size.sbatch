#!/bin/bash
#SBATCH --job-name=gsota-nhs1
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# size of the nonhost_confirmed_reads member inside a built .s1 (repo container code, default zstd level), plus a read-back check
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; BARE=$W/repos/singlet-py.git
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/s8/gsota_nh python | tar -x -C $SCR/src
B=$W/jobs/gsota-gatenh/dpi2/new/nonhost/nonhost_confirmed_reads.bin
PY=; for c in $W/venvs/singlet/bin/python $W/envs/*/bin/python; do $c -c 'import zstandard' 2>/dev/null && { PY=$c; break; }; done; echo python=$PY
PYTHONPATH=$SCR/src/python $PY -I -c "
import sys; sys.path.insert(0,'$SCR/src/python')
from singlet.s1 import container as C
raw=open('$B','rb').read()
w=C.S2Writer('dpi2'); kind=w.add_stored('nonhost/nonhost_confirmed_reads.bin', raw, {}, 0)
blob=w.build(); open('$SCR/x.s1','wb').write(blob)
print('raw bytes',len(raw),'kind',kind,'s1 bytes',len(blob),'ratio',round(len(raw)/len(blob),2))
n=int.from_bytes(raw[12:16],'little'); print('reads',n,'bytes/read raw',round(len(raw)/n,2),'bytes/read s1',round(len(blob)/n,2))
"
