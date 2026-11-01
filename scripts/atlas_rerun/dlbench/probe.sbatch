#!/bin/bash
#SBATCH --job-name=s9-probe
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9 Phase 1a: where does a Clipper compute node's traffic to NCBI / S3 / GCS leave the cluster? Also installs the download tools
# (aria2, s5cmd, sra-tools) into $WORK/tools/dl with micromamba, and resolves SDL locations for the benchmark runs. No data is fetched.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
echo "== node $(hostname) $(date -Is)"
echo "== NIC"; for f in /sys/class/net/*/speed; do echo "$f $(cat $f 2>/dev/null)"; done; ip -br addr 2>/dev/null
echo "== routes"; ip route 2>/dev/null; ip route get 8.8.8.8 2>/dev/null
echo "== egress ip"; curl -s --max-time 10 https://api.ipify.org; echo; curl -s --max-time 10 https://ifconfig.me; echo
echo "== login node egress for comparison is not measured (login node is off limits)"
for h in sra-downloadb.be-md.ncbi.nlm.nih.gov sra-pub-run-odp.s3.amazonaws.com storage.googleapis.com ftp.sra.ebi.ac.uk; do
  echo "== path to $h"; getent hosts $h | head -2
  (command -v traceroute && traceroute -n -m 15 -w 2 $h) 2>&1 | head -20 || true
  (command -v tracepath && tracepath -n -m 15 $h) 2>&1 | head -20 || true
done
echo "== memory/cpu"; nproc; free -g | head -2; df -h /tmp | tail -1
echo "== sysctl"; sysctl net.core.rmem_max net.ipv4.tcp_rmem net.ipv4.tcp_congestion_control 2>&1
echo "== tools present"; for t in aria2c s5cmd prefetch fasterq-dump curl wget python3 zstd traceroute tracepath mtr; do printf "%s: " $t; command -v $t || echo no; done
echo "== micromamba env"
MM=$W/tools/mm/bin/micromamba; ls -l $MM
export MAMBA_ROOT_PREFIX=$W/tools/mm/root
if [ ! -x $W/tools/dl/bin/aria2c ]; then
  $MM create -y -p $W/tools/dl -c conda-forge -c bioconda aria2 s5cmd sra-tools zstd 2>&1 | tail -15
fi
ls $W/tools/dl/bin | grep -E "aria2c|s5cmd|prefetch|fasterq" 
echo "== SDL for candidate runs"
for a in SRR25376447 SRR21102187 SRR23353990; do
  echo "-- $a"; curl -s "https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=$a&filetype=run&accept-alternate-locations=yes" | python3 -c '
import json,sys
d=json.load(sys.stdin)
for f in d["result"][0]["files"]:
    for l in f["locations"]:
        print(f["name"],f["size"],f.get("md5"),l["service"],l.get("region"),l["link"][:110])'
done
echo "== GCS anonymous probe"
curl -s -o /dev/null -w "gcs sra-pub-run-1 %{http_code}\n" -r 0-99 "https://storage.googleapis.com/sra-pub-run-1/SRR21102187/SRR21102187.1"
curl -s -o /dev/null -w "s3 odp %{http_code} %{speed_download}\n" -r 0-99999999 https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR21102187/SRR21102187
echo "== other lab jobs"; squeue -h -o "%u %j %P %T" | sort | uniq -c | sort -rn | head
echo "== done"
