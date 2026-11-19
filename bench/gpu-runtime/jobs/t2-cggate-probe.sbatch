#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 27 --mem=8G --time=00:06:00 --exclude=g050 -J t2-cggate-probe -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-cggate-probe: two minutes on an H100 node (the shape of the 365G job): the node's MemFree, Cached, Shmem every 10 s and the figure of today's gate,
# free = MemFree + 0.25 x (Cached - Shmem) (GiB), to decide whether the mode-0 run of three deep items can be admitted by today's gate (needs about 505-510 GiB, T2-peakmodel.md 3c).
for i in $(seq 1 12); do
  awk -v t="$(date +%H:%M:%S)" -v h="$(hostname -s)" '/^MemFree:/{f=$2} /^Cached:/{c=$2} /^Shmem:/{s=$2} /^MemTotal:/{m=$2} END{printf "%s %s MemTotal %.0f MemFree %.0f Cached %.0f Shmem %.0f GiB  today-free(0.25) %.0f  cache-share-1.0 %.0f\n", t, h, m/1048576, f/1048576, c/1048576, s/1048576, (f+0.25*(c-s))/1048576, (f+(c-s))/1048576}' /proc/meminfo
  sleep 10
done
