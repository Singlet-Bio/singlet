#!/bin/bash
#SBATCH --job-name=rg-wlpin
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# The whitelists next to the binary must be the pinned commit's own (python/singlet/data/whitelists). Report which of the copied
# pipeline-benchmark lists differed, then overlay the pin's files (the extras, e.g. the GEM-X v4 list, stay).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; RG=$W/release-gate; BARE=$W/repos/singlet-py.git
S=$(mktemp -d); trap 'rm -rf $S' EXIT
git -C $BARE archive $(cat $RG/bin/d9834b8a/commit) python/singlet/data/whitelists | tar -x -C $S
cd $S/python/singlet/data/whitelists
for f in *; do
  g=${f%.gz}; [ "$f" = "$g" ] || { gunzip -c $f > /tmp/rg_wl_$$ ; src=/tmp/rg_wl_$$; }; [ "$f" = "$g" ] && src=$f
  if [ -e $RG/bin/whitelists/$g ]; then cmp -s $src $RG/bin/whitelists/$g && echo "same   $g" || echo "DIFFER $g (copied $(wc -l < $RG/bin/whitelists/$g) lines, pin $(wc -l < $src))"; else echo "new    $g"; fi
done
rm -f /tmp/rg_wl_$$
[ -d $RG/bin/whitelists_pb_copy ] || cp -a $RG/bin/whitelists $RG/bin/whitelists_pb_copy
for f in *; do case $f in *.gz) [ "$f" = 737K-cratac-v1.txt.gz ] && gunzip -c $f > $RG/bin/whitelists/${f%.gz}.pin && continue; cp $f $RG/bin/whitelists/$f ;; ATTRIBUTION.md) ;; *) cp $f $RG/bin/whitelists/$f ;; esac; done
