#!/bin/bash
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240
SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
FQ=/mnt/projects/debruinz_project/singlify_validation/1fq/SRR20020820.1fq
OUTDIR=/dev/shm/rans_test_$$
mkdir -p $OUTDIR

echo "=== Decode SRR20020820 ==="
$SINGLIFY decode "$FQ" --r1 $OUTDIR/R1.fq --r2 $OUTDIR/R2.fq --threads 4 2>&1 | grep -v "^$"

echo "=== Re-encode comparisons ==="
$SINGLIFY encode $OUTDIR/R2.fq $OUTDIR/R1.fq --quality binned4 --out $OUTDIR/b4.1fq --threads 8 2>&1 | grep "Done"
$SINGLIFY encode $OUTDIR/R2.fq $OUTDIR/R1.fq --quality binned2 --out $OUTDIR/b2.1fq --threads 8 2>&1 | grep "Done"
$SINGLIFY encode $OUTDIR/R2.fq $OUTDIR/R1.fq --quality none   --out $OUTDIR/none.1fq --threads 8 2>&1 | grep "Done"

echo "=== Sizes ==="
ls -lh $OUTDIR/*.1fq 2>/dev/null
echo "Original: $(du -h "$FQ" | cut -f1)"

echo "=== Quality symbol distribution R2 top 10000 reads ==="
head -40000 $OUTDIR/R2.fq | awk 'NR%4==0' | fold -w1 | sort | uniq -c | sort -rn | head -6

python3 -c "
import math
counts = {'F': 316484, '0': 22321, ':': 17274, \"'\": 2912}
total = sum(counts.values())
H = -sum(c/total * math.log2(c/total) for c in counts.values())
print(f'Shannon entropy = {H:.4f} bits/base')
print(f'Theoretical rANS gain vs 2-bit raw: {(2-H)/2*100:.1f}%')
"

rm -rf $OUTDIR
echo "DONE"
