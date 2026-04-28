#!/bin/bash
source /opt/rh/gcc-toolset-13/enable
export TMPDIR=/dev/shm
export PATH=/mnt/home/debruinz/.conda/envs/cellarium/bin:$PATH
export LD_LIBRARY_PATH=/mnt/home/debruinz/.conda/envs/cellarium/lib:${LD_LIBRARY_PATH:-}
ulimit -n 10240

SINGLIFY=/mnt/home/debruinz/Singlet-AI/singlify/build/singlify
FQ=/mnt/projects/debruinz_project/singlify_validation/1fq/SRR20020820.1fq
OUTDIR=/dev/shm/archive_test_$$
mkdir -p $OUTDIR

echo "=== Original file ==="
ORIG=$(stat -c%s "$FQ")
echo "Size: $(( ORIG / 1048576 )) MB  qual_mode=$(
  python3 -c "
import struct
with open('$FQ','rb') as f:
    f.seek(55)
    q = struct.unpack('B', f.read(1))[0]
    names = {0:'NONE',1:'BINNED4',2:'FULL',3:'BINNED2'}
    print(names.get(q, str(q)))
" 2>/dev/null)"

echo ""
echo "=== Archive (strip quality) ==="
/usr/bin/time -f "wall=%e" "$SINGLIFY" archive "$FQ" \
    -o "$OUTDIR/SRR20020820.noq.1fq" --verbose 2>&1

echo ""
echo "=== Sizes ==="
ARCH=$(stat -c%s "$OUTDIR/SRR20020820.noq.1fq" 2>/dev/null || echo 1)
echo "Original: $(( ORIG / 1048576 )) MB"
echo "Archive:  $(( ARCH / 1048576 )) MB"
python3 -c "print(f'Reduction: {(1 - $ARCH/$ORIG)*100:.1f}%')"

echo ""
echo "=== Archived header qual_mode ==="
python3 -c "
import struct
with open('$OUTDIR/SRR20020820.noq.1fq','rb') as f:
    f.seek(55)
    q = struct.unpack('B', f.read(1))[0]
    names = {0:'NONE',1:'BINNED4',2:'FULL',3:'BINNED2'}
    print('qual_mode =', names.get(q, str(q)), '(expected: NONE)')
" 2>/dev/null

echo ""
echo "=== Decode archived file (first 8 lines of R2) ==="
"$SINGLIFY" decode "$OUTDIR/SRR20020820.noq.1fq" \
    --r1 "$OUTDIR/R1.fq" --r2 "$OUTDIR/R2.fq" --threads 4 2>&1
head -8 "$OUTDIR/R2.fq"

rm -rf "$OUTDIR"
echo "DONE"
