# Every GPU index (nvidia-smi numbering) on which a copy of one throughput point held memory, from the sampler's samples.jsonl ("gproc":
# {copy: {gpu index: MB}}). Prints the indices space-separated, sorted. Used by tp_gc.sbatch's pin check.
import json
import sys

used = set()
for line in open(sys.argv[1]):
    try:
        d = json.loads(line)
    except ValueError:
        continue
    for per_gpu in (d.get('gproc') or {}).values():
        used.update(per_gpu.keys())
print(' '.join(sorted(used)))
