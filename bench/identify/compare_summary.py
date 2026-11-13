"""Compare two scored runs row by row on verdict, protocol, protocol_tag and action.   usage: compare_summary.py OLD_summary.tsv NEW_DIR_OR_FILE..."""
import csv, glob, os, sys

def load(paths):
    rows = {}
    for p in paths:
        for f in (glob.glob(os.path.join(p, '**', 'summary.tsv'), recursive=True) if os.path.isdir(p) else [p]):
            for r in csv.DictReader(open(f), delimiter='\t'):
                rows[r['id']] = r
    return rows

old, new = load([sys.argv[1]]), load(sys.argv[2:])
keys = ('verdict', 'protocol', 'protocol_tag', 'action', 'ambiguous_with')
diff = [(i, k, old[i].get(k), new[i].get(k)) for i in old if i in new for k in keys if old[i].get(k) != new[i].get(k)]
print(f'rows old {len(old)}, new {len(new)}, compared {len([i for i in old if i in new])}, missing in new {len([i for i in old if i not in new])}')
print(f'differences in verdict/protocol/protocol_tag/action/ambiguous_with: {len(diff)}')
for d in diff: print('\t'.join(map(str, d)))
