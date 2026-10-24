"""m_w5a: run IMMEDIATELY before applying the repair. HEADs bundles again and reports any whose ETag / size / Last-Modified differs
from results/m_w5a/live_head.tsv (= the objects the repair SQL was computed from). A changed object means its repair row is wrong
(the compare-and-set guard protects against D1 changes, not against the object being rewritten again).

  python -P tools/m_w5a_precheck.py            # all 8,202 bundles (8,202 HEAD requests, ~9 minutes at 8 parallel)
  python -P tools/m_w5a_precheck.py 200        # random sample: N stale + N unchanged bundles
Writes results/m_w5a/precheck_changed.tsv (gse_id, kind, before, after). Exit status 1 if anything changed.
"""
import csv, os, random, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m_w5a_head import head  # noqa: E402
from m_w5a_validate import load, stale_ids  # noqa: E402

R = os.path.join(HERE, '..', 'results', 'm_w5a')


def main():
    bi, man, live = load()
    st = stale_ids(bi, live)
    ids = sorted(live)
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
        rnd = random.Random()
        stl = set(st)
        ids = sorted(rnd.sample(st, min(n, len(st))) + rnd.sample([g for g in ids if g not in stl], n))
    changed = []
    with ThreadPoolExecutor(8) as ex:
        for g, status, clen, etag, lmod, _ar, err in ex.map(head, ids):
            b = live[g]
            if (str(status), clen, etag, lmod) != (b['status'], b['content_length'], b['etag'], b['last_modified']):
                changed.append((g, 'changed', f"{b['status']} {b['content_length']} {b['etag']} {b['last_modified']}", f'{status} {clen} {etag} {lmod} {err}'))
    with open(os.path.join(R, 'precheck_changed.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('gse_id\tkind\tbefore\tafter\n')
        for c in changed:
            f.write('\t'.join(c) + '\n')
    print(f'checked {len(ids)}; changed since the computation: {len(changed)}')
    sys.exit(1 if changed else 0)


if __name__ == '__main__':
    main()
