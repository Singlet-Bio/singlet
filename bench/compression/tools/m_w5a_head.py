"""m_w5a step 2: HEAD every public bundle -> results/m_w5a/live_head.tsv (resumable).

Ids = union of the three D1 listings (d1_bundle_index.json, d1_gse.json, d1_manifest.json).
Public HTTPS HEAD only, User-Agent set, 8 workers. Retries transient errors; a final status of 404 is recorded, not retried.
Usage: python -P tools/m_w5a_head.py
"""
import json, os, sys, time, threading, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'm_w5a')
OUT = os.path.join(R, sys.argv[2] if len(sys.argv) > 2 else 'live_head.tsv')
UA = 'singlet-m_w5a-index-audit/1.0 (zach@contigtech.com)'
BASE = 'https://data.singlet.bio/data/{g}/{g}.singlet'


def ids():
    if len(sys.argv) > 2:  # python m_w5a_head.py <ids.json> <out.tsv>: HEAD an explicit id list (extras not in D1)
        return sorted(json.load(open(os.path.join(R, sys.argv[1]))))
    s = set()
    for f, k in (('d1_bundle_index.json', 'gse_id'), ('d1_gse.json', 'id'), ('d1_manifest.json', 'gse_id')):
        s |= {r[k] for r in json.load(open(os.path.join(R, f)))[0]['results']}
    return sorted(s)


def head(g):
    last = ''
    for attempt in range(5):
        req = urllib.request.Request(BASE.format(g=g), method='HEAD', headers={'User-Agent': UA})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                h = r.headers
                return (g, r.status, h.get('Content-Length', ''), (h.get('ETag') or '').strip('"'), h.get('Last-Modified', ''), h.get('Accept-Ranges', ''), '')
        except urllib.error.HTTPError as e:
            if e.code in (404, 403, 410):
                return (g, e.code, '', '', '', '', '')
            last = f'HTTP {e.code}'
        except Exception as e:  # DNS, timeout, reset
            last = type(e).__name__ + ':' + str(e)[:80]
        time.sleep(1.5 * (attempt + 1))
    return (g, 0, '', '', '', '', last)


def main():
    done = {}
    if os.path.exists(OUT):
        for line in open(OUT, encoding='utf-8').read().splitlines()[1:]:
            f = line.split('\t')
            if f[1] != '0':
                done[f[0]] = line
    todo = [g for g in ids() if g not in done]
    print(f'{len(done)} done, {len(todo)} to do', flush=True)
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(8) as ex:
        for row in ex.map(head, todo):
            done[row[0]] = '\t'.join(str(x) for x in row)
            n += 1
            if n % 1000 == 0:
                print(n, flush=True)
    with open(OUT, 'w', encoding='utf-8', newline='\n') as f:
        f.write('gse_id\tstatus\tcontent_length\tetag\tlast_modified\taccept_ranges\terror\n')
        for g in sorted(done):
            f.write(done[g] + '\n')
    print('wrote', OUT, len(done))


if __name__ == '__main__':
    main()
