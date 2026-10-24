"""m_w5a step 3: for every bundle whose D1 index row looks stale, fetch the central directory and compute the new row.

Stale test (bundle_index): live status 200 and (live Content-Length != D1 bytes  OR  Last-Modified > indexed_at).
Output: results/m_w5a/computed.jsonl (one line per study, resumable) with the validated port from m_w5a_zip.py.
Per study: 1 range GET for the 256 KiB tail (+1 if the zip64 EOCD or the central directory is outside the tail), plus (unless
--no-check) 1 small range GET that reads the local header of the entry with the largest offset and requires signature PK\\3\\4 and
the same name, which proves the offsets are valid for the live object.
Usage: python -P tools/m_w5a_compute.py [--no-check] [--limit N] [--workers 8]
"""
import argparse, hashlib, json, os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m_w5a_zip import parse_zip_source, entries_json, gsm_count  # noqa: E402
from m_w5a_net import RangeSource  # noqa: E402
from m_w5a_validate import load, stale_ids  # noqa: E402

R = os.path.join(HERE, '..', 'results', 'm_w5a')
OUT = os.path.join(R, 'computed.jsonl')


def work(args):
    g, h, check = args
    size = int(h['content_length'])
    rec = {'gse_id': g, 'bytes': size, 'etag': h['etag'], 'last_modified': h['last_modified'], 'err': '',
           'fetched_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    src = RangeSource(g, size, h['etag'])
    try:
        nbytes, entries, info = parse_zip_source(size, src.rng)
        rec.update(entries=entries_json(entries), n_files=len(entries), n_gsms=gsm_count(entries), n_entries_eocd=info['n_entries_eocd'],
                   n_cen=info['n_cen_records'], cd_size=info['cd_size'], cd_consumed=info['cd_consumed'], cd_offset=info['cd_offset'],
                   zip64=info['zip64'], cd_in_tail=info['cd_in_tail'])
        problems = []
        if info['n_cen_records'] != info['n_entries_eocd']:
            problems.append(f"cen records {info['n_cen_records']} != eocd count {info['n_entries_eocd']}")
        if info['cd_consumed'] != info['cd_size']:
            problems.append(f"consumed {info['cd_consumed']} != cd_size {info['cd_size']}")
        if not entries:
            problems.append('no entries')
        if entries and max(e['o'] + e['c'] for e in entries) > info['cd_offset']:
            problems.append('an entry extends past the central directory start')
        if check and entries:
            last = max(entries, key=lambda e: e['o'])
            nb = last['p'].encode('utf-8')
            blob = src.rng(last['o'], last['o'] + 29 + len(nb))
            sig_ok = blob[:4] == b'PK\x03\x04'
            nl = blob[26] | (blob[27] << 8)
            name_ok = nl == len(nb) and blob[30:30 + len(nb)] == nb
            rec['hdr_check'] = bool(sig_ok and name_ok)
            if not (sig_ok and name_ok):
                problems.append(f'local header check failed at {last["o"]} ({last["p"]})')
        rec['problems'] = problems
    except Exception as e:
        rec['err'] = repr(e)[:300]
    rec['requests'] = src.n
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-check', action='store_true')
    ap.add_argument('--check-every', type=int, default=4, help='local-header check for 1 in K studies (hash of id), plus all zip64 ones')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    bi, man, live = load()
    ids = stale_ids(bi, live)
    print('stale candidates', len(ids), flush=True)
    done = set()
    if os.path.exists(OUT):
        for line in open(OUT, encoding='utf-8'):
            r = json.loads(line)
            if not r['err']:
                done.add(r['gse_id'])
    todo = [g for g in ids if g not in done]
    if a.limit:
        todo = todo[: a.limit]
    print(len(done), 'already done;', len(todo), 'to do', flush=True)
    lock = threading.Lock()
    n = 0
    with ThreadPoolExecutor(a.workers) as ex, open(OUT, 'a', encoding='utf-8', newline='\n') as f:
        for rec in ex.map(work, [(g, live[g], (not a.no_check) and (int(hashlib.md5(g.encode()).hexdigest(), 16) % a.check_every == 0)) for g in todo]):
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
                f.flush()
            n += 1
            if n % 500 == 0:
                print(n, flush=True)
    print('done', n)


if __name__ == '__main__':
    main()
