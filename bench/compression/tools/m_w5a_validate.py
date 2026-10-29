"""m_w5a step 1: validate the Python port against D1 rows of bundles unchanged since the September indexing.

  python -P tools/m_w5a_validate.py select   -> results/m_w5a/validate_ids.txt + validate_query.sql (read-only SELECT to run once)
  python -P tools/m_w5a_validate.py check    -> compares results/m_w5a/d1_validate.json with the recomputed JSON, byte for byte
"""
import csv, json, os, random, sys
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m_w5a_zip import parse_zip_source, entries_json  # noqa: E402
from m_w5a_net import RangeSource  # noqa: E402

R = os.path.join(HERE, '..', 'results', 'm_w5a')
N = 20


def load():
    bi = {r['gse_id']: r for r in json.load(open(os.path.join(R, 'd1_bundle_index.json')))[0]['results']}
    man = {r['gse_id']: r for r in json.load(open(os.path.join(R, 'd1_manifest.json')))[0]['results']}
    live = {r['gse_id']: r for r in csv.DictReader(open(os.path.join(R, 'live_head.tsv'), encoding='utf-8'), delimiter='\t')}
    return bi, man, live


def lm(s):
    return datetime.strptime(s, '%a, %d %b %Y %H:%M:%S %Z').replace(tzinfo=timezone.utc)


def idx_time(s):
    return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=timezone.utc)


def unchanged(bi, live):
    out = []
    for g, b in bi.items():
        h = live.get(g)
        if not h or h['status'] != '200':
            continue
        if int(h['content_length']) == b['bytes'] and lm(h['last_modified']) < idx_time(b['indexed_at']):
            out.append(g)
    return sorted(out)


def stale_ids(bi, live):
    out = []
    for g in sorted(bi):
        h = live.get(g)
        if not h or h['status'] != '200':
            continue
        b = bi[g]
        if int(h['content_length']) != b['bytes'] or lm(h['last_modified']) > idx_time(b['indexed_at']):
            out.append(g)
    return out


def backupq(k=3):
    """Split the stale studies into k query files of about equal expected size (sum of manifest n_files); one SELECT each."""
    bi, man, live = load()
    st = stale_ids(bi, live)
    w = {g: (man.get(g, {}).get('n_files') or 100) for g in st}
    groups = [[] for _ in range(k)]
    tot = [0] * k
    for g in sorted(st, key=lambda g: -w[g]):
        i = tot.index(min(tot))
        groups[i].append(g)
        tot[i] += w[g]
    for i, grp in enumerate(groups):
        grp.sort()
        ids = ','.join("'" + g + "'" for g in grp)
        open(os.path.join(R, f'backup_query_{i}.sql'), 'w').write(f"SELECT gse_id, bytes, entries, indexed_at FROM bundle_index WHERE gse_id IN ({ids})")
        print(i, len(grp), 'studies, est entries', tot[i], 'est MB', round(tot[i] * 105 / 1e6, 1), 'chars', len(ids))
    print('stale total', len(st))


def select():
    bi, man, live = load()
    un = unchanged(bi, live)
    print(len(un), 'bundles with live size == D1 bytes and Last-Modified < indexed_at')
    by_files = sorted((g for g in un if g in man and man[g]['n_files']), key=lambda g: man[g]['n_files'])
    by_bytes = sorted(un, key=lambda g: int(live[g]['content_length']))
    pick = []
    for g in by_files[-3:] + by_files[:3] + by_bytes[-4:]:  # most/fewest members, largest objects (zip64 offsets)
        if g not in pick:
            pick.append(g)
    rnd = random.Random(20261022)
    rest = [g for g in un if g not in pick]
    rnd.shuffle(rest)
    pick += rest[: N - len(pick)]
    est = sum((man.get(g, {}).get('n_files') or 0) * 110 for g in pick)
    print('picked', len(pick), 'est entries JSON bytes', est)
    for g in pick:
        print(g, live[g]['content_length'], man.get(g, {}).get('n_files'), bi[g]['indexed_at'], live[g]['last_modified'])
    open(os.path.join(R, 'validate_ids.txt'), 'w').write('\n'.join(pick) + '\n')
    ids = ','.join("'" + g + "'" for g in pick)
    open(os.path.join(R, 'validate_query.sql'), 'w').write(f"SELECT gse_id, bytes, entries FROM bundle_index WHERE gse_id IN ({ids})\n")


def check():
    bi, man, live = load()
    d1 = {r['gse_id']: r for r in json.load(open(os.path.join(R, 'd1_validate.json')))[0]['results']}
    ids = [l.strip() for l in open(os.path.join(R, 'validate_ids.txt')) if l.strip()]
    ok = 0
    rows = []
    for g in ids:
        h = live[g]
        size = int(h['content_length'])
        src = RangeSource(g, size, h['etag'])
        try:
            nbytes, entries, info = parse_zip_source(size, src.rng)
            mine = entries_json(entries)
            theirs = d1[g]['entries']
            same = mine == theirs
            same_bytes = nbytes == d1[g]['bytes']
            first = ''
            if not same:
                n = next((i for i in range(min(len(mine), len(theirs))) if mine[i] != theirs[i]), min(len(mine), len(theirs)))
                first = f'first diff at char {n}: mine={mine[max(0, n - 40):n + 60]!r} d1={theirs[max(0, n - 40):n + 60]!r}'
            rows.append((g, same and same_bytes, len(entries), len(mine), info['zip64'], info['cd_in_tail'], info['requests'], first))
            ok += int(same and same_bytes)
        except Exception as e:
            rows.append((g, False, 0, 0, '', '', 0, 'EXC ' + repr(e)))
    with open(os.path.join(R, 'validation.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('gse_id\tbyte_identical\tn_entries\tjson_chars\tzip64\tcd_in_tail\trequests\tnote\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    for r in rows:
        print(r)
    print(f'VALIDATION {ok}/{len(ids)} byte-identical (entries JSON and bytes)')


if __name__ == '__main__':
    {'select': select, 'check': check, 'backupq': backupq}[sys.argv[1]]()
