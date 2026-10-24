"""m_w5a step 4: build the repair + backup SQL, summary.tsv, expected.tsv, anomalies.tsv. Writes files only; touches nothing remote.

Inputs (results/m_w5a/): d1_bundle_index.json, d1_gse.json, d1_manifest.json (current D1 values), live_head.tsv, computed.jsonl,
optional d1_backup_*.json (SELECT gse_id, bytes, entries, indexed_at FROM bundle_index ...; the OLD index rows).

Design rules
  * One UPDATE per row, every value inlined and escaped (single quotes doubled). Every UPDATE is compare-and-set: it only fires if the
    row still holds the value this script saw (a concurrent fix by the site or another job is not overwritten).
  * bundle_index.entries can exceed D1's 100 KB statement limit: the first UPDATE sets bytes/entries(chunk 1)/indexed_at, follow-up
    UPDATEs append chunks with `entries = entries || '...'` guarded by the NEW bytes + indexed_at. `wrangler d1 execute --file --remote`
    runs the whole file through D1's import API, atomically, so a reader never sees a half-written entries string.
  * indexed_at = the UTC second the central directory was fetched (after the object's Last-Modified), same format as nowIso().
  * gse is separate (repair_gse_*.sql): every UPDATE on gse fires trigger gse_au which re-tokenises title/abstract into fts_gse.
"""
import csv, glob, hashlib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m_w5a_validate import load, lm, idx_time  # noqa: E402

R = os.path.join(HERE, '..', 'results', 'm_w5a')
MAX_STMT = 80_000          # bytes per statement (D1 limit 100,000)
BI_FILE_BYTES = 3_500_000  # target bytes per bundle_index file
BI_FILE_STUDIES = 150
MAN_PER_FILE = 400
GSE_PER_FILE = 150


def q(s):
    return "'" + s.replace("'", "''") + "'"


def n_or_null(v):
    return 'NULL' if v is None else str(int(v))


def is_(col, v):
    return f"{col} IS {n_or_null(v)}" if v is None or isinstance(v, int) else f"{col} IS {q(str(v))}"


def chunks(js, limit=MAX_STMT - 1500):
    out, i, n = [], 0, len(js)
    while i < n:
        j = min(n, i + limit)
        while len(q(js[i:j]).encode('utf-8')) > limit:
            j = i + max(1, int((j - i) * 0.9))
        if j < n:
            k = js.rfind('},', i, j)
            if k > i:
                j = k + 2
        out.append(js[i:j])
        i = j
    assert ''.join(out) == js
    return out or ['']


def bi_statements(g, new_bytes, new_entries, new_at, old_bytes, old_at, unconditional_entries_old=None):
    """Statements that set the row to (new_bytes, new_entries, new_at) if it still holds (old_bytes, old_at)."""
    parts = chunks(new_entries)
    guard = f"WHERE gse_id = {q(g)} AND bytes IS {n_or_null(old_bytes)} AND indexed_at IS {q(old_at) if old_at is not None else 'NULL'}"
    s = [f"UPDATE bundle_index SET bytes = {new_bytes}, entries = {q(parts[0])}, indexed_at = {q(new_at)} {guard};"]
    have = len(parts[0])
    for p in parts[1:]:  # length(entries) = chars so far makes a re-run a no-op (the bytes/indexed_at guard alone would append again)
        s.append(f"UPDATE bundle_index SET entries = entries || {q(p)} WHERE gse_id = {q(g)} AND bytes = {new_bytes} AND indexed_at = {q(new_at)} AND length(entries) = {have};")
        have += len(p)
    return s


def bi_restore_statements(g, old_bytes, old_entries, old_at):
    parts = chunks(old_entries)
    s = [f"UPDATE bundle_index SET bytes = {n_or_null(old_bytes)}, entries = {q(parts[0])}, indexed_at = {q(old_at)} WHERE gse_id = {q(g)};"]
    have = len(parts[0])
    for p in parts[1:]:
        s.append(f"UPDATE bundle_index SET entries = entries || {q(p)} WHERE gse_id = {q(g)} AND bytes IS {n_or_null(old_bytes)} AND indexed_at = {q(old_at)} AND length(entries) = {have};")
        have += len(p)
    return s


def write(path, lines):
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(lines) + '\n')


def batches(items, size_fn, max_bytes, max_n):
    cur, cur_b = [], 0
    for it in items:
        b = size_fn(it)
        if cur and (cur_b + b > max_bytes or len(cur) >= max_n):
            yield cur
            cur, cur_b = [], 0
        cur.append(it)
        cur_b += b
    if cur:
        yield cur


def main():
    bi, man, live = load()
    gse = {r['id']: r for r in json.load(open(os.path.join(R, 'd1_gse.json')))[0]['results']}
    comp = {}
    for line in open(os.path.join(R, 'computed.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        comp[r['gse_id']] = r
    old_entries = {}
    for fpath in sorted(glob.glob(os.path.join(R, 'd1_backup_*.json'))):
        for blk in json.load(open(fpath)):
            for r in blk.get('results', []):
                old_entries[r['gse_id']] = r
    print('computed', len(comp), 'backup rows', len(old_entries))

    anomalies = []
    ok = {}
    for g, r in sorted(comp.items()):
        if r['err']:
            anomalies.append((g, 'parse_failed', r['err']))
        elif r.get('problems'):
            anomalies.append((g, 'parse_problem', '; '.join(r['problems'])))
        else:
            ok[g] = r
    for g in sorted(set(bi) | set(gse) | set(man)):
        h = live.get(g)
        if not h or h['status'] != '200':
            anomalies.append((g, 'live_missing', f"HTTP {h['status'] if h else 'n/a'}; in D1: bundle_index={g in bi} gse={g in gse} bundle_manifest={g in man}"))
    for g in sorted(set(bi) - set(man)):
        anomalies.append((g, 'no_bundle_manifest_row', 'bundle_index row but no bundle_manifest row (manifest not repaired)'))

    # ---- bundle_index
    bi_set, bi_noop = [], []
    for g, r in ok.items():
        o = old_entries.get(g)
        if o is not None and o['bytes'] == r['bytes'] and o['entries'] == r['entries']:
            bi_noop.append(g)
        else:
            bi_set.append(g)
    # ---- bundle_manifest
    man_set = []
    for g, r in ok.items():
        m = man.get(g)
        if m is not None and (m['bytes'] != r['bytes'] or m['n_files'] != r['n_files']):
            man_set.append(g)
    # ---- gse (all live bundles, not only the recomputed ones)
    gse_set = [g for g in sorted(gse) if g in live and live[g]['status'] == '200' and gse[g]['r2_bundle_bytes'] != int(live[g]['content_length'])]
    # n_gsms differences (reported, not changed)
    for g, r in sorted(ok.items()):
        m = man.get(g)
        if m is not None and m['n_gsms_in_bundle'] != r['n_gsms']:
            anomalies.append((g, 'n_gsms_in_bundle_differs', f"bundle_manifest.n_gsms_in_bundle={m['n_gsms_in_bundle']} live={r['n_gsms']} (not changed by the repair)"))
        gg = gse.get(g)
        if gg is not None and gg['r2_bundle_n_gsms'] is not None and gg['r2_bundle_n_gsms'] != r['n_gsms']:
            anomalies.append((g, 'gse_r2_bundle_n_gsms_differs', f"gse.r2_bundle_n_gsms={gg['r2_bundle_n_gsms']} live={r['n_gsms']} (not changed by the repair)"))

    for g in [g for g in gse_set if g not in comp]:
        anomalies.append((g, 'gse_bytes_wrong_before_rewrite', f"gse.r2_bundle_bytes={gse[g]['r2_bundle_bytes']} but object (unchanged since {live[g]['last_modified']}) and bundle_index/bundle_manifest say {live[g]['content_length']}"))
    # ---- write repair files
    for old in glob.glob(os.path.join(R, 'repair_*.sql')) + glob.glob(os.path.join(R, 'backup*.sql')):
        os.remove(old)
    nonascii = 0
    files = []
    bi_stmts = {g: bi_statements(g, ok[g]['bytes'], ok[g]['entries'], ok[g]['fetched_at'], bi[g]['bytes'], bi[g]['indexed_at']) for g in bi_set}
    for g in bi_set:
        if any(ord(c) > 127 for c in ok[g]['entries']):
            nonascii += 1
    nb = 0
    for k, grp in enumerate(batches(bi_set, lambda g: sum(len(s.encode('utf-8')) for s in bi_stmts[g]), BI_FILE_BYTES, BI_FILE_STUDIES)):
        name = f'repair_bundle_index_{k:03d}.sql'
        lines = [f'-- m_w5a: bundle_index repair batch {k}: {len(grp)} studies. Compare-and-set; safe to re-run (re-run is a no-op).']
        for g in grp:
            lines += bi_stmts[g]
        write(os.path.join(R, name), lines)
        files.append((name, len(grp)))
        nb += 1
    for k, grp in enumerate(batches(man_set, lambda g: 1, 1 << 30, MAN_PER_FILE)):
        name = f'repair_bundle_manifest_{k:03d}.sql'
        lines = [f'-- m_w5a: bundle_manifest repair batch {k}: {len(grp)} studies (bytes, n_files). Compare-and-set.']
        for g in grp:
            m, r = man[g], ok[g]
            lines.append(f"UPDATE bundle_manifest SET bytes = {r['bytes']}, n_files = {r['n_files']} WHERE gse_id = {q(g)} AND bytes IS {n_or_null(m['bytes'])} AND n_files IS {n_or_null(m['n_files'])};")
        write(os.path.join(R, name), lines)
        files.append((name, len(grp)))
    gse_stale = [g for g in gse_set if g in comp]
    gse_other = [g for g in gse_set if g not in comp]  # bundle untouched since June, but gse.r2_bundle_bytes already differed from the object size
    for tag, lst in (('stale', gse_stale), ('other', gse_other)):
        for k, grp in enumerate(batches(lst, lambda g: 1, 1 << 30, GSE_PER_FILE)):
            name = f'repair_gse_{tag}_{k:03d}.sql'
            lines = [f'-- m_w5a: gse repair batch ({tag}) {k}: {len(grp)} studies (r2_bundle_bytes). Compare-and-set. Each UPDATE fires trigger gse_au (FTS delete+insert).']
            for g in grp:
                lines.append(f"UPDATE gse SET r2_bundle_bytes = {int(live[g]['content_length'])} WHERE id = {q(g)} AND r2_bundle_bytes IS {n_or_null(gse[g]['r2_bundle_bytes'])};")
            write(os.path.join(R, name), lines)
            files.append((name, len(grp)))

    # ---- backup: small tables in backup.sql; bundle_index old rows in backup_bundle_index_NNN.sql mirroring the repair batches
    lines = ['-- m_w5a backup.sql: restores the values every repair UPDATE overwrites (bundle_manifest, gse). bundle_index: backup_bundle_index_*.sql.']
    for g in man_set:
        m = man[g]
        lines.append(f"UPDATE bundle_manifest SET bytes = {n_or_null(m['bytes'])}, n_files = {n_or_null(m['n_files'])} WHERE gse_id = {q(g)};")
    for g in gse_set:
        lines.append(f"UPDATE gse SET r2_bundle_bytes = {n_or_null(gse[g]['r2_bundle_bytes'])} WHERE id = {q(g)};")
    write(os.path.join(R, 'backup.sql'), lines)
    missing_backup = [g for g in bi_set if g not in old_entries]
    if not missing_backup:
        for k, grp in enumerate(batches(bi_set, lambda g: sum(len(s.encode('utf-8')) for s in bi_stmts[g]), BI_FILE_BYTES, BI_FILE_STUDIES)):
            lines = [f'-- m_w5a backup of bundle_index rows overwritten by repair_bundle_index_{k:03d}.sql (unconditional restore).']
            for g in grp:
                o = old_entries[g]
                lines += bi_restore_statements(g, o['bytes'], o['entries'], o['indexed_at'] if 'indexed_at' in o else bi[g]['indexed_at'])
            write(os.path.join(R, f'backup_bundle_index_{k:03d}.sql'), lines)
    else:
        print('WARNING: old entries missing for', len(missing_backup), 'studies; backup_bundle_index_*.sql NOT written')

    # ---- summary.tsv / expected.tsv / anomalies.tsv
    allset = sorted(set(bi_set) | set(man_set) | set(gse_set) | set(bi_noop))
    with open(os.path.join(R, 'summary.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('\t'.join(['gse_id', 'old_bytes', 'live_bytes', 'old_entries', 'new_entries', 'last_modified', 'old_indexed_at', 'old_entries_src',
                           'gse_old_bytes', 'manifest_old_bytes', 'manifest_old_n_files', 'new_n_gsms', 'changes']) + '\n')
        for g in allset:
            r = ok.get(g)
            o = old_entries.get(g)
            if o is not None:
                oe, src = len(json.loads(o['entries'])), 'bundle_index.entries'
            else:
                oe, src = (man[g]['n_files'] if g in man else ''), 'bundle_manifest.n_files (proxy)'
            ch = [t for t, s in (('bundle_index', bi_set), ('bundle_manifest', man_set), ('gse', gse_set)) if g in s]
            f.write('\t'.join(str(x) for x in [g, bi[g]['bytes'] if g in bi else '', live[g]['content_length'], oe, r['n_files'] if r else '', live[g]['last_modified'],
                                              bi[g]['indexed_at'] if g in bi else '', src, gse[g]['r2_bundle_bytes'] if g in gse else '',
                                              man[g]['bytes'] if g in man else '', man[g]['n_files'] if g in man else '', r['n_gsms'] if r else '', '+'.join(ch)]) + '\n')
    with open(os.path.join(R, 'expected.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('gse_id\tbytes\tn_files\tentries_chars\tentries_sha256\tindexed_at\n')
        for g in sorted(ok):
            r = ok[g]
            f.write(f"{g}\t{r['bytes']}\t{r['n_files']}\t{len(r['entries'])}\t{hashlib.sha256(r['entries'].encode('utf-8')).hexdigest()}\t{r['fetched_at']}\n")
    with open(os.path.join(R, 'anomalies.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('gse_id\tkind\tdetail\n')
        for a in anomalies:
            f.write('\t'.join(a) + '\n')

    # ---- row-write estimate
    n_stmts = sum(len(v) for v in bi_stmts.values())
    print(f'bundle_index: {len(bi_set)} rows to update ({len(bi_noop)} already identical), {n_stmts} statements; non-ASCII entries in {nonascii}')
    print(f'bundle_manifest: {len(man_set)} rows;  gse: {len(gse_set)} rows')
    print(f'rows written: bundle_index {len(bi_set)} (+0 index) + bundle_manifest {len(man_set)} = {len(bi_set) + len(man_set)};'
          f' gse {len(gse_set)} base rows + FTS shadow rows (fires gse_au on every UPDATE)')
    print(f'gse: {len(gse_stale)} on rewritten bundles, {len(gse_other)} on bundles untouched since June (gse already wrong before the rewrite)')
    print('files:', len(files))
    print('anomalies:', len(anomalies))
    import collections
    print(collections.Counter(a[1] for a in anomalies))


if __name__ == '__main__':
    main()
