"""m_w5a: rehearse the repair + backup SQL against a LOCAL in-memory SQLite copy of the affected D1 rows (no network).

Builds bundle_index (old rows from d1_backup_*.json), bundle_manifest and gse (from the listings), then:
  1. checks the statement format (one complete statement per line, longest statement < 100,000 bytes)
  2. applies every repair_*.sql, checks every row against the live values / computed entries, checks untouched rows
  3. applies the repair again (must change nothing: compare-and-set)
  4. applies the backup files and checks the database equals the original snapshot exactly
  5. a guard test: a row changed by someone else before the repair is not overwritten
"""
import copy, glob, json, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from m_w5a_validate import load  # noqa: E402

R = os.path.join(HERE, '..', 'results', 'm_w5a')


def files(pat):
    return sorted(glob.glob(os.path.join(R, pat)))


def check_format():
    longest, n = 0, 0
    for f in files('repair_*.sql') + files('backup*.sql'):
        for ln, line in enumerate(open(f, encoding='utf-8', newline='').read().split('\n')):
            if not line or line.startswith('--'):
                continue
            assert line.endswith(';'), (f, ln)
            assert sqlite3.complete_statement(line), (f, ln)
            assert line.startswith('UPDATE '), (f, ln)
            longest = max(longest, len(line.encode('utf-8')))
            n += 1
    return n, longest


def build():
    bi, man, live = load()
    gse = {r['id']: r for r in json.load(open(os.path.join(R, 'd1_gse.json')))[0]['results']}
    db = sqlite3.connect(':memory:', isolation_level=None)
    db.executescript('''CREATE TABLE bundle_index (gse_id TEXT PRIMARY KEY, bytes INTEGER, entries TEXT, indexed_at TEXT);
    CREATE TABLE bundle_manifest (gse_id TEXT PRIMARY KEY, bytes INTEGER, n_files INTEGER, n_gsms_in_bundle INTEGER, manifest_n_gsms INTEGER, gsm_ids TEXT, manifest_created_at TEXT, audited_at TEXT NOT NULL DEFAULT '2026-09-04T00:00:00Z', reference_build TEXT, singlet_version TEXT);
    CREATE TABLE gse (id TEXT PRIMARY KEY, r2_bundle_bytes INTEGER, last_updated TEXT);''')
    old = {}
    for f in files('d1_backup_[0-9].json'):
        for r in json.load(open(f))[0]['results']:
            old[r['gse_id']] = r
    db.execute('begin')
    for g, r in old.items():
        db.execute('INSERT INTO bundle_index VALUES (?,?,?,?)', (g, r['bytes'], r['entries'], r['indexed_at']))
    for g, r in bi.items():  # untouched rows get a placeholder entries string
        if g not in old:
            db.execute('INSERT INTO bundle_index VALUES (?,?,?,?)', (g, r['bytes'], '[]', r['indexed_at']))
    for g, r in man.items():
        db.execute('INSERT INTO bundle_manifest (gse_id, bytes, n_files, n_gsms_in_bundle) VALUES (?,?,?,?)', (g, r['bytes'], r['n_files'], r['n_gsms_in_bundle']))
    for g, r in gse.items():
        db.execute('INSERT INTO gse VALUES (?,?,?)', (g, r['r2_bundle_bytes'], r['last_updated']))
    db.execute('commit')
    return db, live


def dump(db):
    return {t: db.execute(f'SELECT * FROM {t} ORDER BY 1').fetchall() for t in ('bundle_index', 'bundle_manifest', 'gse')}


def apply(db, pats):
    t0 = db.total_changes
    for pat in pats:
        for f in files(pat):
            db.executescript(open(f, encoding='utf-8', newline='').read())
    return db.total_changes - t0


def main():
    n, longest = check_format()
    print(f'format ok: {n} statements, longest {longest} bytes (D1 limit 100,000)')
    db, live = build()
    s0 = dump(db)
    comp = {}
    for l in open(os.path.join(R, 'computed.jsonl'), encoding='utf-8'):
        r = json.loads(l)
        comp[r['gse_id']] = r
    ch = apply(db, ['repair_*.sql'])
    print('repair applied, rows changed (sqlite total_changes, no triggers locally):', ch)
    bi = {r[0]: r for r in db.execute('SELECT * FROM bundle_index')}
    bad = 0
    for g, r in comp.items():
        want = (g, r['bytes'], r['entries'], r['fetched_at'])
        if bi[g] != want:
            bad += 1
    print('bundle_index rows != computed:', bad, 'of', len(comp))
    man = {r[0]: r for r in db.execute('SELECT gse_id, bytes, n_files FROM bundle_manifest')}
    badm = sum(1 for g, r in comp.items() if g in man and (man[g][1], man[g][2]) != (r['bytes'], r['n_files']))
    print('bundle_manifest rows != computed:', badm)
    gse = {r[0]: r[1] for r in db.execute('SELECT id, r2_bundle_bytes FROM gse')}
    badg = sum(1 for g in gse if gse[g] != int(live[g]['content_length']))
    print('gse rows != live Content-Length (all 8202):', badg)
    s1 = dump(db)
    untouched = sum(1 for a, b in zip(s0['bundle_index'], s1['bundle_index']) if a[0] not in comp and a != b)
    print('bundle_index rows outside the stale set that changed:', untouched)
    ch2 = apply(db, ['repair_*.sql'])
    print('second application changed rows:', ch2, '(must be 0)')
    assert dump(db) == s1
    ch3 = apply(db, ['backup.sql', 'backup_bundle_index_*.sql'])
    print('backup applied, rows changed:', ch3)
    s2 = dump(db)
    print('state after backup == original snapshot:', s2 == s0)
    # guard test
    db.execute("UPDATE bundle_index SET bytes = bytes + 1 WHERE gse_id = ?", (next(iter(comp)),))
    g0 = next(iter(comp))
    before = db.execute('SELECT * FROM bundle_index WHERE gse_id=?', (g0,)).fetchone()
    apply(db, ['repair_bundle_index_*.sql'])
    after = db.execute('SELECT * FROM bundle_index WHERE gse_id=?', (g0,)).fetchone()
    print('guard: row changed by someone else was left alone:', before == after)


if __name__ == '__main__':
    main()
