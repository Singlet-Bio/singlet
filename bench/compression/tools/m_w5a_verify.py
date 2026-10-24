"""m_w5a post-apply verification (offline compare; the orchestrator runs the 3 listing queries, this script compares).

After applying the repair, dump the current D1 state with exactly these read-only queries (JSON to files in results/m_w5a/):
  after_bundle_index.json : SELECT gse_id, bytes, length(entries) AS entries_chars, indexed_at FROM bundle_index
  after_gse.json          : SELECT id, r2_bundle_bytes FROM gse WHERE r2_bundle_bytes > 0
  after_manifest.json     : SELECT gse_id, bytes, n_files FROM bundle_manifest
and re-run `python -P tools/m_w5a_head.py` (refreshes live_head.tsv) if you want to compare with the current objects. Then:
  python -P tools/m_w5a_verify.py
Prints, per table, how many rows still disagree with the live Content-Length (and with expected.tsv for bundle_index.entries length).
"""
import csv, json, os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'm_w5a')


def rows(name):
    return json.load(open(os.path.join(R, name)))[0]['results']


def main():
    live = {r['gse_id']: r for r in csv.DictReader(open(os.path.join(R, 'live_head.tsv'), encoding='utf-8'), delimiter='\t')}
    exp = {r['gse_id']: r for r in csv.DictReader(open(os.path.join(R, 'expected.tsv'), encoding='utf-8'), delimiter='\t')}
    bi = {r['gse_id']: r for r in rows('after_bundle_index.json')}
    gse = {r['id']: r for r in rows('after_gse.json')}
    man = {r['gse_id']: r for r in rows('after_manifest.json')}
    bad_bi = [g for g, r in bi.items() if g in live and live[g]['status'] == '200' and r['bytes'] != int(live[g]['content_length'])]
    bad_len = [g for g, r in bi.items() if g in exp and r.get('entries_chars') is not None and r['entries_chars'] != int(exp[g]['entries_chars'])]
    bad_gse = [g for g, r in gse.items() if g in live and live[g]['status'] == '200' and r['r2_bundle_bytes'] != int(live[g]['content_length'])]
    bad_man = [g for g, r in man.items() if g in live and live[g]['status'] == '200' and r['bytes'] != int(live[g]['content_length'])]
    bad_nf = [g for g, r in man.items() if g in exp and r['n_files'] != int(exp[g]['n_files'])]
    print(f'bundle_index: {len(bi)} rows, bytes != live: {len(bad_bi)}, entries length != expected: {len(bad_len)}')
    print(f'gse: {len(gse)} rows, r2_bundle_bytes != live: {len(bad_gse)}')
    print(f'bundle_manifest: {len(man)} rows, bytes != live: {len(bad_man)}, n_files != expected: {len(bad_nf)}')
    for name, lst in (('bundle_index.bytes', bad_bi), ('bundle_index.entries', bad_len), ('gse', bad_gse), ('manifest.bytes', bad_man), ('manifest.n_files', bad_nf)):
        if lst:
            print(name, 'first mismatches:', lst[:10])


if __name__ == '__main__':
    main()
