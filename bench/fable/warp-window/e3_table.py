import json, sys, glob, os
# usage: table.py DIR JOBID  -> table of the job's variants from e3_<job>_<label>.json.e3.json
d, job = sys.argv[1], sys.argv[2]
rows = []
for f in sorted(glob.glob(os.path.join(d, 'e3_%s_*.json.e3.json' % job)), key=os.path.getmtime):
    try:
        j = json.load(open(f))
    except Exception as e:
        print('bad', f, e); continue
    rows.append(j)
if not rows: print('no rows'); sys.exit()
base = next((r for r in rows if r['tag'] == 'base'), rows[0])
bmb = base['lanes_mb_ms']
print('%-12s %8s %8s %8s %8s %8s %6s %10s %8s %s' % ('variant', 'A ms', 'M ms', 'B ms', 'M+B', 'GPU ms', 'MB/x', 'identical', 'M dist/s', 'phases A | M | B (p1, scan, assign/sort, p3/warp)'))
for r in rows:
    A, M, B = r['A'], r['M'], r['B']
    ph = ' | '.join(','.join('%.0f' % x for x in L['phase_ms']) for L in (A, M, B))
    print('%-12s %8.1f %8.1f %8.1f %8.1f %8.1f %6.3f %10d %8.3f %s' % (r['tag'], A['ms'], M['ms'], B['ms'], r['lanes_mb_ms'], r['gpu_ms'],
          r['lanes_mb_ms'] / bmb, r['identical_distinct'], r['m_distinct_per_s'], ph))
print('distinct', base['distinct_reads'], ' reads A/M/B', base['A']['reads'], base['M']['reads'], base['B']['reads'], ' heavy', [r['heavy_reads'] for r in rows])
WS = dict(READS=0, SIMPLE=1, FB_NWA=2, FB_LANE=3, OV_STEPS=4, OV_REAL=5, OV_KEEP=6, WIN_NE=7, ROUNDS=8, LANES_ACTIVE=9, STEPS=10, COMPACT=11,
          CLK_LIST=12, CLK_STITCH=13, CLK_MERGE=14, CLK_SELECT=15, CLK_ASSIGN=16, CLK_ASSIGN_BASE=17, CLK_P3_BASE=18, CLK_FALLBACK=19)
for r in rows:
    for ln in ('A', 'M', 'B'):
        L = r[ln]
        if not L.get('warp'): continue
        w = L['ws']; G = L.get('group', 32) or 32
        act = 100.0 * w[9] / (G * w[8]) if w[8] else 0
        print('  %-10s %s: G %d wmin %d reads %d simple %d fb_nwa %d fb_lane %d ov steps/real/keep %d/%d/%d win/read %.2f lanes active %.1f%% steps/read %.1f '
              'clk list %.2e stitch %.2e merge %.2e select %.2e assign %.2e fallback %.2e p3base %.2e' % (
              r['tag'], ln, G, L.get('wmin', 0), w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7] / max(w[0], 1), act, w[10] / max(w[0], 1),
              w[12], w[13], w[14], w[15], w[16], w[19], w[18]))
        print('             hist nW', w[24:32], 'nWA', w[32:40], 'steps/w', w[40:48], 'tr/w', w[48:56])
    he = r.get('host_emul', {})
    if he.get('run'): print('  host_emul', he)
