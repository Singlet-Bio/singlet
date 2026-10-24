"""w4_reader: time-to-bytes model for the .singlet download paths (a model, not a measurement of a live link).

Constants (measured by the coordinator from this laptop to data.singlet.bio): TTFB 0.18-0.27 s per range request
(use 0.22), 32 MB/s single stream, 18 MB/s aggregate with 8 parallel streams. Link RTT 50 ms. Hash/CRC cost 7.2 ms/MB
(measured here: verify_bundle(hashes=True) = 1.35 s for 187 MB on a loaded CPU).
Usage: python -P tools/w4_reader_netmodel.py > results/w4_reader_netmodel.txt
"""
import math

TTFB = 0.22          # s per request (includes one RTT of the 50 ms)
RTT = 0.05
S1 = 32.0            # MB/s single stream (observed ceiling)
AGG8_OBS = 18.0      # MB/s with 8 parallel streams, observed on this laptop
VERIFY = 7.2e-3      # s per MB for zip CRC-32 + SHA-256 on download (today, hashes=True)
CONN = 2 * RTT       # TCP + TLS1.3 handshake


def ramp_loss(rate_mb_s):
    """Seconds lost to TCP slow start before a stream reaches `rate` (initcwnd 10 segments, doubling per RTT)."""
    bdp = rate_mb_s * 1e6 * RTT
    n = max(0.0, math.log2(max(bdp / 14480.0, 1.0)))
    return 0.5 * n * RTT


def stream_time(mb, link_mb_s, streams=1, agg_cap=None, setup=True):
    """One logical transfer of `mb` MB over `streams` parallel connections."""
    rate = min(link_mb_s * 0.95, S1 * streams)
    if streams > 1 and agg_cap is not None:
        rate = min(rate, agg_cap)
    t = (CONN if setup else 0) + TTFB + ramp_loss(rate / streams) + mb / rate
    return t, rate


links = {'100 Mbit/s': 12.5, '1 Gbit/s': 125.0}
# Scenario A: one 10 MB sample (gene-level members: exon + intron + calls) inside a 187 MB, 14-sample bundle (GSE145370:
# 9.6 MB needed of 187 MB). Scenario B: a 500 MB study of which a gene-level load of all samples needs 54% (atlas median).
scen = {
    'A: one sample, 9.6 MB needed of a 187 MB bundle (4 ranges: exon+intron adjacent, calls, vocab, manifest)': dict(need=9.6, bundle=187.0, members=4),
    'B: whole 500 MB study, gene-level (54% = 270 MB needed)': dict(need=270.0, bundle=500.0, members=2 * 20 + 1),
}
print('assumptions: TTFB %.2f s/request, single stream %.0f MB/s, 8-stream aggregate %.0f MB/s (observed on this laptop), RTT %.0f ms'
      % (TTFB, S1, AGG8_OBS, RTT * 1e3))
print('today = urllib GET of the whole bundle, one stream, then SHA-256 + CRC-32 of the whole file (%.1f ms/MB)' % (VERIFY * 1e3))
print('index = member ranges via the service index (1 ETag/size check + one ranged GET per member, pipelined on one connection)')
print('par8  = same ranges on 8 connections. "par8 ideal" = streams scale to 95% of the link (projection, NOT observed here)')
print()
rows = []
for sname, s in scen.items():
    print(sname)
    print('%-12s %-34s %9s %9s' % ('link', 'path', 'seconds', 'MB/s'))
    for lname, lmb in links.items():
        # today
        t, r = stream_time(s['bundle'], lmb, 1)
        t_today = t + s['bundle'] * VERIFY
        rows.append((sname, lname, 'today: whole bundle, 1 stream + verify all', t_today, s['bundle'] / (t_today)))
        # index, sequential on one connection: ETag check + one TTFB per member (pipelining overlaps RTT but not server TTFB)
        t_idx = CONN + TTFB + s['members'] * TTFB + s['need'] / min(lmb * 0.95, S1) + ramp_loss(min(lmb * 0.95, S1)) + s['need'] * VERIFY * 0.5
        rows.append((sname, lname, 'index: member ranges, 1 connection', t_idx, s['need'] / t_idx))
        # index + parallel (8 conns): requests overlap, aggregate observed cap
        for label, cap in (('index + par8 (observed 18 MB/s agg)', AGG8_OBS), ('index + par8 (ideal scaling)', None)):
            rate = min(lmb * 0.95, S1 * 8) if cap is None else min(lmb * 0.95, cap)
            waves = math.ceil(s['members'] / 8)
            t_p = CONN + TTFB + waves * TTFB + ramp_loss(rate / 8) + s['need'] / rate + s['need'] * VERIFY / 8
            rows.append((sname, lname, label, t_p, s['need'] / t_p))
        # single big range for the contiguous bytes? not applicable: members are interleaved with unneeded ones.
    for r in rows:
        if r[0] == sname:
            print('%-12s %-34s %9.2f %9.1f' % (r[1], r[2], r[3], r[4]))
    print()
with open('results/w4_reader_netmodel.tsv', 'w') as f:
    f.write('scenario\tlink\tpath\tseconds\teffective_MBps_of_needed_bytes\n')
    for r in rows:
        f.write('%s\t%s\t%s\t%.3f\t%.2f\n' % r)
