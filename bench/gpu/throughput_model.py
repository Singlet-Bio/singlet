"""Throughput model for the GPU-native Singlet engine (docs/gpu-engine/exploration.md, section 5).

    python bench/gpu/throughput_model.py            # tables for the three scenarios
    python bench/gpu/throughput_model.py --json     # same numbers as JSON

This is a MODEL, not a measurement. Every input is either:
  - measured on the CPU engine (audit-2026-10-27 section 3, PR #41, SeqCache.h, format-1fq.md section 11);
  - a public hardware figure (NVIDIA datasheet, cuCollections blog, Onesweep paper);
  - or an ASSUMPTION, named and tagged with the microbenchmark (Mx) that confirms or kills it.

Unit of memory work: RS = one random 32-byte HBM sector. Contiguous traffic is counted in bytes.
"""

import json
import math
import sys

# ---------------------------------------------------------------- hardware (H100 NVL)
HBM_PEAK = 3.9e12          # B/s, NVIDIA datasheet
SEQ_EFF = 0.60             # ASSUMPTION: achieved fraction of peak on sort/scan kernels (Onesweep reaches ~0.6-0.7 on A100)
# Clean random-sector rate. Derived: cuco static_map find on H100 SXM = 16.8 G finds/s (NVIDIA blog), ~1.2 sectors per
# find at 50% load -> ~20 G RS/s at 3.35 TB/s; scaled to 3.9 TB/s -> ~23 G RS/s. ASSUMPTION confirmed by M1.
RS_CLEAN = 23e9
PCIE = 50e9                # B/s per direction, pinned; 53.8 GB/s measured on H100 HGX (bandwidthTest)
INT_OPS = 132 * 64 * 1.785e9   # INT32 lane-ops/s (64 INT32 lanes/SM)

# ---------------------------------------------------------------- workload (per chemistry)
# d: distinct R2 sequences / reads in one sample. 3': 1 - 0.53 (SeqCache.h, PBMC 1k v3). 5': ASSUMPTION (M2).
# heavy: share of reads in the too-short/repeat class and its CPU cost relative to a normal read (PR #41):
#   3': 4.1% of reads, 19% of mapping -> (19/4.1)/(81/95.9) = 5.5x ; 5': 7.4%, 55% -> (55/7.4)/(45/92.6) = 15.3x
CHEM = {
    "3p": dict(d=0.47, heavy=0.041, k=5.5, unmapped=0.08),
    "5p": dict(d=0.50, heavy=0.074, k=15.3, unmapped=0.10),
}

# Normal-read mix (ASSUMPTION, M3/M4 dumps measure it): perfect / one mismatch / >=2 mismatches or clipped.
MIX = dict(perfect=0.40, mm1=0.35, mm2=0.25)
# Seeds per read in STAR's search order (seedSearchStartLmax=50 on a 90-nt read -> Nstart=2, both directions):
#   perfect: dir0 full-length + dir0 from 45 + dir1 from 45                          = 3 seeds, 1 needs a real search
#   1 mm   : each direction: run to mismatch, restart at mismatch, rest; + istart=1  = 7 seeds, 3 need a real search
#   >=2 mm : ~9 seeds, ~4.5 need a real search
SEEDS = dict(perfect=(3, 1.0), mm1=(7, 3.0), mm2=(9, 4.5))  # (seeds, seeds that cannot be locus-propagated)

RS_FULL = 37.0     # SAi (2) + ~8 binary-search steps x (SA 1.125 + genome 1.125) + findMultRange 2 x ~3 steps -> ~35-37
RS_FULL_SBK = 24.0 # with bucket sample keys: SAi 2 + sample block 2 + ~4 final steps x 2.25 + multRange ~11
RS_CERT = 2.0      # U-array byte (1) + verify compare (genome words, usually already in L1 from the previous seed) (~1)
RS_LOCATE = 6.0    # locator: ~5 minimizer probes, independent (one latency), + verify
RS_PER_SEED_WIN = 2.0   # SA range read for window creation/assignment (contiguous) + chrBin
RS_STITCH = 16.0   # genome around 1-2 windows, sjdb lookups, extension
RS_ANNOT = 10.0    # gene/transcript model, junction table, SNP panel, chrM: per distinct alignment
RS_OUT = 2.0

INSTR_PER_READ = {"port": 60e3, "new": 30e3}   # ASSUMPTION: thread-instructions per distinct read (stitch dominated)


def normal_read_rs(design):
    """Random sectors for one normal (non-heavy) distinct read under a design."""
    total = 0.0
    for cls, w in MIX.items():
        n_seeds, n_hard = SEEDS[cls]
        if design in ("port", "wave"):
            seeds = n_seeds * RS_FULL
        else:
            hard = n_hard
            if design in ("cert+reuse", "cert+reuse+sbk", "all"):
                hard *= 0.70                      # ASSUMPTION: 30% of hard seed searches reused via LCP (M2/M4)
            full_rs = RS_FULL_SBK if design in ("cert+reuse+sbk", "all") else RS_FULL
            first = 0.0
            if design == "all":                   # locator replaces the first search for ~70% of reads (M12)
                located = 0.70 * min(1.0, hard)
                first = located * RS_LOCATE
                hard -= located
            seeds = hard * full_rs + first + (n_seeds - n_hard) * RS_CERT
        total += w * (seeds + n_seeds * RS_PER_SEED_WIN + RS_STITCH + RS_ANNOT + RS_OUT)
    return total


def divergence_eff(chem, design, wave_eff):
    """SIMT efficiency. Megakernel: a warp runs as long as its slowest read. Wavefront: classes separated."""
    if design != "port":
        return wave_eff
    h, k = CHEM[chem]["heavy"], CHEM[chem]["k"]
    p_heavy_in_warp = 1 - (1 - h) ** 32
    mean = (1 - h) + h * k
    # slowest of 32 normal reads assumed 3x the mean normal read (STAR per-read cost is long-tailed)
    return mean / (p_heavy_in_warp * k + (1 - p_heavy_in_warp) * 3.0)


def mapping_rate(chem, design, eta_mem, wave_eff, dedup=True):
    c = CHEM[chem]
    n_rs = normal_read_rs(design)
    port_rs = normal_read_rs("port")
    heavy_rs = c["k"] * port_rs * (1.0 if design == "port" else 0.5)   # heavy kernel: contiguous loci scans
    per_distinct = (1 - c["heavy"]) * n_rs + c["heavy"] * heavy_rs
    per_pair = per_distinct * (c["d"] if dedup else 1.0)
    rs_rate = RS_CLEAN * eta_mem * divergence_eff(chem, design, wave_eff)
    instr = INSTR_PER_READ["port" if design == "port" else "new"]
    instr_rate = INT_OPS * 0.3 / (instr * (c["d"] if dedup else 1.0))   # 30% SIMT x issue efficiency
    return min(rs_rate / per_pair, instr_rate), per_pair


def other_stages(chem, eta_mem):
    """Pairs/s bound of every non-mapping stage (GPU and host)."""
    c = CHEM[chem]
    seq = HBM_PEAK * SEQ_EFF
    rs = RS_CLEAN * eta_mem
    st = {}
    # GPU
    st["H2D (53 B/pair incl. binned quality)"] = PCIE / 53
    st["dedup: hash + 40-bit radix sort + compare"] = seq / (24 + 8 + 5 * 12 * 2 + 24 + 8 + c["d"] * 192)
    st["barcode: WL hash + 1MM (6% x 48 probes)"] = rs / (1 + 0.06 * 48)
    st["expand + (cell,gene,UMI) sort + UMI 1MM"] = 1 / (272 / seq + 2 / rs + 32 / seq)
    st["pileup tuples (junction, SNP, chrM) sort"] = seq / 200
    st["non-host: minimizers of unmapped distinct"] = rs / (c["unmapped"] * c["d"] * 10 * 31)
    # host
    st["host: .1fq decode, 3 threads (67 M/s/thread)"] = 3 * 67e6
    return st


def host_callers(w_core_s, cores_per_gpu=26, sample_pairs=66e6):
    return cores_per_gpu / w_core_s * sample_pairs


DESIGNS = [
    ("port", "STAR port, thread-per-read megakernel"),
    ("wave", "+ wavefront kernels, compact state"),
    ("cert", "+ U-array certificates (locus propagation)"),
    ("cert+reuse", "+ prefix-sorted seed reuse (LCP)"),
    ("cert+reuse+sbk", "+ SA bucket sample keys"),
    ("all", "+ locator for the first seed"),
]
SCEN = {"pessimistic": (0.25, 0.45), "base": (0.35, 0.60), "optimistic": (0.50, 0.70)}


def main():
    out = {}
    for sname, (eta, wave) in SCEN.items():
        out[sname] = {}
        for chem in CHEM:
            rows = {}
            r0, p0 = mapping_rate(chem, "port", eta, wave, dedup=False)
            rows["port, no dedup"] = dict(pairs_s=r0, rs_per_pair=p0)
            for key, _ in DESIGNS:
                r, p = mapping_rate(chem, key, eta, wave)
                rows[key] = dict(pairs_s=r, rs_per_pair=p)
            out[sname][chem] = dict(mapping=rows, other=other_stages(chem, eta))
    if "--json" in sys.argv:
        print(json.dumps(out, indent=1))
        return
    for sname, (eta, wave) in SCEN.items():
        print(f"\n=== scenario {sname}: eta_mem={eta}  wavefront SIMT eff={wave}  RS_clean={RS_CLEAN/1e9:.0f} G/s")
        for chem in CHEM:
            print(f"  -- {chem}: mapping stage (distinct reads), M pairs/s and RS per input pair")
            for k, v in out[sname][chem]["mapping"].items():
                print(f"     {k:18s} {v['pairs_s']/1e6:8.1f} M/s   {v['rs_per_pair']:7.1f} RS/pair")
            print(f"  -- {chem}: other stages, M pairs/s")
            for k, v in out[sname][chem]["other"].items():
                print(f"     {k:46s} {v/1e6:10.0f}")
    # Atlas mean depth from the CPU cost model: 316k core-h = 70k samples x 4 x (300 + 12.79 U) s -> U ~ 294 M pairs.
    print("\n=== host callers + export (CPU), M pairs/s per GPU with 26 cores per GPU (W = core-s per sample)")
    for w in (30, 60, 120, 240, 480):
        print(f"     W = {w:3d}: 66 M-pair sample {host_callers(w)/1e6:6.1f} M/s ; atlas-mean 294 M-pair sample "
              f"{host_callers(w, sample_pairs=294e6)/1e6:6.1f} M/s")
    total_pairs = 70e3 * 294   # M pairs: 70k samples x 294 M pairs (cost-model mean)
    print(f"\n=== campaign: 70k samples x 294 M pairs (cost-model mean) -> {total_pairs/1e6:.1f} T read pairs")
    for rate in (10e6, 20e6, 40e6):
        days = total_pairs * 1e6 / (rate * 6) / 86400
        print(f"     6 GPUs at {rate/1e6:.0f} M pairs/s each: {days:5.1f} days")


if __name__ == "__main__":
    main()
