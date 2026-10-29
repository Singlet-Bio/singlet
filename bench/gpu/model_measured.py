"""Re-run throughput_model.py with the W11 M1/M2/M10 measurements substituted (exploration.md section 12).

    python bench/gpu/model_measured.py

Substitutions (everything else is unchanged from throughput_model.py):
  RS_CLEAN   23e9 -> 28.2e9   (M1: 32 B gather, lower of the 32 GB and 48 GB medians)
  d (3')     0.47 -> 0.474    (M2: pbmc1k_v3, the anchor sample; the spread over 3' samples is reported as a sensitivity)
  d (5')     0.50 -> 0.31     (M2: pbmc5p_v1 0.306, GSM6470208 0.303)
  W          unmeasured -> 2.3 core-s per M pairs (M10 upper bound: export + non-host residual; 1.26 export only)
"""
import throughput_model as m

RS_MEAS, D3, D5 = 28.2e9, 0.474, 0.31
W_PER_M_UP, W_PER_M_EXPORT = 2.295, 1.26


def gpu_e2e(chem, eta, wave):
    rate, _ = m.mapping_rate(chem, "all", eta, wave)
    other = m.other_stages(chem, eta)
    inv = 1 / rate + sum(1 / v for k, v in other.items() if k.startswith(("dedup", "barcode", "expand", "pileup", "non-host")))
    return rate, 1 / inv


def report(tag, rs, d3, d5):
    m.RS_CLEAN = rs
    m.CHEM["3p"]["d"], m.CHEM["5p"]["d"] = d3, d5
    print(f"\n##### {tag}: RS_CLEAN={rs/1e9:.1f} G/s  d3={d3}  d5={d5}")
    res = {}
    for sname, (eta, wave) in m.SCEN.items():
        for chem in ("3p", "5p"):
            r, e = gpu_e2e(chem, eta, wave)
            res[(sname, chem)] = (r / 1e6, e / 1e6)
            print(f"  {sname:12s} {chem}: mapping {r/1e6:6.1f} M/s   GPU stages summed {e/1e6:6.1f} M/s")
    return res


if __name__ == "__main__":
    base = report("original model", 23e9, 0.47, 0.50)
    meas = report("measured (M1, M2)", RS_MEAS, D3, D5)
    print("\n=== sensitivity of 3' base to d (measured RS_CLEAN)")
    for d in (0.30, 0.37, 0.43, 0.474, 0.515, 0.658):
        m.RS_CLEAN = RS_MEAS; m.CHEM["3p"]["d"] = d
        r, e = gpu_e2e("3p", 0.35, 0.60)
        print(f"  d3={d:5.3f}: mapping {r/1e6:6.1f}  GPU summed {e/1e6:6.1f} M pairs/s")
    m.CHEM["3p"]["d"] = D3
    print("\n=== end to end per GPU (base): min(GPU stages summed, host = 26 cores / W x pairs)")
    for chem in ("3p", "5p"):
        m.RS_CLEAN = RS_MEAS; m.CHEM["3p"]["d"], m.CHEM["5p"]["d"] = D3, D5
        _, g = gpu_e2e(chem, 0.35, 0.60)
        for label, w in (("W upper (export + non-host residual)", W_PER_M_UP), ("W export only", W_PER_M_EXPORT)):
            host = 26 / w
            print(f"  {chem} {label:38s}: GPU {g/1e6:5.1f}  host {host:5.1f}  -> e2e {min(g/1e6, host):5.1f} M pairs/s  (x node 1-1.3 M/s: {min(g/1e6, host)/1.3:.0f}-{min(g/1e6, host)/1.0:.0f})")
    for rate in (10e6, 20e6, 40e6):
        print(f"  campaign 20.6 T pairs, 6 GPUs at {rate/1e6:.0f} M/s: {20.6e12/(rate*6)/86400:.1f} days")
