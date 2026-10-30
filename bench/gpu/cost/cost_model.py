"""W11 cost case: GPU node (2x H100 NVL) vs CPU node, per sample and per node, from measured inputs.

    python cost_model.py [--pack-efficiency X] > ../results/cost_case.json

Every number in the output carries a label:
  measured  - read from a job's output (job ids in `sources`)
  derived   - arithmetic on measured numbers only
  projected - uses a measured rate on a sample or setting other than the one it is applied to
  modelled  - uses an outside price or power figure (rates.json) or an assumption stated in `assumptions`

Inputs (all real data):
  CPU baseline: prod/campaign c4a1be38 sources (binary singlet-w3, bb51b063, 3'/5' PGO; source-identical), campaign command line,
    --threads 4, one sample per job on Clipper short-cpu (Xeon Gold 6230R); /usr/bin/time user+sys (runs/single/<sample>/time.txt).
  CPU credit: the per-node non-host index load (a persistent service loads it once per node), per sample from section 12.9.
  GPU node: M5 rate per GPU (section 12.12), distinct fraction d per sample (M2, section 12.2), host work W per sample after the
    pileup (section 12.9), ingest 0.085 core-s per M pairs (section 12.10); pileup + M6 GPU time is < 1% and is included as 0.1 s.
"""
import json
import sys

# ---- measured: CPU baseline (core-s = user + sys; wall), jobs 789532-789535, 789548, 789560 (array 789532 and reruns)
CPU = {  # sample: (M read pairs, user s, sys s, wall s)
    "GSM4006843": (66.7, 2328.79, 269.24, 16 * 60 + 12.32),
    "GSM5411066": (69.6, 1374.86, 264.25, 11 * 60 + 30.60),
    "GSM7901264": (100.3, 3067.82, 318.84, 19 * 60 + 33.34),
    "GSM7036403": (111.6, 2856.44, 348.98, 21 * 60 + 0.97),
    "GSM6470208": (121.8, 4655.83, 625.68, 34 * 60 + 57.85),
    "GSM7656149": (263.4, 11066.58, 594.56, 68 * 60 + 49),
}
CPU_NODE_CORES = 52            # measured: Clipper c010 (2x Xeon Gold 6230R, 26 cores each, no SMT)
# ---- measured (section 12.9, 28969b87 timer build): non-host index load, core-s per sample
NONHOST_LOAD = {"GSM4006843": 33.2, "GSM5411066": 28.6, "GSM7901264": 28.0, "GSM7036403": 53.3, "GSM6470208": 46.2, "GSM7656149": 42.7}
# ---- measured (section 12.9): host work after the pileup, W core-s per sample
W_HOST = {"GSM4006843": 174.9, "GSM5411066": 92.2, "GSM7901264": 121.2, "GSM7036403": 243.6, "GSM6470208": 271.4, "GSM7656149": 694.8}
# ---- measured (section 12.2, M2): distinct-sequence fraction d per sample; None = not measured (pooled value used, projected)
D_FRAC = {"GSM4006843": None, "GSM5411066": 0.392, "GSM7901264": None, "GSM7036403": 0.298, "GSM6470208": 0.303, "GSM7656149": 0.515}
D_POOLED_3P = 0.474            # pbmc1k_v3 (3' v3, measured) used where d is not measured
# ---- measured (section 12.12): M5 on pbmc1k_v3, M distinct reads/s per GPU
M5_RATE = {"whole_sample_job789771": 1.86, "best_2M_set_job789830": 2.40}
INGEST = 0.085                 # core-s per M pairs (section 12.10)
HEAVY = 2.2 * 24 / 66.6        # M5's host lane: 2.2 s on 24 threads per 66.6 M pairs (whole sample, job 789771): an upper bound
GPU_NODE = {"gpus": 2, "host_cores": 55}
PILEUP_M6_S = 0.1              # GPU s per sample, annotation + pileup + M6 (sections 12.8, 12.13): 40-63 ms + ~40 ms

pack_eff = 1.0                 # CPU node: throughput of packed samples relative to the sum of single-sample core-s
for i, a in enumerate(sys.argv):
    if a == "--pack-efficiency":
        pack_eff = float(sys.argv[i + 1])

rates = json.load(open(__file__.replace("cost_model.py", "rates.json"), encoding="utf-8"))
PRICE = {   # modelled (rates.json, accessed 2026-10-27)
    "gpu_hour": {"central": 3.99, "low": 2.59, "high": 6.88},      # $/H100-GPU-hour on-demand (Lambda; RunPod NVL community; AWS p5)
    "core_hour": {"central": 0.05, "low": 0.0409, "high": 0.0893},  # $/physical core-hour (no-SMT HPC; Azure HBv4; AWS c7i SMT)
}
ACCESS = {"anvil_cpu_credit_per_core_hour": 1.0, "anvil_gpu_credit_per_gpu_hour_derived": 67.0}
POWER = {"gpu_node_w": 1400.0, "cpu_node_w_per_core": 650.0 / 192, "pue": 1.56, "usd_per_kwh": 0.1447,
         # the measured CPU speed is the 6230R's: its own power is the consistent pairing (2 x 150 W TDP + ~150 W platform, estimate)
         "cpu_6230r_node_w": 450.0}

out = {"labels": "see module docstring", "assumptions": {
    "cpu_pack_efficiency": {"value": pack_eff, "label": "measured" if pack_eff != 1.0 else "modelled",
                            "note": "1.0 = a node's cores are as productive packed as in single-sample runs; replace with the packing run"},
    "gpu_node_host_cores": "55 of 56 (one for orchestration)",
    "gpu_node_cloud_price": "2 x the per-GPU price (cloud GPU prices include the host's vCPUs and RAM)",
    "cpu_cores": "Clipper's Xeon Gold 6230R (2019, 2.1 GHz); cloud core prices are for newer cores, so the CPU side is probably "
                 "1.5-2x cheaper in the cloud than modelled here: see cpu_speed_sensitivity",
    "cpu_power": "650 W per dual EPYC 9654 server (192 cores) scaled per core; GPU node 1,400 W is an estimate (rates.json)",
}, "samples": {}, "pooled": {}}

tot = {"pairs": 0, "cpu_cs": 0, "cpu_cs_cred": 0, "gpu_s": {k: 0 for k in M5_RATE}, "host_cs": 0}
for s, (mp, us, sy, wall) in CPU.items():
    cs = us + sy
    cred = cs - NONHOST_LOAD[s]
    d = D_FRAC[s] if D_FRAC[s] is not None else D_POOLED_3P
    rec = {
        "M_read_pairs": {"value": mp, "label": "measured"},
        "cpu_core_s": {"value": round(cs, 1), "label": "measured"},
        "cpu_wall_s_4threads": {"value": round(wall, 1), "label": "measured"},
        "cpu_core_s_per_M_pairs": {"value": round(cs / mp, 2), "label": "derived"},
        "cpu_core_s_credited": {"value": round(cred, 1), "label": "derived",
                                "note": "minus the non-host index load (section 12.9 timer build, labelled measured there)"},
        "distinct_fraction": {"value": d, "label": "measured" if D_FRAC[s] is not None else "projected (pbmc1k_v3's d)"},
    }
    for k, r in M5_RATE.items():
        gs = d * mp / r + PILEUP_M6_S
        rec[f"gpu_s_per_sample_{k}"] = {"value": round(gs, 2), "label": "projected",
                                        "note": "M5 rate measured on pbmc1k_v3, applied to this sample's distinct sequences"}
        tot["gpu_s"][k] += gs
    hc = W_HOST[s] + (INGEST + HEAVY) * mp
    rec["gpu_node_host_core_s"] = {"value": round(hc, 1), "label": "derived", "note": "W after the pileup + ingest + M5 host lane (pbmc1k_v3 rate)"}
    out["samples"][s] = rec
    tot["pairs"] += mp; tot["cpu_cs"] += cs; tot["cpu_cs_cred"] += cred; tot["host_cs"] += hc

n = len(CPU)
P = out["pooled"]
P["samples"] = n
P["M_read_pairs_total"] = {"value": round(tot["pairs"], 1), "label": "measured"}
cpu_cs_per_M = tot["cpu_cs_cred"] / tot["pairs"]
P["cpu_core_s_per_M_pairs"] = {"value": round(tot["cpu_cs"] / tot["pairs"], 2), "label": "derived"}
P["cpu_core_s_per_M_pairs_credited"] = {"value": round(cpu_cs_per_M, 2), "label": "derived"}
cpu_node_rate = CPU_NODE_CORES * pack_eff / cpu_cs_per_M          # M pairs/s per CPU node
P["cpu_node_M_pairs_per_s"] = {"value": round(cpu_node_rate, 3), "label": "derived" if pack_eff != 1.0 else "modelled (packing 1.0)"}
avg_pairs = tot["pairs"] / n
P["cpu_node_samples_per_hour"] = {"value": round(cpu_node_rate * 3600 / avg_pairs, 1), "label": P["cpu_node_M_pairs_per_s"]["label"]}
P["cpu_core_hours_per_sample"] = {"value": round(tot["cpu_cs_cred"] / n / 3600, 3), "label": "derived"}
P["cpu_node_hours_per_sample"] = {"value": round(avg_pairs / cpu_node_rate / 3600, 4), "label": P["cpu_node_M_pairs_per_s"]["label"]}

host_cs_per_M = tot["host_cs"] / tot["pairs"]
host_bound = GPU_NODE["host_cores"] / host_cs_per_M
P["gpu_node_host_bound_M_pairs_per_s"] = {"value": round(host_bound, 2), "label": "derived", "note": "step 0 (today's host callers)"}
P["scenarios"] = {}
SCEN = {k: tot["gpu_s"][k] for k in M5_RATE}
SCEN["target_m5_at_host_bound"] = GPU_NODE["gpus"] / host_bound * tot["pairs"]   # not measured: what M5 must reach (section 12.15)
for k in SCEN:
    tot["gpu_s"][k] = SCEN[k]
    gpu_s_per_M = tot["gpu_s"][k] / tot["pairs"]
    gpu_bound = GPU_NODE["gpus"] / gpu_s_per_M
    node = min(gpu_bound, host_bound)
    gpu_h_per_sample = tot["gpu_s"][k] / n / 3600
    sc = {
        "gpu_node_gpu_bound_M_pairs_per_s": {"value": round(gpu_bound, 2), "label": "projected" if k in M5_RATE else "target"},
        "gpu_node_M_pairs_per_s": {"value": round(node, 2), "label": "projected"},
        "gpu_node_samples_per_hour": {"value": round(node * 3600 / avg_pairs, 1), "label": "projected"},
        "gpu_node_hours_per_sample": {"value": round(avg_pairs / node / 3600, 5), "label": "projected"},
        "gpu_hours_per_sample": {"value": round(gpu_h_per_sample, 5), "label": "projected"},
        "gpu_node_host_core_hours_per_sample": {"value": round(tot["host_cs"] / n / 3600, 4), "label": "derived"},
        "cpu_nodes_offset_per_gpu_node": {"value": round(node / cpu_node_rate, 2), "label": "projected"},
    }
    # $ per sample: cloud on-demand
    usd = {}
    for lvl in ("central", "low", "high"):
        gpu_node_hour = GPU_NODE["gpus"] * PRICE["gpu_hour"][lvl]
        cpu_core_hour = PRICE["core_hour"][lvl]
        g = gpu_node_hour * avg_pairs / node / 3600
        c = cpu_core_hour * (tot["cpu_cs_cred"] / n / 3600) / pack_eff
        usd[lvl] = {"gpu_node_usd_per_sample": round(g, 4), "cpu_usd_per_sample": round(c, 4), "cpu_over_gpu": round(c / g, 2)}
    sc["usd_per_sample_cloud_on_demand"] = {"value": usd, "label": "modelled"}
    # CPU cores 2x faster than the 6230R (sensitivity)
    c2 = PRICE["core_hour"]["central"] * (tot["cpu_cs_cred"] / n / 3600) / pack_eff / 2
    sc["cpu_speed_sensitivity"] = {"value": {"cpu_usd_per_sample_if_cores_2x_faster": round(c2, 4),
                                             "cpu_over_gpu": round(c2 / usd["central"]["gpu_node_usd_per_sample"], 2)}, "label": "modelled"}
    # ACCESS credits (Anvil): GPU-hours x 67 (host cores included in the GPU allocation) vs core-hours x 1
    sc["access_credits_per_sample"] = {"value": {
        "gpu": round(gpu_h_per_sample * ACCESS["anvil_gpu_credit_per_gpu_hour_derived"], 3),
        "cpu": round(tot["cpu_cs_cred"] / n / 3600 * ACCESS["anvil_cpu_credit_per_core_hour"] / pack_eff, 3)},
        "label": "modelled", "note": "Anvil GPU rate derived from Purdue's example, not published (rates.json)"}
    # energy per sample (kWh at the wall x PUE)
    e_gpu = POWER["gpu_node_w"] * avg_pairs / node / 3600 / 1000 * POWER["pue"]
    e_cpu = POWER["cpu_node_w_per_core"] * (tot["cpu_cs_cred"] / n / 3600) / pack_eff / 1000 * POWER["pue"]
    e_cpu2 = POWER["cpu_6230r_node_w"] / CPU_NODE_CORES * (tot["cpu_cs_cred"] / n / 3600) / pack_eff / 1000 * POWER["pue"]
    sc["energy_kwh_per_sample"] = {"value": {"gpu_node": round(e_gpu, 4),
                                             "cpu_epyc_power_per_core": round(e_cpu, 4), "cpu_over_gpu_epyc_power": round(e_cpu / e_gpu, 2),
                                             "cpu_6230r_power": round(e_cpu2, 4), "cpu_over_gpu_6230r_power": round(e_cpu2 / e_gpu, 2),
                                             "usd_gpu": round(e_gpu * POWER["usd_per_kwh"], 4), "usd_cpu_6230r": round(e_cpu2 * POWER["usd_per_kwh"], 4)},
                                   "label": "modelled", "note": "no metered power for either node; both GPU and 6230R node figures are estimates"}
    P["scenarios"][k] = sc
out["sources"] = {
    "cpu_baseline": "bench/gpu/cost/run_single.sbatch; $W/jobs/w11/cost/runs/single/<sample>/time.txt",
    "nonhost_load_and_W": "docs/gpu-engine/exploration.md 12.9 (bench/gpu/results/m10b_timers.json)",
    "distinct_fraction": "exploration.md 12.2 (bench/gpu/results/m2_*.json)",
    "m5": "exploration.md 12.12 (bench/gpu/results/m5_pbmc1k_v3_2m*.json, e2e_pbmc1k_v3_m5.json)",
    "rates": "bench/gpu/cost/rates.json",
}
json.dump(out, sys.stdout, indent=1)
print()
