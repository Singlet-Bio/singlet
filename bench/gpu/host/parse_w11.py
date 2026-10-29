#!/usr/bin/env python3
"""Parse [w11t]/[w11n] lines (+ monitor.tsv) of the W11 host-time runs into m10b_timers.json and markdown tables.

usage: parse_w11.py RUNS_DIR OUT_JSON   (RUNS_DIR/<GSM>/{w11.log|run.log,monitor.tsv,small/pileup_stats.json})
Core-seconds: cpu_s is the process CPU delta, thr_cpu_s the calling thread's. A section that runs alone (the serial part of
export, the stages) is exact in cpu_s. Sections inside the concurrent phase 3 (writers + demux thread) use thr_cpu_s.
"""
import json, os, re, sys, collections

LINE = re.compile(r'^\[w11t\] (\S+) wall_s=([\d.]+) cpu_s=([\d.]+) thr_cpu_s=([\d.]+) t0=([\d.]+) tid=(\d+)')
NOTE = re.compile(r'^\[w11n\] (.*)$')
READS = {"GSM4006843": 66.7, "GSM5411066": 69.6, "GSM7901264": 100.3, "GSM7036403": 111.6, "GSM6470208": 121.8, "GSM7656149": 263.4}


def parse_log(path):
    sec = collections.OrderedDict()
    notes = []
    for ln in open(path, errors="replace"):
        m = LINE.match(ln)
        if m:
            n, w, c, t, t0, tid = m.groups()
            sec.setdefault(n, []).append(dict(wall=float(w), cpu=float(c), thr=float(t), t0=float(t0), tid=int(tid)))
            continue
        m = NOTE.match(ln)
        if m:
            notes.append(m.group(1))
    return sec, notes


def tot(sec, name, key):
    return sum(r[key] for r in sec.get(name, []))


def cg_cpu_at(mon, t_epoch):
    """cgroup cpu_s at an epoch time, linear interpolation of monitor.tsv."""
    pts = [(a, b) for a, b in mon]
    if pts and pts[-1][0] < t_epoch <= pts[-1][0] + 8:
        return pts[-1][1]
    for (t1, c1), (t2, c2) in zip(pts, pts[1:]):
        if t1 <= t_epoch <= t2:
            return c1 + (c2 - c1) * (t_epoch - t1) / max(1e-9, (t2 - t1))
    return None


def main(runs, out):
    res = {}
    for gsm in sorted(os.listdir(runs)):
        d = os.path.join(runs, gsm)
        lp = os.path.join(d, "w11.log")
        if not os.path.exists(lp):
            lp = os.path.join(d, "run.log")
        if not os.path.exists(lp):
            continue
        sec, notes = parse_log(lp)
        if "main.total_including_teardown" not in sec:
            res[gsm] = {"status": "incomplete"}
            continue
        origin = next((float(n.split("=")[1]) for n in notes if n.startswith("origin_epoch=")), None)
        mon = []
        mp = os.path.join(d, "monitor.tsv")
        if os.path.exists(mp):
            for ln in open(mp).read().splitlines()[1:]:
                f = ln.split("\t")
                if len(f) >= 4:
                    try:
                        mon.append((float(f[0]), float(f[3])))
                    except ValueError:
                        pass
        reads = READS.get(gsm)
        st = {}
        sp = os.path.join(d, "small", "pileup_stats.json")
        if os.path.exists(sp):
            try:
                st = json.load(open(sp))
            except Exception:
                pass
        smj = {}
        jp = os.path.join(d, "small", "summary.json")
        if os.path.exists(jp):
            try:
                smj = json.load(open(jp))
            except Exception:
                pass
        sections = {n: dict(count=len(v), wall_s=round(sum(r["wall"] for r in v), 4), cpu_s=round(sum(r["cpu"] for r in v), 4),
                            thr_cpu_s=round(sum(r["thr"] for r in v), 4), t0_first=v[0]["t0"]) for n, v in sec.items()}
        e = lambda n, k="cpu": tot(sec, n, k)
        # --- ledger of the export window (exact: export.core cpu is the process delta of the whole export) ---
        serial = ["export.csc_conversion", "cell_qc.compute", "sex_calling", "cell_cycle", "read_stats", "ancestry", "saturation",
                  "saturation_curve", "cell_calling", "ambient", "doublet", "cell_qc.star_read_stats", "cell_qc.write_tsv",
                  "mt.rev_csc_and_reference", "mt.call_and_write_variants"]
        post_join = ["demux.write_assignments", "demux.aggregate_depths", "demux.write_vcfs_coverages", "mt_donor_outputs(G6)"]
        writers = [n for n in sec if n.startswith("write.") and n != "write.polya_total"] + \
            ["gene_agg.thread", "psi.thread_total", "write.polya_total"] + [n for n in sec if n.startswith("mt.cells_only")]
        writers = [n for n in dict.fromkeys(writers) if n in sec and not (n == "write.polya_counts")]
        # write.polya_counts is inside write.polya_total; write.<matrix> via write_matrix; psi/gene_agg threads separate
        wr_set = [n for n in writers if n not in ("write.polya_counts",)]
        led = collections.OrderedDict()
        for n in serial:
            if n in sec:
                led[n] = dict(core_s=round(e(n, "cpu"), 3), wall_s=round(e(n, "wall"), 3), kind="serial")
        for n in post_join:
            if n in sec:
                led[n] = dict(core_s=round(e(n, "cpu"), 3), wall_s=round(e(n, "wall"), 3), kind="serial-after-join")
        led["demux.thread (all phases, one thread)"] = dict(core_s=round(e("demux.thread", "thr"), 3),
                                                            wall_s=round(e("demux.thread", "wall"), 3), kind="concurrent thread")
        wsum = 0.0
        for n in wr_set:
            led["writer:" + n] = dict(core_s=round(e(n, "thr"), 3), wall_s=round(e(n, "wall"), 3), kind="concurrent thread")
            wsum += e(n, "thr")
        att = sum(v["core_s"] for v in led.values())
        exp_cpu = e("export.core", "cpu")
        led["UNATTRIBUTED (helper threads of writers/pz codec, section gaps)"] = dict(core_s=round(exp_cpu - att, 3), wall_s=None, kind="residual")
        # --- demux breakdown ---
        dmx = {}
        dmx["notes"] = [n for n in notes if n.split()[0] in ("demux_input", "demux_cov", "demux_fit", "autoK_input", "autoK_done", "demux_done",
                                                           "staged_explore", "staged_final", "doublet_stage")]
        dmx["total_thread_s"] = round(e("demux.thread", "thr"), 3)
        dmx["extract_covered_snps_cpu"] = round(e("demux.extract_covered_snps", "cpu"), 3)
        dmx["autoK_total_thr"] = round(e("autoK.total", "thr"), 3)
        dmx["autoK_total_cpu"] = round(e("autoK.total", "cpu"), 3)
        dmx["autoK_pca_thr_cpu"] = round(sum(e(f"autoK.pca.f{f}", "thr") for f in (0, 1)), 3)
        per_k = {}
        for n, v in sec.items():
            m = re.match(r'autoK\.K=(\d+)\.total$', n)
            if m:
                per_k[int(m.group(1))] = dict(wall_s=round(v[0]["wall"], 3), cpu_s=round(v[0]["cpu"], 3))
        dmx["autoK_per_K"] = per_k
        parts = collections.defaultdict(float)
        for n, v in sec.items():
            m = re.match(r'autoK\.K=\d+\.f\d\.(\w+)$', n)
            if m and m.group(1) != "fold_total":
                parts[m.group(1)] += sum(r["thr"] for r in v)
        dmx["autoK_fold_phase_thr_cpu_s"] = {k: round(v, 3) for k, v in parts.items()}
        rs = collections.defaultdict(float)
        for n, v in sec.items():
            m = re.match(r'vb_restart\.K=(\d+)\.kind=(\d)\.(\w+)', n)
            if m:
                rs["vb_restart." + m.group(3)] += sum(r["thr"] for r in v)
        dmx["vb_restart_thr_cpu_s_all_calls"] = {k: round(v, 3) for k, v in rs.items()}
        dmx["final_fit"] = {n: dict(wall_s=round(sum(r["wall"] for r in v), 3), cpu_s=round(sum(r["cpu"] for r in v), 3))
                            for n, v in sec.items() if n.startswith("demux.final_fit") or n.startswith("staged.")
                            or n in ("demux.pca_full.K=", "demux.split_merge") or n.startswith("demux.pca_full")}
        for n in ("demux.core_split_and_filter_copy", "demux.doublet_total", "demux.doublet_stage", "demux.doublet_refit_vb", "demux.back_assign"):
            if n in sec:
                dmx[n] = dict(wall_s=round(e(n, "wall"), 3), cpu_s=round(e(n, "cpu"), 3))
        # --- stage / gap ---
        gap = {n: dict(wall_s=round(e(n, "wall"), 3), cpu_s=round(e(n, "cpu"), 3), thr_cpu_s=round(e(n, "thr"), 3))
               for n in ["stage.star_wait_join", "stage.bam_index", "stage.cb_refresh", "stage.pileup", "stage.strand_check", "stage.bam_unlink",
                         "stage.nonhost_em_total", "nonhost.load_indices", "nonhost.load_host_bloom", "nonhost.read_unmapped_fastq",
                         "nonhost.classify", "nonhost.em", "nonhost.secondary_align", "nonhost.cell_matrix", "nonhost.write_tail",
                         "gap.pileup_end_to_export_start", "export.cfg_setup", "export.results_call", "export.core", "export.post_core",
                         "post_export.stats_json_block", "post_export.binary_sha256", "main.total_including_teardown"] if n in sec}
        # cgroup cross-check of the window pileup_end .. main end
        cg = None
        if origin and mon and "gap.pileup_end_to_export_start" in sec:
            t_a = origin + sec["gap.pileup_end_to_export_start"][0]["t0"]
            t_b = origin + sec["main.total_including_teardown"][0]["wall"]
            t_exp = origin + sec["export.core"][0]["t0"]
            ca, cb, ce_ = cg_cpu_at(mon, t_a), cg_cpu_at(mon, t_b), cg_cpu_at(mon, t_exp)
            if None not in (ca, cb, ce_):
                cg = dict(gap_cgroup_cpu_s=round(ce_ - ca, 1), export_to_end_cgroup_cpu_s=round(cb - ce_, 1))
        gap_core = e("gap.pileup_end_to_export_start", "cpu")
        post_core = e("export.post_core", "cpu") + e("post_export.stats_json_block", "cpu")
        tail_core = max(0.0, e("main.total_including_teardown", "cpu") - 0.0)  # whole-process CPU for reference
        r_pairs = reads
        W = dict(gap_core_s=round(gap_core, 2), export_core_s=round(exp_cpu, 2), post_export_core_s=round(post_core, 2),
                 export_cfg_setup_core_s=round(e("export.cfg_setup", "cpu"), 3),
                 W_total_core_s=round(gap_core + e("export.cfg_setup", "cpu") + exp_cpu + post_core, 2))
        if r_pairs:
            W["W_per_M_pairs"] = round(W["W_total_core_s"] / r_pairs, 3)
            W["export_per_M_pairs"] = round(exp_cpu / r_pairs, 3)
        res[gsm] = dict(status="ok", reads_M=reads, total_reads=st.get("total_reads"), cells_called=smj.get("estimated_cells"),
                        sections=sections, ledger=led, W=W, demux=dmx, stages=gap, cgroup=cg, notes=notes)
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, "samples:", [k for k, v in res.items() if v.get("status") == "ok"])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
