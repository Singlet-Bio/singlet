"""Command line: `python -m s1bench [--config FILE] [--set key=value ...] <stage> [options]`.

Stages, in order (see ../PLAN.md): env, fetch, canon, write, decode, ladder, floor, second, minibatch, eco, report.
`all` runs fetch..eco for the selected datasets (minibatch and eco only touch their own panels), then report.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import util


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="s1bench", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key (dots for nesting)")
    sub = ap.add_subparsers(dest="stage", required=True)

    def stage(name, help_, loader_panel=False):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--datasets", default="all", help="'all', ids separated by commas, or tier:S,M")
        p.add_argument("--extensions", action="store_true", help="include the out-of-domain datasets")
        p.add_argument("--force", action="store_true", help="redo work whose results exist")
        return p

    sub.add_parser("env", help="record hardware and package versions in results/env.json")
    stage("fetch", "download and checksum")
    stage("canon", "parse into the canonical matrix")
    p = stage("write", "write every format, measure size, check round trip")
    p.add_argument("--formats", default=None, help="comma-separated format ids (default: config formats)")
    p = stage("decode", "time file -> in-memory CSC")
    p.add_argument("--formats", default=None)
    p.add_argument("--sweep", action="store_true", help="all thread counts in threads.sweep")
    stage("ladder", "bits per non-zero, raw arrays to .s1")
    stage("floor", "cross-validated model code lengths and the sampling term")
    stage("second", "second-pass compression of .s1 files")
    p = stage("minibatch", "time to first minibatch and sustained throughput (loader panel)")
    p.add_argument("--tiers", default=None, help="comma-separated tier names (default: all)")
    stage("eco", "the Fig. 4 calls, timed")
    stage("access", "bytes, requests and index read per random minibatch, from each format's index (untimed)")
    p = stage("netsweep", "round-trip time, requests in flight and DataLoader workers (loader panel)")
    p.add_argument("--r2", action="store_true", help="the real object store (S1_R2_URL) instead of the emulated grids")
    stage("kernels", "one-core decode rate of textbook integer codes against the .s1 member (cpp/kernel_bench)")
    stage("entropy", "byte and stream entropy of each code's output and of the .s1 member, untimed (cpp/kernel_bench)")
    p = stage("writecost", "seconds and peak memory to write each format")
    p.add_argument("--formats", default=None)
    sub.add_parser("report", help="aggregate results into generated/ for the manuscript")
    stage("all", "every stage for the selected datasets, then report")
    # internal: one timed configuration in a fresh process
    p = sub.add_parser("decode-one")
    p.add_argument("unit"), p.add_argument("fmt"), p.add_argument("threads", type=int), p.add_argument("cold", type=int)
    p.add_argument("extra")
    p = sub.add_parser("minibatch-one")
    p.add_argument("unit"), p.add_argument("loader"), p.add_argument("src"), p.add_argument("seed", type=int)
    p.add_argument("threads", type=int)

    p = sub.add_parser("writecost-one")
    p.add_argument("unit"), p.add_argument("fmt"), p.add_argument("threads", type=int)

    a = ap.parse_args(argv)
    cfg = util.load_config(a.config, a.set)

    if a.stage == "env":
        print(util.write_env(cfg))
        return 0
    if a.stage == "report":
        from . import report

        return report.run(cfg)
    if a.stage == "decode-one":
        from . import decode

        print(json.dumps(decode._one(cfg, a.unit, a.fmt, a.threads, bool(a.cold), json.loads(a.extra))))
        return 0
    if a.stage == "minibatch-one":
        from . import minibatch

        print(json.dumps(minibatch.measure_one(cfg, a.unit, a.loader, a.src, a.seed, a.threads)))
        return 0

    if a.stage == "writecost-one":
        from . import writecost

        print(json.dumps(writecost.one(cfg, a.unit, a.fmt, a.threads)))
        return 0

    units = util.select_units(cfg, a.datasets, a.extensions)
    fmts = (a.formats.split(",") if getattr(a, "formats", None) else list(cfg["formats"]))
    stages = ["fetch", "canon", "write", "decode", "ladder", "floor", "second", "minibatch", "eco"] if a.stage == "all" else [a.stage]
    for s in stages:
        util.log(f"stage {s}: {len(units)} matrices")
        if s == "fetch":
            from . import fetch

            fetch.run(cfg, units, a.force)
        elif s == "canon":
            from . import canon

            canon.run(cfg, units, a.force)
        elif s == "write":
            from . import write

            write.run(cfg, units, fmts, a.force)
        elif s == "decode":
            from . import decode

            decode.run(cfg, units, fmts, a.force, getattr(a, "sweep", False), a.set)
        elif s == "ladder":
            from . import ladder

            ladder.run(cfg, units, a.force)
        elif s == "floor":
            from . import floor

            floor.run(cfg, units, a.force)
        elif s == "second":
            from . import second

            second.run(cfg, units, a.force)
        elif s == "minibatch":
            from . import minibatch

            panel = [u for u in units if u.dataset.get("loader_panel") and u.name == "rna"]
            minibatch.run(cfg, panel, a.force, a.set, a.tiers.split(",") if getattr(a, "tiers", None) else None)
        elif s == "access":
            from . import access

            access.run(cfg, units, a.force)
        elif s == "netsweep":
            from . import netsweep

            panel = [u for u in units if u.dataset.get("loader_panel") and u.name == "rna"]
            netsweep.run(cfg, panel, a.force, a.set, r2=a.r2)
        elif s == "writecost":
            from . import writecost

            writecost.run(cfg, units, fmts, a.force, a.set)
        elif s == "kernels":
            from . import kernels

            kernels.run(cfg, units, a.force)
        elif s == "entropy":
            from . import kernels

            kernels.run(cfg, units, a.force, entropy=True)
        elif s == "eco":
            from . import eco

            eco.run(cfg, units, a.force)
    if a.stage == "all":
        from . import report

        return report.run(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
