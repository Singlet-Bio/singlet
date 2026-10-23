#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Annotate the samples of one study: one label Parquet and one summary JSON per sample.

  # v2 study (data.singlet.bio)
  python scripts/annotate_sample.py --gse GSE145370 --out out/
  # legacy bundle on disk, with its restricted sidecar for ancestry
  python scripts/annotate_sample.py --gse GSE164241 --bundle /tmp/GSE164241.singlet \
      --sidecar /tmp/GSE164241.restricted.singlet --af $WORK/data/annotate/1kgp_superpop_af.parquet --out out/

Outputs ``<out>/<GSE>/<GSM>.labels.parquet`` (public schema, checked by ``singlet.annotate.output.assert_public_safe``) and
``<out>/<GSE>/<GSM>.summary.json`` (internal QC: counts, donor-level calls, timings). A sample whose outputs exist with the current
model version is skipped unless ``--force``. Exit status 0 when every sample was written or skipped for a stated reason.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gse", required=True)
    ap.add_argument("--gsm", action="append", help="only these samples (repeatable)")
    ap.add_argument("--bundle", type=Path, help="legacy .singlet bundle on disk (default: read the v2 study)")
    ap.add_argument("--sidecar", type=Path, help="restricted sidecar of the bundle (enables ancestry)")
    ap.add_argument("--af", type=Path, help="1000 Genomes super-population allele frequency table (Parquet)")
    ap.add_argument("--method", default="selftrain", choices=["guide", "selftrain"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-cells", type=int, default=0, help="skip samples with more called cells than this (0 = no limit)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)

    import pandas as pd

    from singlet.annotate import io as IO
    from singlet.annotate import output as O
    from singlet.annotate.pipeline import annotate_sample, model_version

    mv = model_version(a.method)
    af = pd.read_parquet(a.af) if a.af else None
    if a.bundle:
        from singlet.bundle import SingletBundle

        with SingletBundle(a.bundle) as b:
            gsms = list(b.gsm_ids)
    else:
        from singlet.v2 import client as C

        gsms = list(C.open_study(a.gse).samples)
    if a.gsm:
        gsms = [g for g in gsms if g in set(a.gsm)]
    odir = a.out / a.gse
    odir.mkdir(parents=True, exist_ok=True)
    failed = 0
    for gsm in gsms:
        lab, summ = odir / f"{gsm}.labels.parquet", odir / f"{gsm}.summary.json"
        if not a.force and lab.exists() and summ.exists():
            try:
                if json.loads(summ.read_text()).get("model_version") == mv:
                    print(f"{gsm}: up to date", flush=True)
                    continue
            except ValueError:
                pass
        t0 = time.time()
        try:
            s = IO.read_legacy(a.bundle, gsm, sidecar=a.sidecar) if a.bundle else IO.read_v2(a.gse, gsm, sidecar=a.sidecar)
            if a.max_cells and len(s.col) > a.max_cells:
                print(f"{gsm}: {len(s.col)} cells > --max-cells, skipped", flush=True)
                continue
            read_s = time.time() - t0
            df, summary = annotate_sample(s, af_table=af, method=a.method)
            summary["timings"]["read_s"] = read_s
            O.write_labels(df, lab)
            summ.write_text(json.dumps(summary, indent=1, default=str))
            ct = df["cell_type_cl"].notna().mean() if len(df) else 0
            print(f"{gsm}: {len(df)} cells, {ct:.0%} typed, sex {summary.get('sex', {}).get('groups')}, "
                  f"ancestry {'yes' if 'ancestry' in summary else 'no'}, {time.time() - t0:.1f}s", flush=True)
        except Exception as e:  # one bad sample must not stop the study
            failed += 1
            print(f"{gsm}: FAILED {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            (odir / f"{gsm}.error.txt").write_text(traceback.format_exc())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
