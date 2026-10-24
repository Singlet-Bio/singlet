#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the per-donor ancestry inference on real restricted sidecars and compare it with what else is known about the donors.

For each study: download the restricted sidecar to scratch (private R2 bucket; credentials from the environment), read every
``samples/<GSM>/donor/donor<k>_coverage.tsv`` in memory, infer the donor's ancestry, and record the result next to the
pipeline's own sample-level call (``ancestry_call.json``, 50 ancestry-informative markers) and the self-reported race/ethnicity
from GEO when ``--labels`` has it. The sidecar is deleted after each study. The output JSONL holds labels, proportions and site
counts only; no genotype, allele or position leaves memory.

  python scripts/annotate_ancestry_real.py --gse-list gses.txt --af 1kgp_superpop_af.parquet --scratch /tmp/x --out anc.jsonl \
      [--labels race_labels.json] [--n-boot 20]
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from annotate_campaign import RESTRICTED_BUCKET, _sidecar  # noqa: E402

_COV = re.compile(r"^samples/(GSM\d+)/donor/(donor\d+)_coverage\.tsv$")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gse-list", type=Path, required=True)
    ap.add_argument("--af", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--labels", type=Path, help="JSON list of [gse, gsm, self-reported label, ...]")
    ap.add_argument("--n-boot", type=int, default=20)
    a = ap.parse_args(argv)

    import pandas as pd
    from singlet.celllabels import ancestry as ANC

    t0 = time.time()
    idx = ANC.AFIndex(pd.read_parquet(a.af))
    print(
        f"AF index: {len(idx.key):,} sites in {time.time() - t0:.0f}s (bucket {RESTRICTED_BUCKET})",
        flush=True,
    )
    self_rep = {}
    if a.labels:
        for row in json.loads(a.labels.read_text()):
            self_rep[row[1]] = row[2]
    gses = [x.split()[0] for x in a.gse_list.read_text().splitlines() if x.strip()]
    a.scratch.mkdir(parents=True, exist_ok=True)
    done = set()
    if a.out.exists():
        done = {json.loads(x)["gse"] for x in a.out.read_text().splitlines() if x.strip()}
    for gse in gses:
        if gse in done:
            continue
        side = a.scratch / f"{gse}.restricted.singlet"
        t1 = time.time()
        rows = []
        try:
            if not _sidecar(gse, side):
                rows.append(dict(gse=gse, error="no sidecar"))
            else:
                with zipfile.ZipFile(side) as z:
                    names = z.namelist()
                    pipe = {}
                    for n in names:
                        if n.endswith("/ancestry_call.json"):
                            j = json.loads(z.read(n))
                            pipe[n.split("/")[1]] = (
                                j.get("ancestry"),
                                j.get("confidence"),
                                j.get("n_informative_aims"),
                            )
                    for n in names:
                        m = _COV.match(n)
                        if not m:
                            continue
                        gsm, donor = m.groups()
                        sites = pd.read_csv(
                            io.BytesIO(z.read(n)),
                            sep="\t",
                            usecols=["chrom", "pos", "ref", "alt", "dp", "ad"],
                            dtype={"chrom": str, "ref": str, "alt": str},
                        )
                        t2 = time.time()
                        ad, dp, F = idx.align(sites)
                        frac = len(ad) / max(len(sites), 1)
                        if frac < ANC.MIN_MATCH_FRAC:
                            c = ANC.AncestryCall(
                                None, None, {}, int(len(ad)), None, ANC.mid_allele_fraction(ad, dp)
                            )
                        else:
                            c = ANC.infer_ancestry(ad, dp, F, n_boot=a.n_boot)
                        p = pipe.get(gsm, (None, None, None))
                        rows.append(
                            dict(
                                gse=gse,
                                gsm=gsm,
                                donor=donor,
                                n_sites_file=int(len(sites)),
                                match_frac=round(frac, 4),
                                n_sites=c.n_sites,
                                mean_dp=float(dp.mean()) if len(dp) else None,
                                group=c.group,
                                hancestro=c.hancestro,
                                prob=c.prob,
                                q={k: round(v, 4) for k, v in c.q.items()},
                                stability=c.stability,
                                mid_af=c.mid_af,
                                reason=c.reason,
                                alt_read_frac=round(float(ad.sum() / max(dp.sum(), 1)), 4)
                                if len(dp)
                                else None,
                                het_frac=round(ANC.het_fraction(ad, dp), 3) if len(dp) else None,
                                mean_dp_deep=float(dp[dp >= ANC.MID_AF_DP].mean())
                                if (dp >= ANC.MID_AF_DP).any()
                                else None,
                                n_deep=int((dp >= ANC.MID_AF_DP).sum()),
                                seconds=round(time.time() - t2, 2),
                                pipeline_call=p[0],
                                pipeline_conf=p[1],
                                pipeline_n_aims=p[2],
                                self_reported=self_rep.get(gsm),
                            )
                        )
                        del sites, ad, dp, F
        except Exception as e:  # noqa: BLE001
            rows.append(dict(gse=gse, error=f"{type(e).__name__}: {e}"))
        finally:
            side.unlink(missing_ok=True)
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{gse}: {len(rows)} donors in {time.time() - t1:.0f}s", flush=True)
    shutil.rmtree(a.scratch, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
