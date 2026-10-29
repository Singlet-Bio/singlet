"""chrM read depth per called cell across a random sample of atlas samples (public v2 objects on data.singlet.bio).

For each sampled study: the v2 manifest, then for up to --per-study samples a range read of the sample object's
pileup_stats.json and summary.json members (stored zstd); studies not migrated yet are read from the legacy
.singlet bundle (ZIP central directory, then the two members by range). Writes one CSV row per sample.

  python atlas_survey.py sample_meta.csv OUT.csv [--n-studies 400] [--per-study 2] [--seed 1]

sample_meta.csv: gsm,gse,organism,... (the catalog's sample metadata export); only Homo sapiens studies are sampled.
"""
import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.error
import urllib.request
import zipfile

import zstandard

BASE = "https://data.singlet.bio/"
H = {"User-Agent": "singlet-bench/mtdna-lineage"}


def get(path, rng=None, tries=4):
    hdr = dict(H)
    if rng:
        hdr["Range"] = "bytes=%d-%d" % rng
    for t in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(BASE + path, headers=hdr), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 ** t)
        except Exception:
            time.sleep(2 ** t)
    return None


def member_json(key, m):
    raw = get(key, (m["off"], m["off"] + m["len"] - 1))
    if raw is None:
        return None
    if m.get("kind") == 3:
        raw = zstandard.ZstdDecompressor().decompress(raw, max_output_size=m.get("out_len", 1 << 24))
    return json.loads(raw.decode())


class RangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP Range requests (ZIP central directory + single members)."""

    def __init__(self, path):
        self.path, self.p = path, 0
        req = urllib.request.Request(BASE + path, method="HEAD", headers=H)
        self.n = int(urllib.request.urlopen(req, timeout=60).headers["Content-Length"])

    def seekable(self):
        return True

    def readable(self):
        return True

    def seek(self, o, w=0):
        self.p = o if w == 0 else (self.p + o if w == 1 else self.n + o)
        return self.p

    def tell(self):
        return self.p

    def readinto(self, b):
        if self.p >= self.n:
            return 0
        d = get(self.path, (self.p, min(self.n, self.p + len(b)) - 1)) or b""
        b[: len(d)] = d
        self.p += len(d)
        return len(d)


def legacy_samples(gse, per_study):
    """[(gsm, pileup_stats, summary)] from the legacy bundle, or []."""
    try:
        z = zipfile.ZipFile(io.BufferedReader(RangeFile(f"data/{gse}/{gse}.singlet"), buffer_size=1 << 16))
    except Exception:
        return []
    names = set(z.namelist())
    gsms = sorted({n.split("/")[1] for n in names if n.startswith("samples/") and n.endswith("/pileup_stats.json")})
    random.Random(gse).shuffle(gsms)
    out = []
    for g in gsms[:per_study]:
        try:
            ps = json.loads(z.read(f"samples/{g}/pileup_stats.json"))
            sm = json.loads(z.read(f"samples/{g}/summary.json")) if f"samples/{g}/summary.json" in names else {}
        except Exception:
            continue
        out.append((g, ps, sm))
    return out


def row(gse, gsm, ps, sm, n_called=None):
    return dict(gse=gse, gsm=gsm, protocol=sm.get("protocol_name"), n_input_reads=sm.get("n_input_reads"),
                input_reads_capped=sm.get("n_input_reads") == 30000000, total_reads=ps.get("total_reads"),
                n_cells_called=n_called or sm.get("n_cells_called"),
                fraction_reads_in_cells=sm.get("fraction_reads_in_cells"), median_umi=sm.get("median_umi_per_cell"),
                chrm_reads=ps.get("chrm_reads"), mt_pileup_bases=ps.get("mt_pileup_bases"),
                umi_unique=ps.get("umi_unique"), umi_duplicate=ps.get("umi_duplicate"), low_mapq=ps.get("low_mapq"),
                singlet_version=ps.get("version"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("meta"); ap.add_argument("out")
    ap.add_argument("--n-studies", type=int, default=400); ap.add_argument("--per-study", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    gses = sorted({r["gse"] for r in csv.DictReader(open(a.meta)) if r["organism"] == "Homo sapiens"})
    random.Random(a.seed).shuffle(gses)
    cols = ["gse", "gsm", "protocol", "n_input_reads", "input_reads_capped", "total_reads", "n_cells_called",
            "fraction_reads_in_cells", "median_umi", "chrm_reads", "mt_pileup_bases", "umi_unique", "umi_duplicate",
            "low_mapq", "singlet_version"]
    w = csv.DictWriter(open(a.out, "w", newline=""), fieldnames=cols)
    w.writeheader()
    n_ok = n_try = 0
    for gse in gses:
        if n_ok >= a.n_studies:
            break
        n_try += 1
        got = 0
        man = get(f"v2/m/{gse}.json")
        if man:
            samples = json.loads(man).get("samples", [])
            if isinstance(samples, dict):
                samples = list(samples.values())
            samples = [x for x in samples if "pileup_stats.json" in x.get("members", {})]
            random.Random(gse).shuffle(samples)
            for x in samples[: a.per_study]:
                ps = member_json(x["key"], x["members"]["pileup_stats.json"])
                sm = member_json(x["key"], x["members"]["summary.json"]) if "summary.json" in x["members"] else {}
                if ps:
                    w.writerow(row(gse, x["gsm"], ps, sm or {}, x.get("n_called"))); got += 1
        else:
            for g, ps, sm in legacy_samples(gse, a.per_study):
                w.writerow(row(gse, g, ps, sm)); got += 1
        if got:
            n_ok += 1
        if n_try % 25 == 0:
            print(f"tried {n_try} studies, {n_ok} with data", file=sys.stderr, flush=True)
    print(f"done: {n_ok} studies of {n_try} tried", file=sys.stderr)


if __name__ == "__main__":
    main()
