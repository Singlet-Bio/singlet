# SPDX-License-Identifier: MIT
"""Research Alpha QC: stratified random sample of the transcoded studies.

Strata: species (from the manifest's reference build) x chemistry (majority protocol of the study's samples in D1 sample_qc)
x size tercile (s1 bytes). Forced in: the largest studies (bytes and sample count), studies with excluded samples, Drop-seq,
Visium, multiome (nuclei), non-human/non-mouse species, and manifests of studies missing from the D1 catalog. The rest is a
proportional draw per stratum (seeded). Writes the task list (GSE, shard k/N, organism) with large studies split by samples.

    python qc_select.py --sweep sweep.json --gse-meta d1_gse_meta.json --sample-qc d1_sample_qc.json --policy P
                        --n 300 --seed 20261027 --tasks tasks.tsv --table selection.tsv
"""

import argparse
import json
import math
import random
from collections import Counter, defaultdict


def chem_group(p):
    p = (p or "unknown").lower()
    if p in ("10xv3", "10xv3-5p", "10xv4"):
        return "10x_v3"
    if p in ("10xv2", "10xv1"):
        return "10x_v2"
    if p == "10x-arc-gex":
        return "multiome_nuclei"
    if p == "10x-visium":
        return "visium"
    if p in ("dropseq", "indrop", "seqwell", "surecell", "ddseq"):
        return "dropseq_like"
    if p in ("celseq2", "marsseq2", "quartzseq2", "sci-rna-seq3"):
        return "plate"
    if p in ("bd-rhapsody", "microwell-seq", "dnbelab-c4", "splitseq"):
        return "other_droplet_split"
    return "other"


def species(build):
    b = (build or "").lower()
    if "barnyard" in b:
        return "barnyard"
    if b.startswith("grch38"):
        return "human"
    if b.startswith("grcm39"):
        return "mouse"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--gse-meta", required=True)
    ap.add_argument("--sample-qc", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=20261027)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--shard-bytes", type=float, default=6e9, help="s1 bytes per shard (legacy reads are ~4x)")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    sw = json.load(open(a.sweep))
    pol = json.load(open(a.policy))
    X = set(pol["excluded_samples"])
    gm = {r["gse_id"]: r for r in json.load(open(a.gse_meta))[0]["results"]}
    sq = json.load(open(a.sample_qc))[0]["results"]
    prot = defaultdict(Counter)
    excl = Counter()
    for r in sq:
        prot[r["gse_id"]][r["protocol"]] += 1
        if r["gsm_id"] in X:
            excl[r["gse_id"]] += 1
    st = {}
    for m in sw["manifests"]:
        if m.get("status") != 200:
            continue
        g = m["gse"]
        pc = prot.get(g)
        st[g] = {
            "gse": g, "bytes": m["bytes"], "legacy_bytes": m.get("legacy_bytes") or 0, "n_samples": m["n_samples"],
            "species": species(m.get("reference_build")), "build": m.get("reference_build"),
            "chem": chem_group(pc.most_common(1)[0][0] if pc else None), "n_excluded": excl.get(g, 0),
            "in_catalog": g in gm, "organism": (gm.get(g) or {}).get("organism_primary") or "",
            "tissues": (gm.get(g) or {}).get("tissue_groups") or "[]",
        }
    order = sorted(st.values(), key=lambda r: r["bytes"])
    for i, r in enumerate(order):
        r["size"] = ["small", "medium", "large"][min(2, 3 * i // len(order))]
    why = defaultdict(list)

    def force(lst, tag, k):
        for r in lst[:k]:
            why[r["gse"]].append(tag)

    force(sorted(st.values(), key=lambda r: -r["bytes"]), "largest_bytes", 8)
    force(sorted(st.values(), key=lambda r: -r["n_samples"]), "most_samples", 5)
    pool = list(st.values())
    rnd.shuffle(pool)
    force([r for r in pool if r["n_excluded"] > 0], "had_excluded_samples", 40)
    force([r for r in pool if r["chem"] == "dropseq_like"], "dropseq", 10)
    force([r for r in pool if r["chem"] == "visium"], "visium", 8)
    force([r for r in pool if r["chem"] == "multiome_nuclei"], "nuclei_multiome", 8)
    force([r for r in pool if r["chem"] == "plate"], "plate", 6)
    force([r for r in pool if r["species"] == "other"], "other_species", 10)
    force([r for r in pool if r["species"] == "barnyard"], "barnyard", 5)
    force([r for r in pool if not r["in_catalog"]], "not_in_d1_catalog", 10)
    force([r for r in pool if r["n_samples"] >= 8 and r["species"] == "human"], "human_multisample", 15)
    # proportional fill per stratum
    strata = defaultdict(list)
    for r in pool:
        if r["gse"] not in why:
            strata[(r["species"], r["chem"], r["size"])].append(r)
    left = max(0, a.n - len(why))
    tot = sum(len(v) for v in strata.values())
    quota = {k: len(v) * left / tot for k, v in strata.items()}
    take = {k: int(math.floor(q)) for k, q in quota.items()}
    rest = left - sum(take.values())
    for k in sorted(quota, key=lambda k: -(quota[k] - take[k]))[:rest]:
        take[k] += 1
    for k, v in strata.items():
        for r in v[: take[k]]:
            why[r["gse"]].append("stratified")
    sel = [st[g] for g in sorted(why)]
    with open(a.table, "w") as fh:
        cols = ["gse", "species", "chem", "size", "bytes", "legacy_bytes", "n_samples", "n_excluded", "in_catalog",
                "build", "organism", "tissues"]
        fh.write("\t".join(cols + ["why", "n_shards"]) + "\n")
        with open(a.tasks, "w") as ft:
            for r in sorted(sel, key=lambda r: -r["bytes"]):
                k = max(1, min(r["n_samples"], max(math.ceil(r["bytes"] / a.shard_bytes), math.ceil(r["n_samples"] / 50))))
                r["n_shards"] = k
                for i in range(k):
                    ft.write(f"{r['gse']}\t{i}/{k}\t{r['organism']}\n")
                fh.write("\t".join(str(r[c]) for c in cols) + "\t" + ",".join(why[r["gse"]]) + f"\t{k}\n")
    c = Counter((r["species"], r["chem"]) for r in sel)
    print(len(sel), "studies;", sum(r["n_shards"] for r in sel), "tasks;", sum(r["bytes"] for r in sel) / 1e9, "GB s1;",
          sum(r["legacy_bytes"] for r in sel) / 1e9, "GB legacy")
    for k, v in sorted(c.items()):
        print(k, v)


if __name__ == "__main__":
    main()
