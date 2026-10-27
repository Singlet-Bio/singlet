# SPDX-License-Identifier: MIT
"""Research Alpha QC, atlas-wide sweep (cheap checks on every study, over HTTPS from data.singlet.bio):

* every s1/m/<GSE>.json manifest: accession scan against the policy, release-policy version, per-sample provenance fields,
  referenced objects vs the R2 listing (exists, size), referenced packs vs s1/ref/index.json;
* orphan objects (in s1/s/ or s1/a/ but referenced by no manifest): served publicly? member table: withheld layers?
* every withheld study: s1/m manifest and legacy bundle must not be served;
* the old v2/ prefix: what is still served, and what the objects hold.

    python qc_sweep.py --policy P --lists DIR --out OUT.json [--threads 16]
"""

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.environ.get("QC_SRC", ""), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qc_http as H  # noqa: E402
from qc_study import GSM_RE, MARKERS, private_layer, scan_text  # noqa: E402


def listing(d, name):
    out = {}
    with gzip.open(os.path.join(d, f"list_{name}.tsv.gz"), "rt") as fh:
        next(fh)
        for line in fh:
            k, s, e, t = line.rstrip("\n").split("\t")
            out[k] = (int(s), e, t)
    return out


def member_table(key, size):
    from singlet.s1.container import S2Reader

    def fetch(off, ln):
        st, _h, b = H.get_range(key, off, min(off + ln, size) - 1)
        if st not in (200, 206):
            raise H.HttpStatus(st, key)
        return b

    rd = S2Reader(fetch, size=size)
    return rd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--lists", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()
    pol = json.load(open(a.policy))
    X, W = set(pol["excluded_samples"]), set(pol["withheld_studies"])
    Lm, Ls, La, Lr = (listing(a.lists, n) for n in ("s1_m", "s1_s", "s1_a", "s1_ref"))
    Lv2 = listing(a.lists, "v2")
    st_i, _h, idx_raw = H.get("s1/ref/index.json")
    index = {p["id"]: p for p in json.loads(idx_raw)["packs"]}
    R = {"policy_version": pol["version"], "n_manifests_listed": len(Lm)}
    gses = sorted(k.split("/")[2][:-5] for k in Lm)

    def one(g):
        st, hdr, raw = H.get(f"s1/m/{g}.json")
        r = {"gse": g, "status": st}
        if st != 200:
            return r, None
        m = json.loads(raw)
        accs = set(x.group(1).decode() for x in GSM_RE.finditer(raw))
        r["excluded_named"] = sorted(accs & X)
        r["withheld_named"] = sorted(accs & W)
        r["policy_version"] = (m.get("release_policy") or {}).get("version")
        r["format"] = m.get("format")
        r["gse_field"] = m.get("gse")
        r["reference_build"] = m.get("reference_build")
        r["n_samples"] = len(m.get("samples") or [])
        r["legacy_bytes"] = (m.get("legacy") or {}).get("bytes")
        r["packs"] = m.get("packs")
        r["bytes"] = sum(int(s.get("bytes") or 0) for s in m.get("samples") or [])
        r["n_called"] = sum(int(s.get("n_called") or 0) for s in m.get("samples") or [])
        prob = Counter()
        refs = []
        for s in m.get("samples") or []:
            pv = s.get("provenance") or {}
            if pv.get("reads_used") is None:
                prob["no_reads_used"] += 1
            if not isinstance((pv.get("strand") or {}).get("wrong_strand_fraction"), (int, float)):
                prob["no_strand_fraction"] += 1
            if not (pv.get("library_type") or {}).get("verdict"):
                prob["no_libtype_verdict"] += 1
            if pv.get("release_policy") != pol["version"]:
                prob["sample_policy_version"] += 1
            if not s.get("annotations"):
                prob["no_annotations"] += 1
            for nm in (s.get("members") or {}):
                if private_layer(nm):
                    prob["withheld_layer_member:" + nm] += 1
            if "called_ed" not in ((s.get("members") or {}).get("cells") or {}).get("columns", []):
                prob["no_called_ed_column"] += 1
            refs.append((s["gsm"], s["key"], int(s.get("bytes") or -1), s.get("n_called"),
                         [(e["key"], int(e.get("bytes") or -1)) for e in (s.get("annotations") or {}).values()]))
        r["sample_problems"] = dict(prob)
        mh = []
        scan_text(json.dumps({k: v for k, v in m.items() if k != "release_policy"}).encode(), "manifest", MARKERS, mh,
                  None, None, set())
        r["marker_hits"] = Counter(h["marker"] for h in mh)
        return r, refs

    studies, refs_all = [], {}
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for r, refs in ex.map(one, gses):
            studies.append(r)
            if refs is not None:
                refs_all[r["gse"]] = refs
    R["manifests"] = studies
    # referenced objects vs listing
    ref_s, ref_a = {}, {}
    miss, sizebad, dup_gsm = [], [], defaultdict(list)
    for g, refs in refs_all.items():
        for gsm, key, b, _nc, anns in refs:
            ref_s[key] = g
            dup_gsm[gsm].append(g)
            if key not in Ls:
                miss.append([g, key])
            elif Ls[key][0] != b:
                sizebad.append([g, key, Ls[key][0], b])
            for ak, ab in anns:
                ref_a[ak] = g
                if ak not in La:
                    miss.append([g, ak])
                elif La[ak][0] != ab:
                    sizebad.append([g, ak, La[ak][0], ab])
    R["referenced_missing"] = miss
    R["referenced_size_mismatch"] = sizebad
    R["gsm_in_several_studies"] = {k: v for k, v in dup_gsm.items() if len(v) > 1}
    # packs
    used = Counter()
    for s in studies:
        for role, pid in (s.get("packs") or {}).items():
            if pid:
                used[pid] += 1
    R["packs_missing"] = sorted(p for p in used if f"s1/ref/{p}.pack" not in Lr)
    R["packs_not_in_index"] = sorted(p for p in used if p not in index)
    # orphans
    orph_s = sorted(k for k in Ls if k not in ref_s)
    orph_a = sorted(k for k in La if k not in ref_a)
    R["orphans_s1_s"] = len(orph_s)
    R["orphans_s1_a"] = len(orph_a)
    R["orphan_bytes"] = sum(Ls[k][0] for k in orph_s) + sum(La[k][0] for k in orph_a)

    def orphan(key):
        size = Ls[key][0]
        st, h, _b = H.get_range(key, 0, 0)
        o = {"key": key, "bytes": size, "status": st, "last_modified": Ls[key][2]}
        if st in (200, 206):
            try:
                rd = member_table(key, size)
                names = list(rd.members)
                o["withheld_members"] = sorted(n for n in names if private_layer(n))
                o["n_members"] = len(names)
                cm = rd.members.get("cells")
                o["has_labels"] = bool(cm and "labels" in (cm.meta or {}))
            except Exception as e:  # noqa: BLE001
                o["error"] = repr(e)[:200]
        return o

    with cf.ThreadPoolExecutor(a.threads) as ex:
        R["orphan_objects"] = list(ex.map(orphan, orph_s))
    R["orphan_with_withheld_members"] = [o for o in R["orphan_objects"] if o.get("withheld_members")]
    R["orphan_served"] = sum(1 for o in R["orphan_objects"] if o["status"] in (200, 206))

    def orphan_a(key):
        st, _h, _b = H.get_range(key, 0, 0)
        return [key, st]

    with cf.ThreadPoolExecutor(a.threads) as ex:
        oa = list(ex.map(orphan_a, orph_a[:2000]))
    R["orphan_sidecars_served"] = sum(1 for k, st in oa if st in (200, 206))
    R["orphan_sidecars_checked"] = len(oa)

    # withheld studies
    def wh(g):
        s1, _h, _b = H.get(f"s1/m/{g}.json")
        lg, h2, _b2 = H.request(f"{H.BASE}/data/{g}/{g}.singlet", method="HEAD")
        v2, _h3, _b3 = H.get(f"v2/m/{g}.json")
        return [g, s1, lg, v2]

    with cf.ThreadPoolExecutor(a.threads) as ex:
        w = list(ex.map(wh, sorted(W)))
    R["withheld_probe"] = Counter(f"s1m={x[1]} legacy={x[2]} v2m={x[3]}" for x in w)
    R["withheld_served"] = [x for x in w if x[1] == 200 or x[2] in (200, 206) or x[3] == 200]

    # v2 prefix
    v2 = []
    for k, (size, _e, t) in sorted(Lv2.items()):
        st, h, _b = H.get_range(k, 0, 0)
        o = {"key": k, "bytes": size, "status": st, "served_by": h.get("x-served-by"), "last_modified": t}
        if k.startswith("v2/s/") and st in (200, 206):
            gsm = k.split("/")[2]
            o["gsm_excluded"] = gsm in X
            try:
                rd = member_table(k, size)
                o["members"] = sorted(rd.members)
                o["withheld_members"] = sorted(n for n in rd.members if private_layer(n))
            except Exception as e:  # noqa: BLE001
                o["error"] = repr(e)[:200]
        v2.append(o)
    R["v2_objects"] = [o for o in v2 if not o["key"].startswith("v2/ref/")]
    R["v2_ref_served"] = sum(1 for o in v2 if o["key"].startswith("v2/ref/") and o["status"] in (200, 206))
    with open(a.out, "w") as fh:
        json.dump(R, fh, default=str)
    print("sweep done:", len(studies), "manifests;", len(miss), "missing refs;", len(sizebad), "size mismatches;",
          len(orph_s), "orphan objects;", len(R["withheld_served"]), "withheld served", flush=True)


if __name__ == "__main__":
    main()
