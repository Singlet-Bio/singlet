#!/usr/bin/env python
"""w4_atlas step 5: recoding potential of the legacy text members, measured on a byte-quantile sample.

    python -P tools/w4_atlas_recode.py select          # -> results/w4_atlas_recode_sel.tsv
    python -P tools/w4_atlas_recode.py fetch           # ranged downloads -> corpus_w4_atlas/recode/<label>.txt (+ .json)
    python -P tools/w4_atlas_recode.py run             # bin/w4_atlas_bench.exe on every file -> results/w4_atlas_recode_{text,mtx}.tsv

Selection: for each member name, members sorted by in-zip size; the 3 (or 2 for extras) members that contain the 1/6, 3/6, 5/6
points of the cumulative byte distribution (so each stands for an equal third of the category's bytes), taken from different
bundles where possible. Whitelist-like members with only a few distinct contents (CRC) are chosen one per distinct content.
A member whose in-zip size exceeds CAP is read as a PREFIX of its deflate stream (a prefix of deflate decodes from the start);
the text is cut at the last newline. Baseline for a prefix = fetched_compressed * len(cut)/len(decoded).
"""
import csv, collections, json, os, subprocess, sys, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4_atlas_probe as P

ROOT = P.ROOT
RES = P.RES
CORP = os.path.join(ROOT, "corpus_w4_atlas", "recode")
CAP = 10_600_000

CORE = ["star_Solo.out/Gene/raw/matrix.mtx", "star_Solo.out/Gene/filtered/matrix.mtx", "star_Solo.out/Gene/raw/barcodes.tsv",
        "star_Solo.out/Gene/raw/features.tsv", "ambient_profile.tsv", "star_SJ.out.tab", "gene_expression.tsv", "cell_qc_metrics.tsv"]
EXTRA = ["mt/mt_summary.tsv", "donor/donor_assignments.tsv", "cell_cycle_scores.tsv", "auto_barcodes.tsv", "splice_events.tsv",
         "star_Solo.out/Gene/filtered/barcodes.tsv", "cell_calls.tsv", "doublet_scores.tsv", "ambient_contamination.tsv"]


def slug(cat): return cat.replace("/", "__").replace(".", "_")


def select():
    M = list(csv.DictReader(open(os.path.join(RES, "w4_atlas_members.tsv"), encoding="utf-8"), delimiter="\t"))
    by = collections.defaultdict(list)
    for r in M: by[r["category"]].append(r)
    sel = []
    for cat in CORE + EXTRA:
        rr = sorted(by[cat], key=lambda r: (int(r["csize"]), r["gse"], r["gsm"]))
        k = 3 if cat in CORE else 2
        picks = []
        if cat in ("star_Solo.out/Gene/raw/barcodes.tsv", "star_Solo.out/Gene/raw/features.tsv"):
            seen = {}
            for r in rr:
                seen.setdefault((r["usize"], r["crc32"]), []).append(r)
            for key, v in seen.items():
                picks.append(v[len(v) // 2])   # median-sized member of each distinct content
            picks = picks[:3]
        else:
            tot = sum(int(r["csize"]) for r in rr)
            cum = 0
            marks = [(2 * i + 1) / (2 * k) for i in range(k)]
            mi = 0
            used = set()
            for idx, r in enumerate(rr):
                cum += int(r["csize"])
                while mi < len(marks) and cum >= marks[mi] * tot:
                    # nearest unused-bundle neighbour
                    cand = None
                    for d in range(0, 40):
                        for j in (idx - d, idx + d):
                            if 0 <= j < len(rr) and rr[j]["gse"] not in used and rr[j] not in picks:
                                cand = rr[j]; break
                        if cand: break
                    cand = cand or r
                    picks.append(cand); used.add(cand["gse"]); mi += 1
        for i, r in enumerate(picks):
            sel.append(dict(label=f"{slug(cat)}__{i+1}", category=cat, gse=r["gse"], gsm=r["gsm"], path=r["path"], csize=r["csize"], usize=r["usize"],
                            method=r["method"], core=int(cat in CORE)))
    cols = list(sel[0].keys())
    with open(os.path.join(RES, "w4_atlas_recode_sel.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in sel: f.write("\t".join(str(r[c]) for c in cols) + "\n")
    tot = sum(min(int(r["csize"]), CAP) for r in sel)
    print(f"{len(sel)} members selected, ~{tot/1e6:.0f} MB to fetch")
    for r in sel: print(r["label"], r["gse"], r["gsm"], int(r["csize"]) / 1e6, "MB")


def load_sel():
    return list(csv.DictReader(open(os.path.join(RES, "w4_atlas_recode_sel.tsv"), encoding="utf-8"), delimiter="\t"))


def fetch():
    os.makedirs(CORP, exist_ok=True)
    for r in load_sel():
        dst = os.path.join(CORP, r["label"] + ".txt")
        if os.path.exists(dst): continue
        data, info = P.get_member(r["gse"], r["path"], max_comp=CAP)
        full_len = len(data)
        if not info["complete"]:
            cut = data.rfind(b"\n") + 1
            data = data[:cut]
        base = info["csize"] if info["complete"] else info["consumed_comp"] * len(data) / max(full_len, 1)
        open(dst, "wb").write(data)
        meta = dict(r, complete=info["complete"], crc_ok=info["crc_ok"], text_bytes=len(data), decoded_bytes=full_len,
                    baseline_deflate=base, fetched=info["fetched"], consumed_comp=info["consumed_comp"])
        json.dump(meta, open(os.path.join(CORP, r["label"] + ".json"), "w"))
        print(r["label"], f"text {len(data)/1e6:.1f} MB, baseline {base/1e6:.2f} MB, complete={info['complete']}, crc_ok={info['crc_ok']}", flush=True)


def transposed(path, out):
    """columnar transposition of a uniform-width TSV; header '%' lines kept first. Returns False if not uniform."""
    raw = open(path, "rb").read()
    lines = raw.split(b"\n")
    if lines and lines[-1] == b"": lines.pop()
    head = []
    while lines and lines[0].startswith(b"%"): head.append(lines.pop(0))
    if not lines: return False
    w = lines[0].count(b"\t") + 1
    cols = [[] for _ in range(w)]
    for ln in lines:
        f = ln.split(b"\t")
        if len(f) != w: return False
        for i, x in enumerate(f): cols[i].append(x)
    with open(out, "wb") as fo:
        for h in head: fo.write(h + b"\n")
        for c in cols: fo.write(b"\n".join(c) + b"\n")
    return True


def run():
    exe = os.path.join(ROOT, "bin", "w4_atlas_bench.exe")
    tag, skip, only = "", [], []
    for a in sys.argv[2:]:
        if a.startswith("--tag="): tag = "_" + a[6:]
        elif a.startswith("--skip="): skip.append(a[7:])
        else: only.append(a)
    text_tsv = os.path.join(RES, f"w4_atlas_recode_text{tag}.tsv")
    mtx_tsv = os.path.join(RES, f"w4_atlas_recode_mtx{tag}.tsv")
    done = set()
    if os.path.exists(text_tsv):
        done = {l.split("\t")[0] for l in open(text_tsv, encoding="utf-8")}
    flags = 0x00004000 if os.name == "nt" else 0   # BELOW_NORMAL_PRIORITY_CLASS
    for r in load_sel():
        lab = r["label"]
        if only and not any(o in lab for o in only): continue
        if any(sk in lab for sk in skip): continue
        src = os.path.join(CORP, lab + ".txt")
        if not os.path.exists(src): continue
        if os.path.exists(os.path.join(ROOT, "results", ".timing_lock")):
            print("timing lock present; waiting is the caller's job"); return
        if lab not in done:
            subprocess.run([exe, "text", src, lab, text_tsv], cwd=ROOT, creationflags=flags, check=True)
        colf = os.path.join(CORP, lab + ".cols")
        if (lab + "__cols") not in done and r["category"] not in ("star_Solo.out/Gene/raw/barcodes.tsv", "star_Solo.out/Gene/raw/features.tsv"):
            if transposed(src, colf):
                subprocess.run([exe, "text", colf, lab + "__cols", text_tsv], cwd=ROOT, creationflags=flags, check=True)
        if r["category"].endswith("matrix.mtx"):
            subprocess.run([exe, "mtx", src, lab, mtx_tsv], cwd=ROOT, creationflags=flags, check=True)


if __name__ == "__main__":
    {"select": select, "fetch": fetch, "run": run}[sys.argv[1]]()
