#!/usr/bin/env python
"""w4_atlas step 2: classify every member and write the canonical tables.

    python -P tools/w4_atlas_members.py

reads  results/w4_atlas_members_raw.tsv, results/w4_atlas_bundles_raw.tsv   (from w4_atlas_scan.py)
writes results/w4_atlas_members.tsv   one row per member (gse, gsm, path, category, group, method, sizes, crc, restricted, gen)
       results/w4_atlas_bundles.tsv   one row per bundle (size, members, samples, gen, last-modified, etag parts, ...)

Generation rule (member presence, per bundle):
  gen B = every sample directory carries the text/STARsolo members (any of ambient_profile.tsv, gene_expression.tsv,
          cell_qc_metrics.tsv, star_SJ.out.tab, star_Solo.out/...) next to the .1pz members; in this sample these bundles also
          carry donor/donor_assignments.tsv and mt/mt_summary.tsv.
  gen A = .1pz members + small json/tsv only (no text outputs).
  mixed = some samples one way, some the other (none seen).
"""
import csv, collections, importlib.util, math, os, re, sys, email.utils

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
REPO = os.path.abspath(os.path.join(ROOT, "..", ".."))

spec = importlib.util.spec_from_file_location("singlet_restricted", os.path.join(REPO, "python", "singlet", "_restricted.py"))
_r = importlib.util.module_from_spec(spec); sys.modules[spec.name] = _r; spec.loader.exec_module(_r)

SAMPLE_RE = re.compile(r"^samples/(GSM\d+)/(.*)$")
LEGACY_MARK = ("ambient_profile.tsv", "gene_expression.tsv", "cell_qc_metrics.tsv", "star_SJ.out.tab")
# looser name test than the repo rule, to catch anything restricted-looking that the allowlist might miss
LOOSE_RESTRICTED = re.compile(r"(snp_|\.vcf|\.bcf|donor\d*_coverage|ase_counts|mt_consensus|mt_variants|mt_reference|genotype)", re.I)

GROUPS = [
    ("count matrices (.1pz)", ("exon_counts.1pz", "intron_counts.1pz")),
    ("junction matrices (.1pz)", ("sj_counts.1pz", "splice_psi.1pz")),
    ("other .1pz", ("vdj_gene_usage.1pz", "mt_heteroplasmy.1pz")),
    ("STARsolo matrix.mtx", ("star_Solo.out/Gene/raw/matrix.mtx", "star_Solo.out/Gene/filtered/matrix.mtx")),
    ("STARsolo barcodes.tsv", ("star_Solo.out/Gene/raw/barcodes.tsv", "star_Solo.out/Gene/filtered/barcodes.tsv")),
    ("STARsolo features.tsv", ("star_Solo.out/Gene/raw/features.tsv", "star_Solo.out/Gene/filtered/features.tsv")),
    ("ambient_profile.tsv", ("ambient_profile.tsv",)),
    ("star_SJ.out.tab", ("star_SJ.out.tab",)),
    ("gene_expression.tsv", ("gene_expression.tsv",)),
    ("cell_qc_metrics.tsv", ("cell_qc_metrics.tsv",)),
    ("donor/mt public tables", ("mt/mt_summary.tsv", "donor/donor_assignments.tsv", "donor_assignments.tsv")),
    ("other per-cell tsv", ("cell_cycle_scores.tsv", "auto_barcodes.tsv", "splice_events.tsv", "cell_calls.tsv",
                            "doublet_scores.tsv", "ambient_contamination.tsv", "barnyard_classification.tsv")),
    ("study-level json", ("feature_vocab.json", "manifest.json", "study_meta.json")),
]
G_OF = {}
for g, names in GROUPS:
    for n in names: G_OF[n] = g


def category(path):
    m = SAMPLE_RE.match(path)
    return m.group(2) if m else path


def group_of(cat):
    if cat in G_OF: return G_OF[cat]
    return "small logs/json/other"


def main():
    raw = list(csv.DictReader(open(os.path.join(RES, "w4_atlas_members_raw.tsv"), encoding="utf-8"), delimiter="\t"))
    bun = {x["gse"]: x for x in csv.DictReader(open(os.path.join(RES, "w4_atlas_bundles_raw.tsv"), encoding="utf-8"), delimiter="\t")}
    samp = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in raw:
        m = SAMPLE_RE.match(r["name"])
        if m: samp[r["gse"]][m.group(1)].add(m.group(2))

    gen = {}
    for g, ss in samp.items():
        n_leg = sum(1 for s in ss.values() if any(x in s or any(y.startswith("star_Solo") for y in s) for x in LEGACY_MARK))
        gen[g] = "B" if n_leg == len(ss) else ("A" if n_leg == 0 else "mixed")

    # members table
    cols = ["gse", "gen", "gsm", "path", "category", "group", "method", "csize", "usize", "crc32", "lho", "restricted", "restricted_loose"]
    nrestr = 0
    with open(os.path.join(RES, "w4_atlas_members.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in raw:
            cat = category(r["name"])
            m = SAMPLE_RE.match(r["name"])
            gsm = m.group(1) if m else ""
            restr = int(_r.is_restricted_path(r["name"]))
            loose = int(bool(LOOSE_RESTRICTED.search(cat)))
            nrestr += restr
            f.write("\t".join(map(str, [r["gse"], gen.get(r["gse"], "A"), gsm, r["name"], cat, group_of(cat), r["method"], r["csize"], r["usize"],
                                        r["crc32"], r["lho"], restr, loose])) + "\n")
    # bundles table
    bcols = ["gse", "gen", "total", "n_members", "n_samples", "cd_size", "cd_in_tail", "last_modified", "etag", "etag_parts", "ttfb", "sum_csize", "overhead"]
    sums = collections.Counter()
    for r in raw: sums[r["gse"]] += int(r["csize"])
    with open(os.path.join(RES, "w4_atlas_bundles.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(bcols) + "\n")
        for g in sorted(bun):
            b = bun[g]
            et = b["etag"].strip('"')
            parts = int(et.split("-")[1]) if "-" in et else 0
            lm = email.utils.parsedate_to_datetime(b["last_modified"]).strftime("%Y-%m-%d")
            tot = int(b["total"])
            f.write("\t".join(map(str, [g, gen[g], tot, b["n_members"], len(samp[g]), b["cd_size"], b["cd_in_tail"], lm, et, parts,
                                        b["ttfb"], sums[g], tot - sums[g]])) + "\n")
    print(f"members {len(raw)}, bundles {len(bun)}, gen {collections.Counter(gen.values())}, restricted(rule) {nrestr}")


if __name__ == "__main__":
    main()
