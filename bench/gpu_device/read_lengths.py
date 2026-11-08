#!/usr/bin/env python3
"""P3-ENG-15/16: share of samples and read pairs with cDNA reads over 128 bases.

Reads per-read average lengths from SRA's run statistics (efetch db=sra XML, <Statistics><Read average=...>).
The cDNA read is taken as the longest read of a run (10x layouts: I1 8-10, R1 26-28, R2 cDNA).

Inputs
  corpus: benchmark-store/datasets.tsv, column "accessions" (SRR/SRX/SRP/PRJNA/GSE/GSM tokens; 10x URLs are reported apart)
  atlas:  a random sample of human 10x single-cell runs from an Entrez SRA query (the atlas queue stand-in)

  python3 read_lengths.py DATASETS_TSV OUTDIR [N_ATLAS=3000] [SEED=1]
Outputs OUTDIR/corpus_runs.tsv, atlas_runs.tsv, summary.md. NCBI rate limit: 3 requests/s without a key.
"""
import csv, json, random, re, sys, time, urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict

EU = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
UA = {"User-Agent": "singlet-readlen/1.0"}


def get(url, data=None, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, data=data, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            time.sleep(0.4)
            return b
        except Exception as e:  # noqa: BLE001
            time.sleep(2 + 3 * k)
            err = e
    raise RuntimeError(f"{url[:120]}: {err}")


def esearch(term, retmax=100000):
    q = urllib.parse.urlencode({"db": "sra", "term": term, "retmax": retmax, "retmode": "json", "usehistory": "n"})
    j = json.loads(get(EU + "esearch.fcgi?" + q))
    return j["esearchresult"].get("idlist", []), int(j["esearchresult"].get("count", 0))


def efetch_runs(uids):
    """per run: acc, spots, bases, read averages, platform/model, library layout, study, sample, year"""
    out = []
    for i in range(0, len(uids), 150):
        body = urllib.parse.urlencode({"db": "sra", "id": ",".join(uids[i:i + 150]), "rettype": "full", "retmode": "xml"}).encode()
        x = get(EU + "efetch.fcgi", data=body)
        try:
            root = ET.fromstring(x)
        except ET.ParseError:
            continue
        for pkg in root.iter("EXPERIMENT_PACKAGE"):
            model = (pkg.findtext(".//PLATFORM//INSTRUMENT_MODEL") or "").strip()
            study = pkg.find(".//STUDY")
            sacc = study.get("accession") if study is not None else ""
            samp = pkg.find(".//SAMPLE")
            smacc = samp.get("accession") if samp is not None else ""
            gsm = ""
            if samp is not None:
                for ex in samp.iter("EXTERNAL_ID"):
                    if (ex.get("namespace") or "").upper() == "GEO":
                        gsm = (ex.text or "").strip()
            for run in pkg.iter("RUN"):
                acc = run.get("accession") or ""
                spots = int(run.get("total_spots") or 0)
                bases = int(run.get("total_bases") or 0)
                date = run.get("published") or ""
                st = run.find("Statistics")
                avgs = []
                if st is not None:
                    for rd in st.findall("Read"):
                        try:
                            avgs.append(float(rd.get("average") or 0))
                        except ValueError:
                            pass
                est = 0
                if len(avgs) <= 1 and spots:
                    # no per-read split: the stored spot is all reads together. 2 x 150 (+ index) -> half; else the 28-base 10x barcode read off
                    tot = avgs[0] if avgs else bases / spots
                    cd = tot / 2 if tot >= 200 else max(tot - 28, 0)
                    avgs, est = [tot], 1
                else:
                    cd = max(avgs) if avgs else 0.0
                out.append({"run": acc, "study": sacc, "sample": smacc, "gsm": gsm, "model": model, "spots": spots, "bases": bases,
                            "reads": ";".join(f"{a:.0f}" for a in avgs), "cdna": cd, "est": est, "year": date[:4]})
    return out


def uids_for_accessions(accs):
    uids = []
    for i in range(0, len(accs), 40):
        term = " OR ".join(f"{a}[Accession]" if not a.startswith(("GSE", "GSM")) else f"{a}[All Fields]" for a in accs[i:i + 40])
        ids, _ = esearch(term, retmax=20000)
        uids += ids
    return uids


def summarize(rows, label):
    n = len(rows)
    if n == 0:
        return f"### {label}\n\nNo runs.\n"
    long_ = [r for r in rows if r["cdna"] > 128.5]
    sp = sum(r["spots"] for r in rows) or 1
    spl = sum(r["spots"] for r in long_)
    samples = defaultdict(float)
    for r in rows:
        k = r["gsm"] or r["sample"] or r["run"]
        samples[k] = max(samples[k], r["cdna"])
    ns = len(samples)
    nsl = sum(1 for v in samples.values() if v > 128.5)
    hist = defaultdict(int)
    for r in rows:
        c = r["cdna"]
        b = "<=100" if c <= 100.5 else "101-128" if c <= 128.5 else "129-150" if c <= 150.5 else "151-256" if c <= 256.5 else ">256"
        hist[b] += 1
    years = defaultdict(lambda: [0, 0])
    for r in rows:
        years[r["year"]][0] += 1
        years[r["year"]][1] += r["cdna"] > 128.5
    s = [f"### {label}\n",
         "| Measure | Over 128 bases | Total | Share |", "|---|---|---|---|",
         f"| Runs | {len(long_)} | {n} | {len(long_) / n:.1%} |",
         f"| Samples (GSM or BioSample) | {nsl} | {ns} | {nsl / max(ns, 1):.1%} |",
         f"| Read pairs (spots) | {spl / 1e9:.2f} G | {sp / 1e9:.2f} G | {spl / sp:.1%} |",
         f"| Runs with the cDNA length estimated from the spot total | {sum(r['est'] and r['cdna'] > 128.5 for r in rows)} | {sum(r['est'] for r in rows)} | |", "",
         "| cDNA length | Runs |", "|---|---|"] + [f"| {b} | {hist[b]} |" for b in ["<=100", "101-128", "129-150", "151-256", ">256"]] + ["",
         "| Year | Runs | Over 128 |", "|---|---|---|"] + [f"| {y or '?'} | {v[0]} | {v[1]} |" for y, v in sorted(years.items())] + [""]
    return "\n".join(s)


def write(rows, path):
    keys = ["run", "study", "sample", "gsm", "model", "spots", "bases", "reads", "cdna", "est", "year"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ds, outdir = sys.argv[1], sys.argv[2]
    n_atlas = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    tok = re.compile(r"\b((?:SRR|ERR|DRR|SRX|ERX|SRP|ERP|DRP|PRJNA|PRJEB|GSE|GSM)\d+)\b")
    corpus, url_only = {}, []
    with open(ds) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            toks = tok.findall(r.get("accessions", ""))
            if not toks:
                url_only.append(r["dataset_id"])
                continue
            # prefer run-level tokens; otherwise the first study/series token
            runs = [t for t in toks if t[:3] in ("SRR", "ERR", "DRR")]
            corpus[r["dataset_id"]] = runs if runs else toks[:1]
    crow = []
    for d, accs in corpus.items():
        try:
            rows = efetch_runs(uids_for_accessions(accs)[:400])
        except Exception as e:  # noqa: BLE001
            print(f"{d}: {e}", file=sys.stderr)
            rows = []
        for r in rows:
            r["dataset"] = d
        crow += rows
        print(f"corpus {d}: {len(rows)} runs", file=sys.stderr)
    # one row per run, then the corpus summary per dataset (a dataset is "long" if any of its runs is)
    write([{k: v for k, v in r.items() if k != "dataset"} for r in crow], f"{outdir}/corpus_runs.tsv")
    per_ds = defaultdict(float)
    for r in crow:
        per_ds[r["dataset"]] = max(per_ds[r["dataset"]], r["cdna"])
    term = ('"Homo sapiens"[Organism] AND "transcriptomic single cell"[Source] AND "PAIRED"[Layout] '
            'AND ("10x Genomics"[All Fields] OR "chromium"[All Fields] OR "10X"[All Fields])')
    ids, count = esearch(term, retmax=200000)
    random.Random(seed).shuffle(ids)
    arow = efetch_runs(ids[:n_atlas])
    write(arow, f"{outdir}/atlas_runs.tsv")
    with open(f"{outdir}/summary.md", "w") as f:
        f.write("# cDNA reads over 128 bases\n\n")
        f.write(f"Atlas stand-in: {len(arow)} runs from {min(n_atlas, len(ids))} random SRA experiments of {count} matching `{term}` (seed {seed}).\n\n")
        f.write(summarize(arow, "Atlas stand-in, human 10x single-cell runs in SRA"))
        f.write("\n" + summarize(crow, "Corpus runs (benchmark-store/datasets.tsv)"))
        nl = sum(1 for v in per_ds.values() if v > 128.5)
        f.write(f"\nCorpus datasets resolved in SRA: {len(per_ds)}. With any run over 128 bases: {nl}: "
                + ", ".join(sorted(d for d, v in per_ds.items() if v > 128.5)) + ".\n")
        f.write(f"\nCorpus datasets given only as URLs (10x site tars, read lengths not in SRA): {len(url_only)}: {', '.join(url_only)}.\n")
    print(open(f"{outdir}/summary.md").read())


if __name__ == "__main__":
    main()
