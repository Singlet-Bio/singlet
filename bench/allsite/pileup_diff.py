"""Which tool's AD do the raw reads support? Pile up the reads of the sites where engine and cellsnp-lite differ most.

  pileup_diff.py <singlet out dir (mtx)> <cellsnp-lite dir> <tagged BAM> <samtools> <n sites> <out.json>

Sites: the n matched sites with the largest |sum AD(cellsnp-lite) - sum AD(engine)| over the shared cells. For each, every read the
cellsnp-lite 1.2.3 defaults keep (MAPQ >= 20, no UNMAP/SECONDARY/QCFAIL, >= 30 aligned bases) is grouped by (CB, UB). cellsnp-lite counts
one base per UMI (the first read), with no base-quality filter and no consensus (csp_mplp_push in csp.c). Each cellsnp-lite ALT UMI is
classed by what its reads show, so the engine's lower AD can be read as masked errors or as dropped real molecules.
"""
import collections, gzip, json, os, re, subprocess, sys
import numpy as np
import scipy.io


def lines(p):
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt") as f:
        return [l.rstrip("\n") for l in f]


def read_base(cigar, pos, seq, qual, target):
    """Base and quality of the read at 0-based reference position target, or None."""
    rpos, qpos = pos, 0
    for n, op in cigar:
        if op in "M=X":
            if rpos <= target < rpos + n:
                k = qpos + target - rpos
                return seq[k], ord(qual[k]) - 33
            rpos += n; qpos += n
        elif op in "IS": qpos += n
        elif op in "DN":
            if rpos <= target < rpos + n: return None
            rpos += n
    return None


CIG = re.compile(r"(\d+)([MIDNSHP=X])")


def main(sdir, cdir, bam, samtools, nsites, out):
    feats = lines(os.path.join(sdir, "private/allsite/allele_ad_features.tsv.gz"))
    bcs = lines(os.path.join(sdir, "private/allsite/allele_ad_barcodes.tsv.gz"))
    sad = scipy.io.mmread(os.path.join(sdir, "private/allsite/allele_ad.mtx.gz")).tocsr()
    sdp = scipy.io.mmread(os.path.join(sdir, "private/allsite/allele_dp.mtx.gz")).tocsr()
    key = {}
    for i, n in enumerate(feats):
        c, p, ra = n.split(":"); key[(c, int(p), ra[2])] = i
    cbase = [l.split("\t") for l in lines(os.path.join(cdir, "cellSNP.base.vcf.gz")) if not l.startswith("#")]
    csam = lines(os.path.join(cdir, "cellSNP.samples.tsv"))
    cad = scipy.io.mmread(os.path.join(cdir, "cellSNP.tag.AD.mtx")).tocsr()
    rows_c, rows_s = [], []
    for j, t in enumerate(cbase):
        i = key.get((t[0], int(t[1]), t[4]))
        if i is not None: rows_c.append(j); rows_s.append(i)
    bidx = {b: k for k, b in enumerate(bcs)}
    strip = lambda b: b[:-2] if b.endswith("-1") and b[:-2] in bidx else b
    cols_c = [k for k, b in enumerate(csam) if strip(b) in bidx]
    cols_s = [bidx[strip(csam[k])] for k in cols_c]
    A_s = sad[rows_s][:, cols_s].tocsr(); A_c = cad[rows_c][:, cols_c].tocsr()
    del sad, sdp, cad
    psa = np.asarray(A_s.sum(1)).ravel(); pca = np.asarray(A_c.sum(1)).ravel()
    diff = pca - psa
    order = np.argsort(-np.abs(diff))[:nsites]
    cells = {strip(csam[k]) for k in cols_c}
    res = {"sites": int(len(order)), "sum_ad_cellsnp_sites": int(pca[order].sum()), "sum_ad_engine_sites": int(psa[order].sum()),
           "sites_cellsnp_higher": int((diff[order] > 0).sum())}
    cls = collections.Counter(); alt_umis = 0; ref_umis = 0; first_alt_pre = 0
    qhist = collections.Counter()
    for oi in order:
        t = cbase[rows_c[oi]]; chrom, pos1, ref, alt = t[0], int(t[1]), t[3], t[4]
        r = subprocess.run([samtools, "view", bam, f"{chrom}:{pos1}-{pos1}"], capture_output=True, text=True, check=False)
        umis = collections.OrderedDict()
        for ln in r.stdout.splitlines():
            f = ln.split("\t")
            flag, mapq = int(f[1]), int(f[4])
            if mapq < 20 or flag & 772: continue
            cigar = [(int(n), o) for n, o in CIG.findall(f[5])]
            if sum(n for n, o in cigar if o in "M=X") < 30: continue
            cb = ub = None
            for tag in f[11:]:
                if tag.startswith("CB:Z:"): cb = tag[5:]
                elif tag.startswith("UB:Z:"): ub = tag[5:]
            if cb is None or ub is None or strip(cb) not in cells: continue
            rb = read_base(cigar, int(f[3]) - 1, f[9], f[10], pos1 - 1)
            if rb is None: continue
            umis.setdefault((cb, ub), []).append(rb)
        for reads in umis.values():
            first = reads[0][0]
            if first != alt:
                ref_umis += 1; continue
            alt_umis += 1
            n = len(reads); na = sum(1 for b, q in reads if b == alt); q = reads[0][1]
            if n == 1: c = "single_read_q<20" if q < 20 else "single_read_q>=20"
            elif na == n: c = "multi_read_all_alt"
            elif na * 2 > n: c = "multi_read_alt_majority"
            else: c = "multi_read_alt_minority_first_read_alt"
            cls[c] += 1
            qhist["q<10" if q < 10 else "q10-19" if q < 20 else "q20-29" if q < 30 else "q>=30"] += 1
    res["cellsnp_rule_alt_umis_reproduced"] = alt_umis
    res["other_umis"] = ref_umis
    res["alt_umi_classes"] = dict(cls)
    res["alt_umi_first_read_quality"] = dict(qhist)
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5]), sys.argv[6])
