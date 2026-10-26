# SPDX-License-Identifier: MIT
"""V(D)J segment reference built from a genome and its Ensembl / GENCODE annotation.

Every annotated ``TR_*_gene`` / ``IG_*_gene`` (V, D, J, C; pseudogenes excluded) gives one segment:

* **V**: the spliced exons of the gene's canonical transcript from the CDS start (leader + V region, sense strand; the
  5' UTR is cut, it varies between paralogs) and the position of the second conserved cysteine (IMGT Cys104): the last
  in-frame TGT/TGC codon within the last 45 nt.
* **J**: the exon, the frame in which the translation holds the F/W-G-X-G motif, and the position of that F/W codon
  (IMGT Phe/Trp118). The IMGT junction (``cdr3`` of Cell Ranger and TRUST4) runs from Cys104 to Phe/Trp118 inclusive.
* **D**: the exon.
* **C**: the spliced exons of the canonical transcript (the first exon is what V(D)J reads reach).

No IMGT or vendor reference is involved, so the reference has the annotation's licence (Ensembl / GENCODE: unrestricted)
and works for any species whose annotation has IG/TR biotypes. Built once per genome (``build``), stored as a small TSV
(``data/vdj_<species>.tsv.gz``) and loaded with :func:`load`.
"""

from __future__ import annotations

import gzip
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

BIOTYPES = {f"{l}_{r}_gene" for l in ("TR", "IG") for r in ("V", "D", "J", "C")}
CHAINS = ("TRA", "TRB", "TRG", "TRD", "IGH", "IGK", "IGL")
_COMP = str.maketrans("ACGTNacgtn", "TGCANtgcan")
CODON = {}
_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for _i in range(64):
    CODON[_b[_i // 16] + _b[(_i // 4) % 4] + _b[_i % 4]] = _aa[_i]
J_MOTIF = re.compile(r"[FW]G.G")
# a V segment named like one but annotated otherwise (GENCODE v44: TRBV11-2 is "lncRNA", TRBV21-1 a pseudogene) is kept
# when its own sequence is an open V (a frame with at most one stop before an in-frame Cys104), as IMGT's ORFs are
V_NAME = re.compile(r"^(TR[ABGD]|IG[HKL])V\d")
# the same for J segments (GENCODE v44 lacks TRAJ8 and TRAJ15 as TR_J_gene; IMGT has them functional): kept when the
# exon holds an F/W-G-X-G motif where a J region's 3' end puts it
J_NAME = re.compile(r"^(TR[ABGD]|IG[HKL])J\d")
# orphons (IGKV1OR2-108, TRBV20OR9-2 ...): segment copies outside the loci, never rearranged; they only steal votes
ORPHON = re.compile(r"OR\d", re.I)


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def translate(s: str, frame: int = 0) -> str:
    return "".join(CODON.get(s[i:i + 3], "X") for i in range(frame, len(s) - 2, 3))


def chain_of(name: str) -> Optional[str]:
    """TRAV12-1 -> TRA, IGHG1 -> IGH; TRDV genes inside the TRA locus keep their own name (TRDV1 -> TRD); TRAV..DV
    genes (TRAV29/DV5) are TRA (they pair as alpha far more often)."""
    p = name[:3].upper()
    return p if p in CHAINS else None


@dataclass
class Segment:
    name: str
    gene_id: str
    chain: str
    region: str  # V D J C
    seq: str  # sense strand
    anchor: int  # V: Cys104 codon start; J: F/W118 codon start; -1 otherwise / not found
    frame: int  # V: CDS frame of seq[0]; J: frame of the motif; -1 otherwise
    contig: str = ""
    start: int = 0  # 1-based genomic start of the first exon (for provenance)


# ------------------------------------------------------------------------------------------------------- building
def _attrs(s: str) -> Dict[str, str]:
    out = {}
    for m in re.finditer(r'(\S+) "([^"]*)"', s):
        out.setdefault(m.group(1), m.group(2))
    if 'tag "Ensembl_canonical"' in s:
        out["_canonical"] = "1"
    return out


def _read_fasta(path: str, wanted: set) -> Dict[str, str]:
    op = gzip.open if path.endswith(".gz") else open
    seqs: Dict[str, List[str]] = {}
    cur = None
    with op(path, "rt") as f:
        for ln in f:
            if ln.startswith(">"):
                name = ln[1:].split()[0]
                cur = name if name in wanted else None
                if cur:
                    seqs[cur] = []
            elif cur:
                seqs[cur].append(ln.strip())
    return {k: "".join(v).upper() for k, v in seqs.items()}


def parse_gtf(gtf: str):
    """-> {gene_id: dict(name, biotype, contig, strand, transcripts={tid: dict(exons, cds, canonical)})}"""
    op = gzip.open if gtf.endswith(".gz") else open
    genes: Dict[str, dict] = {}
    with op(gtf, "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.rstrip("\n").split("\t")
            if len(p) < 9 or p[2] not in ("exon", "CDS"):
                continue
            if '_gene' not in p[8] and 'gene_name "TR' not in p[8] and 'gene_name "IG' not in p[8]:
                continue
            a = _attrs(p[8])
            bt = a.get("gene_type") or a.get("gene_biotype")
            if bt not in BIOTYPES and not (V_NAME.match(a.get("gene_name", "")) or J_NAME.match(a.get("gene_name", ""))):
                continue
            gid = a["gene_id"].split(".")[0]
            g = genes.setdefault(gid, {"name": a.get("gene_name", gid), "biotype": bt, "contig": p[0], "strand": p[6],
                                       "transcripts": {}})
            t = g["transcripts"].setdefault(a.get("transcript_id", gid), {"exons": [], "cds": [], "canonical": False})
            t["canonical"] |= a.get("_canonical") == "1"
            (t["exons"] if p[2] == "exon" else t["cds"]).append((int(p[3]), int(p[4])))
    return genes


def _spliced(chrom: str, exons, strand: str) -> str:
    ex = sorted(exons)
    s = "".join(chrom[a - 1:b] for a, b in ex)
    return revcomp(s) if strand == "-" else s


def _v_anchor(seq: str, frame: int) -> int:
    lo = max(0, len(seq) - 45)
    best = -1
    for i in range(frame, len(seq) - 2, 3):
        if i >= lo and seq[i:i + 3] in ("TGT", "TGC"):
            best = i
    return best


def _j_anchor(seq: str):
    """F/W118 of a J exon: among F/W-G-X-G motifs in any frame, the one the J region's 3' end puts there (the J region
    ends at the exon's splice donor, 25-45 nt after the F/W codon); some annotated J exons carry 100+ nt upstream
    (GENCODE IGLJ1-3) where a spurious motif can sit in another frame."""
    hits = []
    for fr in range(3):
        aa = translate(seq, fr)
        for m in re.finditer(r"(?=[FW]G.G)", aa):
            pos = fr + 3 * m.start()
            hits.append((abs((len(seq) - pos) - 33), -pos, pos, fr))
    if not hits:
        return -1, -1
    near = [h for h in hits if 20 <= len(seq) - h[2] <= 50]
    _, _, pos, fr = min(near or hits)
    return pos, fr


def _open_v_frame(seq: str) -> int:
    """Frame of an unannotated V: the frame with a Cys104 anchor and at most one stop before it (the reference allele
    can carry a stop that donors' functional alleles lack: GRCh38 TRAV35), preferring one that starts with ATG; -1
    when there is none (a real pseudogene)."""
    best = -1
    for fr in range(3):
        anc = _v_anchor(seq, fr)
        if anc < 0 or translate(seq[fr:anc + 3]).count("*") > 1:
            continue
        if seq[fr:fr + 3] == "ATG":
            return fr
        best = fr if best < 0 else best
    return best


def build(genome_fa: str, gtf: str, extra: Sequence[Tuple[str, str]] = ()) -> List[Segment]:
    """Segments of the primary assembly, plus those of ``extra`` (genome, GTF) pairs (alternate haplotypes, as Cell
    Ranger's ``-alts`` reference has: TRBV4-3, TRBV6-3, TRBV7-8 ... are only on GRCh38 alt loci) whose gene names
    the primary assembly lacks."""
    out = _build_one(genome_fa, gtf)
    for g2, t2 in extra:
        have = {s.name for s in out}
        out += [s for s in _build_one(g2, t2, skip_contigs={s.contig for s in out}) if s.name not in have]
    # one segment per name (an alt locus can repeat a gene in several haplotypes)
    seen, uniq = set(), []
    for s in out:
        if s.name not in seen:
            seen.add(s.name)
            uniq.append(s)
    uniq.sort(key=lambda s: (s.chain, "VDJC".index(s.region), s.name))
    return uniq


def _build_one(genome_fa: str, gtf: str, skip_contigs=frozenset()) -> List[Segment]:
    genes = {k: g for k, g in parse_gtf(gtf).items() if g["contig"] not in skip_contigs}
    chroms = _read_fasta(genome_fa, {g["contig"] for g in genes.values()})
    out: List[Segment] = []
    for gid, g in genes.items():
        if g["contig"] not in chroms:
            continue
        region = g["biotype"][3] if g["biotype"] in BIOTYPES else ("J" if J_NAME.match(g["name"]) else "V")
        chain = chain_of(g["name"])
        if chain is None or ORPHON.search(g["name"]):
            continue
        ts = g["transcripts"]
        tid = next((t for t, v in ts.items() if v["canonical"]), None) or sorted(ts)[0]
        t = ts[tid]
        chrom = chroms[g["contig"]]
        seq = _spliced(chrom, t["exons"], g["strand"])
        anchor, frame = -1, -1
        if region == "V":
            frame = 0
            if t["cds"]:
                # CDS frame relative to the spliced transcript start
                ex = sorted(t["exons"], reverse=g["strand"] == "-")
                cds0 = min(a for a, _ in t["cds"]) if g["strand"] == "+" else max(b for _, b in t["cds"])
                off = 0
                for a, b in ex:
                    if a <= cds0 <= b:
                        off += (cds0 - a) if g["strand"] == "+" else (b - cds0)
                        break
                    off += b - a + 1
                # start at the CDS (leader ATG): annotated 5' UTRs vary between paralogs (IGLV2-8 has 117 nt more
                # than IGLV2-11) and would win k-mer votes from R1 reads that reach the transcript's 5' end
                seq = seq[off:]
            if g["biotype"] not in BIOTYPES:
                frame = _open_v_frame(seq)
                if frame < 0:
                    continue
                seq, frame = seq[frame:], 0
            anchor = _v_anchor(seq, frame)
        elif region == "J":
            anchor, frame = _j_anchor(seq)
            if g["biotype"] not in BIOTYPES and (anchor < 0 or not 20 <= len(seq) - anchor <= 50):
                continue
        start = min(a for a, _ in t["exons"])
        out.append(Segment(g["name"], gid, chain, region, seq, anchor, frame, g["contig"], start))
    return out


def write(segs: List[Segment], path: str, provenance: str = ""):
    with gzip.open(path, "wt") as f:
        if provenance:
            f.write(f"# {provenance}\n")
        f.write("name\tgene_id\tchain\tregion\tanchor\tframe\tcontig\tstart\tseq\n")
        for s in segs:
            f.write(f"{s.name}\t{s.gene_id}\t{s.chain}\t{s.region}\t{s.anchor}\t{s.frame}\t{s.contig}\t{s.start}\t{s.seq}\n")


def load(path_or_species: str = "human") -> List[Segment]:
    path = path_or_species
    if not os.path.exists(path):
        path = os.path.join(os.path.dirname(__file__), "data", f"vdj_{path_or_species}.tsv.gz")
    out = []
    with gzip.open(path, "rt") as f:
        for ln in f:
            if ln.startswith("#") or ln.startswith("name\t"):
                continue
            p = ln.rstrip("\n").split("\t")
            out.append(Segment(p[0], p[1], p[2], p[3], p[8], int(p[4]), int(p[5]), p[6], int(p[7])))
    return out


def equivalence(segs: List[Segment], max_mismatch: int = 1) -> Dict[str, str]:
    """V genes whose coding region up to Cys104 is identical or within ``max_mismatch`` substitutions (IGKV1-39 /
    IGKV1D-39, TRBV6-2 / TRBV6-3 copies): reads cannot tell them apart, so tools name them by convention. Returns
    name -> class label (the class's names joined by "|", sorted); genes in no class map to themselves."""
    vs = [s for s in segs if s.region == "V" and s.anchor >= 0]
    core = {s.name: s.seq[max(0, s.frame):s.anchor + 3] for s in vs}
    parent = {n: n for n in core}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    names = sorted(core)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            sa, sb = core[a], core[b]
            L = min(len(sa), len(sb))
            if L == 0:
                continue
            # align the 3' ends (Cys104): the leader length varies more than the V region
            if sum(x != y for x, y in zip(sa[-L:], sb[-L:])) <= max_mismatch and L >= 250:
                parent[find(a)] = find(b)
    groups: Dict[str, List[str]] = {}
    for n in names:
        groups.setdefault(find(n), []).append(n)
    out = {}
    for g in groups.values():
        lab = "|".join(sorted(g))
        for n in g:
            out[n] = lab
    return out


def summary(segs: List[Segment]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for s in segs:
        d = out.setdefault(s.chain, {})
        d[s.region] = d.get(s.region, 0) + 1
        if s.region in "VJ" and s.anchor < 0:
            d[s.region + "_no_anchor"] = d.get(s.region + "_no_anchor", 0) + 1
    return out


if __name__ == "__main__":  # python -m singlet.vdj.reference GENOME.fa GTF OUT.tsv.gz [ALT_GENOME.fa ALT_GTF]
    import json
    import sys

    extra = [(sys.argv[4], sys.argv[5])] if len(sys.argv) > 5 else []
    segs = build(sys.argv[1], sys.argv[2], extra)
    src = " + ".join(os.path.basename(x) for x in sys.argv[1:3] + sys.argv[4:6])
    write(segs, sys.argv[3], provenance=f"built from {src}")
    print(json.dumps(summary(segs), indent=1))
