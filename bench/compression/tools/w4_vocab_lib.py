"""w4_vocab: shared helpers (loading fetched metadata blocks, parsing feature row names into their structure).

Row name grammar (every build seen so far):
    exon   : <gene_id>_<symbol>_<tag><chrom>:<start>-<end>
    intron : <gene_id>_<symbol>_intron_<tag><chrom>:<start>-<end>
<tag> is empty except in the barnyard build where it is "GRCh38___" or "GRCm39___".
"""
import hashlib, json, os, re, struct
import numpy as np
import zstandard

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "corpus_w4_vocab")


def parse_tlv(raw):
    rows, cols, kv = [], [], {}
    p = 0
    while p < len(raw):
        tag = raw[p]; p += 1
        if tag == 0:
            break
        (ln,) = struct.unpack_from("<I", raw, p); p += 4
        body = raw[p:p + ln]; p += ln
        if tag in (1, 2):
            parts = body.split(b"\0")
            assert parts[-1] == b""
            (rows if tag == 1 else cols).extend(x.decode("utf-8") for x in parts[:-1])
        elif tag == 3:
            parts = body.split(b"\0")[:-1]
            for i in range(0, len(parts), 2):
                kv[parts[i].decode()] = parts[i + 1].decode()
    return rows, cols, kv


def load_meta(path):
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(open(path, "rb").read())
    return parse_tlv(raw)


def load_vocab(path):
    return json.load(open(path))


def names_sha256(lst):
    h = hashlib.sha256()
    for s in lst:
        h.update(s.encode()); h.update(b"\0")
    return h.hexdigest()


class ParseError(Exception):
    pass


_COORD = re.compile(r"^(.*):(\d+)-(\d+)$")


def parse_rows(rows, kind, genes=None, tag=""):
    """Parse exon or intron rows into per-row tuples.

    genes: optional list of (gene_id, gene_name) from feature_vocab.json.  When given, the symbol of a row is taken to be
           gene_name of its gene (the prefix must match exactly), which resolves '_' ambiguities between symbol and
           chromosome; otherwise the chromosome is the text after the last '_' before the coordinates.
    Returns lists gid, sym, chrom, start, end (one entry per row) and raises ParseError on the first row that does not
    follow the grammar.
    """
    gids, syms, chroms, starts, ends = [], [], [], [], []
    gmap = {g: n for g, n in genes} if genes else None
    mid = "intron_" if kind == "intron" else ""
    for i, r in enumerate(rows):
        u = r.find("_")
        if u <= 0:
            raise ParseError("row %d: no gene id separator: %r" % (i, r))
        gid = r[:u]
        rest = r[u + 1:]
        m = _COORD.match(rest)
        if not m:
            raise ParseError("row %d: no coordinates: %r" % (i, r))
        pre, a, b = m.group(1), m.group(2), m.group(3)
        if str(int(a)) != a or str(int(b)) != b:
            raise ParseError("row %d: non canonical integer: %r" % (i, r))
        if gmap is not None and gid in gmap:
            # feature_vocab.json fills an empty symbol with the gene id, rows keep it empty (mouse, rat, macaque)
            cands = [gmap[gid]] + ([""] if gmap[gid] == gid else [])
            for sym in cands:
                head = sym + "_" + mid + tag
                if pre.startswith(head):
                    chrom = pre[len(head):]
                    break
            else:
                raise ParseError("row %d: prefix does not match gene table symbol %r: %r" % (i, cands[0], r))
        else:
            k = pre.rfind("_")
            if k < 0:
                raise ParseError("row %d: no symbol/chrom separator: %r" % (i, r))
            sym, chrom = pre[:k], pre[k + 1:]
            if mid:
                if not sym.endswith("_intron"):
                    raise ParseError("row %d: intron marker missing: %r" % (i, r))
                sym = sym[:-len("_intron")]
            if tag:
                if not chrom.startswith(tag):
                    raise ParseError("row %d: tag missing: %r" % (i, r))
                chrom = chrom[len(tag):]
        if not chrom:
            raise ParseError("row %d: empty chromosome: %r" % (i, r))
        gids.append(gid); syms.append(sym); chroms.append(chrom); starts.append(int(a)); ends.append(int(b))
    return gids, syms, chroms, starts, ends


def group_genes(gids, syms, chroms, starts, ends):
    """Group consecutive rows by gene id.  Returns list of [gid, sym, chrom, [(start,end),...]] or raises on a
    non-contiguous gene / symbol or chromosome change inside a gene."""
    out = []
    seen = set()
    cur = None
    for i in range(len(gids)):
        if cur is None or cur[0] != gids[i]:
            if gids[i] in seen:
                raise ParseError("gene %s is not contiguous (row %d)" % (gids[i], i))
            seen.add(gids[i])
            cur = [gids[i], syms[i], chroms[i], []]
            out.append(cur)
        elif cur[1] != syms[i] or cur[2] != chroms[i]:
            raise ParseError("symbol/chrom changes inside gene %s (row %d)" % (gids[i], i))
        cur[3].append((starts[i], ends[i]))
    return out
