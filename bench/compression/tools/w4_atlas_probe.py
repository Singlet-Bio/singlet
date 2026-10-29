#!/usr/bin/env python
"""w4_atlas step 4: targeted ranged reads of single members and the header audit.

    python -P tools/w4_atlas_probe.py manifests        # manifest.json of ~16 stratified bundles -> results/w4_atlas_manifests.tsv
    python -P tools/w4_atlas_probe.py audit            # header audit of 20 bundles           -> results/w4_atlas_audit.tsv

get_member(gse, name, max_comp=None) does ONE ranged GET [lho, lho+30+len(name)+SLACK+csize) (or a prefix of the stored/deflated
data when max_comp is given), parses the local header, and returns the decoded bytes (inflated for method 8; a deflate PREFIX
is decodable from its start, so a truncated read still yields the first part of the text). CRC32 is checked when the whole
member was read.
"""
import csv, json, os, struct, sys, time, zlib, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import w4_atlas_scan as S

ROOT = S.ROOT
RES = S.OUT
SLACK = 512
_idx = None


def index():
    global _idx
    if _idx is None:
        _idx = {}
        with open(os.path.join(RES, "w4_atlas_members_raw.tsv"), encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                _idx[(r["gse"], r["name"])] = r
    return _idx


def get_member(gse, name, max_comp=None):
    """-> (data bytes, dict(info)). info: csize, usize, method, fetched, complete, crc_ok"""
    r = index()[(gse, name)]
    lho, cs, us, method = int(r["lho"]), int(r["csize"]), int(r["usize"]), int(r["method"])
    want = cs if max_comp is None else min(cs, max_comp)
    nlen = len(name.encode("utf-8"))
    end = lho + 30 + nlen + SLACK + want - 1
    st, h, body, ttfb, tt, tc = S.ranged(gse, f"bytes={lho}-{end}")
    assert st == 206, st
    sig, ver, flags, meth, mt, md, crc, c32, u32, nl, el = struct.unpack_from("<IHHHHHIIIHH", body, 0)
    assert sig == 0x04034B50, hex(sig)
    off = 30 + nl + el
    comp = body[off:off + want]
    complete = want == cs
    if method == 0:
        data = comp
    elif method == 8:
        d = zlib.decompressobj(-15)
        data = d.decompress(comp)
    else:
        raise ValueError(f"method {method}")
    ok = None
    if complete:
        ok = (zlib.crc32(data) & 0xFFFFFFFF) == int(r["crc32"], 16)
    return data, dict(csize=cs, usize=us, method=method, fetched=len(body), consumed_comp=len(comp), complete=complete, crc_ok=ok, ttfb=ttfb)


def pick_manifest_bundles():
    B = list(csv.DictReader(open(os.path.join(RES, "w4_atlas_bundles.tsv"), encoding="utf-8"), delimiter="\t"))
    M = list(csv.DictReader(open(os.path.join(RES, "w4_atlas_members.tsv"), encoding="utf-8"), delimiter="\t"))
    vocab = {r["gse"]: (r["usize"], r["crc32"]) for r in M if r["category"] == "feature_vocab.json"}
    chosen, seen = [], set()
    # one per vocab cluster (prefer the smallest bundle), then fill with smallest A and B bundles
    byc = collections.defaultdict(list)
    for b in B: byc[vocab[b["gse"]]].append(b)
    for k, v in byc.items():
        for gen in ("A", "B"):
            vv = sorted([b for b in v if b["gen"] == gen], key=lambda b: int(b["total"]))
            if vv: chosen.append(vv[0]["gse"]); seen.add(vv[0]["gse"])
    for gen, n in (("A", 4), ("B", 6)):
        vv = sorted([b for b in B if b["gen"] == gen and b["gse"] not in seen], key=lambda b: int(b["total"]))
        step = max(1, len(vv) // n)
        for b in vv[::step][:n]:
            chosen.append(b["gse"]); seen.add(b["gse"])
    return chosen


def cmd_manifests():
    out = []
    for g in pick_manifest_bundles():
        data, info = get_member(g, "manifest.json")
        m = json.loads(data)
        row = dict(gse=g, profile=m.get("profile", ""), reference_build=m.get("reference_build", ""), created_at=m.get("created_at", ""),
                   singlet_version=m.get("singlet_version", ""), schema=m.get("schema_version", ""), n_gsms=m.get("n_gsms", ""),
                   crc_ok=info["crc_ok"], keys=",".join(sorted(m.keys())))
        out.append(row)
        print(row, flush=True)
    cols = list(out[0].keys())
    with open(os.path.join(RES, "w4_atlas_manifests.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in out: f.write("\t".join(str(r[c]) for c in cols) + "\n")


def cmd_audit():
    B = list(csv.DictReader(open(os.path.join(RES, "w4_atlas_bundles.tsv"), encoding="utf-8"), delimiter="\t"))
    B.sort(key=lambda b: int(b["total"]))
    n = len(B)
    idx = sorted(set(int(round(i * (n - 1) / 19)) for i in range(20)))
    sel = [B[i] for i in idx]
    rows = []
    for k, b in enumerate(sel):
        g = b["gse"]
        rec = dict(gse=g, gen=b["gen"], size=b["total"])
        # 1 HEAD on a fresh connection (with Origin)
        S._tl.c = None
        st, h, body, ttfb, tt, tc = S.ranged(g, None, method="HEAD", fresh=True)
        rec.update(head_status=st, head_ttfb=round(ttfb, 4), head_connect=round(tc, 4), head_len=h.get("content-length"),
                   etag=h.get("etag"), cache_control=h.get("cache-control", ""), cf_cache=h.get("cf-cache-status", ""),
                   accept_ranges=h.get("accept-ranges", ""), content_type=h.get("content-type", ""), content_disp=h.get("content-disposition", ""),
                   acao=h.get("access-control-allow-origin", ""), acaeh=h.get("access-control-expose-headers", ""), vary=h.get("vary", ""),
                   age=h.get("age", ""), content_enc=h.get("content-encoding", ""), last_mod=h.get("last-modified", ""),
                   server=h.get("server", ""), alt_svc=h.get("alt-svc", ""), expires=h.get("expires", ""), all_head_keys=",".join(sorted(h.keys())))
        # 2 tail GET (same connection), 3 same again
        for tag in ("tail1", "tail2"):
            st, h, body, ttfb, tt, tc = S.ranged(g, "bytes=-65536")
            rec[f"{tag}_status"] = st; rec[f"{tag}_ttfb"] = round(ttfb, 4); rec[f"{tag}_total"] = round(tt, 4)
            rec[f"{tag}_cf_cache"] = h.get("cf-cache-status", ""); rec[f"{tag}_etag"] = h.get("etag", ""); rec[f"{tag}_cr"] = h.get("content-range", "")
            rec[f"{tag}_cc"] = h.get("cache-control", ""); rec[f"{tag}_acao"] = h.get("access-control-allow-origin", "")
            rec[f"{tag}_age"] = h.get("age", "")
        # 4 OPTIONS preflight (first 6 bundles only), 5 Accept-Encoding (first 4)
        if k < 6:
            st, h, body, ttfb, tt, tc = S.ranged(g, None, method="OPTIONS", extra={"Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "range"})
            rec.update(opt_status=st, opt_acao=h.get("access-control-allow-origin", ""), opt_acam=h.get("access-control-allow-methods", ""),
                       opt_acah=h.get("access-control-allow-headers", ""), opt_allow=h.get("allow", ""))
        if k < 4:
            st, h, body, ttfb, tt, tc = S.ranged(g, "bytes=-4096", extra={"Accept-Encoding": "gzip, deflate, br, zstd"})
            rec.update(ae_status=st, ae_content_encoding=h.get("content-encoding", ""), ae_len=len(body), ae_vary=h.get("vary", ""))
        rows.append(rec)
        print(g, b["gen"], b["total"], rec["head_ttfb"], rec["tail1_ttfb"], rec["tail2_ttfb"], rec["cf_cache"], flush=True)
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(os.path.join(RES, "w4_atlas_audit.tsv"), "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows: f.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")


if __name__ == "__main__":
    {"manifests": cmd_manifests, "audit": cmd_audit}[sys.argv[1]]()
