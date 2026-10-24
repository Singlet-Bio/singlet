#!/usr/bin/env python
"""w4_atlas step 1: read the ZIP central directory of every sampled public bundle with ranged GETs.

    python -P tools/w4_atlas_scan.py w4_atlas_ids.txt            # writes results/w4_atlas_{members,bundles}.tsv

Per bundle: one `Range: bytes=-65536` GET (suffix range; the 206 carries Content-Range total, ETag, Last-Modified and the
cache headers, so no separate HEAD), plus one more ranged GET only if the central directory does not fit in the tail.
Own parser: EOCD, ZIP64 locator, ZIP64 EOCD, central directory with ZIP64 extra fields. No member data is read.
The raw tails are kept in corpus_w4_atlas/tails/ so the parse can be redone offline.
"""
import http.client, json, os, struct, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

HOST = "data.singlet.bio"
UA = "singlet-compression-study/0.1 (curl-compatible)"
ORIGIN = "https://example.org"
TAIL = 65536
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results")
TAILS = os.path.join(ROOT, "corpus_w4_atlas", "tails")

_tl = threading.local()


def conn(fresh=False):
    c = getattr(_tl, "c", None)
    if c is None or fresh:
        if c is not None:
            try: c.close()
            except Exception: pass
        t0 = time.perf_counter()
        c = http.client.HTTPSConnection(HOST, timeout=120)
        c.connect()
        _tl.c = c
        _tl.t_connect = time.perf_counter() - t0
    return c


def ranged(gse, rng, method="GET", extra=None, fresh=False):
    """-> (status, headers dict, body bytes, ttfb_s, total_s, t_connect_s)"""
    path = f"/data/{gse}/{gse}.singlet"
    hdr = {"User-Agent": UA, "Origin": ORIGIN, "Accept": "*/*"}
    if rng: hdr["Range"] = rng
    if extra: hdr.update(extra)
    for attempt in range(4):
        try:
            c = conn(fresh or attempt > 0)
            tc = getattr(_tl, "t_connect", 0.0) if (fresh or attempt > 0) else 0.0
            t0 = time.perf_counter()
            c.request(method, path, headers=hdr)
            r = c.getresponse()
            ttfb = time.perf_counter() - t0
            body = r.read()
            tt = time.perf_counter() - t0
            h = {k.lower(): v for k, v in r.getheaders()}
            if h.get("connection", "").lower() == "close":
                _tl.c = None
            return r.status, h, body, ttfb, tt, tc
        except Exception as e:
            _tl.c = None
            time.sleep(1.5 * (attempt + 1))
            err = e
    raise err


def parse_eocd(tail, tail_start, total):
    """-> dict(total_entries, cd_size, cd_offset, zip64, eocd_pos_abs, z64eocd_off or None, comment_len)"""
    i = tail.rfind(b"PK\x05\x06")
    while i >= 0:
        if i + 22 <= len(tail):
            clen = struct.unpack_from("<H", tail, i + 20)[0]
            if i + 22 + clen == len(tail):
                break
        i = tail.rfind(b"PK\x05\x06", 0, i)
    if i < 0:
        raise ValueError("no EOCD in tail")
    (sig, disk, cdisk, nthis, ntot, cdsize, cdoff, clen) = struct.unpack_from("<IHHHHIIH", tail, i)
    info = dict(total_entries=ntot, cd_size=cdsize, cd_offset=cdoff, zip64=False, eocd_pos=tail_start + i,
                z64_eocd_off=None, comment_len=clen, z64_extensible=0)
    if ntot == 0xFFFF or cdsize == 0xFFFFFFFF or cdoff == 0xFFFFFFFF:
        j = i - 20
        if j < 0 or tail[j:j + 4] != b"PK\x06\x07":
            raise ValueError("ZIP64 locator missing")
        _, d0, z64off, ndisks = struct.unpack_from("<IIQI", tail, j)
        info["zip64"] = True
        info["z64_eocd_off"] = z64off
        info["locator_pos"] = tail_start + j
    return info


def parse_z64_eocd(buf):
    sig, sz, vmade, vneed, d0, d1, nthis, ntot, cdsize, cdoff = struct.unpack_from("<IQHHIIQQQQ", buf, 0)
    if sig != 0x06064B50: raise ValueError("bad ZIP64 EOCD sig")
    return dict(total_entries=ntot, cd_size=cdsize, cd_offset=cdoff, z64_extensible=sz - 44)


def parse_cd(buf, n):
    out = []
    p = 0
    for _ in range(n):
        if buf[p:p + 4] != b"PK\x01\x02":
            raise ValueError(f"bad CD sig at {p}")
        (sig, vmade, vneed, flags, method, mtime, mdate, crc, cs, us, nl, el, cl, disk, ia, ea, lho) = \
            struct.unpack_from("<IHHHHHHIIIHHHHHII", buf, p)
        name = buf[p + 46:p + 46 + nl].decode("utf-8", "replace")
        extra = buf[p + 46 + nl:p + 46 + nl + el]
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack_from("<HH", extra, q)
            if hid == 1:
                z = q + 4
                if us == 0xFFFFFFFF: us = struct.unpack_from("<Q", extra, z)[0]; z += 8
                if cs == 0xFFFFFFFF: cs = struct.unpack_from("<Q", extra, z)[0]; z += 8
                if lho == 0xFFFFFFFF: lho = struct.unpack_from("<Q", extra, z)[0]; z += 8
            q += 4 + hsz
        out.append(dict(name=name, method=method, flags=flags, crc=crc, csize=cs, usize=us, lho=lho,
                        cd_entry_len=46 + nl + el + cl, extra_len=el, dos_date=mdate, dos_time=mtime))
        p += 46 + nl + el + cl
    if p != len(buf):
        raise ValueError(f"CD length mismatch {p} vs {len(buf)}")
    return out


def scan(gse):
    t0 = time.time()
    st, h, tail, ttfb, tt, tc = ranged(gse, f"bytes=-{TAIL}")
    if st != 206:
        raise RuntimeError(f"{gse}: status {st}")
    cr = h["content-range"]  # bytes a-b/total
    total = int(cr.rsplit("/", 1)[1])
    tail_start = total - len(tail)
    os.makedirs(TAILS, exist_ok=True)
    with open(os.path.join(TAILS, f"{gse}.tail"), "wb") as f:
        f.write(tail)
    nreq = 1
    e = parse_eocd(tail, tail_start, total)
    z64_in_tail = None
    if e["zip64"]:
        off = e["z64_eocd_off"]
        if off >= tail_start:
            z64_in_tail = True
            z = parse_z64_eocd(tail[off - tail_start:])
        else:
            z64_in_tail = False
            _, _, b, *_ = ranged(gse, f"bytes={off}-{off + 55}")
            nreq += 1
            z = parse_z64_eocd(b)
        e.update(z)
    cd_off, cd_size, n = e["cd_offset"], e["cd_size"], e["total_entries"]
    cd_in_tail = cd_off >= tail_start
    if cd_in_tail:
        cd = tail[cd_off - tail_start: cd_off - tail_start + cd_size]
    else:
        # fetch the part before the tail, reuse the part we already have
        pre_end = tail_start - 1
        _, _, b, *_ = ranged(gse, f"bytes={cd_off}-{pre_end}")
        nreq += 1
        cd = b + tail[: cd_off + cd_size - tail_start]
    members = parse_cd(cd, n)
    ev = dict(gse=gse, total=total, etag=h.get("etag", ""), last_modified=h.get("last-modified", ""),
              n_members=n, cd_size=cd_size, cd_offset=cd_off, zip64=int(e["zip64"]), cd_in_tail=int(cd_in_tail),
              z64_extensible=e.get("z64_extensible", 0), nreq=nreq, ttfb=ttfb, t_total=tt, t_connect=tc,
              resp_headers=h)
    return ev, members


def main():
    ids = open(sys.argv[1]).read().split()
    os.makedirs(OUT, exist_ok=True)
    bundles, allm, fails = [], [], []
    done = [0]
    lock = threading.Lock()

    def job(g):
        try:
            ev, ms = scan(g)
            with lock:
                bundles.append(ev)
                for i, m in enumerate(ms):
                    m["gse"] = g; m["idx"] = i
                    allm.append(m)
                done[0] += 1
                if done[0] % 20 == 0:
                    print(f"  {done[0]}/{len(ids)}", flush=True)
        except Exception as ex:
            with lock:
                fails.append((g, repr(ex)))
                print(f"FAIL {g}: {ex!r}", flush=True)

    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(job, ids))
    bundles.sort(key=lambda b: b["gse"])
    allm.sort(key=lambda m: (m["gse"], m["idx"]))
    with open(os.path.join(OUT, "w4_atlas_members_raw.tsv"), "w", encoding="utf-8") as f:
        f.write("gse\tidx\tname\tmethod\tflags\tcrc32\tcsize\tusize\tlho\tcd_entry_len\textra_len\n")
        for m in allm:
            f.write(f"{m['gse']}\t{m['idx']}\t{m['name']}\t{m['method']}\t{m['flags']}\t{m['crc']:08x}\t{m['csize']}\t{m['usize']}\t{m['lho']}\t{m['cd_entry_len']}\t{m['extra_len']}\n")
    with open(os.path.join(OUT, "w4_atlas_bundles_raw.tsv"), "w", encoding="utf-8") as f:
        cols = ["gse", "total", "etag", "last_modified", "n_members", "cd_size", "cd_offset", "zip64", "cd_in_tail",
                "z64_extensible", "nreq", "ttfb", "t_total", "t_connect"]
        f.write("\t".join(cols) + "\n")
        for b in bundles:
            f.write("\t".join(str(b[c]) for c in cols) + "\n")
    with open(os.path.join(OUT, "w4_atlas_scan_headers.jsonl"), "w", encoding="utf-8") as f:
        for b in bundles:
            f.write(json.dumps({"gse": b["gse"], "headers": b["resp_headers"]}) + "\n")
    print(f"bundles {len(bundles)}, members {len(allm)}, fails {len(fails)}; requests {sum(b['nreq'] for b in bundles)}")
    for g, e in fails: print("FAIL", g, e)


if __name__ == "__main__":
    main()
