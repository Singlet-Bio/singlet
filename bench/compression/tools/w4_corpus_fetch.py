#!/usr/bin/env python
"""w4_corpus: central-directory inventory and selective download of .singlet bundles (HTTP range requests only).

    python -P tools/w4_corpus_fetch.py inventory            # corpus_w4.tsv -> results/w4_corpus_members.tsv, w4_corpus_bundles.tsv
    python -P tools/w4_corpus_fetch.py scan GSE... [--stop-after N]   # larger bundles: called cells per sample (cell_calls.tsv size)
    python -P tools/w4_corpus_fetch.py fetch [--ids a,b,c] [--max-bytes N]   # download wanted members into corpus_w4_corpus/<id>/

Layout of the output: corpus_w4_corpus/<id>/{exon_counts.1pz,intron_counts.1pz,sj_counts.1pz,splice_psi.1pz,cell_calls.tsv,
summary.json,pileup_stats.json,provenance.json,manifest.json,study_meta.json,sample.json,...}. A cached copy of every central
directory lives in corpus_w4_corpus/_cd/<GSE>.json so nothing is requested twice. Every HTTP request is appended to
results/w4_corpus_requests.tsv (the study rules ask for a few hundred at most).

Everything downloaded is untrusted data: this tool only stores bytes (stored or deflate members, CRC-checked); it executes nothing.
"""
import io, os, sys, json, time, zlib, struct, zipfile, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
BC = os.path.dirname(HERE)
OUT = os.path.join(BC, "corpus_w4_corpus")
RES = os.path.join(BC, "results")
UA = {"User-Agent": "singlet-compression-study/0.1 (curl-compatible)"}
BASE = "https://data.singlet.bio/data/{gse}/{gse}.singlet"
TAIL = 1 << 18          # bytes fetched from the end of a bundle (EOCD + central directory)
MAX_SINGLE = 150 << 20  # cap on any single download (task rule)
REQLOG = os.path.join(RES, "w4_corpus_requests.tsv")
NREQ = 0


def log_req(gse, kind, a, b, nbytes):
    global NREQ
    NREQ += 1
    new = not os.path.exists(REQLOG)
    with open(REQLOG, "a") as f:
        if new:
            f.write("time\tgse\tkind\tstart\tend\tbytes\n")
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')}\t{gse}\t{kind}\t{a}\t{b}\t{nbytes}\n")


def http_range(url, gse, a, b, kind="range"):
    """GET bytes a..b inclusive; verifies a 206 answer so a server that ignores Range cannot make us pull a whole bundle."""
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", **UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                if r.status != 206:
                    raise RuntimeError(f"server answered {r.status} to a range request")
                data = r.read()
            log_req(gse, kind, a, b, len(data))
            return data
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


class TailFile(io.RawIOBase):
    """Read-only seekable view of a remote file whose last TAIL bytes are cached (enough for zipfile to read the
    central directory without further requests). Reads outside the tail are range requests."""
    def __init__(self, gse):
        self.gse = gse
        self.url = BASE.format(gse=gse)
        req = urllib.request.Request(self.url, headers={"Range": f"bytes=-{TAIL}", **UA})
        with urllib.request.urlopen(req, timeout=120) as r:
            if r.status != 206:
                raise RuntimeError(f"{gse}: suffix range not honoured ({r.status})")
            cr = r.headers["Content-Range"]       # bytes a-b/total
            span, total = cr.split(" ")[1].split("/")
            a, b = span.split("-")
            self.tail = r.read()
        self.size = int(total)
        self.tail_off = int(a)
        log_req(gse, "tail", self.tail_off, self.size - 1, len(self.tail))
        self.pos = 0
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=io.SEEK_SET):
        if whence == io.SEEK_SET: self.pos = off
        elif whence == io.SEEK_CUR: self.pos += off
        else: self.pos = self.size + off
        return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        if self.pos >= self.tail_off:
            d = self.tail[self.pos - self.tail_off: self.pos - self.tail_off + n]
        else:
            d = http_range(self.url, self.gse, self.pos, self.pos + n - 1, "zipread")
        self.pos += len(d)
        return d
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)


def read_cd(gse):
    """Central directory of a bundle: list of dict(name, method, csize, usize, hdr_off, crc), cached on disk."""
    cache = os.path.join(OUT, "_cd", f"{gse}.json")
    if os.path.exists(cache):
        return json.load(open(cache))
    tf = TailFile(gse)
    zf = zipfile.ZipFile(io.BufferedReader(tf, buffer_size=1 << 16))
    ents = [dict(name=zi.filename, method=zi.compress_type, csize=zi.compress_size, usize=zi.file_size,
                 hdr_off=zi.header_offset, crc=zi.CRC) for zi in zf.infolist()]
    d = dict(gse=gse, size=tf.size, cd_start=zf.start_dir, entries=ents)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(d, open(cache, "w"))
    return d


def fetch_run(gse, ents, run_a, run_b):
    """Fetch the byte range [run_a, run_b) of the bundle once and slice the members out of it."""
    url = BASE.format(gse=gse)
    buf = http_range(url, gse, run_a, run_b - 1, "members")
    out = {}
    for e in ents:
        o = e["hdr_off"] - run_a
        sig, ver, flag, comp, mt, md, crc, csz, usz, nl, el = struct.unpack_from("<IHHHHHIIIHH", buf, o)
        if sig != 0x04034B50:
            raise RuntimeError(f"{e['name']}: bad local header")
        ds = o + 30 + nl + el
        raw = buf[ds: ds + e["csize"]]
        if len(raw) != e["csize"]:
            raise RuntimeError(f"{e['name']}: short read")
        if e["method"] == 0:
            data = raw
        elif e["method"] == 8:
            data = zlib.decompressobj(-15).decompress(raw, e["usize"] + 1)
        else:
            raise RuntimeError(f"{e['name']}: compression method {e['method']}")
        if (zlib.crc32(data) & 0xFFFFFFFF) != e["crc"] or len(data) != e["usize"]:
            raise RuntimeError(f"{e['name']}: CRC/size mismatch")
        out[e["name"]] = data
    return out


def read_rows(tsv):
    rows = []
    with open(tsv) as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            if line.strip() and not line.startswith("#"):
                rows.append(dict(zip(hdr, line.rstrip("\n").split("\t"))))
    return rows


STUDY_FILES = ["manifest.json", "study_meta.json"]
SAMPLE_WANT = ["exon_counts.1pz", "intron_counts.1pz", "cell_calls.tsv", "summary.json", "pileup_stats.json",
               "provenance.json", "sj_counts.1pz", "splice_psi.1pz"]


def sample_members(cd, gsm):
    pre = f"samples/{gsm}/"
    return [e for e in cd["entries"] if e["name"].startswith(pre)]


def cmd_inventory():
    rows = read_rows(os.path.join(BC, "corpus_w4.tsv"))
    gses = []
    for r in rows:
        if r["gse"] not in gses:
            gses.append(r["gse"])
    os.makedirs(RES, exist_ok=True)
    mt = open(os.path.join(RES, "w4_corpus_members.tsv"), "w")
    mt.write("id\tgse\tgsm\tmember\tmethod\tcsize\tusize\n")
    bt = open(os.path.join(RES, "w4_corpus_bundles.tsv"), "w")
    bt.write("gse\tbundle_bytes\tn_entries\tn_samples_in_bundle\tcd_bytes_est\tsample_ids\n")
    for g in gses:
        try:
            cd = read_cd(g)
        except Exception as e:
            print(f"[{g}] FAILED: {e}", flush=True)
            bt.write(f"{g}\tNA\tNA\tNA\tNA\tFAILED {e}\n")
            continue
        sids = sorted({e["name"].split("/")[1] for e in cd["entries"] if e["name"].startswith("samples/") and e["name"].count("/") >= 2})
        mine = [r for r in rows if r["gse"] == g]
        bt.write(f"{g}\t{cd['size']}\t{len(cd['entries'])}\t{len(sids)}\t{cd['size'] - cd['cd_start']}\t{','.join(r['id'] for r in mine)}\n")
        for r in mine:
            ms = sample_members(cd, r["gsm"])
            if not ms:
                print(f"[{g}] {r['id']} {r['gsm']}: NO samples/{r['gsm']}/ members; bundle samples: {sids[:5]}...", flush=True)
            for e in ms:
                mt.write(f"{r['id']}\t{g}\t{r['gsm']}\t{e['name'].split('/', 2)[2]}\t{e['method']}\t{e['csize']}\t{e['usize']}\n")
        for e in cd["entries"]:
            if not e["name"].startswith("samples/"):
                mt.write(f"-\t{g}\t-\t{e['name']}\t{e['method']}\t{e['csize']}\t{e['usize']}\n")
        print(f"[{g}] {cd['size']/1e6:.1f} MB, {len(cd['entries'])} entries, {len(sids)} samples", flush=True)
    mt.close(); bt.close()
    print(f"requests so far: {NREQ}")


def cmd_scan(args):
    stop_after = None
    if "--stop-after" in args:
        i = args.index("--stop-after"); stop_after = int(args[i + 1]); del args[i:i + 2]
    thresh = 30000
    path = os.path.join(RES, "w4_corpus_scan.tsv")
    new = not os.path.exists(path)
    out = open(path, "a")
    if new:
        out.write("gse\tbundle_bytes\tgsm\tn_called_est\tcell_calls_usize\texon_csize\tintron_csize\thas_sj\thas_psi\tmembers\n")
    found = 0
    for g in args:
        try:
            cd = read_cd(g)
        except Exception as e:
            print(f"[{g}] FAILED {e}", flush=True); continue
        by = {}
        for e in cd["entries"]:
            p = e["name"].split("/")
            if p[0] == "samples" and len(p) >= 3:
                by.setdefault(p[1], {})[p[2]] = e
        best = 0
        for gsm, mem in by.items():
            cc = mem.get("cell_calls.tsv")
            n_called = max(0, (cc["usize"] - 60) // 50) if cc else -1   # ~50 B per line (barcode, TRUE, 4dp, 0.0000e+00, star_solo)
            ex = mem.get("exon_counts.1pz"); inn = mem.get("intron_counts.1pz")
            out.write(f"{g}\t{cd['size']}\t{gsm}\t{n_called}\t{cc['usize'] if cc else -1}\t{ex['csize'] if ex else -1}\t{inn['csize'] if inn else -1}\t"
                      f"{int('sj_counts.1pz' in mem)}\t{int('splice_psi.1pz' in mem)}\t{len(mem)}\n")
            best = max(best, n_called)
            if n_called >= thresh:
                found += 1
        out.flush()
        print(f"[{g}] {cd['size']/1e6:.1f} MB, {len(by)} samples, max est called cells {best}", flush=True)
        if stop_after and found >= stop_after:
            break
    out.close()
    print(f"requests so far: {NREQ}")


def cmd_fetch(args):
    ids = None
    max_bytes = MAX_SINGLE
    if "--ids" in args:
        ids = args[args.index("--ids") + 1].split(",")
    if "--max-bytes" in args:
        max_bytes = int(args[args.index("--max-bytes") + 1])
    rows = read_rows(os.path.join(BC, "corpus_w4.tsv"))
    extra = os.path.join(OUT, "extra.tsv")   # large samples chosen by the scan (same columns)
    if os.path.exists(extra):
        rows += read_rows(extra)
    todo_by = {}
    for r in rows:
        if ids and r["id"] not in ids:
            continue
        todo_by.setdefault(r["gse"], []).append(r)
    fl = open(os.path.join(RES, "w4_corpus_fetch_log.tsv"), "a")
    for g, rs in todo_by.items():
        cd = read_cd(g)
        for r in rs:
            d = os.path.join(OUT, r["id"]); os.makedirs(d, exist_ok=True)
            ms = {e["name"].split("/", 2)[2]: e for e in sample_members(cd, r["gsm"])}
            if not ms:
                print(f"{r['id']}: no members", flush=True); fl.write(f"{r['id']}\t-\tNOMEMBERS\t0\n"); continue
            want = [ms[w] for w in SAMPLE_WANT if w in ms]
            # study files (small) once per study
            sw = [e for e in cd["entries"] if e["name"] in STUDY_FILES]
            todo = [e for e in want if not (os.path.exists(os.path.join(d, e["name"].split("/")[-1])) and
                                           os.path.getsize(os.path.join(d, e["name"].split("/")[-1])) == e["usize"])]
            todo += [e for e in sw if not os.path.exists(os.path.join(d, e["name"]))]
            if not todo:
                print(f"{r['id']}: complete", flush=True); continue
            total = sum(e["csize"] for e in todo)
            if total > max_bytes:
                # drop the largest optional members (sj/psi) first, then refuse
                for opt in ("splice_psi.1pz", "sj_counts.1pz"):
                    if total > max_bytes:
                        todo = [e for e in todo if not e["name"].endswith(opt)]
                        total = sum(e["csize"] for e in todo)
                if total > max_bytes:
                    print(f"{r['id']}: {total/1e6:.0f} MB exceeds cap {max_bytes/1e6:.0f} MB, skipped", flush=True)
                    fl.write(f"{r['id']}\t-\tOVERCAP\t{total}\n"); continue
            # coalesce into runs with gaps <= 1.5 MB (members of one sample are contiguous in the bundle)
            todo.sort(key=lambda e: e["hdr_off"])
            runs = []
            for e in todo:
                end = e["hdr_off"] + 30 + len(e["name"]) + 64 + e["csize"]
                if runs and e["hdr_off"] - runs[-1][1] <= (1536 << 10) and (end - runs[-1][0]) <= MAX_SINGLE:
                    runs[-1][1] = end; runs[-1][2].append(e)
                else:
                    runs.append([e["hdr_off"], end, [e]])
            for a, b, es in runs:
                b = min(b, cd["cd_start"])
                got = fetch_run(g, es, a, b)
                for e in es:
                    nm = e["name"]
                    dst = os.path.join(d, nm.split("/")[-1])
                    with open(dst, "wb") as f:
                        f.write(got[nm])
                    fl.write(f"{r['id']}\t{nm}\tOK\t{len(got[nm])}\n")
                print(f"{r['id']}: fetched {len(got)} members, request {(b - a)/1e6:.1f} MB", flush=True)
            with open(os.path.join(d, "sample.json"), "w") as f:
                json.dump(r, f, indent=1)
            fl.flush()
    fl.close()
    print(f"requests so far: {NREQ}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "inventory": cmd_inventory()
    elif cmd == "scan": cmd_scan(sys.argv[2:])
    elif cmd == "fetch": cmd_fetch(sys.argv[2:])
    else: print(__doc__)
