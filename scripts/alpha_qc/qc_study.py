# SPDX-License-Identifier: MIT
"""Research Alpha QC of one transcoded study (or one shard of its samples), independent of the transcoder's own verify.

Everything is read over HTTPS from the public host (data.singlet.bio): the .s1 manifest, sample objects, packs and label
sidecars through the ``singlet-data`` Worker, the legacy ``.singlet`` bundle through the data guard (exact member ranges only).

Checks (numbers as in the QC brief):
  1 manifest integrity: every object exists, its size and SHA-256 match the manifest (and the listing), its key is its digest,
    its header names the sample; every referenced pack exists and its SHA-256 matches its id and s1/ref/index.json
  2 matrix equality: every legacy .1pz member the policy keeps, decoded with the LEGACY reader (singlet._pz), against the .s1
    member decoded with the client decoder (S1Sample.matrix): shape, nnz, indptr, indices, values, row names, column names, kv
  3 policy: no excluded accession anywhere (manifest, member table, stored blobs, cells table, sidecar); no withheld-layer
    member (rules re-implemented here from the release brief, not imported from singlet.s1.policy); marker scan of decoded text
    and raw object bytes; guard probes of private legacy members
  4 cells and labels: called count = legacy cell_calls; called_ed present; label coverage and validity; distributions
  5 provenance: policy version, reads used, strand fraction, library-type verdict per sample
  8 size: s1 bytes vs legacy bytes

    python qc_study.py GSE --policy alpha-policy.json --out result.json --scratch DIR [--shard K/N] [--sizes s1_s_sizes.tsv]
                           [--organism "Homo sapiens"] [--cl cl-basic.obo] [--full-get-max BYTES]
"""

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time
import traceback
import zlib

sys.path.insert(0, os.path.join(os.environ.get("QC_SRC", ""), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

import qc_http as H  # noqa: E402

GSM_RE = re.compile(rb"(?<![0-9A-Za-z])(GS[EM]\d+)(?![0-9])")
CL_RE = re.compile(r"^CL:\d{7}$")
HANC_RE = re.compile(r"^HANCESTRO:\d{4}$")

# markers of withheld layers in decoded text: (name, regex, strong). strong = content marker (a hit is a finding);
# weak = a layer *name* (a hit is reported with context, for review: e.g. a log line naming a file)
MARKERS = [
    ("vcf_header", rb"##fileformat=VCF", True),
    ("vcf_columns", rb"#CHROM\tPOS\tID\tREF\tALT", True),
    ("rsid", rb"(?<![0-9A-Za-z_])rs\d{4,}(?![0-9])", True),
    ("genotype_format", rb"\tGT:(?:AD|DP|GQ|PL)", True),
    ("snp_", rb"snp_", False),
    ("ase_counts", rb"ase_counts", False),
    ("ancestry_call", rb"ancestry_call", False),
    ("sex_call", rb"sex_call", False),
    ("doublet_scores", rb"doublet_scores", False),
    ("ambient_", rb"ambient_(?:contamination|profile)", False),
    ("mt_variants", rb"mt_variants|_mt_consensus|mt_heteroplasmy", False),
    ("splice_psi", rb"splice_psi", False),
    ("cellsnp_vireo", rb"cellsnp|vireo", False),
]
RAW_MARKERS = [m for m in MARKERS if m[2]]


def private_layer(rel):
    """Layer a sample member belongs to when the Research Alpha release withholds it (from the QC brief), else None."""
    rel = rel.replace("\\", "/")
    for suf in (".local", ".legacy_order"):
        if rel.endswith(".1pz" + suf):
            rel = rel[: -len(suf)]
    rel = rel.split("#", 1)[0]
    name = rel.rsplit("/", 1)[-1]
    if rel.startswith("donor/") and rel != "donor/donor_assignments.tsv":
        return "genotype"
    if name.startswith("snp_") or name == "ase_counts.tsv" or re.search(r"\.(vcf|bcf)(\.gz)?$", name, re.I):
        return "genotype"
    if re.search(r"\.(bam|bai|cram|crai|sam)$", name, re.I):
        return "genotype"
    if rel.startswith("mt/") and rel != "mt/mt_summary.tsv":
        return "mtdna_alleles"
    if name in ("mt_heteroplasmy.1pz", "mt_variants.tsv") or "_mt_consensus" in name or "_mt_variants" in name:
        return "mtdna_alleles"
    if name == "ancestry_call.json":
        return "ancestry_call"
    if name.startswith("ambient_"):
        return "ambient"
    if name.startswith("doublet_scores"):
        return "doublet_scores"
    if "nonhost" in rel and "per_cell" in name:
        return "nonhost_per_cell"
    if name == "splice_psi.1pz":
        return "splice_psi_legacy"
    if name == "sex_call.json":
        return "sex_call_legacy"
    return None


def guard_private(rel):
    """Members the data guard must refuse (genotype and mtDNA alleles, ancestry call): the private subset."""
    lay = private_layer(rel)
    return lay in ("genotype", "mtdna_alleles", "ancestry_call")


class Check:
    def __init__(self):
        self.d = {}

    def add(self, check, ok, what, **ev):
        c = self.d.setdefault(check, {"pass": 0, "fail": 0, "warn": 0, "fail_items": [], "warn_items": []})
        if ok is True:
            c["pass"] += 1
        elif ok == "warn":
            c["warn"] += 1
            if len(c["warn_items"]) < 50:
                c["warn_items"].append({"what": what, **ev})
        else:
            c["fail"] += 1
            if len(c["fail_items"]) < 50:
                c["fail_items"].append({"what": what, **ev})


def scan_text(blob, where, markers, hits, excluded, withheld, found_acc, ctx=40):
    for name, rx, strong in markers:
        for m in re.finditer(rx, blob):
            a, b = max(0, m.start() - ctx), min(len(blob), m.end() + ctx)
            hits.append({"where": where, "marker": name, "strong": strong,
                         "context": blob[a:b].decode("utf-8", "replace")})
            if sum(1 for h in hits if h["where"] == where and h["marker"] == name) >= 3:
                break
    if excluded is not None:
        for m in GSM_RE.finditer(blob):
            acc = m.group(1).decode()
            found_acc.add(acc)


def eq_arrays(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.array_equal(a, b))


def compare_matrix(leg, csc):
    """Legacy dict from singlet._pz.read_1pz vs a resolved client Csc. -> (ok, detail dict)."""
    d = {}
    lm, ln = int(leg["m"]), int(leg["n"])
    lip = np.asarray(leg["indptr"]).astype(np.int64)
    lix = np.asarray(leg["indices"]).astype(np.int64)
    ldt = np.asarray(leg["data"])
    if ldt.dtype.kind == "f":
        ldt = ldt.astype(np.float32).view(np.uint32)
    ldt = ldt.astype(np.uint64)
    d["shape_legacy"], d["shape_s1"] = [lm, ln], [int(csc.m), int(csc.n)]
    d["nnz_legacy"], d["nnz_s1"] = int(lip[-1]) if len(lip) else 0, int(csc.nnz)
    ok_shape = lm == csc.m and ln == csc.n
    ok_nnz = d["nnz_legacy"] == d["nnz_s1"]
    ok_ptr = eq_arrays(lip, np.asarray(csc.indptr).astype(np.int64))
    sdt = np.asarray(csc.data)
    if sdt.dtype.kind == "f":
        sdt = sdt.astype(np.float32).view(np.uint32)
    ok_ix = ok_ptr and eq_arrays(lix, np.asarray(csc.indices).astype(np.int64))
    ok_val = ok_ptr and eq_arrays(ldt, sdt.astype(np.uint64))
    if ok_ptr and not (ok_ix and ok_val):  # element-identical after a per-column sort?
        def canon(ix, dt):
            col = np.repeat(np.arange(len(lip) - 1), np.diff(lip))
            o = np.lexsort((ix, col))
            return ix[o], dt[o]
        a = canon(lix, ldt)
        b = canon(np.asarray(csc.indices).astype(np.int64), sdt.astype(np.uint64))
        d["equal_after_column_sort"] = eq_arrays(a[0], b[0]) and eq_arrays(a[1], b[1])
    ok_rows = list(leg["rownames"]) == list(csc.rownames or [])
    ok_cols = list(leg["colnames"]) == list(csc.colnames or [])
    lkv, skv = dict(leg.get("user_kv") or {}), dict(csc.kv or {})
    ok_kv = lkv == skv
    d.update(shape=ok_shape, nnz=ok_nnz, indptr=ok_ptr, indices=ok_ix, values=ok_val, rownames=ok_rows,
             colnames=ok_cols, kv=ok_kv, n_rownames_legacy=len(leg["rownames"]), n_colnames_legacy=len(leg["colnames"]))
    if not ok_rows and leg["rownames"] and csc.rownames:
        k = next((i for i, (x, y) in enumerate(zip(leg["rownames"], csc.rownames)) if x != y), None)
        d["first_row_diff"] = [k, leg["rownames"][k] if k is not None else None, csc.rownames[k] if k is not None else None]
    if not ok_kv:
        d["kv_legacy_keys"], d["kv_s1_keys"] = sorted(lkv)[:20], sorted(skv)[:20]
    ok = all([ok_shape, ok_nnz, ok_ptr, ok_ix, ok_val, ok_rows, ok_cols, ok_kv])
    return ok, d


def legacy_called(data):
    """Called barcodes of a legacy cell_calls.tsv, with the legacy reader's rule (is_cell column, else every listed row)."""
    from singlet.bundle import _called_from_calls
    import pandas as pd

    if not data.strip():
        return None
    df = pd.read_csv(io.BytesIO(data), sep="\t", dtype=str)
    c = _called_from_calls(df)
    return c


def load_obo_ids(path):
    ids, names = set(), {}
    if not path or not os.path.exists(path):
        return None, names
    cur = None
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("[Term]"):
                cur = None
            elif line.startswith("id: "):
                cur = line[4:].strip()
                ids.add(cur)
            elif line.startswith("name: ") and cur:
                names[cur] = line[6:].strip()
    return ids, names


def species_of(build, organism):
    b = (build or "").lower()
    if "barnyard" in b:
        return "barnyard"
    if b.startswith("grch38"):
        return "human"
    if b.startswith("grcm39"):
        return "mouse"
    o = (organism or "").lower()
    return {"homo sapiens": "human", "mus musculus": "mouse"}.get(o, "other")


# ------------------------------------------------------------------------------------------------------------------ main
def run(a):
    t0 = time.time()
    pol = json.load(open(a.policy))
    excluded, withheld = set(pol["excluded_samples"]), set(pol["withheld_studies"])
    gse = a.gse
    shard_k, shard_n = (int(x) for x in a.shard.split("/"))
    C = Check()
    R = {"gse": gse, "shard": a.shard, "policy_version": pol["version"], "samples": {}, "study": {}}
    sizes = {}
    if a.sizes and os.path.exists(a.sizes):
        op = gzip.open if a.sizes.endswith(".gz") else open
        with op(a.sizes, "rt") as fh:
            for line in fh:
                k, s = line.rstrip("\n").split("\t")[:2]
                if k.startswith("s1/s/") or k.startswith("s1/a/"):
                    sizes[k] = int(s)
    cl_ids, cl_names = load_obo_ids(a.cl)
    R["study"]["cl_ontology_loaded"] = cl_ids is not None and len(cl_ids) > 0

    # ---------------- manifest
    st, hdr, raw = H.get(f"s1/m/{gse}.json")
    R["study"]["manifest_http"] = st
    if gse in withheld:
        C.add("3_policy_withheld_manifest", st == 404, "withheld study manifest status", status=st)
        R["checks"] = C.d
        return R
    if st != 200:
        C.add("1_manifest_exists", False, "manifest not served", status=st)
        R["checks"] = C.d
        return R
    C.add("1_manifest_exists", True, "manifest")
    man = json.loads(raw)
    R["study"]["manifest_bytes"] = len(raw)
    R["study"]["manifest_cache_control"] = hdr.get("cache-control")
    # accession scan of the whole manifest text
    accs = set(m.group(1).decode() for m in GSM_RE.finditer(raw))
    bad = sorted(accs & excluded)
    badw = sorted(accs & withheld)
    C.add("3_policy_manifest_accessions", not bad and not badw, "excluded/withheld accession in manifest",
          excluded=bad[:20], withheld=badw[:20])
    # marker scan of the manifest without the release_policy block (which names the omitted layers by design)
    man_scan = {k: v for k, v in man.items() if k != "release_policy"}
    mh = []
    scan_text(json.dumps(man_scan).encode(), "manifest", MARKERS, mh, None, None, set())
    R["study"]["manifest_marker_hits"] = mh[:30]
    C.add("1_manifest_gse", man.get("gse") == gse and man.get("format") == "singlet-s1", "manifest gse/format",
          gse=man.get("gse"), format=man.get("format"))
    # provenance (study level)
    rp = man.get("release_policy") or {}
    C.add("5_provenance_policy_version", rp.get("version") == pol["version"], "manifest release_policy.version",
          got=rp.get("version"), want=pol["version"])
    R["study"]["release_policy"] = rp
    R["study"]["reference_build"] = man.get("reference_build")
    R["study"]["packs"] = man.get("packs")
    R["study"]["legacy"] = {k: v for k, v in (man.get("legacy") or {}).items() if k in ("key", "bytes", "sha256")}
    R["study"]["n_samples_manifest"] = len(man.get("samples") or [])
    R["study"]["transcode"] = man.get("transcode")
    R["study"]["annotations_default"] = man.get("annotations_default")
    species = species_of(man.get("reference_build"), a.organism)
    R["study"]["species"] = species

    # study files: decoded text scan + accession scan
    sf_text = []
    for name, ent in (man.get("study_files") or {}).items():
        if "zstd_b64" in ent:
            import base64
            import zstandard

            b = zstandard.ZstdDecompressor().decompress(base64.b64decode(ent["zstd_b64"]), max_output_size=1 << 28)
            ok = hashlib.sha256(b).hexdigest() == ent.get("sha256") and len(b) == ent.get("size")
            C.add("1_study_file_inline", ok, f"study file {name} inline bytes vs sha256/size")
            h2 = []
            fa = set()
            scan_text(b, f"study_file:{name}", MARKERS, h2, excluded, withheld, fa)
            sf_text += h2
            bad = sorted(fa & (excluded | withheld))
            C.add("3_policy_study_file_accessions", not bad, f"excluded/withheld accession in study file {name}",
                  accessions=bad[:20])
            if name == "manifest.json":
                try:
                    mj = json.loads(b)
                    inc = mj.get("included_files") or {}
                    bad_m = [f"{g}/{r}" for g, lst in inc.items() for r in (lst or []) if private_layer(str(r))]
                    C.add("3_policy_study_file_members", not bad_m, "legacy manifest.json lists withheld members",
                          members=bad_m[:20])
                    R["study"]["legacy_manifest_release_policy"] = mj.get("release_policy")
                except ValueError:
                    pass
    R["study"]["study_file_marker_hits"] = sf_text[:30]

    # packs
    st_i, _h, idx_raw = H.get("s1/ref/index.json")
    index = {p["id"]: p for p in json.loads(idx_raw).get("packs", [])} if st_i == 200 else {}
    pack_dir = os.path.join(a.scratch, "packs")
    os.makedirs(pack_dir, exist_ok=True)
    for role, pid in (man.get("packs") or {}).items():
        if not pid:
            continue
        dest = os.path.join(pack_dir, pid + ".pack")
        stp, n, sha, _ = H.download(f"s1/ref/{pid}.pack", dest)
        ok = stp == 200 and sha is not None and sha[:32] == pid
        ok_idx = pid in index and index[pid].get("sha256") == sha and int(index[pid].get("bytes", -1)) == n
        C.add("1_pack_exists_sha", ok, f"pack {role}", pack=pid, status=stp, sha256=sha)
        C.add("1_pack_in_index", ok_idx, f"pack {role} in s1/ref/index.json", pack=pid)

    from singlet.s1 import packs as P
    from singlet.s1.sample import S1Sample
    from singlet import _pz

    store = P.PackStore(dirs=[pack_dir], download=True)

    # ---------------- legacy bundle (through the guard)
    leg_key = (man.get("legacy") or {}).get("key") or f"data/{gse}/{gse}.singlet"
    st_h, hh, _ = H.request(f"{H.BASE}/{leg_key}", method="HEAD")
    R["study"]["legacy_head"] = {"status": st_h, "bytes": int(hh.get("content-length", 0) or 0),
                                 "x-alpha-policy": hh.get("x-alpha-policy"), "served_by": hh.get("x-served-by")}
    rz = None
    if st_h == 200:
        try:
            rz = H.RangeZip(leg_key, int(hh["content-length"]))
        except Exception as e:  # noqa: BLE001
            C.add("2_legacy_readable", False, "legacy central directory", error=repr(e)[:300])
    else:
        C.add("2_legacy_readable", False, "legacy bundle HEAD", status=st_h)
    leg_samples = {}
    if rz is not None:
        for nm in rz.entries:
            if nm.startswith("samples/"):
                p = nm.split("/", 2)
                if len(p) == 3:
                    leg_samples.setdefault(p[1], []).append(p[2])
    man_gsms = [s["gsm"] for s in man.get("samples") or []]
    R["study"]["n_samples_legacy"] = len(leg_samples)
    R["study"]["n_legacy_excluded"] = sum(1 for g in leg_samples if g in excluded)
    R["study"]["legacy_private_members"] = sum(1 for g, lst in leg_samples.items() for r in lst if guard_private(r))
    R["study"]["legacy_genotype_members"] = sum(1 for g, lst in leg_samples.items() for r in lst
                                                if private_layer(r) == "genotype")
    R["study"]["legacy_layer_counts"] = {}
    for g, lst in leg_samples.items():
        for r in lst:
            lay = private_layer(r)
            if lay:
                R["study"]["legacy_layer_counts"][lay] = R["study"]["legacy_layer_counts"].get(lay, 0) + 1
    if shard_k == 0 and rz is not None:
        missing = sorted(g for g in leg_samples if g not in excluded and g not in man_gsms)
        extra = sorted(g for g in man_gsms if g not in leg_samples)
        C.add("1_manifest_sample_set", not missing and not extra, "manifest samples vs legacy samples minus excluded",
              missing_included=missing[:30], not_in_legacy=extra[:30], n_missing=len(missing))
        C.add("3_policy_excluded_in_manifest", not (set(man_gsms) & excluded), "excluded GSM as manifest sample",
              gsms=sorted(set(man_gsms) & excluded)[:30])
        if len(man_gsms) != len(set(man_gsms)):
            C.add("1_manifest_sample_set", False, "duplicate GSM in manifest")
        # guard probes: private members must be refused; members of excluded samples: recorded
        probes = {"private": [], "excluded_sample": [], "public": []}
        priv = [f"samples/{g}/{r}" for g, lst in leg_samples.items() for r in lst if guard_private(r)]
        exs = [f"samples/{g}/{r}" for g, lst in leg_samples.items() if g in excluded for r in lst
               if not guard_private(r) and r.endswith((".1pz", ".tsv", ".json"))]
        for nm in priv[:4]:
            stp = rz.probe(nm)
            probes["private"].append([nm, stp])
            C.add("3_guard_private_refused", stp == 403, "guard serves a private legacy member", member=nm, status=stp)
        for nm in exs[:3]:
            stp = rz.probe(nm)
            probes["excluded_sample"].append([nm, stp])
            C.add("3_guard_excluded_sample_refused", stp in (403, 404), "guard serves a member of an excluded sample",
                  member=nm, status=stp)
        R["study"]["guard_probes"] = probes
        # whole-file GET of a small bundle with private members: the bytes of a private member must not be readable
        if priv and a.full_get_max and int(hh.get("content-length", 0) or 0) <= a.full_get_max:
            dest = os.path.join(a.scratch, "legacy_full.singlet")
            stf, nf, _sha, _ = H.download(leg_key, dest)
            res = {"status": stf, "bytes": nf, "private_readable": []}
            if stf == 200:
                with open(dest, "rb") as fh:
                    for nm in priv[:20]:
                        e = rz.entries[nm]
                        fh.seek(e["off"])
                        lh = fh.read(30)
                        if lh[:4] != b"PK\x03\x04":
                            continue
                        nl, el = int.from_bytes(lh[26:28], "little"), int.from_bytes(lh[28:30], "little")
                        fh.seek(e["off"] + 30 + nl + el)
                        d = fh.read(e["csize"])
                        try:
                            d2 = zlib.decompress(d, -15) if e["method"] == 8 else d
                            if (zlib.crc32(d2) & 0xFFFFFFFF) == e["crc"] and e["usize"] > 0:
                                res["private_readable"].append(nm)
                        except zlib.error:
                            pass
                os.remove(dest)
            R["study"]["guard_full_get"] = res
            C.add("3_guard_full_get", stf != 200 or not res["private_readable"],
                  "whole-bundle GET returns private member bytes intact", status=stf,
                  members=res["private_readable"][:10])

    # ---------------- samples (this shard)
    sel = [s for i, s in enumerate(man.get("samples") or []) if i % shard_n == shard_k]
    tot_s1 = 0
    for s in sel:
        g = s["gsm"]
        SR = {"bytes": s.get("bytes"), "n_columns": s.get("n_columns"), "n_called": s.get("n_called")}
        R["samples"][g] = SR
        try:
            sample_checks(a, C, SR, s, g, man, rz, leg_samples, store, S1Sample, _pz, excluded, withheld, sizes,
                          species, cl_ids)
        except Exception as e:  # noqa: BLE001
            C.add("0_sample_error", False, "exception while checking sample", gsm=g,
                  error=traceback.format_exc()[-1500:])
        tot_s1 += int(s.get("bytes") or 0)
    R["study"]["s1_bytes_shard"] = tot_s1
    R["study"]["s1_bytes_total"] = sum(int(s.get("bytes") or 0) for s in man.get("samples") or [])
    R["study"]["elapsed_s"] = round(time.time() - t0, 1)
    R["checks"] = C.d
    return R


def sample_checks(a, C, SR, s, g, man, rz, leg_samples, store, S1Sample, _pz, excluded, withheld, sizes, species,
                  cl_ids):
    key = s["key"]
    # ---- 1: object existence, size, sha256, content address
    C.add("1_key_layout", key.startswith(f"s1/s/{g}/") and key.endswith(".s1"), "sample key layout", gsm=g, key=key)
    dest = os.path.join(a.scratch, g + ".s1")
    t = time.time()
    st, n, sha, hdr = H.download(key, dest)
    SR["download_s"] = round(time.time() - t, 2)
    ok = st == 200 and n == s.get("bytes") and sha == s.get("sha256")
    C.add("1_object_size_sha", ok, "sample object vs manifest", gsm=g, status=st, bytes=n, want_bytes=s.get("bytes"),
          sha256=sha, want_sha256=s.get("sha256"))
    if st != 200:
        return
    d16 = key.rsplit("/", 1)[-1][:-3]
    C.add("1_object_content_address", sha is not None and sha[:32] == d16, "key digest = sha256[:16 bytes]", gsm=g)
    if key in sizes:
        C.add("1_object_listing_size", sizes[key] == n, "R2 listing size vs downloaded", gsm=g, listing=sizes[key],
              bytes=n)
    SR["cache_control"] = hdr.get("cache-control")
    smp = S1Sample(dest, store)
    rd = smp.reader
    C.add("1_object_header", rd.sample_id == g and rd.file_len == n, "object header sample id / file_len", gsm=g,
          sample_id=rd.sample_id, file_len=rd.file_len)
    man_members = set((s.get("members") or {}).keys())
    obj_members = set(rd.members.keys())
    C.add("1_member_table_vs_manifest", man_members == obj_members, "manifest members vs object member table", gsm=g,
          only_manifest=sorted(man_members - obj_members)[:20], only_object=sorted(obj_members - man_members)[:20])
    for nm, mm in (s.get("members") or {}).items():
        mi = rd.members.get(nm)
        if mi is None:
            continue
        if mm.get("off") != mi.off or mm.get("len") != mi.len or mm.get("kind") != mi.kind:
            C.add("1_member_ranges", False, "manifest member range differs from object table", gsm=g, member=nm)
        else:
            C.add("1_member_ranges", True, "")

    # ---- 3: policy on the object
    acc = set()
    hits = []
    mt_ref = None
    for nm, mi in rd.members.items():
        lay = private_layer(nm)
        C.add("3_policy_member_layer", lay is None, "withheld-layer member in public object", gsm=g, member=nm,
              layer=lay)
        meta_txt = (nm + "\0" + json.dumps(mi.meta or {})).encode()
        scan_text(meta_txt, f"{g}:table:{nm}", MARKERS, hits, excluded, withheld, acc)
        if mi.kind in (3, 5):
            try:
                blob = rd.legacy_blob(nm)
            except Exception as e:  # noqa: BLE001
                C.add("1_stored_member_crc", False, "stored member fails CRC/decompress", gsm=g, member=nm,
                      error=repr(e)[:200])
                continue
            C.add("1_stored_member_crc", True, "")
            if len(blob) <= 512 << 20:
                scan_text(blob, f"{g}:{nm}", MARKERS, hits, excluded, withheld, acc)
            if nm == "mt_reference.fa":
                mt_ref = {"sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
                          "first_line": blob.split(b"\n", 1)[0][:80].decode("utf-8", "replace"),
                          "n_N": blob.count(b"N"), "iupac_ambig": len(re.findall(rb"[RYKMSWBDHV]", blob.split(b"\n", 1)[-1]))}
    SR["mt_reference"] = mt_ref
    with open(dest, "rb") as fh:
        rawb = fh.read()
    for name, rx, _strong in RAW_MARKERS:
        k = len(re.findall(rx, rawb))
        if k:
            hits.append({"where": f"{g}:raw_object", "marker": name, "strong": False, "count": k,
                         "context": "(raw compressed bytes; random-match rate applies)"})
    del rawb

    cells = smp.cells()
    if cells is not None:
        for cn in cells.names:
            v = cells.column(cn)
            if cells.dtype(cn) in (8, 9) or (isinstance(v, (list, np.ndarray)) and len(v) and isinstance(v[0], str)):
                txt = "\n".join(str(x) for x in set(v) if x).encode() if len(v) else b""
                scan_text(txt, f"{g}:cells:{cn}", MARKERS, hits, excluded, withheld, acc)
        SR["cells_columns"] = list(cells.names)
        sus = [c for c in cells.names if re.search(r"doublet|ambient|sex_call|ancestry_call|psi|hetero|vcf|snp|genotype", c)]
        SR["cells_columns_from_omitted_layers"] = sus
        if sus:
            C.add("3_cells_columns_omitted_layers", "warn", "cells table carries columns named like an omitted layer",
                  gsm=g, columns=sus)
    bad = sorted(acc & excluded)
    C.add("3_policy_object_accessions", not bad and not (acc & withheld), "excluded/withheld accession inside object",
          gsm=g, excluded=bad[:20], withheld=sorted(acc & withheld)[:20])
    strong = [h for h in hits if h["strong"]]
    C.add("3_policy_marker_strong", not strong, "genotype content marker in decoded object text", gsm=g,
          hits=strong[:5])
    SR["marker_hits"] = hits[:40]

    # ---- annotations sidecar
    for ver, ent in (s.get("annotations") or {}).items():
        dsc = os.path.join(a.scratch, g + ".parquet")
        st2, n2, sha2, _ = H.download(ent["key"], dsc)
        C.add("1_sidecar_size_sha", st2 == 200 and n2 == ent.get("bytes") and sha2 == ent.get("sha256"),
              "label sidecar vs manifest", gsm=g, status=st2, key=ent["key"])
        if st2 == 200:
            import pyarrow.parquet as pq

            tb = pq.read_table(dsc)
            sc = tb.to_pydict()
            sacc = set()
            for col, vals in sc.items():
                if vals and isinstance(vals[0], str):
                    for m in GSM_RE.finditer("\n".join(set(x for x in vals if x)).encode()):
                        sacc.add(m.group(1).decode())
            md = tb.schema.metadata or {}
            for k_, v_ in md.items():
                for m in GSM_RE.finditer(k_ + b"\0" + v_):
                    sacc.add(m.group(1).decode())
            C.add("3_policy_sidecar_accessions", not (sacc & excluded), "excluded accession in label sidecar", gsm=g,
                  excluded=sorted(sacc & excluded)[:10])
            SR["sidecar"] = {"rows": tb.num_rows, "columns": tb.column_names, "n_rows_manifest": ent.get("n_rows")}
            if cells is not None:
                called = np.asarray(cells.column("called"), bool)
                idx = np.flatnonzero(called)
                colv = sc.get("col")
                C.add("4_sidecar_matches_cells", colv is not None and list(idx) == list(colv), "sidecar rows = called cells",
                      gsm=g, n_sidecar=tb.num_rows, n_called=int(called.sum()))
                for lc in ("cell_type_cl", "sex_chrom", "ancestry_hancestro"):
                    if lc in sc and cells.has(lc) and colv is not None and list(idx) == list(colv):
                        cv = [cells.column(lc)[i] or "" for i in idx]
                        C.add("4_sidecar_values", cv == [x or "" for x in sc[lc]], f"sidecar {lc} vs cells table", gsm=g)
            os.remove(dsc)

    # ---- 5: provenance (sample level)
    pv = s.get("provenance") or {}
    stv = (pv.get("strand") or {})
    lt = (pv.get("library_type") or {})
    okp = (pv.get("reads_used") is not None and isinstance(stv.get("wrong_strand_fraction"), (int, float))
           and bool(lt.get("verdict")) and pv.get("release_policy") == json.load(open(a.policy))["version"])
    C.add("5_provenance_sample", okp, "sample provenance fields", gsm=g, reads_used=pv.get("reads_used"),
          wrong_strand_fraction=stv.get("wrong_strand_fraction"), library_type=lt.get("verdict"),
          release_policy=pv.get("release_policy"))
    SR["provenance"] = {"reads_used": pv.get("reads_used"), "reads_used_source": pv.get("reads_used_source"),
                        "wrong_strand_fraction": stv.get("wrong_strand_fraction"), "strand_class": stv.get("class"),
                        "library_type": lt.get("verdict")}

    # ---- 2: matrix equality against the legacy reader
    leg_members = leg_samples.get(g, [])
    called_leg = None
    if rz is None:
        C.add("2_matrix_equal", False, "no legacy bundle to compare", gsm=g)
    else:
        mats = [r for r in leg_members if r.endswith(".1pz") and private_layer(r) is None]
        SR["legacy_1pz"] = mats
        SR["matrices"] = {}
        for r in mats:
            mi = rd.members.get(r)
            if mi is None:
                C.add("2_matrix_present", False, "legacy .1pz member missing from .s1 object", gsm=g, member=r)
                continue
            C.add("2_matrix_present", True, "")
            lpath = os.path.join(a.scratch, "leg_" + r.replace("/", "_"))
            t = time.time()
            rz.read(f"samples/{g}/{r}", dest=lpath)
            if mi.kind == 1:
                leg = _pz.read_1pz(lpath)
                t1 = time.time()
                csc = smp.matrix(r)
                t2 = time.time()
                ok, det = compare_matrix(leg, csc)
                det["legacy_vt_code"] = int(leg.get("vt_code", 0))
                det["s1_value_type"] = int(csc.value_type)
                det["t_legacy_s"], det["t_s1_decode_s"] = round(t1 - t, 2), round(t2 - t1, 2)
                SR["matrices"][r] = det
                C.add("2_matrix_equal", ok, "matrix differs from legacy", gsm=g, member=r,
                      **{k: v for k, v in det.items() if k not in ("t_legacy_s", "t_s1_decode_s")})
                if r == "exon_counts.1pz" and cells is not None:
                    C.add("4_cells_barcodes_vs_matrix", list(cells.barcodes) == list(leg["colnames"]),
                          "cells table barcodes = legacy exon column names", gsm=g)
                    C.add("4_n_columns", cells.n_rows == int(leg["n"]) == int(s.get("n_columns") or -1),
                          "cells rows = legacy n = manifest n_columns", gsm=g, cells=cells.n_rows, legacy=int(leg["n"]),
                          manifest=s.get("n_columns"))
                del leg, csc
            elif mi.kind in (3, 5):
                same = rd.legacy_blob(r) == open(lpath, "rb").read()
                SR["matrices"][r] = {"kind": mi.kind, "bytes_identical": same}
                C.add("2_matrix_equal", same, "stored .1pz member bytes differ from legacy", gsm=g, member=r)
            else:
                SR["matrices"][r] = {"kind": mi.kind, "recipe": (mi.meta or {}).get("recipe")}
                C.add("2_matrix_derived", "warn", "legacy .1pz is a derived member (not compared)", gsm=g, member=r,
                      kind=mi.kind, recipe=(mi.meta or {}).get("recipe"))
            os.remove(lpath)
        if "cell_calls.tsv" in leg_members:
            called_leg = legacy_called(rz.read(f"samples/{g}/cell_calls.tsv"))
        SR["legacy_has_cell_calls"] = "cell_calls.tsv" in leg_members
        if "doublet_scores.tsv" in leg_members and cells is not None and cells.has("qc_doublet_score"):
            try:
                import pandas as pd

                ds = pd.read_csv(io.BytesIO(rz.read(f"samples/{g}/doublet_scores.tsv")), sep="\t")
                SR["doublet_scores_tsv_via_guard"] = {"rows": len(ds), "columns": list(ds.columns)[:10]}
            except Exception as e:  # noqa: BLE001
                SR["doublet_scores_tsv_via_guard"] = {"error": repr(e)[:200]}

    # ---- 4: cells table and labels
    if cells is None:
        C.add("4_cells_table", False, "sample has no cells table", gsm=g)
        return
    called = np.asarray(cells.column("called"), bool)
    nc = int(called.sum())
    SR["n_called_cells_table"] = nc
    C.add("4_called_vs_manifest", nc == int(s.get("n_called") or -1), "cells called vs manifest n_called", gsm=g,
          cells=nc, manifest=s.get("n_called"))
    if called_leg is not None:
        bcs = cells.barcodes
        cset = set(bcs[i] for i in np.flatnonzero(called))
        lset = set(called_leg)
        C.add("4_called_vs_legacy", cset == lset, "called barcodes vs legacy cell_calls.tsv", gsm=g, cells=len(cset),
              legacy=len(lset), only_s1=len(cset - lset), only_legacy=len(lset - cset))
    elif SR.get("legacy_has_cell_calls") is False:
        C.add("4_called_vs_legacy", "warn", "legacy bundle has no cell_calls.tsv", gsm=g)
    has_ed = cells.has("called_ed")
    C.add("4_called_ed_present", has_ed, "called_ed column", gsm=g)
    if has_ed:
        ced = np.asarray(cells.column("called_ed"), bool)
        SR["n_called_ed"] = int(ced.sum())
        C.add("4_called_ed_superset", bool(np.all(ced[called])), "called_ed must contain the stored call", gsm=g)
        meta_ed = ((s.get("members") or {}).get("cells") or {}).get("called_ed") or {}
        SR["called_ed_meta"] = {k: meta_ed.get(k) for k in ("method", "version", "validated", "n_added", "n_called_ed")}
    cmeta = (s.get("members") or {}).get("cells") or {}
    SR["called_source"] = cmeta.get("called_source")

    # labels
    has = {c: cells.has(c) for c in ("cell_type_cl", "cell_type_path", "sex_chrom", "ancestry_hancestro")}
    SR["label_columns"] = has
    if nc == 0:
        return
    idx = np.flatnonzero(called)
    unc = np.flatnonzero(~called)

    def strcol(c):
        v = cells.column(c)
        return [v[i] if v[i] else "" for i in range(len(v))]

    lab = {}
    if has["cell_type_cl"]:
        ct = strcol("cell_type_cl")
        ctc = [ct[i] for i in idx]
        lab["cell_type_frac_called"] = sum(1 for x in ctc if x) / nc
        lab["cell_type_on_uncalled"] = sum(1 for i in unc if ct[i])
        badfmt = sorted(set(x for x in ctc if x and not CL_RE.match(x)))
        notin = sorted(set(x for x in ctc if x and CL_RE.match(x) and cl_ids is not None and x not in cl_ids))
        C.add("4_cell_type_valid_cl", not badfmt and not notin, "invalid CL id", gsm=g, bad_format=badfmt[:10],
              not_in_ontology=notin[:10])
        if has["cell_type_path"]:
            pth = strcol("cell_type_path")
            l1 = {}
            badp = set()
            for i in idx:
                p = pth[i]
                if not p:
                    continue
                parts = [x for x in re.split(r"[>|/;,]", p) if x]
                for x in parts:
                    if not CL_RE.match(x) or (cl_ids is not None and x not in cl_ids):
                        badp.add(x)
                l1[parts[0]] = l1.get(parts[0], 0) + 1
            lab["level1_counts"] = dict(sorted(l1.items(), key=lambda kv: -kv[1])[:12])
            lab["path_example"] = next((pth[i] for i in idx if pth[i]), None)
            C.add("4_cell_type_path_valid", not badp, "invalid CL id in cell_type_path", gsm=g, bad=sorted(badp)[:10])
        top = {}
        for x in ctc:
            top[x] = top.get(x, 0) + 1
        lab["cell_type_top"] = dict(sorted(top.items(), key=lambda kv: -kv[1])[:8])
    for pc in ["cell_type_score"] + [f"cell_type_p{k}" for k in range(1, 7)] + ["sex_score", "ancestry_prob"]:
        if cells.has(pc):
            v = np.asarray(cells.column(pc), np.float64)
            f = v[np.isfinite(v)]
            okr = bool(np.all((f >= 0) & (f <= 1))) if f.size else True
            C.add("4_probabilities_in_range", okr, f"{pc} outside [0,1]", gsm=g, column=pc,
                  min=float(f.min()) if f.size else None, max=float(f.max()) if f.size else None)
    if has["sex_chrom"]:
        sx = strcol("sex_chrom")
        cnt = {}
        for i in idx:
            cnt[sx[i] or "(none)"] = cnt.get(sx[i] or "(none)", 0) + 1
        lab["sex_counts"] = cnt
        lab["sex_frac_called"] = sum(1 for i in idx if sx[i]) / nc
        badsx = sorted(set(cnt) - {"male", "female", "ambiguous", "(none)", "unknown"})
        C.add("4_sex_values", not badsx, "unexpected sex_chrom values", gsm=g, values=badsx[:10])
    if has["ancestry_hancestro"]:
        an = strcol("ancestry_hancestro")
        cnt = {}
        for i in idx:
            cnt[an[i] or "(none)"] = cnt.get(an[i] or "(none)", 0) + 1
        lab["ancestry_counts"] = cnt
        lab["ancestry_frac_called"] = sum(1 for i in idx if an[i]) / nc
        badan = sorted(x for x in cnt if x != "(none)" and not HANC_RE.match(x))
        C.add("4_ancestry_valid", not badan, "invalid HANCESTRO id", gsm=g, values=badan[:10])
        if species != "human" and lab["ancestry_frac_called"] > 0:
            C.add("4_ancestry_only_human", False, "ancestry label on a non-human sample", gsm=g, species=species)
    # expectations
    if species in ("human", "mouse"):
        C.add("4_labels_cell_type_present", has["cell_type_cl"] and lab.get("cell_type_frac_called", 0) > 0,
              "human/mouse sample with called cells has no cell-type labels", gsm=g)
        C.add("4_labels_sex_present", has["sex_chrom"] and lab.get("sex_frac_called", 0) > 0,
              "human/mouse sample with called cells has no sex labels", gsm=g)
    SR["labels"] = lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gse")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--sizes", default="")
    ap.add_argument("--organism", default="")
    ap.add_argument("--cl", default="")
    ap.add_argument("--full-get-max", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    try:
        R = run(a)
    except Exception:  # noqa: BLE001
        R = {"gse": a.gse, "shard": a.shard, "fatal": traceback.format_exc()[-3000:]}
    tmp = a.out + ".part"
    with open(tmp, "w") as fh:
        json.dump(R, fh, default=str)
    os.replace(tmp, a.out)
    nf = sum(v.get("fail", 0) for v in (R.get("checks") or {}).values())
    print(f"{a.gse} shard {a.shard}: {nf} failing items, {R.get('study', {}).get('elapsed_s')}s"
          + (" FATAL" if "fatal" in R else ""), flush=True)


if __name__ == "__main__":
    main()
