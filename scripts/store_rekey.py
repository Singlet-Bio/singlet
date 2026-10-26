#!/usr/bin/env python3
"""Copy the objects published under the old `v2/` prefix to `store/` (docs/rename-to-1-0.md section 3.2).

With R2 S3 credentials (R2_ACCOUNT_ID or R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, as for store_campaign.py):

    python -P scripts/store_rekey.py plan   --out DIR [--local-root ROOT]
    python -P scripts/store_rekey.py apply  --out DIR [--local-root ROOT]
    python -P scripts/store_rekey.py verify --out DIR [--local-root ROOT] [--http https://data.singlet.bio]

Without them (the laptop): plan from the public origin, upload with wrangler, verify over HTTP:

    python -P scripts/store_rekey.py plan     --out DIR --http https://data.singlet.bio --gse GSE145370
    python -P scripts/store_rekey.py wrangler --out DIR > DIR/apply.sh      # review, then: bash DIR/apply.sh
    python -P scripts/store_rekey.py verify   --out DIR --http https://data.singlet.bio

`plan` only reads. Every `v2/` key maps to its `store/` key (`v2/X` -> `store/X`, sample objects `.s2` -> `.s1`). Objects are
copied byte for byte: sample objects, packs, ledgers and sidecars keep their digests because no magic byte changes. Two kinds
are rewritten instead:

* study manifests `v2/m/<GSE>.json`: `format` becomes `singlet-store`, every string value naming a `v2/` key is mapped, and
  `rev` is recomputed (it hashes the sample keys);
* the pack index `v2/ref/index.json`: `format` becomes `singlet-store-packs` (it names no keys).

`plan` writes DIR/plan.json, the rewritten files (and, in --http mode, the downloaded copies, checked against the sha256 the
manifests and the index record) under DIR/files/, and DIR/catalog.sql: the `rev` / manifest size updates for D1, applied right
after schema 036 (which already rewrote the keys). Uploads go copies first, then the pack index, then the manifests, so a
manifest never names a missing object. `apply` refuses to overwrite a different existing object. Nothing under `v2/` is ever
written or deleted.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
_spec = importlib.util.spec_from_file_location("store_campaign", ROOT / "scripts" / "store_campaign.py")
camp = importlib.util.module_from_spec(_spec)
sys.modules["store_campaign"] = camp
_spec.loader.exec_module(camp)

from singlet.store import manifest as M  # noqa: E402

BUCKET = camp.PUBLIC_BUCKET
OLD, NEW = "v2/", "store/"
ORDER = {"copy": 0, "index": 1, "manifest": 2}


def new_key(key: str) -> str:
    assert key.startswith(OLD), key
    k = NEW + key[len(OLD):]
    return k[:-3] + ".s1" if camp.RE_SAMPLE.match(key) else k


def _map_strings(obj):
    if isinstance(obj, str):
        return new_key(obj) if obj.startswith(OLD) else obj
    if isinstance(obj, list):
        return [_map_strings(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _map_strings(v) for k, v in obj.items()}
    return obj


def rewrite_manifest(data: bytes) -> tuple:
    m = M.loads_manifest(data)
    if m.get("format") != "singlet-v2":
        raise SystemExit(f"{m.get('gse')}: format {m.get('format')!r}, expected singlet-v2")
    old_rev = m["rev"]
    if M.compute_rev([s["key"] for s in m["samples"]], m.get("study_files") or {}) != old_rev:
        raise SystemExit(f"{m['gse']}: the stored rev does not recompute from the manifest; stopping")
    n = _map_strings(m)
    n["format"] = "singlet-store"
    n["rev"] = M.compute_rev([s["key"] for s in n["samples"]], n.get("study_files") or {})
    return M.dumps_manifest(n), old_rev, n["rev"]


def _http_get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "singlet-rekey/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


class HttpSource:
    """Read-only view of the `v2/` tree over the public origin. Its keys are the pack index, the packs it lists, and for each
    study the manifest, its sample objects and its ledger: every published `v2/` object (no cell index or sidecar is published
    under `v2/`)."""

    def __init__(self, origin: str, gses) -> None:
        self.origin, self.gses, self._cache = origin.rstrip("/"), list(gses), {}

    def get_bytes(self, bucket: str, key: str) -> bytes:
        if key not in self._cache:
            self._cache[key] = _http_get(self.origin + "/" + key)
        return self._cache[key]

    def list_keys(self, bucket: str, prefix: str):
        idx = OLD + "ref/index.json"
        keys = [idx] + [f"{OLD}ref/{p['id']}.pack" for p in json.loads(self.get_bytes(bucket, idx))["packs"]]
        for g in self.gses:
            mk = f"{OLD}m/{g}.json"
            m = json.loads(self.get_bytes(bucket, mk))
            keys += [mk] + [s["key"] for s in m["samples"]]
            if (m.get("legacy") or {}).get("ledger"):
                keys.append(m["legacy"]["ledger"])
        return [(k, -1 if k.startswith(OLD + "ledger/") else len(self.get_bytes(bucket, k))) for k in keys]

    def recorded_sha256(self) -> dict:
        want = {f"{OLD}ref/{p['id']}.pack": p["sha256"]
                for p in json.loads(self.get_bytes(BUCKET, OLD + "ref/index.json"))["packs"]}
        for g in self.gses:
            want.update({s["key"]: s["sha256"] for s in json.loads(self.get_bytes(BUCKET, f"{OLD}m/{g}.json"))["samples"]})
        return want


def open_store(a):
    return camp.LocalStore(Path(a.local_root)) if a.local_root else camp.S3Store()


def sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _save(out: Path, dst: str, body: bytes) -> str:
    local = out / "files" / dst
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(body)
    return local.relative_to(out).as_posix()


def cmd_plan(a) -> int:
    http = bool(a.http)
    st, out = (HttpSource(a.http, a.gse) if http else open_store(a)), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items, sql = [], ["-- after schema/036_store_rename.sql: rev and manifest size of the rewritten manifests"]
    for key, size in sorted(st.list_keys(BUCKET, OLD)):
        dst = new_key(key)
        if camp.RE_MANIFEST.match(key) or key == OLD + "ref/index.json":
            data = st.get_bytes(BUCKET, key)
            if key.endswith("index.json"):
                doc = json.loads(data)
                if doc.get("format") != "singlet-v2-packs":
                    raise SystemExit(f"{key}: format {doc.get('format')!r}")
                doc["format"] = "singlet-store-packs"
                body, kind = (json.dumps(doc, indent=1) + "\n").encode("utf-8"), "index"
            else:
                body, old_rev, rev = rewrite_manifest(data)
                gse = camp.RE_MANIFEST.match(key).group(1)
                sha = hashlib.sha256(body).hexdigest()
                sql += [f"UPDATE store_sample SET rev = {sql_str(rev)} WHERE gse_id = {sql_str(gse)} AND rev = {sql_str(old_rev)};",
                        f"UPDATE store_study SET rev = {sql_str(rev)}, manifest_bytes = {len(body)}, manifest_sha256 = {sql_str(sha)} "
                        f"WHERE gse_id = {sql_str(gse)} AND rev = {sql_str(old_rev)};"]
                kind = "manifest"
            items.append({"src": key, "dst": dst, "kind": kind, "bytes": len(body),
                          "sha256": hashlib.sha256(body).hexdigest(), "file": _save(out, dst, body)})
        elif http and size < 0:
            items.append({"src": key, "dst": dst, "kind": "copy", "bytes": None, "fetch": "wrangler",
                          "file": (Path("files") / dst).as_posix()})
        elif http:
            data = st.get_bytes(BUCKET, key)
            items.append({"src": key, "dst": dst, "kind": "copy", "bytes": len(data),
                          "sha256": hashlib.sha256(data).hexdigest(), "file": _save(out, dst, data)})
        else:
            h = st.head(BUCKET, key)
            items.append({"src": key, "dst": dst, "kind": "copy", "bytes": size, "etag": h.etag if h else ""})
    if http:
        want = st.recorded_sha256()
        for it in items:
            if it["kind"] == "copy" and it["src"] in want and it["sha256"] != want[it["src"]]:
                raise SystemExit(f"{it['src']}: downloaded bytes do not match the recorded sha256")
    (out / "plan.json").write_text(json.dumps({"bucket": BUCKET, "items": items}, indent=1) + "\n", encoding="utf-8")
    camp.write_lf(out / "catalog.sql", "\n".join(sql) + "\n")
    kinds = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print(f"plan: {len(items)} keys {kinds}, {sum(i['bytes'] or 0 for i in items)} bytes -> {out / 'plan.json'}, {out / 'catalog.sql'}")
    return 0


def _load(a):
    out = Path(a.out)
    return out, sorted(json.loads((out / "plan.json").read_text(encoding="utf-8"))["items"], key=lambda i: ORDER[i["kind"]])


def copy_object(st, src: str, dst: str, size: int) -> None:
    """Server-side byte copy. On R2 the Content-Type / Cache-Control are set to what a store key gets (MetadataDirective
    REPLACE), whatever the source carried; objects here are far below the 5 GB single-copy limit."""
    if isinstance(st, camp.S3Store):
        ctype, cache = camp.object_headers(dst)
        st.c.copy_object(Bucket=BUCKET, Key=dst, CopySource={"Bucket": BUCKET, "Key": src}, MetadataDirective="REPLACE",
                         ContentType=ctype, CacheControl=cache)
    else:
        st.copy(BUCKET, src, BUCKET, dst, size=size)


def cmd_apply(a) -> int:
    st, (out, items) = open_store(a), _load(a)
    for it in items:
        have = st.head(BUCKET, it["dst"])
        if it["kind"] == "copy" and "file" not in it:
            if have is not None:
                if have.size == it["bytes"] and have.etag == it["etag"]:
                    continue
                raise SystemExit(f"refusing to overwrite {it['dst']}: it exists with different content")
            copy_object(st, it["src"], it["dst"], it["bytes"])
        else:
            body = (out / it["file"]).read_bytes()
            if have is not None:
                if st.get_bytes(BUCKET, it["dst"]) == body:
                    continue
                if it["kind"] != "manifest":
                    raise SystemExit(f"refusing to overwrite {it['dst']}: it exists with different content")
            ctype, cache = camp.object_headers(it["dst"])
            st.put_bytes(BUCKET, it["dst"], body, content_type=ctype, cache_control=cache)
        print(f"{it['kind']:8s} {it['src']} -> {it['dst']}")
    print("apply: done; run verify")
    return 0


def cmd_wrangler(a) -> int:
    out, items = _load(a)
    print("#!/usr/bin/env bash")
    print("# generated by scripts/store_rekey.py wrangler: writes store/ keys only, copies first, manifests last")
    print("set -euo pipefail")
    print(f"cd '{out.resolve().as_posix()}'")
    for it in items:
        ctype, cache = camp.object_headers(it["dst"])
        if it.get("fetch") == "wrangler":
            print(f"mkdir -p '{Path(it['file']).parent.as_posix()}'")
            print(f"npx wrangler r2 object get '{BUCKET}/{it['src']}' --remote --file='{it['file']}'")
        print(f"npx wrangler r2 object put '{BUCKET}/{it['dst']}' --remote --file='{it['file']}' "
              f"--content-type='{ctype}' --cache-control='{cache}'")
    return 0


def cmd_verify(a) -> int:
    out, items = _load(a)
    st = open_store(a) if (a.local_root or not a.http) else None
    bad = 0
    for it in items:
        ok = True
        if st is not None:
            h = st.head(BUCKET, it["dst"])
            ok = h is not None and (it["bytes"] is None or h.size == it["bytes"])
            if ok and "file" in it:
                ok = st.get_bytes(BUCKET, it["dst"]) == (out / it["file"]).read_bytes()
            elif ok:
                ok = h.etag == it["etag"]
            if ok and isinstance(st, camp.S3Store):
                want_ct, want_cc = camp.object_headers(it["dst"])
                if (h.content_type, h.cache_control) != (want_ct, want_cc):
                    print(f"headers {it['dst']}: {h.content_type!r} {h.cache_control!r}, expected {want_ct!r} {want_cc!r}")
                    ok = False
        if ok and a.http and it.get("fetch") == "wrangler":
            print(f"not served over HTTP, not checked: {it['dst']}")
        elif ok and a.http:
            want = it.get("sha256") or hashlib.sha256(st.get_bytes(BUCKET, it["src"])).hexdigest()
            try:
                ok = hashlib.sha256(_http_get(a.http.rstrip("/") + "/" + it["dst"])).hexdigest() == want
            except urllib.error.HTTPError:
                ok = False
        if not ok:
            bad += 1
            print(f"MISMATCH {it['dst']}")
    print(f"verify: {len(items) - bad}/{len(items)} ok")
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "apply", "wrangler", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True, help="plan directory")
        s.add_argument("--local-root", help="directories under this root instead of R2 (tests, rehearsals)")
        if name == "plan":
            s.add_argument("--http", help="read the v2/ tree from this public origin instead of the bucket (needs --gse)")
            s.add_argument("--gse", action="append", default=[], help="study whose manifest names the objects (repeatable)")
        if name == "verify":
            s.add_argument("--http", help="fetch every destination from this origin and check its sha256")
    a = p.parse_args(argv)
    if a.cmd == "plan" and a.http and not a.gse:
        p.error("plan --http needs --gse")
    return {"plan": cmd_plan, "apply": cmd_apply, "wrangler": cmd_wrangler, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
