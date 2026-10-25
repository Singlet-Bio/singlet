#!/usr/bin/env python3
"""Copy the objects published under the old `v2/` prefix to `store/` (docs/rename-to-1-0.md section 3.2).

    python -P scripts/store_rekey.py plan   --out DIR [--local-root ROOT]
    python -P scripts/store_rekey.py apply  --out DIR [--local-root ROOT]
    python -P scripts/store_rekey.py verify --out DIR [--local-root ROOT] [--http https://data.singlet.bio]

`plan` only reads. It lists every `v2/` key in the public bucket and maps it to its `store/` key (`v2/X` -> `store/X`, sample
objects `.s2` -> `.s1`). Objects are copied byte for byte (S3 CopyObject, metadata kept): sample objects, packs, ledgers and
sidecars keep their digests because no magic byte changes. Two kinds are rewritten instead:

* study manifests `v2/m/<GSE>.json`: `format` becomes `singlet-store`, every string value naming a `v2/` key is mapped, and
  `rev` is recomputed (it hashes the sample keys);
* the pack index `v2/ref/index.json`: `format` becomes `singlet-store-packs` (it names no keys).

It writes DIR/plan.json, the rewritten files under DIR/files/, and DIR/catalog.sql: the `rev` / manifest size updates for D1, to
be applied right after schema 036 (which already rewrote the keys). `apply` performs the copies, then uploads the rewritten pack
index, then the manifests (last, so a manifest never names a missing object), and refuses to overwrite a different existing
object. `verify` checks every destination: size and ETag equal to the source for copies, bytes equal to the planned file for
rewrites, and with `--http` the sha256 of the served bytes. Nothing under `v2/` is ever written or deleted.

R2 credentials come from the environment as for store_campaign.py (R2_ACCOUNT_ID or R2_ENDPOINT, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
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


def open_store(a):
    return camp.LocalStore(Path(a.local_root)) if a.local_root else camp.S3Store()


def sql_str(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def cmd_plan(a) -> int:
    st, out = open_store(a), Path(a.out)
    (out / "files").mkdir(parents=True, exist_ok=True)
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
                body = (json.dumps(doc, indent=1) + "\n").encode("utf-8")
                kind = "index"
            else:
                body, old_rev, rev = rewrite_manifest(data)
                gse = camp.RE_MANIFEST.match(key).group(1)
                sha = hashlib.sha256(body).hexdigest()
                sql += [f"UPDATE store_sample SET rev = {sql_str(rev)} WHERE gse_id = {sql_str(gse)} AND rev = {sql_str(old_rev)};",
                        f"UPDATE store_study SET rev = {sql_str(rev)}, manifest_bytes = {len(body)}, manifest_sha256 = {sql_str(sha)} "
                        f"WHERE gse_id = {sql_str(gse)} AND rev = {sql_str(old_rev)};"]
                kind = "manifest"
            local = out / "files" / dst
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(body)
            items.append({"src": key, "dst": dst, "kind": kind, "bytes": len(body),
                          "sha256": hashlib.sha256(body).hexdigest(), "file": str(local.relative_to(out))})
        else:
            h = st.head(BUCKET, key)
            items.append({"src": key, "dst": dst, "kind": "copy", "bytes": size, "etag": h.etag if h else ""})
    (out / "plan.json").write_text(json.dumps({"bucket": BUCKET, "items": items}, indent=1) + "\n", encoding="utf-8")
    camp.write_lf(out / "catalog.sql", "\n".join(sql) + "\n")
    kinds = {}
    for it in items:
        kinds[it["kind"]] = kinds.get(it["kind"], 0) + 1
    print(f"plan: {len(items)} keys {kinds}, {sum(i['bytes'] for i in items)} bytes -> {out / 'plan.json'}, {out / 'catalog.sql'}")
    return 0


def _load(a):
    out = Path(a.out)
    return out, json.loads((out / "plan.json").read_text(encoding="utf-8"))["items"]


def cmd_apply(a) -> int:
    st, (out, items) = open_store(a), _load(a)
    order = {"copy": 0, "index": 1, "manifest": 2}
    for it in sorted(items, key=lambda i: order[i["kind"]]):
        have = st.head(BUCKET, it["dst"])
        if it["kind"] == "copy":
            if have is not None:
                if have.size == it["bytes"] and have.etag == it["etag"]:
                    continue
                raise SystemExit(f"refusing to overwrite {it['dst']}: it exists with different content")
            st.copy(BUCKET, it["src"], BUCKET, it["dst"], size=it["bytes"])
        else:
            body = (out / it["file"]).read_bytes()
            if have is not None and st.get_bytes(BUCKET, it["dst"]) == body:
                continue
            if have is not None and it["kind"] == "index":
                raise SystemExit(f"refusing to overwrite {it['dst']}: it exists with different content")
            ctype, cache = camp.object_headers(it["dst"])
            st.put_bytes(BUCKET, it["dst"], body, content_type=ctype, cache_control=cache)
        print(f"{it['kind']:8s} {it['src']} -> {it['dst']}")
    print("apply: done; run verify")
    return 0


def cmd_verify(a) -> int:
    st, (out, items) = open_store(a), _load(a)
    bad = 0
    for it in items:
        h = st.head(BUCKET, it["dst"])
        ok = h is not None and h.size == it["bytes"]
        if ok and it["kind"] == "copy":
            ok = h.etag == it["etag"]
        elif ok:
            ok = st.get_bytes(BUCKET, it["dst"]) == (out / it["file"]).read_bytes()
        if ok and a.http:
            want = it.get("sha256") or hashlib.sha256(st.get_bytes(BUCKET, it["src"])).hexdigest()
            req = urllib.request.Request(a.http.rstrip("/") + "/" + it["dst"], headers={"User-Agent": "singlet-rekey/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                ok = hashlib.sha256(r.read()).hexdigest() == want
        if not ok:
            bad += 1
            print(f"MISMATCH {it['dst']}")
    print(f"verify: {len(items) - bad}/{len(items)} ok")
    return 1 if bad else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "apply", "verify"):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True, help="plan directory")
        s.add_argument("--local-root", help="directories under this root instead of R2 (tests, rehearsals)")
        if name == "verify":
            s.add_argument("--http", help="also fetch every destination from this origin and check its sha256")
    a = p.parse_args(argv)
    return {"plan": cmd_plan, "apply": cmd_apply, "verify": cmd_verify}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
