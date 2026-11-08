#!/usr/bin/env python3
"""Second guard on the barcode whitelist, read from one finished sample.

common.require_whitelists checks before a worker starts that <binary dir>/../whitelists holds the 10x files. This check reads what the engine
actually did on a sample: its log (singlet.log) and its outputs. A 10x protocol that ran without a whitelist takes the sorted-BAM pileup and
STAR's barcode auto-discovery, which give different counts from the stream path, so the sample must not be published.

It fails when any of these hold:
  - the log says "Whitelist <file> not found for <protocol> - using barcode auto-discovery" (a 10x protocol without its file);
  - the log says it cannot read the .1fq header for whitelist auto-resolution;
  - the log names a 10x protocol in a "using barcode auto-discovery" line;
  - the log has the engine's warning that a CB_UMI_Simple protocol found no whitelist file;
  - any output JSON (pileup_stats.json, provenance.json) records whitelist_missing = true;
  - no output JSON records whitelist_missing at all (an engine too old to report it, or an incomplete output).
A whitelist-free protocol (its "is whitelist-free - using barcode auto-discovery" line) passes.

Usage:  python3 whitelist_check.py <singlet.log> <out dir>      exit 0 = pass, 1 = fail, 2 = files missing
Import: whitelist_check.check(log_path, out_dir) -> dict(ok, why, line, resolved)
"""
import json
import os
import sys

AUTO = "using barcode auto-discovery"
NOT_FOUND = "not found for "
NO_HEADER = "cannot read .1fq header"
WL_FREE = "is whitelist-free"
CB_WARN = "is a barcode-whitelist type (CB_UMI_Simple) but no whitelist file was found"
RESOLVED = "Auto-resolved whitelist:"
JSONS = ("pileup_stats.json", "provenance.json", "summary.json")


def _find_key(obj, key):
    """-> every value of `key` anywhere in a JSON object."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            out.extend(_find_key(v, key))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_find_key(v, key))
    return out


def check(log_path, out_dir):
    """-> dict(ok, why, line, resolved). `resolved` is the engine's "Auto-resolved whitelist" line when there is one."""
    resolved = None
    try:
        with open(log_path, errors="replace") as fh:
            for raw in fh:
                line = raw.strip()
                if RESOLVED in line and resolved is None:
                    resolved = line[:300]
                if CB_WARN in line:
                    return {"ok": False, "why": "the engine found no whitelist for a CB_UMI_Simple protocol", "line": line[:300], "resolved": resolved}
                if AUTO not in line:
                    continue
                if NOT_FOUND in line:
                    return {"ok": False, "why": "a 10x protocol ran without its whitelist file", "line": line[:300], "resolved": resolved}
                if NO_HEADER in line:
                    return {"ok": False, "why": "the .1fq header was unreadable, so no whitelist was resolved", "line": line[:300], "resolved": resolved}
                if "10x" in line.lower() and WL_FREE not in line:
                    return {"ok": False, "why": "a 10x protocol used barcode auto-discovery", "line": line[:300], "resolved": resolved}
    except OSError as e:
        return {"ok": None, "why": "log not readable: %s" % e, "line": None, "resolved": None}
    seen = []
    for name in JSONS:
        p = os.path.join(out_dir, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p) as fh:
                vals = _find_key(json.load(fh), "whitelist_missing")
        except (OSError, ValueError):
            continue
        if any(v is True or str(v).lower() == "true" for v in vals):
            return {"ok": False, "why": "%s records whitelist_missing = true" % name, "line": None, "resolved": resolved}
        seen.extend(vals)
    if not seen:
        return {"ok": False, "why": "no output JSON in %s records whitelist_missing" % out_dir, "line": None, "resolved": resolved}
    return {"ok": True, "why": "whitelist_missing = false" + ("; " + resolved if resolved else "; whitelist-free protocol"), "line": None,
            "resolved": resolved}


def main(argv):
    if len(argv) != 3:
        sys.stderr.write(__doc__)
        return 2
    log_path, out_dir = argv[1], argv[2]
    if not os.path.isfile(log_path) or not os.path.isdir(out_dir):
        print("WHITELIST CHECK: missing %s" % (log_path if not os.path.isfile(log_path) else out_dir))
        return 2
    r = check(log_path, out_dir)
    print("WHITELIST CHECK: %s: %s%s" % ("PASS" if r["ok"] else "FAIL", r["why"], (" | " + r["line"]) if r["line"] else ""))
    return 0 if r["ok"] else (2 if r["ok"] is None else 1)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
