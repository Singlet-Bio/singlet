# SPDX-License-Identifier: MIT
"""Generic per-cell TSV <-> typed cells-table columns (owner: W3).

The pipeline writes a dozen per-cell tables (cell QC, doublet scores, cell-cycle scores, ambient contamination,
donor assignments, mt summary, ...) as text with printf-style number formatting.  ``fit_column`` finds, for one text
column, a typed numpy column plus a format string such that ``render_column`` gives back the identical text; the
table is then regenerated from cells-table columns (``tsvcols-v1``) and the typed columns are what the container
stores (they compress several times better than the text).  Columns that are exact functions of data the container
already holds are not stored at all (see recipes_tables.py for the derived ones).

Format kinds (JSON-able dict, stored in the recipe params):
  {"k": "int",   "dt": "u8|u16|u32|i32", "na": null|"NA"}      integers, optionally with one non-numeric token
  {"k": "float", "dt": "f32|f64", "fmt": "%.4f", "na": ...}   floats printed with a printf format
  {"k": "bool",  "labels": ["False", "True"]}                  two-token booleans
  {"k": "cat",   "labels": [...]}                              <= 255 distinct strings, stored as u8 codes
  {"k": "str"}                                                 anything else, stored as strings
"""
from __future__ import annotations

import re

import numpy as np

_INT = re.compile(r"^-?(0|[1-9][0-9]*)$")
_FIX = re.compile(r"^-?[0-9]+\.([0-9]+)$")
_SCI = re.compile(r"^-?[0-9](?:\.([0-9]+))?e[+-][0-9]{2,3}$")
_NUM = re.compile(r"^-?(?:[0-9]+\.?[0-9]*|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$")
NA_TOKENS = ("NA", "nan", "NaN", "", "NULL", "null", "None", "-", ".")

_DT = {"u8": np.uint8, "u16": np.uint16, "u32": np.uint32, "i32": np.int32, "f32": np.float32, "f64": np.float64}


def _is_num(s):
    return bool(_NUM.match(s))


def render_column(fit: dict, arr) -> list:
    """Text of a fitted column."""
    k = fit["k"]
    if k == "int":
        a = np.asarray(arr)
        out = [str(int(x)) for x in a]
        if fit.get("na") is not None:
            na_code = fit["na_code"]
            out = [fit["na"] if int(x) == na_code else s for x, s in zip(a, out)]
        return out
    if k == "float":
        fmt = fit["fmt"]
        na = fit.get("na")
        a = np.asarray(arr)
        if na is None:
            return [fmt % x for x in a.astype(np.float64)]
        return [na if x != x else fmt % x for x in a.astype(np.float64)]
    if k == "bool":
        lab = fit["labels"]
        return [lab[1] if x else lab[0] for x in np.asarray(arr, dtype=bool)]
    if k == "cat":
        lab = fit["labels"]
        return [lab[int(x)] for x in np.asarray(arr)]
    if k == "str":
        return [str(x) for x in arr]
    raise ValueError(fit)


def _fit_float(vals, na_tok):
    """Find (fit, array) for numeric strings (None entries = NA token).  Returns None on failure."""
    idx = [i for i, v in enumerate(vals) if v is not None]
    present = [vals[i] for i in idx]
    if not present:
        return None
    nums = np.array([float(v) for v in present], dtype=np.float64)
    cands = []
    decs = set()
    ok_fix = True
    for v in present:
        m = _FIX.match(v)
        if not m:
            ok_fix = False
            break
        decs.add(len(m.group(1)))
    if ok_fix and len(decs) == 1:
        cands.append("%%.%df" % decs.pop())
    ok_sci = all(_SCI.match(v) for v in present)
    if ok_sci:
        ds = {len(_SCI.match(v).group(1) or "") for v in present}
        if len(ds) == 1:
            cands.append("%%.%de" % ds.pop())
    for p in (6, 7, 8, 9, 10, 12, 15, 17):
        cands.append("%%.%dg" % p)
    for fmt in cands:
        for dt in ("f32", "f64"):
            arr = np.full(len(vals), np.nan, dtype=np.float64)
            arr[idx] = nums
            arr = arr.astype(_DT[dt])
            back = arr.astype(np.float64)[idx]
            if not np.array_equal(back == back, np.ones(len(back), bool)):
                continue
            # float32 overflow / precision: render and compare
            txt = [fmt % x for x in back]
            if txt == present:
                fit = {"k": "float", "dt": dt, "fmt": fmt}
                if na_tok is not None:
                    fit["na"] = na_tok
                return fit, arr
    return None


def fit_column(values):
    """Fit a list of strings.  Returns (fit, array); falls back to a string column, never fails."""
    n = len(values)
    toks = set(values)
    if n == 0:
        return {"k": "str"}, np.array([], dtype=object)
    # booleans
    for lab in (("False", "True"), ("FALSE", "TRUE"), ("false", "true"), ("F", "T"), ("N", "Y"), ("no", "yes")):
        if toks <= set(lab):
            return {"k": "bool", "labels": list(lab)}, np.array([v == lab[1] for v in values], dtype=bool)
    non_num = {v for v in toks if not _is_num(v)}
    na_tok = None
    if len(non_num) == 1 and next(iter(non_num)) in NA_TOKENS and len(non_num) < len(toks):
        na_tok = next(iter(non_num))
    if not non_num or na_tok is not None:
        vals = [None if v == na_tok else v for v in values]
        present = [v for v in vals if v is not None]
        if all(_INT.match(v) for v in present):
            nums = np.array([int(v) for v in present], dtype=np.int64)
            lo, hi = int(nums.min()), int(nums.max())
            # keep one value free for the NA sentinel when needed
            if lo >= 0 and hi < (255 if na_tok else 256):
                dt, code = "u8", 255
            elif lo >= 0 and hi < (65535 if na_tok else 65536):
                dt, code = "u16", 65535
            elif lo >= 0 and hi < (4294967295 if na_tok else 4294967296):
                dt, code = "u32", 4294967295
            elif -(2**31) < lo and hi < 2**31 - 1:
                dt, code = "i32", -(2**31)
            else:
                dt = None
            if dt:
                arr = np.full(n, code if na_tok else 0, dtype=np.int64)
                arr[[i for i, v in enumerate(vals) if v is not None]] = nums
                fit = {"k": "int", "dt": dt}
                if na_tok is not None:
                    fit["na"] = na_tok
                    fit["na_code"] = int(code)
                return fit, arr.astype(_DT[dt])
        else:
            r = _fit_float(vals, na_tok)
            if r:
                return r
    # categorical / string
    labs = []
    seen = {}
    for v in values:
        if v not in seen:
            if len(seen) >= 255:
                return {"k": "str"}, np.array(values, dtype=object)
            seen[v] = len(seen)
            labs.append(v)
    return {"k": "cat", "labels": labs}, np.array([seen[v] for v in values], dtype=np.uint8)


# ---------------------------------------------------------------------------------------------------
# table parsing
# ---------------------------------------------------------------------------------------------------
def parse_tsv(b: bytes, has_header=True):
    """-> (header[list[str]] or None, columns[list[list[str]]]) or None if the file is not a plain rectangular TSV
    that ends with a newline (CR, empty fields at the end of the table and ragged rows are refused)."""
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if "\r" in s or (s and not s.endswith("\n")):
        return None
    lines = s.split("\n")[:-1] if s else []
    if not lines:
        return None
    rows = [ln.split("\t") for ln in lines]
    w = len(rows[0])
    if any(len(r) != w for r in rows):
        return None
    header = None
    if has_header:
        header, rows = rows[0], rows[1:]
    cols = [[r[j] for r in rows] for j in range(w)]
    return header, cols


def render_table(header, text_cols) -> bytes:
    n = len(text_cols[0]) if text_cols else 0
    out = []
    if header is not None:
        out.append("\t".join(header))
    out.extend("\t".join(c[i] for c in text_cols) for i in range(n))
    return ("\n".join(out) + "\n").encode("utf-8")
