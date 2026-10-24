#!/usr/bin/env python
"""w4_mtx test 4: STARsolo raw/matrix.mtx versus gene sums of the .1pz exon/intron interval matrices of the same sample.
    python -P tools/w4_mtx_t4.py <GSE.GSM> [...]      -> results/w4_mtx_t4_<sample>.json + stdout
"""
import os, sys, json, math, zlib, lzma, collections
import numpy as np, zstandard
sys.path.insert(0, '../../python')
from singlet._pz import read_1pz
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE)
CORP = os.path.join(BC, "corpus_w4_mtx"); RES = os.path.join(BC, "results")
P = "star_Solo.out__Gene__"


def H(counts):
    c = np.asarray(counts, dtype=np.float64); c = c[c > 0]; n = c.sum()
    return float(-(c * np.log2(c / n)).sum()) if n > 0 else 0.0   # total bits (n * entropy)


def varint(a):
    out = bytearray()
    for v in np.asarray(a, dtype=np.uint64).tolist():
        while v >= 0x80:
            out.append((v & 0x7f) | 0x80); v >>= 7
        out.append(v)
    return bytes(out)


def standalone(rows, cols, vals, ncols):
    """rows/cols sorted by (col,row). CSC cost: per-col nnz + row gaps + values, each stream by zstd-19 / xz-9 / order-0 entropy."""
    nz_per_col = np.bincount(cols, minlength=ncols)
    first = np.concatenate([[True], cols[1:] != cols[:-1]])
    prev = np.concatenate([[0], rows[:-1]])
    gaps = np.where(first, rows, rows - prev)
    res = {}
    if not os.environ.get("W4_FAST"):
        streams = {"colnnz": varint(nz_per_col[nz_per_col > 0]), "gaps": varint(gaps), "vals": varint(vals - 1)}
        z = zstandard.ZstdCompressor(level=19)
        for k, b in streams.items():
            res[k] = {"raw": len(b), "zstd19": len(z.compress(b)), "xz": len(lzma.compress(b, preset=9))}
    g_h = H(np.bincount(gaps)) / 8; v_h = H(np.bincount(vals)) / 8
    nz = nz_per_col[nz_per_col > 0]; c_h = H(np.bincount(nz)) / 8
    res["o0_gaps_bytes"] = g_h; res["o0_vals_bytes"] = v_h; res["o0_colnnz_bytes"] = c_h
    if res.get("gaps"):
        res["total_zstd19"] = sum(res[k]["zstd19"] for k in ("colnnz", "gaps", "vals")); res["total_xz"] = sum(res[k]["xz"] for k in ("colnnz", "gaps", "vals"))
    res["total_o0"] = g_h + v_h + c_h
    return res


def lookup(u, s, q):
    i = np.searchsorted(u, q); i2 = np.minimum(i, len(u) - 1); hit = (i < len(u)) & (u[i2] == q)
    return np.where(hit, s[i2], 0), hit


def main():
    for d in sys.argv[1:]:
        out = {"sample": d}
        b = open(os.path.join(CORP, d, P + "raw__matrix.mtx"), "rb").read()
        pos = 0
        while True:
            e = b.index(b"\n", pos); ln = b[pos:e + 1]; pos = e + 1
            if not ln.startswith(b"%"): break
        G, NB, NNZ = map(int, ln.split())
        T = np.fromstring(b[pos:], dtype=np.int64, sep=" ").reshape(-1, 3)
        mr, mc, mv = T[:, 0] - 1, T[:, 1] - 1, T[:, 2]
        out.update(G=G, NB=NB, mtx_nnz=int(NNZ), mtx_total_umi=int(mv.sum()))
        feat = [l.split("\t")[0] for l in open(os.path.join(CORP, d, P + "raw__features.tsv")).read().split("\n")[:-1]]
        gidx = {g: i for i, g in enumerate(feat)}
        bcs = open(os.path.join(CORP, d, P + "raw__barcodes.tsv")).read().split("\n")[:-1]
        bidx = {x: i for i, x in enumerate(bcs)}
        gsum = {}
        for kind in ("exon", "intron"):
            r = read_1pz(os.path.join(CORP, d, f"{kind}_counts.1pz"))
            rg = np.array([gidx.get(n.split("_")[0], -1) for n in r["rownames"]], dtype=np.int64)
            assert (rg >= 0).all(), "unmapped interval gene"
            ip, ix, dv = r["indptr"].astype(np.int64), r["indices"].astype(np.int64), r["data"].astype(np.int64)
            cn = np.repeat(np.arange(r["n"]), np.diff(ip))
            bc = np.array([bidx.get(x, -1) for x in r["colnames"]], dtype=np.int64)
            miss = int((bc < 0).sum())
            col = bc[cn]; ok = col >= 0
            key = col[ok] * G + rg[ix[ok]]
            u, inv = np.unique(key, return_inverse=True)
            s = np.bincount(inv, weights=dv[ok]).astype(np.int64)
            gsum[kind] = (u, s)
            gsum[kind + "_cols"] = np.unique(bc[bc >= 0])
            out[f"{kind}_1pz"] = dict(m=int(r["m"]), n=int(r["n"]), nnz=int(r["nnz"]), cols_not_in_raw_whitelist=miss,
                                      gene_barcode_nnz=int(len(u)), total=int(dv.sum()), vt_code=int(r["vt_code"]))
        mkey = mc * G + mr
        order = np.argsort(mkey, kind="stable"); mkey_s = mkey[order]; mval_s = mv[order]
        ue, se = gsum["exon"]; ui, si = gsum["intron"]
        uu = np.union1d(ue, ui)
        e_u, _ = lookup(ue, se, uu); i_u, _ = lookup(ui, si, uu)
        ei = e_u + i_u
        mcols = np.unique(mc)
        pcols = np.union1d(gsum["exon_cols"], gsum["intron_cols"])
        out["mtx_nonzero_columns"] = int(len(mcols)); out["pz_columns_union"] = int(len(pcols))
        out["mtx_cols_in_pz"] = int(np.isin(mcols, pcols).sum()); out["pz_cols_in_mtx_nonzero"] = int(np.isin(pcols, mcols).sum())
        in_pz_col = np.isin(mc, pcols)
        out["mtx_nnz_in_pz_columns"] = int(in_pz_col.sum()); out["mtx_umi_in_pz_columns"] = int(mv[in_pz_col].sum())
        for tag, (keys, vals) in {"exon+intron": (uu, ei), "exon": (ue, se)}.items():
            mval_in_pz, hit = lookup(keys, vals, mkey)          # pz value at each mtx nonzero
            n_both = int(hit.sum()); n_m = len(mkey)
            p_in_m, phit = lookup(mkey_s, mval_s, keys)          # mtx value at each pz nonzero
            n_p = len(keys); n_p_only = int((~phit).sum())
            d_ = mv[hit] - mval_in_pz[hit]
            dc = collections.Counter(np.clip(d_, -5, 5).tolist())
            r_ = {"mtx_nnz": n_m, "pz_nnz": n_p, "both": n_both, "mtx_only": n_m - n_both, "pz_only": n_p_only,
                  "frac_mtx_nnz_also_in_pz": n_both / n_m, "frac_pz_nnz_also_in_mtx": (n_p - n_p_only) / n_p,
                  "frac_equal_among_both": float((d_ == 0).mean()) if n_both else None,
                  "diff_mtx_minus_pz_clipped_hist": {str(k): int(v) for k, v in sorted(dc.items())},
                  "mean_diff": float(d_.mean()) if n_both else None,
                  "umi_mtx_total": int(mv.sum()), "umi_pz_total": int(vals.sum()),
                  "umi_mtx_on_both": int(mv[hit].sum()), "umi_pz_on_both": int(mval_in_pz[hit].sum())}
            ctxv = np.minimum(vals, 20)
            mval_on_pz = np.where(phit, p_in_m, 0)
            mclip = np.minimum(mval_on_pz, 63)
            joint = np.zeros((21, 64), dtype=np.int64); np.add.at(joint, (ctxv, mclip), 1)
            Hc = sum(H(joint[c]) for c in range(21))
            esc = mval_on_pz[mval_on_pz >= 63]
            esc_bits = float(sum(2 * math.log2(v - 62 + 1) + 1 for v in esc.tolist())) if len(esc) else 0.0
            half = ((keys // G) % 2 == 0)
            jt = np.zeros((21, 64), dtype=np.int64); np.add.at(jt, (ctxv[half], mclip[half]), 1)
            pr = (jt + 0.02) / (jt + 0.02).sum(1, keepdims=True)
            oos_bits = float(-np.log2(pr[ctxv[~half], mclip[~half]]).sum())
            r_["cond_cost_on_pz_support"] = {
                "support_entries": n_p, "plugin_bits_total": Hc + esc_bits, "plugin_bytes": (Hc + esc_bits) / 8,
                "heldout_odd_barcodes_bits": oos_bits, "heldout_entries": int((~half).sum()),
                "heldout_bits_per_entry": oos_bits / max(1, int((~half).sum())),
                "plugin_bits_per_mtx_nonzero": (Hc + esc_bits) / n_m}
            mo = ~hit
            if mo.any():
                o = np.argsort(mkey[mo], kind="stable")
                k = mkey[mo][o]
                r_["mtx_only_standalone"] = standalone(k % G, k // G, mv[mo][o], NB)
            out[f"vs_{tag}"] = r_
        o = np.argsort(mkey, kind="stable")
        out["standalone_mtx_csc"] = standalone(mkey[o] % G, mkey[o] // G, mv[o], NB)
        out["mtx_text_bytes"] = len(b); out["mtx_text_deflate9"] = len(zlib.compress(b, 9))
        if not os.environ.get("W4_FAST"): out["mtx_text_zstd19"] = len(zstandard.ZstdCompressor(level=19).compress(b))
        json.dump(out, open(os.path.join(RES, f"w4_mtx_t4_{d}.json"), "w"), indent=1, default=int)
        print(json.dumps(out, indent=1, default=int)[:7000])


main()
