"""Port of the ATAC barcode layout / whitelist choice (atac_barcodes.h detect_layout, atac_pipeline.h choose) with the two
sampling rules: the PR #48 prefix rule (first N records in stream order) and rule A2 (every record with qname_hash % 64 == 0)."""
from __future__ import annotations

M64 = (1 << 64) - 1
K = 16


def qname_hash(s: str) -> int:
    """FNV-1a over the bytes, murmur finaliser: AtacBarcodeCorrector::qname_hash (the engine's hash_qname)."""
    h = 14695981039346656037
    for b in s.encode():
        h = ((h ^ b) * 1099511628211) & M64
    h ^= h >> 33; h = (h * 0xff51afd7ed558ccd) & M64
    h ^= h >> 33; h = (h * 0xc4ceb9fe1a85ec53) & M64
    h ^= h >> 33
    return h


def is_layout_sample(qname: str) -> bool:
    return (qname_hash(qname) & 63) == 0


_RC = str.maketrans("ACGTacgt", "TGCAtgca")


def window(raw: str, off: int, rc: bool):
    if len(raw) < off + K:
        return None
    if not rc:
        w = raw[off:off + K]
    else:
        r = raw.translate(_RC)[::-1]
        w = r[off:off + K]
    return w if set(w) <= set("ACGT") else None


def detect_layout(sample: list[str], wl: set[str]):
    """Highest exact hit rate over (rc, offset); strict '>' so ties go to forward, then the smaller offset."""
    best = (0, False, 0.0)
    maxlen = max((len(s) for s in sample), default=0)
    for rc in (False, True):
        for off in range(0, maxlen - K + 1):
            n = hits = 0
            for s in sample:
                if len(s) < off + K:
                    continue
                n += 1
                w = window(s, off, rc)
                hits += w is not None and w in wl
            r = hits / n if n else 0.0
            if r > best[2]:
                best = (off, rc, r)
    return best


def choose(sample: list[str], whitelists: list[set[str]]):
    """Best hit rate over whitelist candidates, ties to the earlier one: (index, (offset, rc, rate))."""
    best_i, best = 0, None
    for i, wl in enumerate(whitelists):
        lay = detect_layout(sample, wl)
        if best is None or lay[2] > best[2]:
            best_i, best = i, lay
    return best_i, best


def prefix_sample(records: list[tuple[str, str]], n: int) -> list[str]:
    """PR #48: the raw barcodes of the first n records in stream order. records = [(qname, raw)]."""
    return [raw for _, raw in records[:n]]


def hash_sample(records: list[tuple[str, str]]) -> list[str]:
    """A2: every record whose qname hashes to 0 mod 64."""
    return [raw for q, raw in records if is_layout_sample(q)]
