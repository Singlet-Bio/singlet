"""m_w5a: exact Python port of functions/_shared/bundle-reader.ts parseZipSource() + JSON.stringify(entries).

Source of truth: C:/Users/zacha/Projects/singlet-website/functions/_shared/bundle-reader.ts (lines 111-190).
Port notes (each is a deliberate equivalence with the TypeScript):
  * bytes = HEAD Content-Length; tail = last min(256 KiB, bytes) bytes.
  * EOCD: scan i from len(tail)-22 down to 0 for u32 == 0x06054b50 (first hit from the end wins).
  * nEntries=u16(eocd+10), cdSize=u32(eocd+12), cdOffset=u32(eocd+16). zip64 iff nEntries==0xffff or cdSize==0xffffffff or
    cdOffset==0xffffffff; then locator at eocd-20 (sig 0x07064b50), z64 record offset=u64(loc+8), read from the tail when
    z64Offset >= bytes-tailLen else a separate 56-byte range; sig 0x06064b50; nEntries=u64(rec+32), cdSize=u64(rec+40),
    cdOffset=u64(rec+48).
  * central directory = range(cdOffset, cdOffset+cdSize-1) (one extra request unless it lies inside the tail; slicing the tail
    gives the same bytes).
  * loop while p+46 <= len(cd) and u32(cd,p)==0x02014b50: method=u16(p+10), csz=u32(p+20), usz=u32(p+24), nameLen=u16(p+28),
    extraLen=u16(p+30), commentLen=u16(p+32), off=u32(p+42), name=utf-8 decode (TextDecoder: BOM stripped, U+FFFD on bad bytes).
    If any of usz/csz/off == 0xffffffff, walk the extra field to id 0x0001 and read u64 for usz, csz, off in that order
    (only those that were 0xffffffff). Entries whose name ends with "/" are dropped. Order is central-directory order.
  * entry = {p: name, n: method, c: csz, u: usz, o: off} serialised as JSON.stringify does (key order p,n,c,u,o; no spaces;
    non-ASCII kept raw; control chars as \\u00xx lowercase; " and \\ escaped).
  * `o` is the offset of the LOCAL HEADER (not of the payload).
"""
import json
import struct

EOCD_SIG = 0x06054B50
ZIP64_LOC_SIG = 0x07064B50
ZIP64_EOCD_SIG = 0x06064B50
CEN_SIG = 0x02014B50
TAIL_BYTES = 256 * 1024
M32 = 0xFFFFFFFF


def u16(b, o):
    return b[o] | (b[o + 1] << 8)


def u32(b, o):
    return b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)


def u64(b, o):
    # DataView.getBigUint64 throws RangeError when out of range; struct.error is the analogue here.
    return struct.unpack_from('<Q', b, o)[0]


def _decode_name(raw):
    s = bytes(raw).decode('utf-8', errors='replace')
    if s.startswith('\ufeff'):  # TextDecoder default ignoreBOM=false strips a leading BOM
        s = s[1:]
    return s


def parse_zip_source(size, rng):
    """size: int (HEAD content-length). rng(start, end_inclusive) -> bytes. Returns (bytes, entries list of dicts, info dict)."""
    nbytes = size
    tail_len = min(TAIL_BYTES, nbytes)
    tail_start = nbytes - tail_len
    tail = rng(tail_start, nbytes - 1)
    info = {'requests': 1, 'tail_len': tail_len}

    eocd = -1
    for i in range(len(tail) - 22, -1, -1):
        if u32(tail, i) == EOCD_SIG:
            eocd = i
            break
    if eocd < 0:
        raise ValueError('End of central directory not found')

    n_entries = u16(tail, eocd + 10)
    cd_size = u32(tail, eocd + 12)
    cd_offset = u32(tail, eocd + 16)
    info['zip64'] = False
    if n_entries == 0xFFFF or cd_size == M32 or cd_offset == M32:
        info['zip64'] = True
        loc_at = eocd - 20
        if loc_at < 0 or u32(tail, loc_at) != ZIP64_LOC_SIG:
            raise ValueError('zip64 locator not found')
        z64 = u64(tail, loc_at + 8)
        if z64 >= tail_start:
            rec = tail[z64 - tail_start:]
        else:
            rec = rng(z64, z64 + 55)
            info['requests'] += 1
        if len(rec) < 4 or u32(rec, 0) != ZIP64_EOCD_SIG:
            raise ValueError('zip64 EOCD record not found')
        n_entries = u64(rec, 32)
        cd_size = u64(rec, 40)
        cd_offset = u64(rec, 48)
    info['n_entries_eocd'] = n_entries
    info['cd_size'] = cd_size
    info['cd_offset'] = cd_offset

    if cd_offset >= tail_start and cd_offset + cd_size <= nbytes:
        cd = tail[cd_offset - tail_start: cd_offset - tail_start + cd_size]
        info['cd_in_tail'] = True
    else:
        cd = rng(cd_offset, cd_offset + cd_size - 1)
        info['requests'] += 1
        info['cd_in_tail'] = False
    if len(cd) != cd_size:
        raise ValueError(f'central directory short read: {len(cd)} != {cd_size}')

    entries = []
    n_cen = 0
    p = 0
    while p + 46 <= len(cd) and u32(cd, p) == CEN_SIG:
        n_cen += 1
        method = u16(cd, p + 10)
        csz = u32(cd, p + 20)
        usz = u32(cd, p + 24)
        name_len = u16(cd, p + 28)
        extra_len = u16(cd, p + 30)
        comment_len = u16(cd, p + 32)
        off = u32(cd, p + 42)
        name = _decode_name(cd[p + 46: p + 46 + name_len])

        if usz == M32 or csz == M32 or off == M32:
            e = p + 46 + name_len
            extra_end = e + extra_len
            while e + 4 <= extra_end:
                eid = u16(cd, e)
                esz = u16(cd, e + 2)
                if eid == 0x0001:
                    q = e + 4
                    if usz == M32:
                        usz = u64(cd, q)
                        q += 8
                    if csz == M32:
                        csz = u64(cd, q)
                        q += 8
                    if off == M32:
                        off = u64(cd, q)
                        q += 8
                    break
                e += 4 + esz
        if not name.endswith('/'):
            entries.append({'p': name, 'n': method, 'c': csz, 'u': usz, 'o': off})
        p += 46 + name_len + extra_len + comment_len
    info['n_cen_records'] = n_cen
    info['cd_consumed'] = p
    return nbytes, entries, info


def entries_json(entries):
    """JSON.stringify(entries) as V8 produces it."""
    return json.dumps(entries, ensure_ascii=False, separators=(',', ':'))


def gsm_count(entries):
    """catalog-refresh.ts: distinct GSM ids matched by ^samples/(GSM\\d+)/ over entry paths."""
    import re
    s = set()
    for e in entries:
        m = re.match(r'^samples/(GSM[0-9]+)/', e['p'])  # JS \d is ASCII-only
        if m:
            s.add(m.group(1))
    return len(s)
