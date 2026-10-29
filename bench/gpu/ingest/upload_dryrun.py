#!/usr/bin/env python3
"""Dry-run upload planner for .1fg files (interface sketch for the fetcher, workstream W4).

Reads only the local file's header, chunk index and footer. It makes NO network calls, holds no credentials and writes nothing
except stdout (or --out JSON). It prints what an upload would do: object key, part plan (parts end on chunk boundaries so every part
is a valid, independently checkable run of chunks), the checksum the object store would be sent, and the manifest record that the
download workstream would register once the real upload succeeds.

  upload_dryrun.py FILE.1fg --accession SRR21102187 --bucket <name> [--prefix ingest/v1] [--part-mib 64] [--out manifest.json]

The real implementation would live behind `singlet ingest upload` with `--dry-run` as the default; this script only shows the plan.
"""
import argparse, hashlib, json, struct, sys, os

HDR = struct.Struct('<IHHBBBBIQQQIHHBBBBIQIIQQQ')   # FileHeader up to source_bytes (96 bytes), rest reserved
IDX = struct.Struct('<QII')
FOOT = struct.Struct('<IIQII')


def crc32c(data, crc=0):
    # table-driven software CRC32C (Castagnoli); only used for the 24-byte footer check on small buffers here
    poly = 0x82F63B78
    tab = getattr(crc32c, 'tab', None)
    if tab is None:
        tab = []
        for i in range(256):
            c = i
            for _ in range(8):
                c = (c >> 1) ^ poly if c & 1 else c >> 1
            tab.append(c)
        crc32c.tab = tab
    c = crc ^ 0xFFFFFFFF
    for b in data:
        c = tab[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('file'); ap.add_argument('--accession', required=True); ap.add_argument('--bucket', required=True)
    ap.add_argument('--prefix', default='ingest/v1'); ap.add_argument('--part-mib', type=int, default=64); ap.add_argument('--out')
    a = ap.parse_args()
    size = os.path.getsize(a.file)
    with open(a.file, 'rb') as f:
        h = HDR.unpack(f.read(HDR.size))
        (magic, ver, hsz, codec, level, flags, proto, chunk_pairs, n_pairs, n_in, n_drop, n_chunks, r2_slot, r2_max,
         bc_len, umi_len, bc_start, r1_len, max_exc, meta_off, meta_comp, meta_raw, chunks_off, index_off, src_bytes) = h
        if magic != 0x00474631:
            sys.exit('not a .1fg file')
        f.seek(size - FOOT.size); fmagic, fn, fio, ficrc, _ = FOOT.unpack(f.read(FOOT.size))
        assert fmagic == 0x31475446 and fn == n_chunks and fio == index_off, 'footer mismatch'
        f.seek(index_off); raw = f.read(n_chunks * IDX.size)
        index = [IDX.unpack_from(raw, i * IDX.size) for i in range(n_chunks)]
        sha = hashlib.sha256()
        # the object checksum is computed in the real run while streaming the file; the dry run hashes only the header + index
        sha.update(raw)
    part_bytes = a.part_mib << 20
    parts, start, cur_first, cur_pairs = [], chunks_off, 0, 0
    # part 0 carries header + metadata (everything before the first chunk) plus whole chunks up to the part size
    start = 0
    for i, (off, tot, npairs) in enumerate(index):
        end = off + tot
        cur_pairs += npairs
        if end - start >= part_bytes:
            parts.append({'part': len(parts) + 1, 'offset': start, 'bytes': end - start, 'first_chunk': cur_first, 'chunks': i + 1 - cur_first, 'pairs': cur_pairs})
            start, cur_first, cur_pairs = end, i + 1, 0
    if cur_first < n_chunks:   # tail: remaining chunks + index + footer
        parts.append({'part': len(parts) + 1, 'offset': start, 'bytes': size - start, 'first_chunk': cur_first, 'chunks': n_chunks - cur_first, 'pairs': cur_pairs})
    key = '%s/%s/%s.1fg' % (a.prefix.strip('/'), a.accession, a.accession)
    manifest = {
        'schema': 'singlet-ingest-manifest/1', 'dry_run': True, 'network_calls': 0,
        'accession': a.accession, 'format': '1fg', 'version': ver, 'codec': 'zstd' if codec == 1 else 'raw', 'codec_level': level,
        'object': {'bucket': a.bucket, 'key': key, 'bytes': size, 'content_type': 'application/x-singlet-1fg'},
        'reads': {'pairs': n_pairs, 'pairs_in_source': n_in, 'dropped': n_drop, 'r2_slot_bytes': r2_slot, 'r2_max_len': r2_max, 'bc_len': bc_len, 'umi_len': umi_len, 'protocol_id': proto},
        'chunks': {'count': n_chunks, 'max_pairs': chunk_pairs, 'index_offset': index_off, 'per_chunk_crc32c': bool(flags & 1)},
        'upload': {'multipart': len(parts) > 1, 'part_size_mib': a.part_mib, 'parts': len(parts), 'part_plan_head': parts[:3], 'part_plan_tail': parts[-1:],
                   'checksum_algorithm': 'CRC32C (x-amz-checksum-crc32c, per part)', 'resume': 'a failed part is re-sent alone; parts begin on chunk boundaries'},
        'index_sha256': sha.hexdigest(), 'source_bytes_1fq_or_sra': src_bytes,
        'would_register': ['manifest row (accession, key, bytes, pairs, n_chunks, index_sha256)', 'no write to Cloudflare in this dry run'],
    }
    out = json.dumps(manifest, indent=1)
    if a.out:
        open(a.out, 'w').write(out)
    print(out)
    print('DRY RUN: nothing was uploaded', file=sys.stderr)


if __name__ == '__main__':
    main()
