The ``.1pz`` binary format
==========================

``.1pz`` (``TP1Z v1``) is singlify's native sparse-matrix format —
a **VOCSC** (Value-Offset Compressed Sparse Column) file with zstd
block compression. It is designed for:

1. **Fast random column (cell) access** — a column decompresses in ~10 µs
2. **High compression ratio** — UMI count matrices compress 40–80× vs MTX
3. **Direct read into R/Python sparse objects** — no intermediate copy
4. **Embedded metadata** — the GEO context travels with the file so
   pipeline outputs are self-describing

The canonical binary spec lives at ``singlify/docs/1PZ_FORMAT_SPEC.md``
in the repo. The authoritative reference implementation is the
header-only C++ reader at
``singlify/include/singlet-pileup/pz_reader.h``, which is bit-exact
with the writer at ``pz_writer.h``. Both this Python package and the
R package bind to the same C++ reader.

File layout (overview)
----------------------

.. code-block:: text

    [ PZHeader      : 96 bytes ]
    [ permutation   : zstd-compressed ]
    [ column counts : zstd-compressed (ptr_width bytes per column) ]
    [ chunk table   : num_chunks × uint32 (blob sizes) ]
    [ chunk 0       : 20-byte header + zstd blob ]
    [ chunk 1       : ... ]
    ...
    [ colsums_z     : zstd-compressed column sums ]
    [ metadata      : zstd-compressed TLV stream (optional) ]
    [ PZFooter      : 16 bytes ]

The ``PZHeader`` carries ``m`` (rows), ``n`` (cols), ``nnz``, ``vt_code``
(value-type hint), flag bits, chunk count, and byte offsets for the
metadata and transpose blocks. The ``PZFooter`` carries a CRC32 over
the entire file body (everything before the footer) for corruption
detection.

Value types
-----------

The ``vt_code`` byte records the narrowest integer width that
losslessly holds every stored value — **not** the writer's C++ template
parameter. The writer assigns:

* ``1`` — all values fit in ``uint8``  (max ≤ 255)
* ``2`` — all values fit in ``uint16`` (max ≤ 65535)
* ``3`` — all values fit in ``uint32`` (max ≤ 2³²−1)

On-disk values are **varint-encoded** inside the VOCSC column TLV
stream, so ``vt_code`` only affects downstream memory layout (the
Python binding casts to the matching narrow numpy dtype), not the
decoding pipeline itself.

Metadata block (TLV)
--------------------

When the ``HAS_METADATA`` flag is set, the metadata block is a
zstd-compressed stream of ``(tag: uint8, length: uint32LE, payload)``
records terminated by a ``META_TAG_END`` (0) tag. Tag IDs:

============  ==========================  ======================================
Tag           Meaning                      Payload format
============  ==========================  ======================================
0             ``META_TAG_END``             No payload (terminator)
1             ``META_TAG_ROWNAMES``        ``\0``-delimited UTF-8 strings
2             ``META_TAG_COLNAMES``        ``\0``-delimited UTF-8 strings
3             ``META_TAG_USER_KV``         ``\0``-delimited ``key\0value\0`` pairs
============  ==========================  ======================================

Unknown tags must be silently skipped by readers for forward
compatibility. The pipeline's GEO context (``gsm_id``, ``gse_id``,
``srr_ids``, ``organism``, ``protocol``, ``singlify_version``,
``pipeline_date``, ``read_count``) is stored in the ``USER_KV`` block.

VOCSC chunks
------------

Each chunk covers up to ``chunk_cols`` consecutive columns. A chunk
blob has a 20-byte header (``ng32 | msz32 | zsz | chunk_crc32 |
psz32``) followed by ``zsz`` bytes of zstd-compressed body.

The decompressed body is the output of ``bitmap_pack(prefilter)``
(first ``(pf_sz + 7) / 8`` bytes = bitmap, then non-zero bytes in
order), where ``prefilter`` itself is laid out as:

.. code-block:: text

    [0 .. msz]                         # column TLV metadata (varint stream)
    [msz .. msz + 8*ceil(ng/8)]        # bit-plane 0 of the gap LSBs
    [msz + ... .. msz + ... + (gw-1)*ng]  # raw high bytes of the gaps

The column TLV metadata is, per column, a varint ``ng_col`` (number of
distinct values) followed by ``ng_col`` repetitions of
``(varint cv, varint cnt, <cnt gap entries from the gap stream>)``.
Within a value group, row indices are gap-encoded (delta from the
previous row index + 1) and the decoded rows are in **permuted** space;
the reader applies the inverse of the permutation block to recover
original rows, then sorts each column's non-zeros by original row.

The Python / R decoders must invert every transform in the correct
order:

#. zstd decompress chunk body → packed buffer of size ``psz32``
#. Verify the 20-byte header's ``chunk_crc32`` matches ``CRC32(packed)``
#. ``bitmap_unpack(packed, psz32)`` → prefilter buffer of size ``pf_sz``
#. ``bit_planes_decode(prefilter[msz ..], ng32)`` → gap low bytes
#. Byte-merge low + raw high-byte planes → uint16/uint32 gap stream
#. Walk column TLV; for each (cv, cnt) group, consume ``cnt`` gaps
   and gap-decode rows within that group
#. Apply ``perm[row]`` to map permuted rows back to original rows
#. Sort each column's entries by original row

All of this is ~400 lines of C++ in ``pz_reader.h``. The Python and R
bindings add another 100 lines each of pure marshalling.

Storage savings
---------------

Typical compression ratios on singlify pipeline outputs:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20 20

   * - Sample
     - Matrix
     - Raw (MTX)
     - ``.1pz``
   * - GSM5293863 gene_counts
     - 38606 × 16079
     - ~170 MB
     - 3.9 MB
   * - GSM5293863 exon_counts
     - 310797 × 16079
     - ~900 MB
     - 8.0 MB
   * - GSM5293863 sj_counts
     - 1854 × 16079
     - ~40 MB
     - 3.5 MB

Drop-policy
-----------

The recent pipeline revision drops these empirically-redundant files
from the NFS copy step:

* ``gene_counts.1pz`` — bit-exactly equal to
  ``spliced + unspliced + ambiguous`` for every observed 10xv3 and
  drop-seq sample. Verifiable property; the Python test suite enshrines
  it via ``test_gene_counts_equals_velocity_trio``.
* ``ambiguous.1pz`` — ``nnz = 0`` for every 10x and drop-seq protocol
  in the catalog (STARsolo only emits non-zero ambiguous for
  paired-end full-length Smart-seq-style protocols).
* ``splice_psi.1pz`` — same per-junction feature axis as
  ``sj_counts``, binary-valued (every non-zero entry = 1), and a
  strict subset of ``sj_counts`` non-zero positions. It encodes "this
  junction participates in a callable PSI event in this cell" and is
  derivable from ``sj_counts + splice_events.tsv``.

Reader code should never assume any of these files is present.

Future drops under evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These have been **verified** as derivable. The reader infrastructure
to reconstruct them on read is now in place, but the pipeline still
ships the per-gene matrices for now (decision pending):

* ``spliced.1pz`` — ``sum(spliced) == sum(exon_counts)`` exactly on
  every test sample. ``spliced[gene, cell]`` is the per-gene
  aggregation of ``exon_counts[exons-of-gene, cell]``.
* ``unspliced.1pz`` — same logic vs ``intron_counts.1pz``.

When the pipeline drops these, the readers reconstruct them
**transparently**:

.. code-block:: python

    from singlify.io import aggregate_features_to_gene, read_matrix

    ex, meta = read_matrix("exon_counts.1pz")
    spliced, gene_ids = aggregate_features_to_gene(ex, meta.rownames)
    # spliced is now a per-gene csc_matrix that drops in for spliced.1pz

The same aggregator is wired into
:func:`singlify.interop.anndata.read_anndata` — when ``spliced.1pz``
is missing but ``exon_counts.1pz`` is present, the AnnData reader
auto-derives ``layers["spliced"]`` and records the source in
``adata.uns["singlify_derived_layers"]`` so users can audit the
provenance.

The R package has the parallel ``singlify::aggregate_features_to_gene()``
function plus the same auto-derive wiring in ``as_sce()`` and
``as_seurat()`` (recorded in ``metadata(sce)$singlify_derived_assays``
and ``obj@misc$singlify_derived_assays`` respectively).

The aggregation parses the leading ``ENSG...`` prefix from the exon
rowname format ``ENSG..._GENE_chr:start-end`` (Ensembl IDs never contain
underscores). The aggregation itself is O(n_features + nnz) via a
sparse projection matrix multiplication.

Storage savings if the drop happens
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Per-sample (typical 10xv3 with ~10K cells):

* ``spliced.1pz``    ~3.4 MB
* ``unspliced.1pz``  ~1.0 MB
* Total              ~4.4 MB / sample

Across the full ~200K processable human-catalog target:

* ~880 GB additional savings on top of the gene_counts + ambiguous +
  splice_psi drops already in place
