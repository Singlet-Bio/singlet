Reading pipeline outputs
========================

The :mod:`singlify.io` submodule reads ``.1pz`` sparse-matrix files
produced by the singlify single-cell reprocessing pipeline. All decode
work happens in a header-only C++ reader accessed via the
:mod:`singlify._pz_io` pybind11 binding; the Python surface is a thin
marshalling layer around it.

Reading a single file
---------------------

:func:`singlify.io.read_matrix` returns a
(:class:`scipy.sparse.csc_matrix`, :class:`~singlify.io.PZMetadata`) pair:

.. code-block:: python

    import singlify.io as sio

    mat, meta = sio.read_matrix(
        "quant/scrna/GSE174/GSE174399/GSM5293863/gene_counts.1pz"
    )
    print(mat.shape)                   # (38606, 16079)
    print(mat.dtype)                   # uint16 (narrow-width hint from writer)
    print(meta.user_kv["gsm_id"])      # 'GSM5293863'
    print(meta.rownames[:3])           # ['ENSG00000000003', ...]
    print(meta.colnames[:3])           # cell barcodes

The matrix dtype is the narrowest type that losslessly holds every stored
value:

* ``vt_code == 1`` → :class:`numpy.uint8`
* ``vt_code == 2`` → :class:`numpy.uint16`
* ``vt_code == 3`` → :class:`numpy.uint32`

This narrow typing is a signal from the writer — the on-disk values are
varint-encoded, so the width doesn't affect decoding, only downstream
memory use.

Reading a whole pipeline directory
----------------------------------

:func:`singlify.io.read_dir` walks every ``.1pz`` file in a sample
directory and returns a :class:`singlify.io.PipelineDirectory`:

.. code-block:: python

    dd = sio.read_dir("quant/scrna/GSE174/GSE174399/GSM5293863")

    # Dict-like access
    dd["gene_counts"].shape
    "spliced" in dd
    list(dd.keys())

    # Metadata from any one matrix (they all carry the same user_kv)
    dd.metadata.user_kv["gsm_id"]
    dd.metadata.rownames[:5]

Filter the load with ``include`` / ``exclude``:

.. code-block:: python

    dd = sio.read_dir(
        "quant/scrna/GSE174/GSE174399/GSM5293863",
        include=["spliced", "unspliced", "ambiguous"],
    )

Peeking at metadata without decoding the matrix
-----------------------------------------------

For the common "what GSM is this?" case you don't want to pay for a full
decode. :func:`singlify.io.open_pz` is a pure-Python context manager
that parses the 96-byte header and returns file metadata:

.. code-block:: python

    from singlify.io import open_pz, read_metadata

    with open_pz("gene_counts.1pz") as pz:
        print(pz.header.m, "×", pz.header.n)  # rows × cols
        print(pz.header.nnz)
        print(pz.header.vt_code.name)         # 'UINT16'

        md = read_metadata(pz)
        print(md.user_kv["gsm_id"])           # 'GSM5293863'

Pass ``verify_crc=True`` to also run a full-file CRC32 check against the
16-byte footer before returning — this catches silent corruption:

.. code-block:: python

    with open_pz("gene_counts.1pz", verify_crc=True) as pz:
        ...  # raises PZError if the CRC mismatches

Error handling
--------------

All the readers raise :exc:`ValueError` (via
:class:`singlify.io.PZError`) on any structural failure: missing file,
truncated file, wrong magic, CRC mismatch, unsupported version, bad
``vt_code``, unexpected chunk count. The error message always includes
the file path so CI logs are useful.

.. code-block:: python

    try:
        mat, _ = sio.read_matrix("garbage.1pz")
    except ValueError as e:
        print(f"decode failed: {e}")
