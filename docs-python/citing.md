# How to cite

In a paper, cite the original study of every dataset you use (its paper and GEO accession; the Cite button on each study page at [singlet.bio](https://singlet.bio) gives both) and singlet.bio with its release and the date you downloaded the data:

> DeBruine Z. singlet.bio: an open atlas of public single-cell RNA-seq studies reprocessed from raw reads. Singlet Research Alpha (October 2026). https://singlet.bio (accessed YYYY-MM-DD).

```bibtex
@misc{debruine2026singletbio,
  author       = {DeBruine, Zachary},
  title        = {{singlet.bio}: an open atlas of public single-cell {RNA-seq} studies reprocessed from raw reads},
  year         = {2026},
  howpublished = {\url{https://singlet.bio}},
  url          = {https://singlet.bio},
  note         = {Singlet Research Alpha (October 2026). Accessed YYYY-MM-DD}
}
```

If you used this package, cite it with the version you ran (`singlet.__version__`):

> DeBruine Z. singlet: Python, R and C++ client and engine for singlet.bio. Version X.Y.Z. 2026. https://github.com/Singlet-Bio/singlet

```bibtex
@misc{debruine2026singlet,
  author       = {DeBruine, Zachary},
  title        = {singlet: {Python}, {R} and {C++} client and engine for {singlet.bio}},
  year         = {2026},
  howpublished = {\url{https://github.com/Singlet-Bio/singlet}},
  url          = {https://github.com/Singlet-Bio/singlet},
  note         = {Python package version X.Y.Z}
}
```

For reproducibility, also report each study's pipeline version (shown on its study page). RIS and plain text:
[singlet.bio/about#cite](https://singlet.bio/about#cite).
