# singlet

**An open single-cell RNA-seq atlas — find, load, and analyze processed
datasets with one line of Python.**

Singlet reprocesses public single-cell RNA-seq studies into a uniform,
analysis-ready atlas. You work with two simple things: GEO accession strings
(`GSE…` / `GSM…`) and `.singlet` files. Everything downloads as
[AnnData](https://anndata.readthedocs.io/), ready for scanpy or PyTorch.

Data is CC0 (public domain); code is MIT. No login, no API keys, no usage
pricing — public data is free to download.

```{include} ../python/README.md
:start-after: "## Install"
:end-before: "## Natural-language search"
```

See the [full README on GitHub](https://github.com/Singlet-Bio/singlet/blob/main/python/README.md)
for natural-language search, format conversion, the GPU/PyTorch extras, and
the pipeline CLI.

```{toctree}
:hidden:
:maxdepth: 2

installation
api
GitHub <https://github.com/Singlet-Bio/singlet>
Data catalog <https://singlet.bio>
```
