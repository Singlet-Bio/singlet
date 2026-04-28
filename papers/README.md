# Singlet AI Papers

Scientific manuscripts, application notes, and whitepapers for the Singlet AI project.

## Structure

```
papers/
├── templates/                    Shared LaTeX style and templates
│   ├── singletai-preprint.sty    Two-column preprint style
│   ├── manuscript.tex            Full paper template
│   ├── appnote.tex               Application note template (2–4 pp)
│   └── whitepaper.tex            Single-column whitepaper template
├── shared/
│   ├── refs.bib                  Shared bibliography
│   └── figures/                  Shared figure assets
├── manuscripts/                  One subdirectory per manuscript
│   └── <name>/
│       ├── main.tex
│       ├── refs.bib              Local references
│       └── figures/
│           ├── render.R          ggplot2 figure rendering script
│           └── *.pdf             Rendered figures
└── Makefile
```

## Building

```bash
# Single manuscript
make <manuscript-name>

# All manuscripts
make all

# Validate BibTeX (check for missing DOIs)
make validate

# Render figures
make figures-<manuscript-name>
```

Requires TeX Live (`latexmk`, `pdflatex`, `bibtex`). For figures: R with `ggplot2`.

## Creating a New Manuscript

```bash
# Copy a template
mkdir -p manuscripts/my-paper/figures
cp templates/manuscript.tex manuscripts/my-paper/main.tex
cp shared/refs.bib manuscripts/my-paper/refs.bib
# Edit main.tex, then: make my-paper
```

## Part of Singlet AI

| Repository | Purpose |
|-----------|---------|
| [geo-reprocess](https://github.com/Singlet-AI/geo-reprocess) | HPC pipeline |
| [singlet](https://github.com/Singlet-AI/singlet) | Python client |
| [singlepress](https://github.com/Singlet-AI/singlepress) | Compression |
| [singlet-intelligence](https://github.com/Singlet-AI/singlet-intelligence) | ML models |
| [singlet-strategy](https://github.com/Singlet-AI/singlet-strategy) | Strategic planning |
| [singletai-website](https://github.com/Singlet-AI/singletai-website) | Website & dashboard |
| **papers** | Manuscripts & reports |
