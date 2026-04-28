# Singlet AI Papers — Build System
#
# Usage:
#   make <manuscript-name>    Build a single manuscript (e.g., make genomics-intelligence)
#   make all                  Build all manuscripts
#   make clean                Remove build artifacts
#   make validate             Check BibTeX entries for missing DOIs/URLs
#
# Requirements: latexmk, pdflatex, bibtex (TeX Live recommended)

MANUSCRIPTS := $(notdir $(wildcard manuscripts/*/main.tex))
MANUSCRIPT_DIRS := $(dir $(wildcard manuscripts/*/main.tex))
MANUSCRIPT_NAMES := $(patsubst manuscripts/%/main.tex,%,$(wildcard manuscripts/*/main.tex))

.PHONY: all clean validate help $(MANUSCRIPT_NAMES)

all: $(MANUSCRIPT_NAMES)

# Build a specific manuscript (pdflatex + bibtex, 3 passes)
$(MANUSCRIPT_NAMES):
	@echo "Building $@..."
	cd manuscripts/$@ && TEXINPUTS=../../templates//:$$TEXINPUTS pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	cd manuscripts/$@ && bibtex main > /dev/null 2>&1 || true
	cd manuscripts/$@ && TEXINPUTS=../../templates//:$$TEXINPUTS pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	cd manuscripts/$@ && TEXINPUTS=../../templates//:$$TEXINPUTS pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
	@echo "Output: manuscripts/$@/main.pdf"

# Clean all build artifacts
clean:
	@for dir in $(MANUSCRIPT_DIRS); do \
		cd "$$dir" && rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz *.run.xml *.bcf 2>/dev/null; \
		cd - > /dev/null; \
	done
	@echo "Cleaned all build artifacts."

# Validate BibTeX: every @article/@inproceedings must have doi or url
validate:
	@echo "Validating BibTeX entries..."
	@for bib in shared/refs.bib manuscripts/*/refs.bib; do \
		if [ -f "$$bib" ]; then \
			echo "Checking $$bib:"; \
			grep -c '@' "$$bib" | xargs -I{} echo "  {} entries"; \
			missing=$$(grep -B20 '^\s*}' "$$bib" | \
				awk '/^@/{key=$$0} /doi|url/{found=1} /^\s*}/{if(!found && key){print key}; found=0; key=""}'); \
			if [ -n "$$missing" ]; then \
				echo "  WARNING: Entries without doi/url:"; \
				echo "$$missing" | sed 's/^/    /'; \
			else \
				echo "  All entries have doi or url. ✓"; \
			fi; \
		fi; \
	done

# Render figures using R/ggplot2 (if figures/render.R exists)
figures-%:
	@if [ -f manuscripts/$*/figures/render.R ]; then \
		echo "Rendering figures for $*..."; \
		cd manuscripts/$*/figures && Rscript render.R; \
	else \
		echo "No figures/render.R found for $*"; \
	fi

help:
	@echo "Singlet AI Papers Build System"
	@echo ""
	@echo "  make all              Build all manuscripts"
	@echo "  make <name>           Build one manuscript"
	@echo "  make clean            Remove build artifacts"
	@echo "  make validate         Check BibTeX for missing DOIs"
	@echo "  make figures-<name>   Render ggplot2 figures for a manuscript"
	@echo ""
	@echo "Available manuscripts:"
	@for name in $(MANUSCRIPT_NAMES); do echo "  $$name"; done
