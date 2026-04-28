# SinglePress manuscript — figure reproduction
# Usage:
#   make figures     # generate all figure PDFs from pre-computed data
#   make clean       # remove generated PDFs

RSCRIPT := Rscript

.PHONY: figures clean

figures: figures/fig1_compression.pdf figures/fig2_performance.pdf \
         figures/fig3_io.pdf figures/figS1_structure.pdf \
         figures/fig4_frontier.pdf figures/fig5_ecosystem.pdf

figures/fig1_compression.pdf figures/fig2_performance.pdf figures/fig3_io.pdf figures/figS1_structure.pdf: figures/figures_main.R $(wildcard data/*.csv) $(wildcard data/*.json)
	cd figures && $(RSCRIPT) figures_main.R

figures/fig4_frontier.pdf: figures/figures_frontier.R data/compression_frontier.csv
	cd figures && $(RSCRIPT) figures_frontier.R

figures/fig5_ecosystem.pdf: figures/figures_new.R $(wildcard data/*.csv)
	cd figures && $(RSCRIPT) figures_new.R

clean:
	rm -f figures/fig1_compression.pdf figures/fig2_performance.pdf \
	      figures/fig3_io.pdf figures/figS1_structure.pdf \
	      figures/fig4_frontier.pdf
