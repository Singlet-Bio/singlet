# w4_reader: time the production R read path (repo r/src/singlet.dll + r/R code) on one sample's members.
# The installed R package named 'singlet' on this laptop is a different package (no pz_read_1pz_raw), so the repo's
# prebuilt r/src/singlet.dll is loaded directly and the two R functions are sourced from r/R. Download excluded.
# Usage (from bench/compression): Rscript tools/w4_reader_r_time.R
suppressMessages({library(Rcpp); library(Matrix)})
repo <- normalizePath("../..")
dll <- dyn.load(file.path(repo, "r", "src", "singlet.dll"))
pz_read_1pz_raw <- function(path) .Call("_singlet_pz_read_1pz_raw", path)
source(file.path(repo, "r", "R", "read.R"), local = TRUE)  # read_1pz()
source(file.path(repo, "r", "R", "bundle.R"), local = TRUE) # .bundle_aggregate_to_vocab()
best <- function(f, reps = 3) min(vapply(seq_len(reps), function(i) system.time(f())[["elapsed"]], numeric(1))) * 1000
ex <- normalizePath("corpus_w4_reader/GSM4317426_exon.1pz"); it <- normalizePath("corpus_w4_reader/GSM4317426_intron.1pz")
m_ex <- read_1pz(ex)
cat(sprintf("exon: dim %d x %d, nnz %d\n", nrow(m_ex), ncol(m_ex), length(m_ex@x)))
cat(sprintf("pz_read_1pz_raw (decode + copies, double values, names to CHARSXP) exon: %.0f ms, intron: %.0f ms\n",
            best(function() pz_read_1pz_raw(ex)), best(function() pz_read_1pz_raw(it))))
cat(sprintf("read_1pz (raw + sparseMatrix(i = indices + 1L) + dimnames) exon: %.0f ms, intron: %.0f ms\n",
            best(function() read_1pz(ex)), best(function() read_1pz(it))))
fv <- jsonlite::fromJSON("corpus_w4_reader/feature_vocab_GSE145370.json", simplifyVector = FALSE)
gene_ids <- vapply(fv$genes, function(g) g$gene_id, character(1))
cat(sprintf("n_genes %d; .bundle_aggregate_to_vocab (match + sparseMatrix + proj %%*%% mat), exon: %.0f ms\n",
            length(gene_ids), best(function() .bundle_aggregate_to_vocab(m_ex, gene_ids))))
cat(sprintf("dgCMatrix exon object size %.1f MB (double values: 12 bytes per nonzero)\n", as.numeric(object.size(m_ex)) / 1e6))
