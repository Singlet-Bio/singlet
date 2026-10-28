# Write and read a count matrix as a .s1 sample object of your own (docs/format-s1.md).
#
# write_s1() stores one integer count matrix with its gene ids and cell barcodes as a sample object holding a single pz2
# member ("counts"), coded by the same C++ encoder as the Python package: singlet.write_s1() in Python writes the same bytes
# for the same matrix. read_s1() reads it back exactly; load("x.s1") routes such a file to read_s1().

.s1_counts_of <- function(x, assay) {
    if (methods::is(x, "SingleCellExperiment") || methods::is(x, "SummarizedExperiment")) {
        return(SummarizedExperiment::assay(x, assay))
    }
    if (methods::is(x, "Seurat")) {
        if (requireNamespace("SeuratObject", quietly = TRUE) && "LayerData" %in% getNamespaceExports("SeuratObject")) {
            m <- SeuratObject::LayerData(x, layer = assay)
        } else {
            m <- SeuratObject::GetAssayData(x, slot = assay)
        }
        return(m)
    }
    x
}

#' Write a count matrix as a .s1 sample object
#'
#' Stores one integer count matrix, its gene ids (row names) and cell barcodes (column names) as a \code{.s1} sample
#' object with a single matrix member. The bytes are identical to what \code{singlet.write_s1()} writes in Python for the
#' same matrix. Read it back with \code{\link{read_s1}} or \code{load("x.s1")}; stream minibatches from it in Python with
#' \code{singlet.torch.S1Cells}.
#'
#' @param x A \code{dgCMatrix} or base matrix (genes x cells, with dimnames), a \code{SingleCellExperiment} or
#'   \code{SummarizedExperiment} (assay \code{assay}), or a \code{Seurat} object (layer \code{assay} of the default assay).
#'   Values must be non-negative integers below 2^32.
#' @param path File to write. It is written under a temporary name and renamed.
#' @param sample_id ASCII id stored in the object header, at most 16 bytes.
#' @param assay Assay (SCE) or layer (Seurat) holding the counts.
#' @param member Name of the matrix member.
#' @param gene_pack \code{NULL} to store gene ids inline, or the 16-hex-character id of a vocabulary pack that holds them.
#' @param threads Encoder threads (the output does not depend on it).
#' @return \code{path}, invisibly.
#' @examples
#' m <- Matrix::rsparsematrix(100, 20, 0.1, rand.x = function(n) sample(1:5, n, TRUE))
#' dimnames(m) <- list(paste0("g", 1:100), paste0("c", 1:20))
#' f <- tempfile(fileext = ".s1")
#' write_s1(m, f)
#' identical(as.matrix(read_s1(f)), as.matrix(m))
#' @export
write_s1 <- function(x, path, sample_id = "sample", assay = "counts", member = "counts", gene_pack = NULL, threads = 4L) {
    m <- .s1_counts_of(x, assay)
    if (is.null(rownames(m)) || is.null(colnames(m))) {
        stop("write_s1: the matrix needs row names (gene ids) and column names (cell barcodes)")
    }
    m <- methods::as(methods::as(methods::as(m, "CsparseMatrix"), "generalMatrix"), "dMatrix")
    m <- Matrix::drop0(m)
    if (length(m@x) && (any(m@x < 0) || any(m@x != floor(m@x)))) {
        stop("write_s1: values must be non-negative integers (counts, not normalised data)")
    }
    pack <- if (is.null(gene_pack)) "" else tolower(as.character(gene_pack))
    raw <- s1_write_(m@p, m@i, m@x, nrow(m), ncol(m), as.character(rownames(m)), as.character(colnames(m)),
                     as.character(sample_id), as.character(member), pack, as.integer(threads))
    path <- path.expand(path)
    tmp <- paste0(path, ".part")
    writeBin(raw, tmp)
    if (!file.rename(tmp, path)) {
        unlink(tmp)
        stop("write_s1: could not write ", path)
    }
    invisible(path)
}

.s1_read_head <- function(path) {
    con <- file(path, "rb")
    on.exit(close(con))
    head <- readBin(con, "raw", n = 65536L)
    if (length(head) >= 32L) {
        table_end <- 64 + sum(as.numeric(head[25:32]) * 256^(0:7))
        if (table_end > length(head) && table_end <= file.size(path)) {
            head <- c(head, readBin(con, "raw", n = table_end - length(head)))
        }
    }
    head
}

#' The matrix member of a .s1 object that is not an atlas sample (none of the atlas members): "counts" if it has one, else
#' its only pz2 matrix member. NULL for an atlas sample, a file that is not a sample object, or several matrices and no
#' "counts". As user_member() in python/singlet/s1/write.py.
#' @keywords internal
#' @noRd
.s1_user_member <- function(path) {
    info <- tryCatch(s1_members_(.s1_read_head(path)), error = function(e) NULL)
    if (is.null(info) || any(c("exon_counts.1pz", "intron_counts.1pz", "cells") %in% info$name)) return(NULL)
    k <- match("counts", info$name)
    if (!is.na(k) && info$kind[[k]] == 1L) return("counts")
    mats <- info$name[info$kind == 1L]
    if (length(mats) == 1L) mats else NULL
}

#' Is a file a one-matrix .s1 object, such as one written by write_s1?
#' @keywords internal
#' @noRd
.s1_is_user_object <- function(path) !is.null(.s1_user_member(path))

#' Read a .s1 object written by write_s1
#'
#' @param path A \code{.s1} file written by \code{\link{write_s1}} (or by \code{singlet.write_s1()} in Python).
#' @param as \code{"matrix"} (a \code{dgCMatrix}, genes x cells), \code{"sce"} or \code{"seurat"}.
#' @param member Name of the matrix member.
#' @param genes Gene ids, for an object written with \code{gene_pack} (the pack is not read here).
#' @param threads Decoder threads.
#' @return The counts as \code{as} asks, with gene ids as row names and barcodes as column names.
#' @export
read_s1 <- function(path, as = c("matrix", "sce", "seurat"), member = "counts", genes = NULL, threads = 4L) {
    as <- match.arg(as)
    path <- path.expand(path)
    info <- s1_members_(.s1_read_head(path))
    k <- match(member, info$name)
    if (is.na(k)) stop(sprintf("read_s1: %s has no member '%s' (members: %s)", path, member, paste(info$name, collapse = ", ")))
    if (info$kind[[k]] != 1L) stop(sprintf("read_s1: member '%s' is not a matrix", member))
    con <- file(path, "rb")
    on.exit(close(con))
    seek(con, info$off[[k]])
    body <- readBin(con, "raw", n = info$len[[k]])
    h <- pz2_header_(body)
    d <- pz2_decode_(body, numeric(), list(), 0, h$n, "canonical", NULL, NULL, 0L, as.integer(threads), "auto")
    rn <- pz2_names_(body, TRUE)
    cn <- pz2_names_(body, FALSE)
    rows <- if (length(rn$names)) rn$names else genes
    if (is.null(rows)) stop(sprintf("read_s1: the gene ids are in vocabulary pack %s; pass genes =", rn$pack_id))
    if (length(rows) != h$m) stop(sprintf("read_s1: %d gene ids for %d genes", length(rows), h$m))
    m <- methods::new("dgCMatrix", i = d$i, p = d$p, x = d$x, Dim = c(as.integer(h$m), as.integer(h$n)),
                      Dimnames = list(as.character(rows), as.character(cn$names)))
    if (as == "matrix") return(m)
    .require_sce("read_s1(as = \"sce\")")
    sce <- SingleCellExperiment::SingleCellExperiment(assays = list(counts = m))
    S4Vectors::metadata(sce)$s1 <- list(sample_id = info$sample_id, member = member)
    if (as == "sce") return(sce)
    .require_seurat("read_s1(as = \"seurat\")")
    .sce_to_seurat(sce)
}
