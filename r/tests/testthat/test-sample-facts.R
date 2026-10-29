# Structured sample facts from the study API as colData columns: flattening, reply parsing and when the API is called.
# No network: the fetch is either skipped (offline, local, loopback) or mocked.

k562 <- list(sample_kind = "cell_line", tissue_type = "cell line", cell_line = list(list(id = "CVCL_0004", name = "K-562")))
adj <- list(sample_kind = "primary", tissue_type = "tissue", tumor_sample_type = "adjacent normal", donor_status = "diseased",
            disease = list(list(id = "MONDO:0005061", label = "lung adenocarcinoma")))

test_that("facts flatten to the eight columns, NA where not stated", {
    f <- singlet:::.sample_facts_flatten(k562)
    expect_identical(names(f), singlet:::.sample_fact_columns)
    expect_identical(unname(f[c("tissue_type", "cell_line", "cell_line_cellosaurus_id")]), c("cell line", "K-562", "CVCL_0004"))
    expect_true(is.na(f[["disease"]]) && is.na(f[["tumor_sample_type"]]) && is.na(f[["donor_status"]]))
    g <- singlet:::.sample_facts_flatten(adj)
    expect_identical(g[["disease"]], "lung adenocarcinoma")
    expect_identical(g[["disease_ontology_term_id"]], "MONDO:0005061")
    expect_true(all(is.na(singlet:::.sample_facts_flatten(NULL))))
})

test_that("several lines join in order; healthy is PATO normal; a missing label falls back to the id", {
    f <- singlet:::.sample_facts_flatten(list(cell_line = list(list(id = "CVCL_9771", name = "WA01"),
                                                               list(id = "CVCL_B207", name = "HUES 8"))))
    expect_identical(f[["cell_line"]], "WA01; HUES 8")
    expect_identical(f[["cell_line_cellosaurus_id"]], "CVCL_9771; CVCL_B207")
    h <- singlet:::.sample_facts_flatten(list(donor_status = "healthy", disease = list(list(id = "PATO:0000461", label = "normal"))))
    expect_identical(unname(h[c("disease", "disease_ontology_term_id", "donor_status")]), c("normal", "PATO:0000461", "healthy"))
    n <- singlet:::.sample_facts_flatten(list(disease = list(list(id = "MONDO:0100096", label = NULL))))
    expect_identical(n[["disease"]], "MONDO:0100096")
})

test_that("the study reply parses to the samples that carry facts", {
    payload <- list(series = list(id = "GSE1"),
                    samples = list(list(gsm_id = "gsm1", facts = k562), list(gsm_id = "GSM2", facts = list()), list(gsm_id = "GSM3")))
    p <- singlet:::.sample_facts_parse(payload)
    expect_identical(names(p), "GSM1")
    expect_identical(singlet:::.sample_facts_parse(NULL), list())
})

test_that("no fetch offline, for a local directory or a loopback server", {
    st <- function(base) list(gse = "GSE1", transport = list(base = base))
    expect_identical(singlet:::.sample_facts_for_study(st("C:/data/singlet")), list())
    expect_identical(singlet:::.sample_facts_for_study(st("http://127.0.0.1:8000")), list())
    expect_identical(singlet:::.sample_facts_for_study(st("http://localhost:9")), list())
    withr::local_envvar(SINGLET_OFFLINE = "1")
    expect_identical(singlet:::.sample_facts_fetch("GSE999999999"), list())
})

test_that("an unreachable API gives no facts and no error", {
    skip_on_cran()
    withr::local_envvar(SINGLET_API_BASE = "http://127.0.0.1:9", SINGLET_OFFLINE = "")
    expect_identical(singlet:::.sample_facts_fetch("GSE999999998"), list())
})

test_that("fact columns become factors with NA for samples that state nothing", {
    cd <- singlet:::.s1_label_rbind(list(
        c(list(gsm_id = rep("A", 2)), singlet:::.sample_facts_cols(k562, 2)),
        c(list(gsm_id = rep("B", 1)), singlet:::.sample_facts_cols(adj, 1))
    ))
    cd <- singlet:::.sample_facts_factorize(cd)
    expect_true(is.factor(cd$tissue_type) && is.factor(cd$disease))
    expect_identical(as.character(cd$cell_line), c("K-562", "K-562", NA))
    expect_identical(as.character(cd$tumor_sample_type), c(NA, NA, "adjacent normal"))
    expect_false(is.factor(cd$gsm_id))
})

test_that("a cell-line sample's cells carry the line as cell type; other samples keep the labeller's call", {
    n <- 3L
    lab <- list(cell_type = c("transit amplifying cell", NA, "T cell"), cell_type_ontology_term_id = c("CL:0009010", NA, "CL:0000084"),
                cell_type_level = c(2L, 0L, 2L), cell_type_score = c(0.9, NA, 0.8), cell_type_deepest = c("transit amplifying cell", NA, "T cell"),
                cell_type_path = c("CL:a>CL:b", NA, "CL:c"))
    k562 <- list(tissue_type = "cell line", cell_line = list(list(id = "CVCL_0004", name = "K-562")))
    r <- singlet:::.sample_facts_relabel(lab, k562, n)
    expect_equal(r$cell_type, rep("K-562", n))
    expect_equal(r$cell_type_ontology_term_id, rep("CVCL_0004", n))
    expect_equal(r$cell_type_deepest, rep("K-562", n))
    expect_null(r$cell_type_deepest_ontology_term_id)
    expect_equal(r$cell_type_level, rep(0L, n))
    expect_true(all(is.na(r$cell_type_score)) && all(is.na(r$cell_type_path)))
    u <- singlet:::.sample_facts_relabel(lab, list(tissue_type = "cell line"), n)
    expect_equal(unique(u$cell_type), "cultured cell")
    expect_equal(unique(u$cell_type_ontology_term_id), "CL:0000010")
    expect_identical(singlet:::.sample_facts_relabel(lab, list(tissue_type = "tissue"), n), lab)
    expect_identical(singlet:::.sample_facts_relabel(lab, NULL, n), lab)
})
