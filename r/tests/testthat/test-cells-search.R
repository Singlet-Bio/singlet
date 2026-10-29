# SPDX-License-Identifier: MIT
# Atlas-wide cell search (R/cells_search.R): request building, answer parsing, paging and gather's grouping, all offline
# (the API calls are mocked).

answers <- function() {
    list(
        cells = list(totals = list(cells = 1631, samples = 1, studies = 1),
                     studies = list(list(gse_id = "GSE114724", cells = 1631, samples = 1)),
                     samples = list(list(gsm_id = "GSM3148582", gse_id = "GSE114724", cells = 1631)),
                     facets = list(sex = list()), unmapped = list(tissue = list(samples = 10)),
                     caveats = list("previews"), basis = list(index_version = "v1")),
        slice1 = list(cells = list(list("GSM1", list(0L, 2L)), list("GSM2", list(5L))),
                      objects = list(list(gsm_id = "GSM1", gse_id = "GSE10"), list(gsm_id = "GSM2", gse_id = "GSE20")),
                      progress = list(cells_total = 4), next_cursor = "c2"),
        slice2 = list(cells = list(list("GSM3", list(1L))), objects = list(list(gsm_id = "GSM3", gse_id = "GSE10")),
                      progress = list(cells_total = 4), next_cursor = NULL)
    )
}

test_that("where is sent as JSON and the answer becomes data frames", {
    seen <- list()
    local_mocked_bindings(.singlet_api_json = function(path, params = list(), what = "") {
        seen[[length(seen) + 1L]] <<- list(path = path, params = params)
        answers()$cells
    })
    hits <- singlet_search_cells(list(cell_type = "T cell", sex = "female"), limit = 5)
    expect_s3_class(hits, "singlet_cell_search")
    expect_equal(seen[[1]]$path, "cells")
    expect_equal(jsonlite::fromJSON(seen[[1]]$params$where), list(cell_type = "T cell", sex = "female"))
    expect_equal(seen[[1]]$params$limit, 5L)
    expect_null(seen[[1]]$params$include_low_qc)
    expect_equal(hits$totals$cells, 1631)
    expect_equal(hits$studies$gse_id, "GSE114724")
    expect_equal(hits$samples$cells, 1631)
    expect_equal(hits$unmapped$tissue$samples, 10)
    expect_output(print(hits), "1,631 cells in 1 samples of 1 studies")
    expect_error(singlet_search_cells(), "where")
})

test_that("an empty answer gives empty tables", {
    local_mocked_bindings(.singlet_api_json = function(...) list(totals = list(cells = 0, samples = 0, studies = 0),
                                                                  studies = list(), samples = list()))
    hits <- singlet_search_cells(question = "nothing")
    expect_equal(nrow(hits$studies), 0L)
    expect_named(hits$samples, c("gsm_id", "gse_id", "cells"))
})

test_that("the slice is paged until the cursor runs out or max_cells is reached", {
    calls <- 0L
    local_mocked_bindings(.singlet_api_json = function(path, params = list(), what = "") {
        calls <<- calls + 1L
        if (is.null(params$cursor)) answers()$slice1 else answers()$slice2
    })
    s <- singlet_cells_slice(list(cell_type = "T cell"))
    expect_equal(calls, 2L)
    expect_true(s$complete)
    expect_equal(s$total, 4)
    expect_equal(vapply(s$cells, function(x) x$gsm, ""), c("GSM1", "GSM2", "GSM3"))
    expect_equal(s$cells[[1]]$columns, c(0L, 2L))
    expect_equal(unname(s$gse[c("GSM1", "GSM2", "GSM3")]), c("GSE10", "GSE20", "GSE10"))

    calls <- 0L
    s <- singlet_cells_slice(list(cell_type = "T cell"), max_cells = 2L)
    expect_equal(calls, 1L)
    expect_false(s$complete)
})

test_that("gather groups cells by study and merges repeated samples", {
    got <- list()
    local_mocked_bindings(
        .singlet_cells_by_column = function(gse, cols_by_gsm, cache_dir = NULL, ...) {
            got[[gse]] <<- cols_by_gsm
            gse
        },
        .combine_sces = function(sces) unlist(sces)
    )
    out <- singlet_gather(list(list(gsm = "GSM1", columns = c(2L, 0L)), list("GSM3", 1L), list(gsm = "GSM1", columns = 2L),
                               list(gsm = "GSM2", columns = 5L)),
                          gse = c(GSM1 = "GSE10", GSM2 = "GSE20", GSM3 = "GSE10"))
    expect_equal(sort(out), c("GSE10", "GSE20"))
    expect_equal(got$GSE10, list(GSM1 = c(0L, 2L), GSM3 = 1L))
    expect_equal(got$GSE20, list(GSM2 = 5L))
    expect_error(singlet_gather(list()), "no cells")
    # a named list GSM = columns works too
    got <- list()
    singlet_gather(list(GSM2 = 5L), gse = c(GSM2 = "GSE20"))
    expect_equal(got$GSE20, list(GSM2 = 5L))
})

test_that("the cell search hits the live API", {
    skip_on_cran()
    skip_if_offline("singlet.bio")
    skip_if(nzchar(Sys.getenv("SINGLET_OFFLINE")))
    hits <- singlet_search_cells(list(cell_type = "T cell"), limit = 2)
    expect_gt(hits$totals$cells, 0)
    f <- singlet_cell_fields()
    expect_true(all(c("cell_type", "sex", "ancestry", "tissue", "disease") %in% f$name))
})
