# Structured per-sample facts from the singlet.bio study API (GET /api/gse/<GSE>, field `facts` of each sample), as colData
# columns: the R counterpart of singlet.s1.sample_facts in Python. The facts come from the sample's own GEO record only; a field
# the record does not state is NA, never guessed.

# colData columns built from the facts, in order (ontology pairs follow the CELLxGENE naming)
.sample_fact_columns <- c("tissue_type", "cell_line", "cell_line_cellosaurus_id", "tumor_sample_type", "disease",
                          "disease_ontology_term_id", "donor_status", "sample_kind")

.sample_facts_memo <- new.env(parent = emptyenv())

.sample_facts_join <- function(items, key) {
    v <- vapply(items, function(x) {
        val <- if (is.list(x)) x[[key]] else NULL
        if (is.null(val) || length(val) == 0L || is.na(val[[1L]])) "" else as.character(val[[1L]])
    }, character(1))
    v <- v[nzchar(v)]
    if (length(v)) paste(v, collapse = "; ") else NA_character_
}

# One sample's `facts` (a list parsed from JSON) -> named character vector over .sample_fact_columns (NA when not stated).
.sample_facts_flatten <- function(f) {
    if (!is.list(f)) f <- list()
    out <- structure(rep(NA_character_, length(.sample_fact_columns)), names = .sample_fact_columns)
    for (k in c("tissue_type", "tumor_sample_type", "donor_status", "sample_kind")) {
        v <- f[[k]]
        if (!is.null(v) && length(v) && !is.na(v[[1L]]) && nzchar(as.character(v[[1L]]))) out[[k]] <- as.character(v[[1L]])
    }
    lines <- if (is.list(f$cell_line)) f$cell_line else list()
    name <- .sample_facts_join(lines, "name")
    out[["cell_line"]] <- if (is.na(name)) .sample_facts_join(lines, "id") else name
    out[["cell_line_cellosaurus_id"]] <- .sample_facts_join(lines, "id")
    dis <- if (is.list(f$disease)) f$disease else list()
    lab <- .sample_facts_join(dis, "label")
    out[["disease"]] <- if (is.na(lab)) .sample_facts_join(dis, "id") else lab
    out[["disease_ontology_term_id"]] <- .sample_facts_join(dis, "id")
    out
}

# /api/gse/<GSE> answer (parsed with simplifyVector = FALSE) -> named list GSM -> facts, for the samples that carry any.
.sample_facts_parse <- function(payload) {
    out <- list()
    if (!is.list(payload) || !is.list(payload$samples)) return(out)
    for (s in payload$samples) {
        if (!is.list(s) || is.null(s$gsm_id) || !is.list(s$facts) || length(s$facts) == 0L) next
        out[[toupper(as.character(s$gsm_id))]] <- s$facts
    }
    out
}

.sample_facts_offline <- function() {
    !tolower(trimws(Sys.getenv("SINGLET_OFFLINE", ""))) %in% c("", "0", "false", "no")
}

# Facts of a study from the live API, memoised per session; an empty list offline or when the API cannot be reached.
.sample_facts_fetch <- function(gse) {
    gse <- toupper(trimws(gse))
    if (exists(gse, envir = .sample_facts_memo, inherits = FALSE)) return(get(gse, envir = .sample_facts_memo))
    if (.sample_facts_offline()) return(list())
    facts <- tryCatch(.sample_facts_parse(.singlet_api_json(paste0("gse/", gse), what = "study facts")),
                      error = function(e) NULL)
    if (is.null(facts)) return(list())  # not memoised: a later call may reach the API
    assign(gse, facts, envir = .sample_facts_memo)
    facts
}

# Facts for a study read over HTTP(S) from a non-loopback host; a local directory or a local test server has no study API.
.sample_facts_for_study <- function(study) {
    base <- as.character(study$transport$base %||% "")
    if (!grepl("^https?://", base)) return(list())
    host <- tolower(sub("^https?://(\\[[^]]*\\]|[^/:]*).*$", "\\1", base))
    if (host %in% c("localhost", "[::1]") || startsWith(host, "127.")) return(list())
    .sample_facts_fetch(study$gse)
}

# The fact columns for n cells of one sample.
.sample_facts_cols <- function(f, n) {
    flat <- .sample_facts_flatten(f)
    structure(lapply(flat, function(v) rep(v, n)), names = names(flat))
}

# Fact columns of a colData data.frame as factors.
.sample_facts_factorize <- function(cd) {
    for (c in intersect(.sample_fact_columns, names(cd))) cd[[c]] <- factor(cd[[c]])
    cd
}
