# SPDX-License-Identifier: MIT
# Design catalog readers: the sidecar s1/d/<GSE>.design.json of a study (schema singlet-design-catalog-0.1,
# docs/CONTRASTS_CATALOG_SCHEMA.md). It exists before any counts do. Same transports, cache layout and ETag revalidation as
# the study manifest (singlet_study); mirrors python/singlet/s1/design.py.

.S1_DESIGN_TIERS <- c("confirmed", "candidate", "unresolved")
.S1_DESIGN_GRADES <- c("A", "B", "C", "D")  # best first

.S1_DESIGN_CAVEAT <- paste0(
    "tier 'candidate' means model-curated and unverified: the factors, levels and contrasts were read from the study's ",
    "records by a language model and no person has checked them against the paper; check each contrast's evidence before use")

.s1_design_caveat <- function(tier) {
    tier <- if (is.null(tier) || is.na(tier)) "" else tier
    if (identical(tier, "confirmed")) return("tier 'confirmed': checked by a person against the paper")
    if (identical(tier, "unresolved")) {
        return(paste0("tier 'unresolved': the design could not be settled from the study's records; treat every field as ",
                      "incomplete. ", .S1_DESIGN_CAVEAT))
    }
    .S1_DESIGN_CAVEAT
}

.s1_design_gse <- function(accession) {
    gse <- toupper(trimws(as.character(accession)[[1L]]))
    if (!grepl("^GSE[0-9]+$", gse)) stop(sprintf("not a GEO Series accession (GSE...): %s", accession), call. = FALSE)
    gse
}

.s1_design_check <- function(doc, gse) {
    schema <- if (is.list(doc)) doc$schema else NULL
    if (!is.character(schema) || !startsWith(schema, "singlet-design-catalog-")) {
        .s1_stop(sprintf("%s: not a singlet design catalog sidecar", gse))
    }
    major <- suppressWarnings(as.integer(strsplit(substring(schema, nchar("singlet-design-catalog-") + 1L), ".", fixed = TRUE)[[1L]][1L]))
    if (is.na(major)) .s1_stop(sprintf("%s: unreadable sidecar schema %s", gse, schema))
    if (major > 0L) {
        .s1_stop(sprintf("%s: sidecar schema %s is newer than this client supports; update the singlet package", gse, schema))
    }
    if (!is.null(doc$study) && !identical(doc$study, gse)) {
        .s1_stop(sprintf("%s: the sidecar at this key describes %s", gse, doc$study))
    }
    doc
}

# The parsed sidecar of `gse`, plus where it came from. Signals singlet_design_not_found (also singlet_s1_not_found).
.s1_design_doc <- function(gse, base = NULL, cache_dir = NULL) {
    tr <- .s1_transport(base)
    key <- sprintf("s1/d/%s.design.json", gse)
    not_found <- function(e) {
        .s1_stop(sprintf(paste0("%s has no design sidecar at %s: it is not in this catalog yet. Check the data host (base= or ",
                                "SINGLET_DATA_BASE; Beta is https://beta-data.singlet.bio) and the accession"),
                         gse, paste0(tr$base, "/", key)),
                 c("singlet_design_not_found", "singlet_s1_not_found"))
    }
    if (tr$kind == "dir") {
        r <- tryCatch(.s1_get(tr, key), singlet_s1_not_found = not_found)
        return(list(doc = .s1_design_check(.s1_json(r$body), gse), source = "local"))
    }
    root <- .s1_root(cache_dir)
    ddir <- file.path(root, "d")
    cpath <- file.path(ddir, paste0(gse, ".design.json"))
    epath <- file.path(ddir, paste0(gse, ".etag"))
    etag <- if (file.exists(cpath) && file.exists(epath)) trimws(readLines(epath, warn = FALSE)[1L]) else NULL
    r <- tryCatch(.s1_get(tr, key, if_none_match = etag),
                  singlet_s1_not_found = not_found,
                  singlet_s1_error = function(e) {
                      if (file.exists(cpath)) {
                          warning(sprintf("could not revalidate the design of %s (%s); using the cached sidecar", gse, conditionMessage(e)),
                                  call. = FALSE)
                          list(status = -1L)
                      } else stop(e)
                  })
    if (r$status == 304L || r$status == -1L) {
        doc <- .s1_json(paste(readLines(cpath, warn = FALSE, encoding = "UTF-8"), collapse = "\n"))
        return(list(doc = .s1_design_check(doc, gse), source = if (r$status == 304L) "cache-304" else "cache-offline"))
    }
    doc <- .s1_design_check(.s1_json(r$body), gse)
    if (!dir.exists(ddir)) dir.create(ddir, recursive = TRUE, showWarnings = FALSE)
    tmp <- paste0(cpath, ".", Sys.getpid(), ".part")
    writeBin(r$body, tmp)
    file.rename(tmp, cpath)
    if (!is.null(r$etag)) writeLines(r$etag, epath) else unlink(epath)
    list(doc = doc, source = "network")
}

.s1_chr <- function(x) if (is.null(x) || length(x) == 0L) NA_character_ else as.character(x[[1L]])
.s1_lgl <- function(x) if (is.null(x) || length(x) == 0L) NA else as.logical(x[[1L]])
.s1_num <- function(x) if (is.null(x) || length(x) == 0L) NA_real_ else as.numeric(x[[1L]])

.s1_design_frames <- function(doc) {
    tier <- .s1_chr(doc$tier)
    facs <- doc$factors
    fid <- vapply(facs, function(f) .s1_chr(f$id), "")
    factors <- data.frame(
        factor_id = fid,
        name = vapply(facs, function(f) .s1_chr(f$name), ""),
        type = vapply(facs, function(f) .s1_chr(f$type), ""),
        unit = vapply(facs, function(f) .s1_chr(f$unit), ""),
        source = vapply(facs, function(f) .s1_chr(f$source), ""),
        confidence = vapply(facs, function(f) .s1_chr(f$confidence), ""),
        evidence = vapply(facs, function(f) .s1_chr(f$evidence), ""),
        n_levels = vapply(facs, function(f) length(f$levels), 0L),
        stringsAsFactors = FALSE)
    lv <- do.call(c, lapply(facs, function(f) lapply(f$levels, function(l) c(list(factor_id = .s1_chr(f$id)), l))))
    levels <- data.frame(
        factor_id = vapply(lv, function(l) .s1_chr(l$factor_id), ""),
        level_id = vapply(lv, function(l) .s1_chr(l$id), ""),
        label = vapply(lv, function(l) .s1_chr(l$label), ""),
        standard = vapply(lv, function(l) .s1_chr(l$standard), ""),
        role = vapply(lv, function(l) .s1_chr(l$role), ""),
        stringsAsFactors = FALSE)
    levels$ontology <- lapply(lv, function(l) as.character(unlist(l$ontology)))
    us <- doc$units
    units <- data.frame(
        unit_id = vapply(us, function(u) .s1_chr(u$unit_id), ""),
        gsm = vapply(us, function(u) .s1_chr(u$gsm), ""),
        library = vapply(us, function(u) .s1_chr(u$library), ""),
        biosample_key = vapply(us, function(u) .s1_chr(u$biosample_key), ""),
        donor = vapply(us, function(u) .s1_chr(u$donor), ""),
        pool_type = vapply(us, function(u) .s1_chr(u$pool$type), ""),
        pool_n_members = vapply(us, function(u) .s1_num(u$pool$n_members), 0),
        pool_mapping = vapply(us, function(u) .s1_chr(u$pool$mapping), ""),
        status = vapply(us, function(u) .s1_chr(u$status), ""),
        in_campaign_sample_list = vapply(us, function(u) .s1_lgl(u$in_campaign_sample_list), NA),
        stringsAsFactors = FALSE)
    for (f in fid) units[[f]] <- vapply(us, function(u) .s1_chr(u$levels[[f]]), "")
    cov_names <- unique(unlist(lapply(us, function(u) names(u$covariates))))
    for (k in cov_names) {
        units[[paste0("cov_", k)]] <- vapply(us, function(u) .s1_chr(u$covariates[[k]]), "")
    }
    out <- list(factors = factors, levels = levels, units = units)
    for (n in names(out)) {
        attr(out[[n]], "tier") <- tier
        attr(out[[n]], "caveat") <- .s1_design_caveat(tier)
    }
    out
}

.s1_status_frame <- function(doc) {
    tier <- .s1_chr(doc$tier)
    us <- doc$units
    df <- data.frame(
        study = rep(.s1_chr(doc$study), length(us)),
        unit_id = vapply(us, function(u) .s1_chr(u$unit_id), ""),
        gsm = vapply(us, function(u) .s1_chr(u$gsm), ""),
        library = vapply(us, function(u) .s1_chr(u$library), ""),
        status = vapply(us, function(u) .s1_chr(u$status), ""),
        pool_type = vapply(us, function(u) .s1_chr(u$pool$type), ""),
        in_campaign_sample_list = vapply(us, function(u) .s1_lgl(u$in_campaign_sample_list), NA),
        tier = rep(tier, length(us)),
        stringsAsFactors = FALSE)
    attr(df, "tier") <- tier
    attr(df, "caveat") <- .s1_design_caveat(tier)
    df
}

#' Read the design of a study
#'
#' Reads the design catalog sidecar \code{s1/d/<GSE>.design.json} of a study: its factors, their levels, the units
#' (samples) with the level of each factor, the contrasts and where they came from. The sidecar exists before any counts
#' do, so this works for studies that are scheduled, held or still running. It is fetched like the study manifest
#' (conditional on the cached ETag).
#'
#' The \code{tier} says how far the design can be trusted. \code{"candidate"} means model-curated and unverified: a
#' language model read the study's records and no person has checked the design against the paper. \code{"confirmed"}
#' means a person did. \code{"unresolved"} means the design could not be settled. The tier is kept on every returned object
#' and printing repeats the caveat.
#'
#' @param accession A GEO Series accession (\code{"GSE..."}).
#' @param base Data host (default \code{SINGLET_DATA_BASE} or \code{https://data.singlet.bio}; use
#'   \code{https://beta-data.singlet.bio} for Beta; a directory with the server layout also works).
#' @param cache_dir Directory for the sidecar cache (default as for \code{\link{singlet_study}}).
#' @return A \code{singlet_design} list with \code{gse}, \code{tier}, \code{summary}, \code{design_class}, \code{flags},
#'   \code{factors}, \code{levels} and \code{units} (data frames; \code{units} has one column per factor id holding the
#'   level id of each unit), \code{contrasts} (a list of lists, as stored), \code{provenance}, \code{caveat},
#'   \code{source} and the full sidecar as \code{doc}. A study without a sidecar signals an error of class
#'   \code{singlet_design_not_found}.
#' @examples
#' \donttest{
#' if (curl::has_internet()) {
#'   d <- s1_design("GSE190760", base = "https://beta-data.singlet.bio",
#'                  cache_dir = file.path(tempdir(), "singlet-example"))
#'   d$factors
#' }
#' }
#' @seealso \code{\link{s1_contrasts}}, \code{\link{s1_sample_status}}, \code{\link{singlet_study}}
#' @export
s1_design <- function(accession, base = NULL, cache_dir = NULL) {
    gse <- .s1_design_gse(accession)
    r <- .s1_design_doc(gse, base, cache_dir)
    doc <- r$doc
    fr <- .s1_design_frames(doc)
    structure(list(gse = gse, tier = .s1_chr(doc$tier), summary = .s1_chr(doc$summary),
                   design_class = .s1_chr(doc$design_class), flags = doc$flags,
                   factors = fr$factors, levels = fr$levels, units = fr$units,
                   contrasts = doc$contrasts, provenance = doc$provenance,
                   caveat = .s1_design_caveat(doc$tier), source = r$source, doc = doc),
              class = "singlet_design")
}

#' @exportS3Method print singlet_design
print.singlet_design <- function(x, ...) {
    cat(sprintf("<singlet_design %s: tier %s, class %s, %d factors, %d units, %d contrasts>\n", x$gse, x$tier, x$design_class,
                nrow(x$factors), nrow(x$units), length(x$contrasts)))
    if (!is.na(x$summary)) cat("  ", x$summary, "\n", sep = "")
    cat("  note: ", x$caveat, "\n", sep = "")
    invisible(x)
}

#' Contrasts of the studies you name
#'
#' Filters the contrasts in the design sidecars of the studies you name (one request per study; the filtering is
#' client-side). Each contrast keeps the \code{tier} of its study. \code{"candidate"} means model-curated and unverified:
#' check the evidence of a contrast before relying on it.
#'
#' @param studies GEO Series accessions to read.
#' @param factor_type One or more factor types (\code{"time"}, \code{"disease"}, \code{"treatment"}, \code{"tissue_site"},
#'   ...); \code{NULL} keeps all.
#' @param min_grade \code{"A"} to \code{"D"}: keep contrasts at this grade or better (\code{"B"} keeps A and B).
#' @param tier One or more of \code{"confirmed"}, \code{"candidate"}, \code{"unresolved"}: keep studies of these tiers.
#'   The default keeps every tier.
#' @param base,cache_dir As for \code{\link{s1_design}}.
#' @param missing What to do about a study without a sidecar: \code{"warn"} (default), \code{"skip"} or \code{"error"}.
#' @return A \code{singlet_contrasts} list of the stored contrasts, each with \code{study} and \code{tier} added.
#'   \code{as.data.frame()} flattens the main fields. Printing repeats the tier caveat.
#' @examples
#' \donttest{
#' if (curl::has_internet()) {
#'   cs <- s1_contrasts(c("GSE190760", "GSE190487"), factor_type = "time", min_grade = "B",
#'                      base = "https://beta-data.singlet.bio", cache_dir = file.path(tempdir(), "singlet-example"))
#'   as.data.frame(cs)
#' }
#' }
#' @seealso \code{\link{s1_design}}
#' @export
s1_contrasts <- function(studies, factor_type = NULL, min_grade = NULL, tier = NULL, base = NULL, cache_dir = NULL,
                         missing = c("warn", "skip", "error")) {
    missing <- match.arg(missing)
    if (!is.null(min_grade)) {
        min_grade <- toupper(trimws(as.character(min_grade)[[1L]]))
        if (!(min_grade %in% .S1_DESIGN_GRADES)) {
            stop(sprintf("min_grade must be one of %s, got %s", paste(.S1_DESIGN_GRADES, collapse = ", "), min_grade), call. = FALSE)
        }
    }
    if (!is.null(tier)) {
        bad <- setdiff(tier, .S1_DESIGN_TIERS)
        if (length(bad)) stop(sprintf("unknown tier %s; tiers are %s", paste(bad, collapse = ", "), paste(.S1_DESIGN_TIERS, collapse = ", ")), call. = FALSE)
    }
    out <- list()
    for (s in as.character(studies)) {
        gse <- .s1_design_gse(s)
        doc <- tryCatch(.s1_design_doc(gse, base, cache_dir)$doc, singlet_design_not_found = function(e) {
            if (missing == "error") stop(e)
            if (missing == "warn") warning(conditionMessage(e), call. = FALSE)
            NULL
        })
        if (is.null(doc)) next
        st <- .s1_chr(doc$tier)
        if (!is.null(tier) && !(st %in% tier)) next
        for (cn in doc$contrasts) {
            if (!is.null(factor_type) && !(.s1_chr(cn$factor_type) %in% factor_type)) next
            if (!is.null(min_grade)) {
                g <- match(.s1_chr(cn$grade), .S1_DESIGN_GRADES)
                if (is.na(g) || g > match(min_grade, .S1_DESIGN_GRADES)) next
            }
            cn$study <- gse
            cn$tier <- st
            out[[length(out) + 1L]] <- cn
        }
    }
    structure(out, class = "singlet_contrasts")
}

#' @export
as.data.frame.singlet_contrasts <- function(x, row.names = NULL, optional = FALSE, ...) {
    f <- function(fn) vapply(unclass(x), fn, "")
    df <- data.frame(
        study = f(function(c) .s1_chr(c$study)), id = f(function(c) .s1_chr(c$id)),
        factor = f(function(c) .s1_chr(c$factor)), factor_type = f(function(c) .s1_chr(c$factor_type)),
        test = f(function(c) .s1_chr(c$test)), reference = f(function(c) .s1_chr(c$reference)),
        grade = f(function(c) .s1_chr(c$grade)), status = f(function(c) .s1_chr(c$status)),
        tier = f(function(c) .s1_chr(c$tier)), replication_unit = f(function(c) .s1_chr(c$replication_unit)),
        pairing = f(function(c) .s1_chr(c$pairing)),
        n_test_units = vapply(unclass(x), function(c) .s1_num(c$n$test_units), 0),
        n_reference_units = vapply(unclass(x), function(c) .s1_num(c$n$reference_units), 0),
        stringsAsFactors = FALSE)
    attr(df, "caveat") <- .s1_contrasts_caveat(x)
    df
}

.s1_contrasts_caveat <- function(x) {
    tiers <- unique(vapply(unclass(x), function(c) .s1_chr(c$tier), ""))
    if (length(tiers) == 0L || "candidate" %in% tiers) .S1_DESIGN_CAVEAT else .s1_design_caveat(tiers[[1L]])
}

#' @exportS3Method print singlet_contrasts
print.singlet_contrasts <- function(x, ...) {
    n <- length(x)
    cat(sprintf("<singlet_contrasts: %d contrast%s>\n", n, if (n == 1L) "" else "s"))
    if (n) print(utils::head(as.data.frame(x)[, c("study", "id", "factor_type", "test", "reference", "grade", "tier")], 20L), row.names = FALSE)
    if (n > 20L) cat(sprintf("  ... %d more\n", n - 20L))
    cat("  note: ", .s1_contrasts_caveat(x), "\n", sep = "")
    invisible(x)
}

#' Processing status of a study's samples
#'
#' The status of each unit of a study as written in its design sidecar: \code{scheduled}, \code{held},
#' \code{running}, \code{counts_available} or \code{needs_debug}. It is the status when the sidecar was written, so a study
#' that has finished since can read \code{scheduled}; \code{\link{singlet_study}} is the check for what is available.
#'
#' @inheritParams s1_design
#' @return A data frame with \code{study}, \code{unit_id}, \code{gsm}, \code{library}, \code{status}, \code{pool_type},
#'   \code{in_campaign_sample_list} and \code{tier}; the tier and its caveat are also attributes.
#' @examples
#' \donttest{
#' if (curl::has_internet()) {
#'   s1_sample_status("GSE190760", base = "https://beta-data.singlet.bio",
#'                    cache_dir = file.path(tempdir(), "singlet-example"))
#' }
#' }
#' @seealso \code{\link{s1_design}}
#' @export
s1_sample_status <- function(accession, base = NULL, cache_dir = NULL) {
    gse <- .s1_design_gse(accession)
    .s1_status_frame(.s1_design_doc(gse, base, cache_dir)$doc)
}
