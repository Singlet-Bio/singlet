# SPDX-License-Identifier: MIT
test_that("read_cohort loads multiple sample directories", {
    dir <- skip_if_no_fixture()
    if (!file.exists(file.path(dir, "spliced.1pz"))) {
        skip("fixture lacks spliced.1pz")
    }

    # Build a synthetic 3-sample cohort by symlinking the same fixture
    # three times into a temp dir.
    tmp_root <- tempfile("cohort_test_")
    dir.create(tmp_root)
    on.exit(unlink(tmp_root, recursive = TRUE))

    sample_dirs <- character(3)
    for (i in 1:3) {
        sd <- file.path(tmp_root, sprintf("sample_%d", i))
        dir.create(sd)
        for (f in list.files(dir, full.names = TRUE)) {
            file.symlink(f, file.path(sd, basename(f)))
        }
        sample_dirs[i] <- sd
    }

    cohort <- read_cohort(sample_dirs, matrix_name = "spliced")
    expect_length(cohort, 3L)
    # All matrices should have the same shape (same source)
    shapes <- unique(lapply(cohort, dim))
    expect_length(shapes, 1L)
    # Each entry should be a dgCMatrix
    for (nm in names(cohort)) {
        expect_s4_class(cohort[[nm]], "dgCMatrix")
    }
})

test_that("read_cohort skips directories that lack the requested matrix", {
    tmp_root <- tempfile("cohort_empty_")
    dir.create(tmp_root)
    on.exit(unlink(tmp_root, recursive = TRUE))
    empty1 <- file.path(tmp_root, "empty1")
    empty2 <- file.path(tmp_root, "empty2")
    dir.create(empty1)
    dir.create(empty2)

    cohort <- read_cohort(c(empty1, empty2), matrix_name = "spliced")
    expect_length(cohort, 0L)
})
