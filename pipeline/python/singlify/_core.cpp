// singlify Python bindings via pybind11
//
// Exposes the singlet-pileup engine as a Python module returning scipy CSC matrices.
// The C++ engine runs entirely in C++; Python only handles config and result I/O.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "singlet-pileup/pileup_engine.h"
#include "singlet-pileup/sparse_accumulator.h"
#include "singlet-pileup/mt_heteroplasmy.h"
#include "singlet-pileup/donor_demux.h"

namespace py = pybind11;

// Convert a CSC matrix to a dict with numpy arrays (for scipy.sparse.csc_matrix)
template <typename T>
static py::dict csc_to_dict(const typename singlet::SparseAccumulator<T>::CSCMatrix& csc,
                            const std::vector<std::string>& features) {
    py::dict d;
    d["shape"] = py::make_tuple(csc.nrows, csc.ncols);
    d["indptr"] = py::array_t<int32_t>(csc.indptr.size(), csc.indptr.data());
    d["indices"] = py::array_t<int32_t>(csc.indices.size(), csc.indices.data());
    d["data"] = py::array_t<T>(csc.data.size(), csc.data.data());
    d["features"] = features;
    return d;
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "singlify: streaming BAM pileup engine for single-cell multi-omics";

    // ── PileupConfig ──
    py::class_<singlet::PileupConfig>(m, "PileupConfig")
        .def(py::init<>())
        .def_readwrite("snp_path", &singlet::PileupConfig::snp_path)
        .def_readwrite("exon_gtf_path", &singlet::PileupConfig::exon_gtf_path)
        .def_readwrite("barcode_path", &singlet::PileupConfig::barcode_path)
        .def_readwrite("out_chrm_bam", &singlet::PileupConfig::out_chrm_bam)
        .def_readwrite("threads", &singlet::PileupConfig::threads)
        .def_readwrite("min_mapq", &singlet::PileupConfig::min_mapq)
        .def_readwrite("min_baseq", &singlet::PileupConfig::min_baseq)
        .def_readwrite("count_introns", &singlet::PileupConfig::count_introns)
        .def_readwrite("count_sj", &singlet::PileupConfig::count_sj)
        .def_readwrite("umi_dedup", &singlet::PileupConfig::umi_dedup)
        .def_readwrite("stranded", &singlet::PileupConfig::stranded)
        .def_readwrite("count_mt", &singlet::PileupConfig::count_mt);

    // ── PileupStats ──
    py::class_<singlet::PileupStats>(m, "PileupStats")
        .def(py::init<>())
        .def_readonly("total_reads", &singlet::PileupStats::total_reads)
        .def_readonly("mapped_reads", &singlet::PileupStats::mapped_reads)
        .def_readonly("barcoded_reads", &singlet::PileupStats::barcoded_reads)
        .def_readonly("snp_hits", &singlet::PileupStats::snp_hits)
        .def_readonly("exon_hits", &singlet::PileupStats::exon_hits)
        .def_readonly("intron_hits", &singlet::PileupStats::intron_hits)
        .def_readonly("sj_hits", &singlet::PileupStats::sj_hits)
        .def_readonly("chrm_reads", &singlet::PileupStats::chrm_reads)
        .def_readonly("mt_pileup_bases", &singlet::PileupStats::mt_pileup_bases)
        .def_readonly("low_mapq", &singlet::PileupStats::low_mapq)
        .def_readonly("secondary_reads", &singlet::PileupStats::secondary_reads)
        .def_readonly("no_barcode", &singlet::PileupStats::no_barcode)
        .def_readonly("no_umi", &singlet::PileupStats::no_umi)
        .def_readonly("umi_unique", &singlet::PileupStats::umi_unique)
        .def_readonly("umi_duplicate", &singlet::PileupStats::umi_duplicate)
        .def_readonly("multimapper_reads", &singlet::PileupStats::multimapper_reads)
        .def_readonly("multigene_reads", &singlet::PileupStats::multigene_reads)
        .def_readonly("wrong_strand", &singlet::PileupStats::wrong_strand)
        .def_readonly("wall_time_s", &singlet::PileupStats::wall_time_s);

    // ── Main pileup function ──
    m.def("pileup", [](
        const std::string& bam_path,
        const std::string& barcode_path,
        const std::string& exon_gtf_path,
        const std::string& snp_path,
        int threads,
        int min_mapq,
        int min_baseq,
        bool count_introns,
        bool count_sj,
        bool umi_dedup,
        bool count_mt
    ) -> py::dict {
        // Configure engine
        singlet::PileupConfig config;
        config.barcode_path = barcode_path;
        config.exon_gtf_path = exon_gtf_path;
        config.snp_path = snp_path;
        config.threads = threads;
        config.min_mapq = min_mapq;
        config.min_baseq = min_baseq;
        config.count_introns = count_introns;
        config.count_sj = count_sj;
        config.umi_dedup = umi_dedup;
        config.count_mt = count_mt;

        // Create and run engine
        singlet::PileupEngine engine(config);
        if (!engine.load_references()) {
            throw std::runtime_error("Failed to load pileup references");
        }

        singlet::PileupStats stats;
        {
            py::gil_scoped_release release;
            stats = engine.run(bam_path);
        }

        if (stats.total_reads == 0) {
            throw std::runtime_error("No reads processed from BAM");
        }

        // Build result dict
        py::dict result;
        result["stats"] = stats;
        result["barcodes"] = engine.barcodes();

        // Convert accumulators to CSC arrays
        if (!config.snp_path.empty()) {
            auto snp_ad_csc = engine.snp_ad().to_csc();
            auto snp_dp_csc = engine.snp_dp().to_csc();
            result["snp_ad"] = csc_to_dict<uint8_t>(snp_ad_csc, engine.snp_names());
            result["snp_dp"] = csc_to_dict<uint8_t>(snp_dp_csc, engine.snp_names());
        }
        if (!config.exon_gtf_path.empty()) {
            auto exon_csc = engine.exons().to_csc();
            result["exons"] = csc_to_dict<uint16_t>(exon_csc, engine.exon_names());
            if (config.count_introns && engine.gene_model().n_introns() > 0) {
                auto intron_csc = engine.introns().to_csc();
                result["introns"] = csc_to_dict<uint16_t>(intron_csc, engine.intron_names());
            }
            if (config.count_sj && !engine.sj_names().empty()) {
                auto sj_csc = engine.splice_junctions().to_csc();
                result["splice_junctions"] = csc_to_dict<uint16_t>(sj_csc, engine.sj_names());
            }
        }
        if (config.count_mt && engine.mt_alleles().nnz_raw() > 0) {
            auto mt_csc = engine.mt_alleles().to_csc();
            result["mt_alleles"] = csc_to_dict<uint16_t>(mt_csc, singlet::mt::mt_feature_names());
        }

        return result;
    },
    py::arg("bam_path"),
    py::arg("barcode_path"),
    py::arg("exon_gtf_path") = "",
    py::arg("snp_path") = "",
    py::arg("threads") = 4,
    py::arg("min_mapq") = 20,
    py::arg("min_baseq") = 10,
    py::arg("count_introns") = true,
    py::arg("count_sj") = true,
    py::arg("umi_dedup") = true,
    py::arg("count_mt") = false,
    R"doc(
    Run streaming BAM pileup for single-cell multi-omics feature extraction.

    Processes a BAM file in a single pass, extracting per-barcode counts for
    SNP genotypes, exon/intron expression, splice junctions, and chrM alleles.

    Parameters
    ----------
    bam_path : str
        Path to input BAM file (sorted or unsorted).
    barcode_path : str
        Path to filtered barcode list (one per line).
    exon_gtf_path : str, optional
        Path to GTF file for gene model (exon/intron counting).
    snp_path : str, optional
        Path to VCF file with SNP positions.
    threads : int, default 4
        BAM decompression threads.
    min_mapq : int, default 20
        Minimum mapping quality.
    min_baseq : int, default 10
        Minimum base quality for SNP calls.
    count_introns : bool, default True
        Count intronic reads.
    count_sj : bool, default True
        Extract splice junctions.
    umi_dedup : bool, default True
        Deduplicate by UMI.
    count_mt : bool, default False
        Enable chrM allele pileup.

    Returns
    -------
    dict
        Dictionary with keys:
        - ``stats``: PileupStats object with read counts
        - ``barcodes``: list of barcode strings
        - ``exons``: dict with CSC arrays (if exon_gtf_path given)
        - ``introns``: dict with CSC arrays (if count_introns)
        - ``splice_junctions``: dict with CSC arrays (if count_sj)
        - ``snp_ad``: dict with CSC arrays for allele depth (if snp_path given)
        - ``snp_dp``: dict with CSC arrays for total depth (if snp_path given)
        - ``mt_alleles``: dict with CSC arrays (if count_mt)

        Each CSC dict contains: ``shape``, ``indptr``, ``indices``, ``data``, ``features``.
    )doc");
}
