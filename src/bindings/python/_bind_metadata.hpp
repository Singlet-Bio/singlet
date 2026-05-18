// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_metadata.hpp
//
// pybind11 bindings for singlet::gpu::core::Metadata.
//
// Exposes every GEO field, provenance string, and the rownames / colnames
// vectors as read-only Python properties.  Mutation is intentionally not
// supported — Metadata is produced by the loader and is authoritative.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>       // std::vector<std::string> ↔ Python list
#include <pybind11/chrono.h>

#include <singlet/gpu/core/memory.h>  // Metadata definition

namespace py = pybind11;

// ---------------------------------------------------------------------------
// bind_metadata
//
// Call from PYBIND11_MODULE to register the Metadata class.
// Example:
//   bind_metadata(m);
// ---------------------------------------------------------------------------
inline void bind_metadata(py::module_& m) {
    py::class_<singlet::gpu::core::Metadata>(m, "Metadata",
        R"doc(
        GEO metadata and provenance embedded in a .1pz file.

        All string fields are UTF-8.  Fields absent from the .1pz TLV
        block are empty strings or zero-valued integers.

        Attributes
        ----------
        gsm_id : str
            GEO sample accession (e.g. "GSM4037629").
        gse_id : str
            GEO series accession (e.g. "GSE127918").
        organism : str
            Scientific name (e.g. "Homo sapiens").
        taxon_id : int
            NCBI taxonomy ID (e.g. 9606 for human).
        protocol : str
            Library preparation protocol (e.g. "10xv3").
        modality : str
            Assay modality (e.g. "scrna", "cite", "multiome").
        srr_ids : list[str]
            SRA run accessions.
        read_count : int
            Approximate read count from the catalog.
        geo_title : str
            GEO sample title.
        geo_source_name : str
            GEO source name / tissue annotation.
        singlet_version : str
            singlet version that produced this file.
        pipeline_date : str
            ISO-8601 date of pipeline run.
        rownames : list[str]
            Feature names (genes), one per row.  Empty if not embedded.
        colnames : list[str]
            Cell barcodes, one per column.  Empty if not embedded.
        )doc"
    )
    // GEO identifiers
    .def_readonly("gsm_id",           &singlet::gpu::core::Metadata::gsm_id)
    .def_readonly("gse_id",           &singlet::gpu::core::Metadata::gse_id)

    // Organism / taxonomy
    .def_readonly("organism",         &singlet::gpu::core::Metadata::organism)
    .def_readonly("taxon_id",         &singlet::gpu::core::Metadata::taxon_id)

    // Assay
    .def_readonly("protocol",         &singlet::gpu::core::Metadata::protocol)
    .def_readonly("modality",         &singlet::gpu::core::Metadata::modality)
    .def_readonly("srr_ids",          &singlet::gpu::core::Metadata::srr_ids)
    .def_readonly("read_count",       &singlet::gpu::core::Metadata::read_count)

    // GEO text
    .def_readonly("geo_title",        &singlet::gpu::core::Metadata::geo_title)
    .def_readonly("geo_source_name",  &singlet::gpu::core::Metadata::geo_source_name)

    // Provenance
    .def_readonly("singlet_version", &singlet::gpu::core::Metadata::singlet_version)
    .def_readonly("pipeline_date",    &singlet::gpu::core::Metadata::pipeline_date)

    // Names
    .def_readonly("rownames",         &singlet::gpu::core::Metadata::rownames)
    .def_readonly("colnames",         &singlet::gpu::core::Metadata::colnames)

    // Convenience: serialize to a plain Python dict (used by io.py to populate
    // adata.uns['singlet']).
    .def("to_dict", [](const singlet::gpu::core::Metadata& m) -> py::dict {
        py::dict d;
        d["gsm_id"]           = m.gsm_id;
        d["gse_id"]           = m.gse_id;
        d["organism"]         = m.organism;
        d["taxon_id"]         = m.taxon_id;
        d["protocol"]         = m.protocol;
        d["modality"]         = m.modality;
        d["srr_ids"]          = m.srr_ids;
        d["read_count"]       = m.read_count;
        d["geo_title"]        = m.geo_title;
        d["geo_source_name"]  = m.geo_source_name;
        d["singlet_version"] = m.singlet_version;
        d["pipeline_date"]    = m.pipeline_date;
        return d;
    }, "Return a plain dict of all GEO + provenance fields (no rownames/colnames).")

    .def("__repr__", [](const singlet::gpu::core::Metadata& m) {
        return "<Metadata gsm_id='" + m.gsm_id + "' gse_id='" + m.gse_id +
               "' protocol='" + m.protocol + "' organism='" + m.organism + "'>";
    });
}
