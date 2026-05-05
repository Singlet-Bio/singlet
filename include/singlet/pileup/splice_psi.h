#pragma once
// singlet-pileup: splice_psi.h
// G-PSI: Per-cell splice junction PSI (Percent Spliced In).
//
// For each annotated splice junction, compute per-cell:
//   PSI = junction_count / sum(all same-site junctions)
// Junctions sharing a donor (5' splice site) or acceptor (3' splice site)
// form alternative splicing "events". Only events with ≥2 junctions are kept.
//
// Input : sj_csc (junctions × cells, uint16_t counts), sj_names (chrom:start-end)
// Output: PSI CSC (junctions × cells, float [0,1]) + SpliceEvent table

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

struct SpliceEvent {
    std::string event_id;                    ///< e.g. "chr1:1000-D" (donor) or "chr1:2000-A" (acceptor)
    std::vector<uint32_t> junction_indices;  ///< row indices into sj_csc
    std::string event_type;                  ///< "alt_donor" or "alt_acceptor"
};

/// PSI sparse matrix result (same row/col dimensions as input sj_csc, float values).
struct PSIResult {
    std::vector<int32_t> indptr;   ///< CSC column pointers (size ncols+1)
    std::vector<int32_t> indices;  ///< row indices (junction)
    std::vector<float> data;       ///< PSI values in [0, 1]
    uint32_t nrows;                ///< = n_junctions
    uint32_t ncols;                ///< = n_cells
    std::vector<SpliceEvent> events;
};

/// Parse junction name "chrom:start-end" → components. Returns false on failure.
inline bool parse_junction_name(const std::string& name,
                                std::string& chrom,
                                uint32_t& start,
                                uint32_t& end) {
    const auto colon = name.find(':');
    if (colon == std::string::npos) return false;
    const auto dash = name.find('-', colon + 1);
    if (dash == std::string::npos) return false;
    chrom = name.substr(0, colon);
    try {
        start = static_cast<uint32_t>(std::stoul(name.substr(colon + 1, dash - colon - 1)));
        end = static_cast<uint32_t>(std::stoul(name.substr(dash + 1)));
    } catch (...) {
        return false;
    }
    return true;
}

/// Group junctions into splice events by shared donor or acceptor site.
/// Only groups with ≥2 junctions are included.
inline std::vector<SpliceEvent> build_splice_events(
    const std::vector<std::string>& sj_names) {
    // donor_key (chrom:start-D) → junction indices
    std::unordered_map<std::string, std::vector<uint32_t>> donor_groups;
    // acceptor_key (chrom:end-A) → junction indices
    std::unordered_map<std::string, std::vector<uint32_t>> acceptor_groups;

    for (uint32_t i = 0; i < static_cast<uint32_t>(sj_names.size()); ++i) {
        std::string chrom;
        uint32_t start, end;
        if (!parse_junction_name(sj_names[i], chrom, start, end)) continue;
        donor_groups[chrom + ":" + std::to_string(start) + "-D"].push_back(i);
        acceptor_groups[chrom + ":" + std::to_string(end) + "-A"].push_back(i);
    }

    std::vector<SpliceEvent> events;

    for (auto& [key, jidxs] : donor_groups) {
        if (jidxs.size() >= 2) {
            SpliceEvent ev;
            ev.event_id = key;
            ev.junction_indices = std::move(jidxs);
            ev.event_type = "alt_donor";
            events.push_back(std::move(ev));
        }
    }
    for (auto& [key, jidxs] : acceptor_groups) {
        if (jidxs.size() >= 2) {
            SpliceEvent ev;
            ev.event_id = key;
            ev.junction_indices = std::move(jidxs);
            ev.event_type = "alt_acceptor";
            events.push_back(std::move(ev));
        }
    }

    // Sort deterministically by event_id
    std::sort(events.begin(), events.end(),
              [](const SpliceEvent& a, const SpliceEvent& b) {
                  return a.event_id < b.event_id;
              });
    return events;
}

/// Compute per-cell PSI matrix from sj_csc.
/// Each junction is assigned to the first event it belongs to (donor events
/// take priority over acceptor events due to sorting order).
/// CSCType must expose: indptr, indices, data (vectors), nrows, ncols.
template <typename CSCType>
PSIResult compute_psi(const CSCType& sj_csc,
                      const std::vector<std::string>& sj_names) {
    PSIResult result;
    result.nrows = sj_csc.nrows;
    result.ncols = sj_csc.ncols;
    result.indptr.assign(sj_csc.ncols + 1, 0);

    if (sj_names.empty() || sj_csc.data.empty() || sj_csc.ncols == 0)
        return result;

    result.events = build_splice_events(sj_names);
    if (result.events.empty())
        return result;

    const uint32_t n_events = static_cast<uint32_t>(result.events.size());

    // Map each junction to its first (priority) event index (-1 = no event).
    std::vector<int32_t> junction_event(sj_csc.nrows, -1);
    for (int32_t e = 0; e < static_cast<int32_t>(n_events); ++e) {
        for (uint32_t jidx : result.events[e].junction_indices) {
            if (jidx < static_cast<uint32_t>(junction_event.size()) &&
                junction_event[jidx] == -1) {
                junction_event[jidx] = e;
            }
        }
    }

    // Per-event running sum (reset after each cell).
    std::vector<float> event_sum(n_events, 0.0f);

    // Build PSI CSC column-by-column (same column order as sj_csc).
    result.indices.reserve(sj_csc.data.size());
    result.data.reserve(sj_csc.data.size());

    for (uint32_t c = 0; c < sj_csc.ncols; ++c) {
        const int32_t begin = sj_csc.indptr[c];
        const int32_t end_ = sj_csc.indptr[c + 1];

        // Accumulate per-event sums for this cell.
        std::vector<uint32_t> active_events;
        for (int32_t k = begin; k < end_; ++k) {
            const uint32_t j = static_cast<uint32_t>(sj_csc.indices[k]);
            if (j < static_cast<uint32_t>(junction_event.size()) &&
                junction_event[j] >= 0) {
                const uint32_t e = static_cast<uint32_t>(junction_event[j]);
                if (event_sum[e] == 0.0f)
                    active_events.push_back(e);
                event_sum[e] += static_cast<float>(sj_csc.data[k]);
            }
        }

        // Emit PSI entries for junctions that have a positive event sum.
        for (int32_t k = begin; k < end_; ++k) {
            const uint32_t j = static_cast<uint32_t>(sj_csc.indices[k]);
            if (j < static_cast<uint32_t>(junction_event.size()) &&
                junction_event[j] >= 0) {
                const uint32_t e = static_cast<uint32_t>(junction_event[j]);
                const float ev_sum = event_sum[e];
                if (ev_sum > 0.0f) {
                    result.indices.push_back(static_cast<int32_t>(j));
                    result.data.push_back(static_cast<float>(sj_csc.data[k]) / ev_sum);
                }
            }
        }

        // Reset active event sums.
        for (uint32_t e : active_events)
            event_sum[e] = 0.0f;

        result.indptr[c + 1] = static_cast<int32_t>(result.indices.size());
    }

    return result;
}

/// Write splice_events.tsv: event_id, event_type, n_junctions, junction_names (comma-sep).
inline bool write_splice_events(const std::string& path,
                                const std::vector<SpliceEvent>& events,
                                const std::vector<std::string>& sj_names) {
    std::ofstream f(path);
    if (!f) return false;

    f << "event_id\tevent_type\tn_junctions\tjunction_names\n";
    for (const auto& ev : events) {
        f << ev.event_id << "\t" << ev.event_type << "\t"
          << ev.junction_indices.size() << "\t";
        for (size_t i = 0; i < ev.junction_indices.size(); ++i) {
            if (i) f << ",";
            const uint32_t ji = ev.junction_indices[i];
            f << (ji < sj_names.size() ? sj_names[ji] : ("J" + std::to_string(ji)));
        }
        f << "\n";
    }
    return true;
}

}  // namespace singlet
