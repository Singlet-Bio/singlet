// nonhost_sylph.h — B-G7-2: Coverage-corrected sketch screener (Sylph-port API)
//
// Provides a thin wrapper over NonHostScreener with Sylph-style naming:
//   • NonhostHit { species_taxid, containment, abundance }
//   • NonhostSylphScreener::screen(read) → vector<NonhostHit>
//
// `containment` maps to hit_rate from NonHostScreener::classify_multi().
// `abundance`   is set to 0.0 at classify time; callers fill it from EM output.
//
// This is the Phase-1 gate: only reads with containment > threshold pass
// to downstream EM deconvolution.
//
// Namespace: singlet::nonhost
#pragma once

#include <string>
#include <string_view>
#include <vector>
#include <cstdint>
#include "min_sketch.h"
#include "nonhost_screener.h"

namespace singlet {
namespace nonhost {

// ── NonhostHit ───────────────────────────────────────────────────────────────
struct NonhostHit {
    uint32_t species_taxid;  // species index in the MinSketchIndex
    double   containment;    // fraction of read k-mers matching this species
    double   abundance;      // EM-estimated relative abundance (filled post-EM)
};

// ── NonhostSketchDB ──────────────────────────────────────────────────────────
// Wraps two MinSketchIndex objects (viral + microbial) as a single DB handle.
// Constructed from pre-loaded indices; owns no resources of its own.
struct NonhostSketchDB {
    const MinSketchIndex& viral;
    const MinSketchIndex& microbial;
    float                 min_hit_rate;  // containment threshold (e.g. 0.05)

    NonhostSketchDB(const MinSketchIndex& v,
                    const MinSketchIndex& m,
                    float threshold = 0.05f)
        : viral(v), microbial(m), min_hit_rate(threshold) {}
};

// ── NonhostSylphScreener ─────────────────────────────────────────────────────
// B-G7-2 Sylph-port screener.  Per-read multi-species classification.
//
// Usage:
//   NonhostSylphScreener screener(db);
//   auto hits = screener.screen(read_seq);  // one call per read
//
class NonhostSylphScreener {
public:
    explicit NonhostSylphScreener(const NonhostSketchDB& db)
        : screener_(db.viral, db.microbial, db.min_hit_rate, {})
        , min_hit_rate_(db.min_hit_rate) {}

    // Screen one read.  Returns all species with containment ≥ min_hit_rate.
    // abundance is 0.0 (placeholder; fill from em_deconvolve results).
    std::vector<NonhostHit> screen(std::string_view read) const {
        std::string seq(read);
        auto mh = screener_.classify_multi(seq);  // min_hit_rate_ is baked into screener_

        std::vector<NonhostHit> out;
        out.reserve(mh.viral_hits.size() + mh.microbial_hits.size());

        for (const auto& [sid, rate] : mh.viral_hits) {
            NonhostHit h;
            h.species_taxid = sid;
            h.containment   = static_cast<double>(rate);
            h.abundance     = 0.0;
            out.push_back(h);
        }
        for (const auto& [sid, rate] : mh.microbial_hits) {
            NonhostHit h;
            h.species_taxid = sid;
            h.containment   = static_cast<double>(rate);
            h.abundance     = 0.0;
            out.push_back(h);
        }
        return out;
    }

    // Batch screen.  Returns one vector<NonhostHit> per read.
    // Faster than calling screen() in a loop — uses classify_multi_batch.
    std::vector<std::vector<NonhostHit>> screen_batch(
            const std::vector<std::string>& reads,
            int n_threads = 1) const {
        auto multi_hits = screener_.classify_multi_batch(reads, 0.25f, n_threads);  // min_hit_rate_ baked in

        std::vector<std::vector<NonhostHit>> result;
        result.reserve(reads.size());

        for (const auto& mh : multi_hits) {
            std::vector<NonhostHit> hits;
            hits.reserve(mh.viral_hits.size() + mh.microbial_hits.size());
            for (const auto& [sid, rate] : mh.viral_hits) {
                NonhostHit h{ sid, static_cast<double>(rate), 0.0 };
                hits.push_back(h);
            }
            for (const auto& [sid, rate] : mh.microbial_hits) {
                NonhostHit h{ sid, static_cast<double>(rate), 0.0 };
                hits.push_back(h);
            }
            result.push_back(std::move(hits));
        }
        return result;
    }

private:
    NonHostScreener screener_;
    float           min_hit_rate_;
};

} // namespace nonhost
} // namespace singlet
