// Track C codecs. Owned by the Track C agent; see PLAN.md section 4 and notes/track_c.md.
// Every codec satisfies decode(encode(M)) == M for all CanonicalMatrix fields.
#pragma once
#include "common.h"
#include "track_c_codec.h"

namespace cs {
inline void register_track_c(std::vector<Codec>& v) {
    using tc::Params;
    auto add = [&](std::string name, bool file_classes, bool adapt, bool use_model, bool insample, int kappa) {
        Params p; p.name = name; p.file_classes = file_classes; p.adapt = adapt; p.use_model = use_model; p.insample = insample; p.kappa_e = kappa;
        v.push_back(tc::make_tc_codec(p));
    };
    // rung 1: trained profile classes + trained (bucket,class) presence table, static (out of sample)
    add("c_r1", false, false, true, false, 1);
    // rung 1b: semi-static: per-feature presence-count class sent as side info (coded with trained CPT), trained table
    add("c_r1b", true, false, true, false, 1);
    // rung 2: as 1b + online adaptation of the (bucket,class) table (exact integer counts mirrored in the decoder)
    add("c_r2", true, true, true, false, 1);
    // rung 2 on profile classes (no side info)
    add("c_r2p", false, true, true, false, 1);
    // model-free variant (side info coded adaptively, flat prior): shows what the trained model buys
    add("c_r2_nomodel", true, true, false, false, 1);
    // upper bounds: model trained including the file's own study
    add("c_r1_insample", false, false, true, true, 1);
    add("c_r2_insample", true, true, true, true, 1);
    // rung 3: as c_r2 + head features (2048 most frequent in training) coded per feature under an adaptive
    // (cluster, depth tier) table initialised from out-of-sample k-means cell-type profiles; cluster id per column is side info
    {
        Params p; p.name = "c_r3"; p.latent = true; v.push_back(tc::make_tc_codec(p));
        Params q; q.name = "c_r3_insample"; q.latent = true; q.insample = true; v.push_back(tc::make_tc_codec(q));
    }
}
}  // namespace cs
