// Codec registry. Each experiment track owns one header that exposes
//     void register_track_<x>(std::vector<cs::Codec>& v);
// and pushes its codecs there. Nobody edits another track's header; this file is fixed.
#pragma once
#include "common.h"
#include "ref_zstd_csc.h"
#include "track_a.h"
#include "track_b.h"
#include "track_c.h"
#include "track_d.h"

namespace cs {
inline std::vector<Codec> all_codecs() {
    std::vector<Codec> v;
    v.push_back(make_ref_identity_1pz());
    v.push_back(make_ref_zstd_csc(3));
    v.push_back(make_ref_zstd_csc(19));
    register_track_a(v);
    register_track_b(v);
    register_track_c(v);
    register_track_d(v);
    return v;
}
}  // namespace cs
