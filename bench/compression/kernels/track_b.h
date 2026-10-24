// Track B codecs. Owned by the Track B agent; see PLAN.md section 4 and notes/track_b.md.
// Every codec satisfies decode(encode(M)) == M for all CanonicalMatrix fields.
#pragma once
#include "track_b_codecs.h"
#include "track_b_leb.h"

namespace cs {
inline void register_track_b(std::vector<Codec>& v) {
    tb::register_all(v);
    tb::register_leb(v);
}
}  // namespace cs
