#pragma once
// Parallel CB:Z tag count over an indexed, coordinate-sorted BAM (the "CB-tag barcode refresh" of singlet.cpp).
//
// The serial scan reads the whole BAM in file order and does `cb_freq[cb]++` for every primary record whose CB tag is
// set and not STAR's "-" sentinel. Its result is later sorted by count, and equal counts keep the unordered_map's
// iteration order, which depends on the order in which keys were first inserted. To give the very same map, the
// parallel scan splits the file into contiguous runs of references (in file order, then the unplaced reads), records
// for each key its count and the position of its first occurrence in that run, and then inserts the keys into the
// caller's map in the order of their first occurrence in the file, with the summed counts. Same keys, same counts,
// same insertion sequence: the same map, bucket for bucket.
#include <htslib/hts.h>
#include <htslib/sam.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace singlet {

// Fills `cb_freq` (expected empty, already reserve()d by the caller exactly as for the serial scan). Returns false
// (and leaves cb_freq empty) when the BAM has no index or cannot be read, so the caller can fall back to the serial
// scan.
inline bool cb_tag_counts_parallel(const std::string& bam_path, int n_threads,
                                   std::unordered_map<std::string, uint32_t>& cb_freq) {
    if (n_threads < 2) return false;
    samFile* fp = sam_open(bam_path.c_str(), "rb");
    if (!fp) return false;
    bam_hdr_t* hdr = sam_hdr_read(fp);
    hts_idx_t* idx = hdr ? sam_index_load(fp, bam_path.c_str()) : nullptr;
    if (!idx) {
        if (hdr) bam_hdr_destroy(hdr);
        sam_close(fp);
        return false;
    }
    const int n_ref = hdr->n_targets;
    // Work per reference (mapped + unmapped records) to balance contiguous runs.
    std::vector<uint64_t> w(n_ref + 1, 0);
    uint64_t total = 0;
    for (int t = 0; t < n_ref; ++t) {
        uint64_t m = 0, u = 0;
        hts_idx_get_stat(idx, t, &m, &u);
        w[t] = m + u;
        total += w[t];
    }
    w[n_ref] = hts_idx_get_n_no_coor(idx);
    total += w[n_ref];
    hts_idx_destroy(idx);
    bam_hdr_destroy(hdr);
    sam_close(fp);

    // Contiguous runs [b, e) over "units" 0..n_ref (n_ref = the unplaced reads), about total/n_runs records each.
    const int n_runs = std::max(1, std::min(n_threads * 4, n_ref + 1));
    std::vector<std::pair<int, int>> runs;
    {
        int b = 0;
        uint64_t acc = 0;
        for (int u = 0; u <= n_ref; ++u) {
            acc += w[u];
            const uint64_t target = (total * (runs.size() + 1)) / n_runs;
            if (acc >= target || u == n_ref) {
                runs.push_back({b, u + 1});
                b = u + 1;
            }
        }
    }

    struct KeyStat {
        uint32_t count = 0;
        uint64_t first = 0;  // index of the first occurrence within the run
    };
    std::vector<std::unordered_map<std::string, KeyStat>> per_run(runs.size());
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    auto worker = [&]() {
        samFile* f = sam_open(bam_path.c_str(), "rb");
        bam_hdr_t* h = f ? sam_hdr_read(f) : nullptr;
        hts_idx_t* ix = h ? sam_index_load(f, bam_path.c_str()) : nullptr;
        if (!ix) {
            failed = true;
            if (h) bam_hdr_destroy(h);
            if (f) sam_close(f);
            return;
        }
        bam1_t* rec = bam_init1();
        for (size_t r; (r = next.fetch_add(1)) < runs.size();) {
            auto& m = per_run[r];
            uint64_t seq = 0;
            for (int u = runs[r].first; u < runs[r].second; ++u) {
                hts_itr_t* itr = (u < n_ref) ? sam_itr_queryi(ix, u, 0, HTS_POS_MAX)
                                             : sam_itr_queryi(ix, HTS_IDX_NOCOOR, 0, 0);
                if (!itr) continue;  // a reference without reads
                int ret;
                while ((ret = sam_itr_next(f, itr, rec)) >= 0) {
                    if (rec->core.flag & (BAM_FSECONDARY | BAM_FSUPPLEMENTARY)) continue;
                    uint8_t* cb_aux = bam_aux_get(rec, "CB");
                    if (cb_aux && *cb_aux == 'Z') {
                        const char* cb_str = bam_aux2Z(cb_aux);
                        if (cb_str && cb_str[0] && cb_str[0] != '-') {
                            auto it = m.try_emplace(cb_str).first;
                            if (it->second.count == 0) it->second.first = seq;
                            it->second.count++;
                            ++seq;
                        }
                    }
                }
                if (ret < -1) failed = true;  // truncated / corrupt
                hts_itr_destroy(itr);
            }
        }
        bam_destroy1(rec);
        hts_idx_destroy(ix);
        bam_hdr_destroy(h);
        sam_close(f);
    };
    std::vector<std::thread> pool;
    const int nt = std::min<int>(n_threads, static_cast<int>(runs.size()));
    for (int i = 0; i < nt; ++i) pool.emplace_back(worker);
    for (auto& t : pool) t.join();
    if (failed) return false;

    // Merge: total counts, and each key's first occurrence = (first run containing it, position in that run).
    std::unordered_map<std::string, std::tuple<uint32_t, uint32_t, uint64_t>> merged;  // count, run, first
    for (uint32_t r = 0; r < per_run.size(); ++r) {
        for (auto& kv : per_run[r]) {
            auto ins = merged.try_emplace(kv.first, 0u, r, kv.second.first);
            std::get<0>(ins.first->second) += kv.second.count;  // runs are visited in order: first run wins
        }
        per_run[r].clear();
    }
    std::vector<std::tuple<uint32_t, uint64_t, const std::string*, uint32_t>> order;
    order.reserve(merged.size());
    for (auto& kv : merged)
        order.emplace_back(std::get<1>(kv.second), std::get<2>(kv.second), &kv.first, std::get<0>(kv.second));
    std::sort(order.begin(), order.end(), [](const auto& a, const auto& b) {
        return std::get<0>(a) != std::get<0>(b) ? std::get<0>(a) < std::get<0>(b) : std::get<1>(a) < std::get<1>(b);
    });
    for (const auto& o : order) cb_freq[*std::get<2>(o)] = std::get<3>(o);
    return true;
}

}  // namespace singlet
