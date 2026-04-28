#pragma once
// singlet-pileup: interval_tree.h
// Static augmented interval tree for fast overlap queries.
// Used for exon-level counting: lookup which exon(s) a read position overlaps.
//
// Design:
// - Build once from sorted intervals (exons from GTF)
// - O(log n + k) query time for k overlapping intervals
// - Cache-friendly flat array layout (implicit tree, no pointers)

#include <algorithm>
#include <cstdint>
#include <vector>

namespace singlet {

struct Interval {
    int32_t start;   // 0-based, inclusive
    int32_t end;     // 0-based, exclusive
    uint32_t index;  // exon index (row in output matrix)
};

class IntervalTree {
public:
    IntervalTree() = default;

    // Build from a vector of intervals. Must be called before query().
    // Intervals are sorted by start position internally.
    void build(std::vector<Interval> intervals) {
        intervals_ = std::move(intervals);
        std::sort(intervals_.begin(), intervals_.end(),
            [](const Interval& a, const Interval& b) {
                return a.start < b.start || (a.start == b.start && a.end < b.end);
            });
        n_ = intervals_.size();
        if (n_ == 0) return;

        // Build augmented max_end array for implicit balanced BST
        max_end_.resize(n_);
        build_augmented(0, 0, n_);
    }

    // Find all intervals overlapping [qstart, qend).
    // Returns indices into the original interval array.
    void query(int32_t qstart, int32_t qend, std::vector<uint32_t>& results) const {
        results.clear();
        if (n_ == 0) return;
        query_impl(0, 0, n_, qstart, qend, results);
    }

    size_t size() const { return n_; }

private:
    std::vector<Interval> intervals_;
    std::vector<int32_t> max_end_;  // augmented max end for subtree
    size_t n_ = 0;

    // Build augmented max_end for implicit BST rooted at midpoint
    int32_t build_augmented(size_t node, size_t lo, size_t hi) {
        if (lo >= hi) return 0;
        size_t mid = lo + (hi - lo) / 2;
        int32_t left_max = build_augmented(mid, lo, mid);
        int32_t right_max = build_augmented(mid, mid + 1, hi);
        max_end_[mid] = std::max({intervals_[mid].end, left_max, right_max});
        return max_end_[mid];
    }

    void query_impl(size_t node, size_t lo, size_t hi,
                    int32_t qstart, int32_t qend,
                    std::vector<uint32_t>& results) const {
        if (lo >= hi) return;
        size_t mid = lo + (hi - lo) / 2;

        // Check if this interval overlaps query
        const auto& iv = intervals_[mid];
        if (iv.start < qend && iv.end > qstart) {
            results.push_back(iv.index);
        }

        // Search left subtree if it could contain overlaps
        if (lo < mid && max_end_[lo + (mid - lo) / 2] > qstart) {
            query_impl(mid, lo, mid, qstart, qend, results);
        }

        // Search right subtree if query extends past this node's start
        if (mid + 1 < hi && qend > intervals_[mid].start) {
            query_impl(mid, mid + 1, hi, qstart, qend, results);
        }
    }
};

} // namespace singlet
