// SPDX-License-Identifier: MIT
// singlet::pz2c -- client-side helpers over decoded pz2 buffers, shared by the Python (_pz2.cpp) and R (pz2_io.cpp) bindings.
//
// Pure C++17, no Python/R headers.  Everything works in place on the arrays the decoder filled, so the bindings allocate
// the output arrays once (numpy / R vectors), decode into them, and trim:
//   * select_columns_inplace  keep a sorted subset of the decoded columns (the called cells of a sample);
//   * aggregate_genes         map row ids to gene ids, sum duplicates, sort gene ids per column (feature -> gene level).
// Both are deterministic and independent of the thread count.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace singlet {
namespace pz2c {

inline int auto_threads(int t) {
    if (t > 0) return t;
    unsigned h = std::thread::hardware_concurrency();
    return int(h ? std::min(4u, h) : 1u);
}

/// Run f(thread_index, block_index) for every block in [0, n_blocks) on up to `threads` threads (dynamic scheduling).
template <class F>
inline void parallel_blocks(size_t n_blocks, int threads, F&& f) {
    threads = std::max(1, std::min<int>(threads, int(std::min<size_t>(n_blocks, 1024))));
    if (threads == 1 || n_blocks <= 1) {
        for (size_t b = 0; b < n_blocks; ++b) f(size_t(0), b);
        return;
    }
    std::atomic<size_t> next{0};
    std::exception_ptr eptr;
    std::mutex mu;
    std::vector<std::thread> ts;
    ts.reserve(size_t(threads));
    for (int t = 0; t < threads; ++t)
        ts.emplace_back([&, t] {
            try {
                for (;;) {
                    size_t b = next.fetch_add(1);
                    if (b >= n_blocks) break;
                    f(size_t(t), b);
                }
            } catch (...) {
                std::lock_guard<std::mutex> g(mu);
                if (!eptr) eptr = std::current_exception();
                next.store(n_blocks);
            }
        });
    for (auto& th : ts) th.join();
    if (eptr) std::rethrow_exception(eptr);
}

/// stored row id -> gene: out[s] = gene_of_canonical[perm[s]] (perm empty = identity).  -1 marks a row that maps to no gene.
inline std::vector<int32_t> compose_gene_map(const int32_t* gene_of_canonical, size_t m_canonical, const uint32_t* perm, size_t m_stored) {
    std::vector<int32_t> out(m_stored);
    if (perm == nullptr) {
        if (m_stored != m_canonical) throw std::invalid_argument("gene map length != number of rows");
        std::memcpy(out.data(), gene_of_canonical, m_stored * sizeof(int32_t));
        return out;
    }
    for (size_t s = 0; s < m_stored; ++s) {
        uint32_t c = perm[s];
        if (c >= m_canonical) throw std::invalid_argument("row permutation entry outside the gene map");
        out[s] = gene_of_canonical[c];
    }
    return out;
}

/// Keep the columns listed in `cols` (strictly ascending, each < n_in) of a decoded CSC, compacting indices/data in place.
/// `indptr` has n_in+1 entries; `out_indptr` receives n_sel+1 entries.  Returns the new nnz.
template <class PtrT, class IdxT, class ValT>
inline PtrT select_columns_inplace(const PtrT* indptr, size_t n_in, const uint32_t* cols, size_t n_sel, IdxT* indices, ValT* data, PtrT* out_indptr) {
    PtrT w = 0;
    out_indptr[0] = 0;
    uint32_t prev = 0;
    for (size_t k = 0; k < n_sel; ++k) {
        const uint32_t c = cols[k];
        if (c >= n_in || (k > 0 && c <= prev)) throw std::invalid_argument("column selection must be strictly ascending and in range");
        prev = c;
        const PtrT a = indptr[c], b = indptr[c + 1];
        const size_t len = size_t(b - a);
        if (len && a != w) {
            std::memmove(indices + w, indices + a, len * sizeof(IdxT));
            std::memmove(data + w, data + a, len * sizeof(ValT));
        }
        w += PtrT(len);
        out_indptr[k + 1] = w;
    }
    return w;
}

/// Aggregate rows into genes in place.  Input: a CSC with n columns (indptr n+1, indices < m, data), ascending or not.
/// `row_to_gene[r]` is the gene (0 <= g < n_genes) of row r, or -1 to drop the row.  Entries with value 0 are dropped (as a
/// sparse matrix product would).  Output (same buffers): gene ids strictly ascending within each column, values summed;
/// `indptr` is rewritten.  Returns the new nnz.
template <class PtrT, class ValT>
inline PtrT aggregate_genes(PtrT* indptr, size_t n, int32_t* indices, ValT* data, const int32_t* row_to_gene, int32_t n_genes, int threads) {
    const PtrT nnz_in = indptr[n];
    if (n == 0) return 0;
    if (n_genes <= 0) {
        for (size_t j = 0; j <= n; ++j) indptr[j] = 0;
        return 0;
    }
    threads = auto_threads(threads);
    // blocks of columns with about equal input nnz
    std::vector<size_t> bnd;
    bnd.push_back(0);
    const size_t target = std::max<size_t>(8192, size_t(nnz_in) / (size_t(threads) * 8 + 1));
    {
        size_t acc = 0;
        for (size_t j = 0; j < n; ++j) {
            acc += size_t(indptr[j + 1] - indptr[j]);
            if (acc >= target) { bnd.push_back(j + 1); acc = 0; }
        }
        if (bnd.back() != n) bnd.push_back(n);
    }
    const size_t n_blocks = bnd.size() - 1;
    std::vector<uint32_t> cnt(n, 0);
    std::vector<size_t> block_out(n_blocks, 0);
    const size_t nwords = (size_t(n_genes) + 63) / 64;
    struct Scratch { std::vector<ValT> acc; std::vector<uint64_t> bits; std::vector<int32_t> touched; };
    std::vector<Scratch> scratch(size_t(std::max(1, threads)));

    parallel_blocks(n_blocks, threads, [&](size_t t, size_t b) {
        Scratch& S = scratch[t];
        if (S.acc.empty()) { S.acc.assign(size_t(n_genes), ValT(0)); S.bits.assign(nwords, 0); }
        ValT* acc = S.acc.data();
        uint64_t* bits = S.bits.data();
        const size_t c0 = bnd[b], c1 = bnd[b + 1];
        PtrT w = indptr[c0];
        const PtrT region0 = w;
        for (size_t j = c0; j < c1; ++j) {
            const PtrT s = indptr[j], e = indptr[j + 1];
            const size_t k = size_t(e - s);
            if (k == 0) continue;
            const size_t w0 = size_t(w);
            if (k * 12 < nwords) {  // few entries: gather the touched genes and sort them
                S.touched.clear();
                for (PtrT i = s; i < e; ++i) {
                    const uint32_t r = uint32_t(indices[i]);
                    const int32_t g = row_to_gene[r];
                    const ValT v = data[i];
                    if (g < 0 || v == ValT(0)) continue;
                    if (acc[g] == ValT(0)) S.touched.push_back(g);
                    acc[g] += v;
                }
                std::sort(S.touched.begin(), S.touched.end());
                for (int32_t g : S.touched) { indices[w] = g; data[w] = acc[g]; acc[g] = ValT(0); ++w; }
            } else {  // many entries: mark genes in a bitmap and scan it in order
                for (PtrT i = s; i < e; ++i) {
                    const uint32_t r = uint32_t(indices[i]);
                    const int32_t g = row_to_gene[r];
                    const ValT v = data[i];
                    if (g < 0 || v == ValT(0)) continue;
                    acc[g] += v;
                    bits[size_t(g) >> 6] |= uint64_t(1) << (g & 63);
                }
                for (size_t wi = 0; wi < nwords; ++wi) {
                    uint64_t x = bits[wi];
                    if (!x) continue;
                    bits[wi] = 0;
                    while (x) {
#if defined(__GNUC__) || defined(__clang__)
                        const int bit = __builtin_ctzll(x);
#else
                        int bit = 0; while (!((x >> bit) & 1)) ++bit;
#endif
                        const int32_t g = int32_t(wi * 64 + size_t(bit));
                        indices[w] = g; data[w] = acc[g]; acc[g] = ValT(0); ++w;
                        x &= x - 1;
                    }
                }
            }
            cnt[j] = uint32_t(size_t(w) - w0);
        }
        block_out[b] = size_t(w - region0);
    });

    // compact the block regions (they only move down), then rebuild indptr from the per-column counts
    PtrT w = 0;
    for (size_t b = 0; b < n_blocks; ++b) {
        const PtrT region0 = indptr[bnd[b]];
        const size_t len = block_out[b];
        if (len && region0 != w) {
            std::memmove(indices + w, indices + region0, len * sizeof(int32_t));
            std::memmove(data + w, data + region0, len * sizeof(ValT));
        }
        w += PtrT(len);
    }
    indptr[0] = 0;
    for (size_t j = 0; j < n; ++j) indptr[j + 1] = indptr[j] + PtrT(cnt[j]);
    return w;
}

}  // namespace pz2c
}  // namespace singlet
