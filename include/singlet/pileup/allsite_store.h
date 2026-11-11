// SPDX-License-Identifier: MIT
// include/singlet/pileup/allsite_store.h
//
// Storage of the all-site member (private/allsite/, owner-only): the donor genotype plus a per-cell delta against it.
//
//  * Layer 1, the donor genotypes at the member's sites: donors_allsite.vcf.gz, written from the per-donor sums (write_donors_vcf).
//  * Layer 2, the per-cell delta, one rule for every sample. Each called cell is compared with the genotype of its assigned donor and keeps
//    an entry (ALT and total UMI counts) only where it is informative:
//      - donor homozygous: sites where the cell has molecules of the other allele;
//      - donor heterozygous, or the donor has no genotype call: every covered site (the allele-specific expression signal);
//      - sites where the library's donors differ: every covered cell, so demultiplexing can be recomputed from the store;
//      - doublets and unassigned cells: stored against the reference genome (ALT molecules only), the assignment recorded per cell,
//        except at the sites where the donors differ.
//    Concordant coverage at homozygous sites is dropped. sites.tsv.gz keeps the per-site summed DP and AD over every barcode.
//
// The delta is a binary sparse file with one byte range per cell, 64-bit offsets and no entry limit (allele_delta.bin): a header, one
// zlib block per cell (row gaps, ALT and REF counts as varints), then an index of {offset, compressed bytes, entries} per cell. The matrices
// of allsite.h (CSC with int32 column pointers) are not built for the member; entries are read from the merged records.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <zlib.h>

#include "allsite.h"
#include "genotype_export.h"

namespace singlet { namespace allsite {

constexpr const char* kStoreRule = "donor-delta-v1";
constexpr int kRef = -2;   // genotype index of a cell stored against the reference genome

// Which entries a cell keeps, given the genotype index of its donor (0, 1, 2, -1 no call) or kRef, and whether the library's donors differ.
inline bool keep_entry(int g, bool donors_differ, uint32_t ad, uint32_t dp) {
    if (dp == 0) return false;
    if (donors_differ) return true;
    if (g == 0 || g == kRef) return ad > 0;
    if (g == 2) return dp > ad;
    return true;   // heterozygous, or no donor call
}

struct DonorCalls {
    int K = 0;
    uint32_t sites = 0;
    std::vector<int8_t> gt;          // [site * K + donor]: 0, 1, 2 or -1
    std::vector<uint8_t> differs;    // per site: two donors with different calls
    std::vector<uint8_t> variant;    // per site: some donor is 0/1 or 1/1
    std::vector<uint32_t> sad, sdp;  // [site * K + donor] summed over the donor's cells
    uint64_t sites_variant = 0;
    int g_of(uint32_t site, int donor) const { return donor < 0 || K == 0 ? kRef : gt[static_cast<size_t>(site) * K + donor]; }
};

// Per-donor sums over the cells assigned to a single donor (donor_of[barcode] >= 0), and the donors' genotype calls. The sums are the rule
// of genotype::write_donors_vcf, so the genotypes are those of donors.vcf.gz at the same sites.
// f(unit, entries, worker) over every unit of a kernel 1 result, r.threads workers taking units in order
template <class F> inline void par_units(const Result& r, F&& f) {
    const unsigned nth = static_cast<unsigned>(std::max<size_t>(1, std::min<size_t>(r.threads, r.n_units)));
    std::atomic<size_t> next{0};
    auto work = [&](unsigned w) { std::vector<Result::Pair> v; for (size_t i; (i = next.fetch_add(1)) < r.n_units;) { r.unit_fn(i, v); f(i, v, w); } };
    std::vector<std::thread> th; for (unsigned w = 1; w < nth; ++w) th.emplace_back(work, w);
    work(0); for (auto& x : th) x.join();
}

inline DonorCalls donor_calls(const Result& r, const std::vector<int32_t>& donor_of, int K) {
    DonorCalls d; d.K = K; d.sites = static_cast<uint32_t>(r.sites.size());
    d.differs.assign(d.sites, 0); d.variant.assign(d.sites, 0);
    if (K <= 0) return d;
    std::vector<uint64_t> sad(static_cast<size_t>(d.sites) * K, 0), sdp(static_cast<size_t>(d.sites) * K, 0);
    auto add = [&](uint32_t cell, uint32_t row, uint32_t ad, uint32_t dp) {
        if (cell >= donor_of.size() || donor_of[cell] < 0) return;
        const size_t i = static_cast<size_t>(row) * K + static_cast<size_t>(donor_of[cell]);
        sad[i] += ad; sdp[i] += dp;
    };
    if (r.n_units && r.unit_fn) par_units(r, [&](size_t, const std::vector<Result::Pair>& v, unsigned) { for (const auto& p : v) add(p.cell, p.row, p.ad, p.dp); });   // units own their rows
    else r.for_each_pair(add);
    d.gt.assign(sad.size(), -1); d.sad.resize(sad.size()); d.sdp.resize(sad.size());
    const uint32_t chunk = 1u << 16, nch = (d.sites + chunk - 1) / chunk;
    std::atomic<uint32_t> next{0};
    auto call = [&](uint32_t s) {
        int lo = 3, hi = -1;
        for (int k = 0; k < K; ++k) {
            const size_t i = static_cast<size_t>(s) * K + k;
            d.sad[i] = static_cast<uint32_t>(std::min<uint64_t>(sad[i], UINT32_MAX)); d.sdp[i] = static_cast<uint32_t>(std::min<uint64_t>(sdp[i], UINT32_MAX));
            const int g = genotype::donor_gt_index(d.sad[i], d.sdp[i]);
            d.gt[i] = static_cast<int8_t>(g);
            if (g >= 0) { lo = std::min(lo, g); hi = std::max(hi, g); }
            if (g >= 1) d.variant[s] = 1;
        }
        d.differs[s] = (hi >= 0 && lo != hi) ? 1 : 0;
    };
    auto work = [&]() { for (uint32_t c; (c = next.fetch_add(1)) < nch;) for (uint32_t s = c * chunk, e = std::min(d.sites, s + chunk); s < e; ++s) call(s); };
    const unsigned nth = std::max(1u, std::min<unsigned>(r.threads, nch));
    std::vector<std::thread> th; for (unsigned w = 1; w < nth; ++w) th.emplace_back(work);
    work(); for (auto& x : th) x.join();
    for (uint32_t s = 0; s < d.sites; ++s) d.sites_variant += d.variant[s];
    return d;
}

// The called cells' kept entries, cell by cell (columns in the order of `cells`), rows ascending within a cell.
struct DeltaStore {
    uint32_t n_sites = 0;
    int K = 0;
    std::vector<uint64_t> off;                 // n_cells + 1 entry offsets
    std::vector<uint32_t> row;
    std::vector<uint16_t> ad, dp;
    uint64_t full_entries = 0;                 // merged (cell, site) entries of the called cells, before the rule
    uint32_t n_cells() const { return off.empty() ? 0 : static_cast<uint32_t>(off.size() - 1); }
    uint64_t n_entries() const { return row.size(); }
};

// cells: sorted, unique barcode indices of the called cells. donor_of: per barcode, the assigned donor or -1 (doublet, unassigned, none).
inline void build_store(const Result& r, uint32_t n_barcodes, const std::vector<uint32_t>& cells, const std::vector<int32_t>& donor_of,
                        const DonorCalls& calls, DeltaStore& s) {
    s = DeltaStore(); s.n_sites = calls.sites; s.K = calls.K;
    std::vector<int32_t> col_of(n_barcodes, -1);
    for (size_t j = 0; j < cells.size(); ++j) if (cells[j] < n_barcodes) col_of[cells[j]] = static_cast<int32_t>(j);
    auto keep = [&](uint32_t cell, uint32_t row, uint32_t ad, uint32_t dp) {
        const int d = cell < donor_of.size() ? donor_of[cell] : -1;
        return keep_entry(calls.g_of(row, d), calls.differs[row] != 0, ad, dp);
    };
    std::vector<uint64_t> cnt(cells.size() + 1, 0);
    if (r.n_units && r.unit_fn) {   // units in parallel: each keeps its stored entries, then they are placed in unit order (rows ascend)
        struct E { uint32_t col, row; uint16_t ad, dp; };
        std::vector<std::vector<E>> ue(r.n_units); std::vector<uint64_t> full(r.n_units, 0);
        par_units(r, [&](size_t u, const std::vector<Result::Pair>& v, unsigned) {
            auto& o = ue[u];
            for (const auto& p : v) {
                if (p.cell >= n_barcodes || col_of[p.cell] < 0) continue;
                ++full[u];
                if (keep(p.cell, p.row, p.ad, p.dp)) o.push_back({static_cast<uint32_t>(col_of[p.cell]), p.row, static_cast<uint16_t>(p.ad), static_cast<uint16_t>(p.dp)});
            }
            o.shrink_to_fit();
        });
        for (size_t u = 0; u < r.n_units; ++u) { s.full_entries += full[u]; for (const auto& e : ue[u]) ++cnt[static_cast<size_t>(e.col) + 1]; }
        for (size_t j = 0; j < cells.size(); ++j) cnt[j + 1] += cnt[j];
        s.off = cnt;
        s.row.resize(cnt.back()); s.ad.resize(cnt.back()); s.dp.resize(cnt.back());
        std::vector<uint64_t> pos(cnt.begin(), cnt.end() - 1);
        for (size_t u = 0; u < r.n_units; ++u) {
            for (const auto& e : ue[u]) { const uint64_t k = pos[e.col]++; s.row[k] = e.row; s.ad[k] = e.ad; s.dp[k] = e.dp; }
            std::vector<E>().swap(ue[u]);
        }
        return;
    }
    r.for_each_pair([&](uint32_t cell, uint32_t row, uint32_t ad, uint32_t dp) {
        if (cell >= n_barcodes || col_of[cell] < 0) return;
        ++s.full_entries;
        if (keep(cell, row, ad, dp)) ++cnt[static_cast<size_t>(col_of[cell]) + 1];
    });
    for (size_t j = 0; j < cells.size(); ++j) cnt[j + 1] += cnt[j];
    s.off = cnt;
    s.row.resize(cnt.back()); s.ad.resize(cnt.back()); s.dp.resize(cnt.back());
    std::vector<uint64_t> pos(cnt.begin(), cnt.end() - 1);
    r.for_each_pair([&](uint32_t cell, uint32_t row, uint32_t ad, uint32_t dp) {   // contigs ascend, so rows ascend within a cell
        if (cell >= n_barcodes || col_of[cell] < 0 || !keep(cell, row, ad, dp)) return;
        const uint64_t k = pos[static_cast<size_t>(col_of[cell])]++;
        s.row[k] = row; s.ad[k] = static_cast<uint16_t>(ad); s.dp[k] = static_cast<uint16_t>(dp);
    });
}

namespace detail_store {
inline void put_varint(std::string& o, uint32_t v) { while (v >= 128) { o.push_back(static_cast<char>((v & 127) | 128)); v >>= 7; } o.push_back(static_cast<char>(v)); }
inline uint32_t get_varint(const uint8_t*& p, const uint8_t* end) {
    uint32_t v = 0; int sh = 0;
    while (p < end) { const uint8_t b = *p++; v |= static_cast<uint32_t>(b & 127) << sh; if (!(b & 128)) break; sh += 7; }
    return v;
}
constexpr char kMagic[8] = {'S', 'G', 'A', 'L', 'L', 'E', 'L', '1'};
constexpr size_t kHeader = 64;
struct Ix { uint64_t off; uint32_t len, n; };
}  // namespace detail_store

// Raw (uncompressed) bytes of one cell's block: per entry the row gap, the ALT count and the REF count as varints.
inline std::string encode_cell_raw(const DeltaStore& s, uint32_t j) {
    std::string raw; raw.reserve((s.off[j + 1] - s.off[j]) * 4);
    uint32_t prev = 0;
    for (uint64_t k = s.off[j]; k < s.off[j + 1]; ++k) {
        detail_store::put_varint(raw, s.row[k] - prev); prev = s.row[k];
        detail_store::put_varint(raw, s.ad[k]); detail_store::put_varint(raw, static_cast<uint32_t>(s.dp[k]) - s.ad[k]);
    }
    return raw;
}

// Writes the store; returns the file size, or 0 on failure. Atomic (temporary name, then rename).
inline uint64_t write_store(const std::string& path, const DeltaStore& s) {
    using detail_store::Ix;
    const std::string tmp = path + ".tmp";
    std::ofstream f(tmp, std::ios::binary);
    if (!f) return 0;
    char h[detail_store::kHeader] = {0};
    f.write(h, sizeof(h));   // header placeholder
    std::vector<Ix> ix(s.n_cells());
    uint64_t at = detail_store::kHeader;
    std::vector<uint8_t> z;
    for (uint32_t j = 0; j < s.n_cells(); ++j) {
        const std::string raw = encode_cell_raw(s, j);
        uLongf zl = 0;
        if (!raw.empty()) {
            zl = compressBound(static_cast<uLong>(raw.size()));
            z.resize(zl);
            if (compress2(z.data(), &zl, reinterpret_cast<const Bytef*>(raw.data()), static_cast<uLong>(raw.size()), 6) != Z_OK) { f.close(); std::remove(tmp.c_str()); return 0; }
            f.write(reinterpret_cast<const char*>(z.data()), static_cast<std::streamsize>(zl));
        }
        ix[j] = {at, static_cast<uint32_t>(zl), static_cast<uint32_t>(s.off[j + 1] - s.off[j])};
        at += zl;
    }
    const uint64_t index_off = at;
    if (!ix.empty()) f.write(reinterpret_cast<const char*>(ix.data()), static_cast<std::streamsize>(ix.size() * sizeof(Ix)));
    at += ix.size() * sizeof(Ix);
    std::memcpy(h, detail_store::kMagic, 8);
    const uint32_t ver = 1, nc = s.n_cells(), ns = s.n_sites, K = static_cast<uint32_t>(s.K);
    const uint64_t ne = s.n_entries();
    std::memcpy(h + 8, &ver, 4); std::memcpy(h + 12, &nc, 4); std::memcpy(h + 16, &ns, 4); std::memcpy(h + 20, &K, 4);
    std::memcpy(h + 24, &ne, 8); std::memcpy(h + 32, &index_off, 8);
    f.seekp(0); f.write(h, sizeof(h));
    f.close();
    if (!f) { std::remove(tmp.c_str()); return 0; }
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return 0; }
    return at;
}

// Reads the whole store back (tests, tools). False on a malformed file.
inline bool read_store(const std::string& path, DeltaStore& s) {
    using detail_store::Ix;
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char h[detail_store::kHeader];
    if (!f.read(h, sizeof(h)) || std::memcmp(h, detail_store::kMagic, 8) != 0) return false;
    uint32_t ver, nc, ns, K; uint64_t ne, index_off;
    std::memcpy(&ver, h + 8, 4); std::memcpy(&nc, h + 12, 4); std::memcpy(&ns, h + 16, 4); std::memcpy(&K, h + 20, 4);
    std::memcpy(&ne, h + 24, 8); std::memcpy(&index_off, h + 32, 8);
    if (ver != 1) return false;
    std::vector<Ix> ix(nc);
    f.seekg(static_cast<std::streamoff>(index_off));
    if (nc && !f.read(reinterpret_cast<char*>(ix.data()), static_cast<std::streamsize>(nc * sizeof(Ix)))) return false;
    s = DeltaStore(); s.n_sites = ns; s.K = static_cast<int>(K); s.off.assign(1, 0);
    s.row.reserve(ne); s.ad.reserve(ne); s.dp.reserve(ne);
    std::vector<uint8_t> z; std::string raw;
    for (uint32_t j = 0; j < nc; ++j) {
        if (ix[j].n) {
            z.resize(ix[j].len);
            f.seekg(static_cast<std::streamoff>(ix[j].off));
            if (!f.read(reinterpret_cast<char*>(z.data()), ix[j].len)) return false;
            raw.assign(static_cast<size_t>(ix[j].n) * 11 + 16, '\0');   // at most 5 + 3 + 3 bytes per entry
            uLongf rl = static_cast<uLongf>(raw.size());
            if (uncompress(reinterpret_cast<Bytef*>(&raw[0]), &rl, z.data(), ix[j].len) != Z_OK) return false;
            const uint8_t* p = reinterpret_cast<const uint8_t*>(raw.data()); const uint8_t* end = p + rl;
            uint32_t prev = 0;
            for (uint32_t e = 0; e < ix[j].n; ++e) {
                prev += detail_store::get_varint(p, end);
                const uint32_t a = detail_store::get_varint(p, end), r2 = detail_store::get_varint(p, end);
                s.row.push_back(prev); s.ad.push_back(static_cast<uint16_t>(a)); s.dp.push_back(static_cast<uint16_t>(a + r2));
            }
        }
        s.off.push_back(s.row.size());
    }
    return s.row.size() == ne;
}

// cells.tsv.gz: one line per column of the store: barcode, the assignment (donorN, doublet, unassigned, none), kept entries.
inline bool write_cells_tsv(const std::string& path, const std::vector<std::string>& barcodes, const std::vector<uint32_t>& cells,
                            const std::vector<std::string>& assignment, const DeltaStore& s) {
    const std::string tmp = path + ".tmp";
    gzFile f = gzopen(tmp.c_str(), "wb6");
    if (!f) return false;
    std::string line = "barcode\tassignment\tentries\n";
    bool ok = gzwrite(f, line.data(), static_cast<unsigned>(line.size())) > 0;
    for (size_t j = 0; ok && j < cells.size(); ++j) {
        line = barcodes[cells[j]] + "\t" + assignment[j] + "\t" + std::to_string(s.off[j + 1] - s.off[j]) + "\n";
        ok = gzwrite(f, line.data(), static_cast<unsigned>(line.size())) > 0;
    }
    if (gzclose(f) != Z_OK) ok = false;
    if (ok) ok = std::rename(tmp.c_str(), path.c_str()) == 0;
    if (!ok) std::remove(tmp.c_str());
    return ok;
}

// Layer 1: private/genotype/donors_allsite.vcf.gz, the donor genotypes at the sites with depth in the single-donor cells (same fields as
// donors.vcf.gz). Returns the rows written, -1 on a write failure, 0 without donors.
inline long long write_donors_vcf(const std::string& path, const Result& r, const std::vector<std::string>& tid_names, const DonorCalls& d) {
    if (d.K <= 0) return 0;
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    genotype::GzOut f(path, true);
    if (!f.ok()) return -1;
    f.put("##fileformat=VCFv4.2\n##source=singlet allsite_store\n"
          "##singlet_genotype_model=binomial on donor-summed UMI counts, error 0.01, flat prior; GT when the posterior >= 0.99\n"
          "##singlet_cells=called cells (cell_status cell) with donor prob_max >= 0.9\n");
    long long n = 0;
    std::vector<uint8_t> row_on(d.sites, 0);
    for (uint32_t s = 0; s < d.sites; ++s) for (int k = 0; k < d.K; ++k) if (d.sdp[static_cast<size_t>(s) * d.K + k] > 0) row_on[s] = 1;
    std::string seen;
    for (uint32_t s = 0; s < d.sites; ++s) {
        if (!row_on[s]) continue;
        const std::string& c = tid_names[static_cast<size_t>(r.sites[s].tid)];
        if (c != seen) { seen = c; f.put("##contig=<ID=" + c + ">\n"); }
    }
    f.put("##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
          "##INFO=<ID=NS,Number=1,Type=Integer,Description=\"Number of donors with coverage at the site\">\n"
          "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype; ./. when the best posterior is below 0.99\">\n"
          "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"Phred-scaled 1 - posterior of the most likely genotype, capped at 99\">\n"
          "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Ref and alt UMI counts summed over the donor's cells\">\n"
          "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Ref + alt UMI count summed over the donor's cells\">\n"
          "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"Phred-scaled genotype likelihoods, normalized to the best\">\n"
          "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT");
    for (int k = 0; k < d.K; ++k) f.put("\tdonor" + std::to_string(k));
    f.put("\n");
    for (uint32_t s = 0; s < d.sites; ++s) {
        if (!row_on[s]) continue;
        const Site& st = r.sites[s];
        int ns = 0; for (int k = 0; k < d.K; ++k) if (d.sdp[static_cast<size_t>(s) * d.K + k] > 0) ++ns;
        std::string line = tid_names[static_cast<size_t>(st.tid)] + "\t" + std::to_string(st.pos + 1) + "\t" + site_name(st, tid_names) + "\t" + st.ref + "\t" + st.alt +
                           "\t.\tPASS\tNS=" + std::to_string(ns) + "\tGT:GQ:AD:DP:PL";
        for (int k = 0; k < d.K; ++k) { line += '\t'; line += genotype::donor_fields(d.sad[static_cast<size_t>(s) * d.K + k], d.sdp[static_cast<size_t>(s) * d.K + k]); }
        line += '\n';
        f.put(line); ++n;
    }
    return n;
}

// A standard per-cell VCF from the stored entries: rows are the sites where some donor is 0/1 or 1/1 and a cell kept an entry, columns the
// called cells in store order. FORMAT GT:AD:DP:DC per cell (GT the donor's genotype, ./. when it has none or the cell has no donor; DC the
// UMIs that contradict it); a cell without a stored entry is ".". Returns the rows, -1 on a write failure, -2 when rows x cells exceeds
// max_fields (the VCF is dense in cells; the store stays the record).
inline long long write_cells_vcf(const std::string& path, const DeltaStore& s, const Result& r, const std::vector<std::string>& tid_names,
                                 const std::vector<std::string>& barcodes, const std::vector<uint32_t>& cells, const std::vector<int32_t>& donor_of,
                                 const DonorCalls& d, uint64_t max_fields = 4000000000ull) {
    if (d.K <= 0) return 0;
    std::vector<int64_t> slot(d.sites, -1); std::vector<uint32_t> rows;
    for (uint64_t k = 0; k < s.row.size(); ++k) if (d.variant[s.row[k]]) slot[s.row[k]] = 0;
    for (uint32_t rw = 0; rw < d.sites; ++rw) if (slot[rw] == 0) { slot[rw] = static_cast<int64_t>(rows.size()); rows.push_back(rw); }
    if (static_cast<uint64_t>(rows.size()) * cells.size() > max_fields) return -2;
    std::vector<std::vector<std::array<uint32_t, 3>>> by_row(rows.size());   // per kept row: (column, ad, dp)
    for (uint32_t j = 0; j < s.n_cells(); ++j)
        for (uint64_t k = s.off[j]; k < s.off[j + 1]; ++k) if (slot[s.row[k]] >= 0) by_row[static_cast<size_t>(slot[s.row[k]])].push_back({j, s.ad[k], s.dp[k]});
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    genotype::GzOut f(path, true);
    if (!f.ok()) return -1;
    f.put("##fileformat=VCFv4.2\n##source=singlet allsite_store (per-cell delta)\n"
          "##singlet_cells=called cells; an entry is kept where the cell is informative against its donor genotype (rule " + std::string(kStoreRule) + ")\n");
    std::string seen;
    for (uint32_t rw : rows) { const std::string& c = tid_names[static_cast<size_t>(r.sites[rw].tid)]; if (c != seen) { seen = c; f.put("##contig=<ID=" + c + ">\n"); } }
    f.put("##FILTER=<ID=PASS,Description=\"All filters passed\">\n"
          "##INFO=<ID=DGT,Number=.,Type=String,Description=\"Genotype of each donor (donor0, donor1, ...)\">\n"
          "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype of the cell's donor; ./. when not called or the cell has no donor\">\n"
          "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Ref and alt UMI counts of the cell\">\n"
          "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Ref + alt UMI count of the cell\">\n"
          "##FORMAT=<ID=DC,Number=1,Type=Integer,Description=\"UMIs contradicting the donor genotype: ALT at 0/0, REF at 1/1, 0 at 0/1\">\n"
          "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT");
    for (uint32_t c : cells) { f.put("\t"); f.put(barcodes[c]); }
    f.put("\n");
    static const char* kGt[3] = {"0/0", "0/1", "1/1"};
    std::string line;
    for (size_t i = 0; i < rows.size(); ++i) {
        const uint32_t rw = rows[i]; const Site& st = r.sites[rw];
        line = tid_names[static_cast<size_t>(st.tid)]; line += '\t'; line += std::to_string(st.pos + 1); line += '\t'; line += site_name(st, tid_names);
        line += '\t'; line += st.ref; line += '\t'; line += st.alt; line += "\t.\tPASS\tDGT=";
        for (int k = 0; k < d.K; ++k) { if (k) line += ','; const int g = d.gt[static_cast<size_t>(rw) * d.K + k]; line += g < 0 ? "./." : kGt[g]; }
        line += "\tGT:AD:DP:DC";
        auto& v = by_row[i]; std::sort(v.begin(), v.end()); size_t e = 0;
        for (uint32_t j = 0; j < cells.size(); ++j) {
            line += '\t';
            if (e < v.size() && v[e][0] == j) {
                const uint32_t a = v[e][1], dpv = v[e][2]; ++e;
                const int dn = cells[j] < donor_of.size() ? donor_of[cells[j]] : -1;
                const int g = dn < 0 ? -1 : d.gt[static_cast<size_t>(rw) * d.K + dn];
                line += g < 0 ? "./." : kGt[g]; line += ':'; line += std::to_string(dpv - a); line += ','; line += std::to_string(a); line += ':'; line += std::to_string(dpv); line += ':';
                line += g < 0 ? std::string(".") : std::to_string(g == 0 ? a : g == 2 ? dpv - a : 0u);
            } else line += '.';
        }
        line += '\n'; f.put(line);
        std::vector<std::array<uint32_t, 3>>().swap(v);
    }
    return static_cast<long long>(rows.size());
}

struct StoreStats {
    uint32_t cells = 0, assigned = 0, sites = 0, sites_variant = 0, sites_differ = 0;
    int n_donors = 0;
    uint64_t entries = 0, full_entries = 0, bytes = 0;
    uint64_t coverage_block_bytes_est = 0;   // dropped (concordant) entries, at the measured bytes per kept entry, two of three fields
};

inline StoreStats store_stats(const DeltaStore& s, const DonorCalls& d, const std::vector<int32_t>& donor_of, const std::vector<uint32_t>& cells, uint64_t bytes) {
    StoreStats t; t.cells = s.n_cells(); t.sites = s.n_sites; t.n_donors = s.K; t.entries = s.n_entries(); t.full_entries = s.full_entries; t.bytes = bytes;
    t.sites_variant = static_cast<uint32_t>(d.sites_variant);
    for (uint8_t x : d.differs) t.sites_differ += x;
    for (uint32_t c : cells) if (c < donor_of.size() && donor_of[c] >= 0) ++t.assigned;
    const uint64_t dropped = t.full_entries - t.entries;
    t.coverage_block_bytes_est = t.entries ? static_cast<uint64_t>(static_cast<double>(dropped) * (static_cast<double>(bytes) / static_cast<double>(t.entries)) * 2.0 / 3.0) : 0;
    return t;
}

inline bool write_store_json(const std::string& path, const StoreStats& t) {
    std::ofstream f(path);
    if (!f) return false;
    f << "{\n  \"rule\": \"" << kStoreRule << "\",\n  \"n_donors\": " << t.n_donors << ",\n  \"called_cells\": " << t.cells
      << ",\n  \"cells_assigned_to_a_donor\": " << t.assigned << ",\n  \"sites\": " << t.sites << ",\n  \"sites_donor_variant\": " << t.sites_variant
      << ",\n  \"sites_donors_differ\": " << t.sites_differ << ",\n  \"full_entries\": " << t.full_entries << ",\n  \"stored_entries\": " << t.entries
      << ",\n  \"file_bytes\": " << t.bytes << ",\n  \"bytes_per_cell\": " << (t.cells ? t.bytes / t.cells : 0)
      << ",\n  \"coverage_blocks_bytes_estimate\": " << t.coverage_block_bytes_est
      << ",\n  \"note\": \"allele_delta.bin holds, per called cell, the entries informative against the donor genotype; sites.tsv.gz dp and ad sum every barcode\"\n}\n";
    return static_cast<bool>(f);
}

}}  // namespace singlet::allsite
