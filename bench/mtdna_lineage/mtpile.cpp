// mtpile: per-cell chrM allele counts by strand, at read level and at molecule (UMI) level, from a coordinate-sorted
// BAM with cell (CB) and UMI (UB) tags. Benchmark tool for docs/mtdna-lineage.md; not part of the pipeline.
//
//   mtpile BAM CELLS.txt OUT_PREFIX [--min-mapq 20] [--min-bq 10] [--sites SITES.txt] [--contig chrM] [--threads 4]
//
// Read level: one count per aligned M/=/X base with base quality >= min-bq (what Singlet's pileup_mt_bases counts).
// Molecule level: reads of one (cell, UMI) are collapsed to one consensus base per position, in two steps:
//   1. reads with the same start, strand, cell and UMI (PCR duplicates of one fragment) vote per position;
//   2. fragments of the same (cell, UMI) at a position vote again, weighted by their reads.
//   The consensus base needs >= 75% of the reads; ties or weaker majorities drop the molecule at that position.
//   umi1 counts every molecule, umi3 only molecules with >= 3 reads at the position (maegatk --min-reads 3).
// Outputs (little endian):
//   OUT.{read,umi1,umi3}.bin  'MTP1' u32 n_cells u64 n_rec, then n_rec x {u32 cell, u32 pos, u32 count[8]}
//                              count[b + 4*rev], b = A,C,G,T; pos 0-based
//   OUT.sites.bin             (only with --sites; 1-based positions, one per line) molecule consensus at those sites:
//                              'MTS1' u64 n, then n x {u32 cell, u32 umi, u16 site_index, u8 base, u8 reads(capped 255)}
//   OUT.span.tsv              per-molecule histograms: span (max end - min start over its fragments, N skips
//                             included), fragments, covered (reference bases read by its longest fragment)
//   OUT.stats.json            read accounting
#include <htslib/sam.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

static constexpr uint32_t L = 16569;

static std::string strip_suffix(const std::string& s) {
    auto p = s.find('-');
    return p == std::string::npos ? s : s.substr(0, p);
}

static int base_idx(uint8_t code4) {  // htslib 4-bit
    switch (code4) { case 1: return 0; case 2: return 1; case 4: return 2; case 8: return 3; default: return -1; }
}

static bool encode_umi(const char* u, uint32_t& out) {
    uint32_t v = 0; int n = 0;
    for (const char* p = u; *p; ++p, ++n) {
        int b;
        switch (*p) { case 'A': b = 0; break; case 'C': b = 1; break; case 'G': b = 2; break; case 'T': b = 3; break; default: return false; }
        if (n >= 16) return false;
        v = (v << 2) | b;
    }
    out = v | (static_cast<uint32_t>(n) << 28);  // length in the top bits keeps different lengths apart
    return n > 0;
}

struct Frag {               // reads of one (start, strand, cell, umi)
    uint32_t cell, umi;
    uint8_t rev;
    uint16_t nreads = 0;
    std::vector<std::array<uint16_t, 4>> votes;   // by reference offset from start
};

struct MolObs {             // one fragment's consensus base at one position
    uint64_t key;           // cell << 32 | umi
    uint8_t base, rev;
    uint16_t reads;
};

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: mtpile BAM CELLS.txt OUT_PREFIX [options]\n"; return 2; }
    std::string bam_path = argv[1], cells_path = argv[2], out = argv[3];
    int min_mapq = 20, min_bq = 10, threads = 4;
    std::string sites_path, contig = "chrM";
    for (int i = 4; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if (a == "--min-mapq") min_mapq = std::stoi(next());
        else if (a == "--min-bq") min_bq = std::stoi(next());
        else if (a == "--sites") sites_path = next();
        else if (a == "--contig") contig = next();
        else if (a == "--threads") threads = std::stoi(next());
        else { std::cerr << "unknown option " << a << "\n"; return 2; }
    }
    std::unordered_map<std::string, uint32_t> cell_of;
    {
        std::ifstream f(cells_path); std::string s; uint32_t i = 0;
        while (std::getline(f, s)) if (!s.empty()) cell_of.emplace(strip_suffix(s), i++);
    }
    const uint32_t nc = static_cast<uint32_t>(cell_of.size());
    std::vector<int32_t> site_of(L, -1);
    if (!sites_path.empty()) {
        std::ifstream f(sites_path); std::string s; int k = 0;
        while (std::getline(f, s)) if (!s.empty()) { uint32_t p = std::stoul(s); if (p >= 1 && p <= L && site_of[p - 1] < 0) site_of[p - 1] = k++; }
    }
    std::cerr << "cells " << nc << "\n";
    const size_t cells_x = static_cast<size_t>(L) * 8;
    std::vector<uint32_t> rd(nc * cells_x, 0), u1(nc * cells_x, 0), u3(nc * cells_x, 0);

    samFile* in = sam_open(bam_path.c_str(), "r");
    if (!in) { std::cerr << "cannot open " << bam_path << "\n"; return 1; }
    hts_set_threads(in, threads);
    bam_hdr_t* hdr = sam_hdr_read(in);
    hts_idx_t* idx = sam_index_load(in, bam_path.c_str());
    if (!idx) { std::cerr << "no index\n"; return 1; }
    hts_itr_t* itr = sam_itr_querys(idx, hdr, contig.c_str());
    if (!itr) { std::cerr << "contig " << contig << " not found\n"; return 1; }
    bam1_t* b = bam_init1();

    uint64_t n_total = 0, n_flag = 0, n_mapq = 0, n_nocell = 0, n_noumi = 0, n_used = 0, n_bases = 0;
    std::unordered_map<uint64_t, Frag> frags;                   // current start: key (cell<<32|umi)*2+rev -> fragment
    int64_t cur_start = -1;
    std::vector<std::vector<MolObs>> posbuf(L);        // fragment consensus per position, flushed in order
    uint32_t next_flush = 0;
    std::unordered_map<uint64_t, std::pair<uint16_t, uint16_t>> span;  // molecule -> (min start, max end)
    std::unordered_map<uint64_t, uint32_t> nfrag, covered;
    std::vector<uint8_t> site_rec;                    // packed site records
    uint64_t n_site = 0, n_mol_amb = 0, n_mol_ok = 0;

    auto finalize_frags = [&]() {
        for (auto& kv : frags) {
            Frag& g = kv.second;
            const uint64_t key = (static_cast<uint64_t>(g.cell) << 32) | g.umi;
            for (size_t o = 0; o < g.votes.size(); ++o) {
                auto& v = g.votes[o];
                uint32_t t = v[0] + v[1] + v[2] + v[3];
                if (!t) continue;
                int bi = static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
                if (v[bi] * 4 < t * 3) continue;   // weak fragment consensus: drop this position
                uint32_t p = static_cast<uint32_t>(cur_start + o);
                if (p >= L) continue;
                posbuf[p].push_back({key, static_cast<uint8_t>(bi), g.rev, static_cast<uint16_t>(t)});
            }
            // extent = first..last reference position read (includes N skips: STAR aligns some chrM reads with
            // multi-kb N gaps); covered = reference positions read by the fragment (the read-length-limited quantity)
            int64_t f0 = -1, f1 = -1; uint32_t ncov = 0;
            for (size_t o = 0; o < g.votes.size(); ++o)
                if (g.votes[o][0] + g.votes[o][1] + g.votes[o][2] + g.votes[o][3]) { if (f0 < 0) f0 = o; f1 = o; ++ncov; }
            { auto& cv = covered[key]; cv = std::max(cv, ncov); }
            if (f0 >= 0) {
                uint16_t s = static_cast<uint16_t>(std::min<int64_t>(cur_start + f0, L - 1));
                uint16_t e = static_cast<uint16_t>(std::min<int64_t>(cur_start + f1 + 1, L));
                auto it = span.find(key);
                if (it == span.end()) span.emplace(key, std::make_pair(s, e));
                else { it->second.first = std::min(it->second.first, s); it->second.second = std::max(it->second.second, e); }
            }
            nfrag[key]++;
        }
        frags.clear();
    };
    auto flush_until = [&](uint32_t upto) {   // positions < upto are complete
        for (; next_flush < upto && next_flush < L; ++next_flush) {
            auto& v = posbuf[next_flush];
            std::sort(v.begin(), v.end(), [](const MolObs& a, const MolObs& c) { return a.key < c.key; });
            for (size_t i = 0; i < v.size();) {
                size_t j = i; uint32_t c[4] = {0, 0, 0, 0}, rr = 0, t = 0;
                while (j < v.size() && v[j].key == v[i].key) { c[v[j].base] += v[j].reads; rr += v[j].rev ? v[j].reads : 0; t += v[j].reads; ++j; }
                int bi = 0; for (int k = 1; k < 4; ++k) if (c[k] > c[bi]) bi = k;
                if (c[bi] * 4 >= t * 3) {
                    const uint32_t cell = static_cast<uint32_t>(v[i].key >> 32);
                    const int rev = rr * 2 > t ? 1 : 0;
                    const size_t off = cell * cells_x + static_cast<size_t>(next_flush) * 8 + bi + 4 * rev;
                    u1[off]++; if (t >= 3) u3[off]++;
                    ++n_mol_ok;
                    if (site_of[next_flush] >= 0) {
                        uint32_t umi = static_cast<uint32_t>(v[i].key & 0xffffffffu);
                        uint16_t si = static_cast<uint16_t>(site_of[next_flush]);
                        uint8_t bb = static_cast<uint8_t>(bi), nr = static_cast<uint8_t>(std::min<uint32_t>(t, 255));
                        uint8_t buf[12];
                        std::memcpy(buf, &cell, 4); std::memcpy(buf + 4, &umi, 4); std::memcpy(buf + 8, &si, 2);
                        buf[10] = bb; buf[11] = nr;
                        site_rec.insert(site_rec.end(), buf, buf + 12); ++n_site;
                    }
                } else ++n_mol_amb;
                i = j;
            }
            std::vector<MolObs>().swap(v);
        }
    };

    while (sam_itr_next(in, itr, b) >= 0) {
        ++n_total;
        const auto& c = b->core;
        if (c.flag & (BAM_FUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY | BAM_FQCFAIL)) { ++n_flag; continue; }
        if (c.qual < min_mapq) { ++n_mapq; continue; }
        uint8_t* cbt = bam_aux_get(b, "CB");
        if (!cbt) { ++n_nocell; continue; }
        auto it = cell_of.find(strip_suffix(bam_aux2Z(cbt)));
        if (it == cell_of.end()) { ++n_nocell; continue; }
        const uint32_t cell = it->second;
        uint8_t* ubt = bam_aux_get(b, "UB");
        uint32_t umi = 0;
        bool has_umi = ubt && encode_umi(bam_aux2Z(ubt), umi);
        if (!has_umi) ++n_noumi;
        ++n_used;
        if (c.pos != cur_start) {
            if (!frags.empty()) finalize_frags();
            cur_start = c.pos;
            flush_until(static_cast<uint32_t>(std::min<int64_t>(cur_start, L)));
        }
        const int rev = (c.flag & BAM_FREVERSE) ? 1 : 0;
        Frag* fr = nullptr;
        if (has_umi) {
            uint64_t k = (((static_cast<uint64_t>(cell) << 32) | umi) << 1) | rev;
            auto [fit, ins] = frags.try_emplace(k);
            fr = &fit->second;
            if (ins) { fr->cell = cell; fr->umi = umi; fr->rev = static_cast<uint8_t>(rev); }
            fr->nreads++;
        }
        const uint32_t* cig = bam_get_cigar(b);
        const uint8_t* seq = bam_get_seq(b);
        const uint8_t* q = bam_get_qual(b);
        int64_t rp = c.pos; int qp = 0;
        for (uint32_t k = 0; k < c.n_cigar; ++k) {
            const int op = bam_cigar_op(cig[k]); const int len = bam_cigar_oplen(cig[k]);
            if (op == BAM_CMATCH || op == BAM_CEQUAL || op == BAM_CDIFF) {
                for (int t = 0; t < len; ++t) {
                    const int64_t p = rp + t; const int qq = qp + t;
                    if (p >= L) break;
                    if (q[qq] < min_bq) continue;
                    const int bi = base_idx(bam_seqi(seq, qq));
                    if (bi < 0) continue;
                    rd[cell * cells_x + static_cast<size_t>(p) * 8 + bi + 4 * rev]++;
                    ++n_bases;
                    if (fr) {
                        const size_t o = static_cast<size_t>(p - cur_start);
                        if (fr->votes.size() <= o) fr->votes.resize(o + 1, {0, 0, 0, 0});
                        fr->votes[o][bi]++;
                    }
                }
                rp += len; qp += len;
            } else if (op == BAM_CINS || op == BAM_CSOFT_CLIP) qp += len;
            else if (op == BAM_CDEL || op == BAM_CREF_SKIP) rp += len;
        }
    }
    if (!frags.empty()) finalize_frags();
    flush_until(L);

    auto write_counts = [&](const std::string& path, const std::vector<uint32_t>& a) {
        std::vector<uint32_t> recs;
        uint64_t n = 0;
        for (uint32_t cc = 0; cc < nc; ++cc)
            for (uint32_t p = 0; p < L; ++p) {
                const uint32_t* x = &a[cc * cells_x + static_cast<size_t>(p) * 8];
                bool any = false; for (int k = 0; k < 8; ++k) any |= x[k] != 0;
                if (!any) continue;
                recs.push_back(cc); recs.push_back(p); recs.insert(recs.end(), x, x + 8); ++n;
            }
        FILE* f = std::fopen(path.c_str(), "wb");
        std::fwrite("MTP1", 1, 4, f); std::fwrite(&nc, 4, 1, f); std::fwrite(&n, 8, 1, f);
        std::fwrite(recs.data(), 4, recs.size(), f); std::fclose(f);
        return n;
    };
    uint64_t nr = write_counts(out + ".read.bin", rd), n1 = write_counts(out + ".umi1.bin", u1), n3 = write_counts(out + ".umi3.bin", u3);
    if (!sites_path.empty()) {
        FILE* f = std::fopen((out + ".sites.bin").c_str(), "wb");
        std::fwrite("MTS1", 1, 4, f); std::fwrite(&n_site, 8, 1, f); std::fwrite(site_rec.data(), 1, site_rec.size(), f); std::fclose(f);
    }
    {
        std::map<uint32_t, uint64_t> h; std::map<uint32_t, uint64_t> hf;
        for (auto& kv : span) h[(kv.second.second - kv.second.first) / 10 * 10]++;
        for (auto& kv : nfrag) hf[std::min<uint32_t>(kv.second, 50)]++;
        std::map<uint32_t, uint64_t> hc;
        for (auto& kv : covered) hc[kv.second / 10 * 10]++;
        std::ofstream f(out + ".span.tsv");
        f << "kind\tbin\tn_molecules\n";
        for (auto& kv : h) f << "span\t" << kv.first << "\t" << kv.second << "\n";
        for (auto& kv : hf) f << "fragments\t" << kv.first << "\t" << kv.second << "\n";
        for (auto& kv : hc) f << "covered\t" << kv.first << "\t" << kv.second << "\n";
    }
    std::ofstream js(out + ".stats.json");
    js << "{\"cells\": " << nc << ", \"reads_total\": " << n_total << ", \"reads_flag\": " << n_flag << ", \"reads_lowmapq\": " << n_mapq
       << ", \"reads_not_cell\": " << n_nocell << ", \"reads_no_umi\": " << n_noumi << ", \"reads_used\": " << n_used
       << ", \"bases_counted\": " << n_bases << ", \"molecules\": " << span.size() << ", \"mol_pos_ok\": " << n_mol_ok
       << ", \"mol_pos_ambiguous\": " << n_mol_amb << ", \"rec_read\": " << nr << ", \"rec_umi1\": " << n1 << ", \"rec_umi3\": " << n3
       << ", \"site_obs\": " << n_site << ", \"min_mapq\": " << min_mapq << ", \"min_bq\": " << min_bq << "}\n";
    std::cerr << "done: reads used " << n_used << " of " << n_total << ", molecules " << span.size() << "\n";
    bam_destroy1(b); hts_itr_destroy(itr); hts_idx_destroy(idx); bam_hdr_destroy(hdr); sam_close(in);
    return 0;
}
