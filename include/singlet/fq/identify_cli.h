// SPDX-License-Identifier: MIT
// fq/identify_cli.h — head readers and the `singlet identify` subcommand
//
//   singlet identify SOURCE [--reads N] [--genome-dir DIR] [--whitelist-dir DIR] [--hint assay=...] [--json OUT] [--route auto|sra|ena]
//   singlet identify --list samples.tsv --out-dir DIR [same options]      (rows: id TAB source [TAB hint [TAB genome-dir]]; lists stay loaded)
//
// SOURCE forms (runs of one sample separated by ';')
//   SRR/ERR/DRR accession   the archive's own spot access through SraReader (libncbi-vdb fetches blocks lazily; row limit stops it)
//                           --route ena: ranged GET of the run's FASTQ files from ENA instead
//   URL[,URL...]            ranged GET of each gzip FASTQ (one URL per segment, in spot order), decompressed on the fly
//   FILE[,FILE...]          local FASTQ (.gz or plain), one file per segment
//   bam:FILE                Cell Ranger style BAM through samtools (CR+UR tags, then the read)
//   FILE.1fq                a staged run
// The head is the first --reads spots per run (default 200000); the transfer stops there.

#pragma once

#include "identify.h"
#include "reader.h"
#include "sra_reader.h"

#include <zlib.h>

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <unistd.h>

namespace singlet::fq::identify {

namespace cli_detail {

inline std::string sq(const std::string& s) {
    std::string o = "'";
    for (char c : s) o += (c == '\'') ? std::string("'\\''") : std::string(1, c);
    return o + "'";
}
inline bool is_url(const std::string& s) { return s.rfind("http://", 0) == 0 || s.rfind("https://", 0) == 0 || s.rfind("ftp://", 0) == 0; }
inline bool is_accession(const std::string& s) {
    if (s.size() < 4 || s.size() > 14) return false;
    if (!(s[0] == 'S' || s[0] == 'E' || s[0] == 'D') || s[1] != 'R' || s[2] != 'R') return false;
    for (size_t i = 3; i < s.size(); ++i) if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    return true;
}
inline std::vector<std::string> split(const std::string& s, char d) {
    std::vector<std::string> v;
    std::string cur;
    for (char c : s) { if (c == d) { v.push_back(cur); cur.clear(); } else cur += c; }
    v.push_back(cur);
    return v;
}
inline std::string host_of(const std::string& u) {
    auto p = u.find("://");
    if (p == std::string::npos) return "local file";
    auto e = u.find('/', p + 3);
    return u.substr(p + 3, e == std::string::npos ? std::string::npos : e - p - 3);
}
inline double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

/// Read up to max_spots sequences from the head of a FASTQ (URL by ranged GET, or local path). Counts the compressed bytes read.
/// A truncated final record is dropped. The pipe is closed as soon as enough spots are held, which stops the transfer.
inline bool fastq_head(const std::string& loc, size_t max_spots, std::vector<std::string>& seqs, int64_t& bytes, std::string& err,
                       uint64_t range_bytes = 192ull << 20) {
    FILE* fp = nullptr;
    bool piped = false;
    if (is_url(loc)) {
        const std::string cmd = "curl -sfL --max-time 900 -r 0-" + std::to_string(range_bytes - 1) + " " + sq(loc);
        fp = popen(cmd.c_str(), "r");
        piped = true;
    } else {
        fp = std::fopen(loc.c_str(), "rb");
    }
    if (!fp) { err = "cannot open " + loc; return false; }
    z_stream z{};
    bool gz = false, inited = false, ended = false;
    std::vector<unsigned char> in(1 << 16), out(1 << 18);
    std::string text, pending;      // decompressed bytes not yet split into lines
    size_t line_no = 0;
    std::string rec_seq;
    bool stop = false;
    auto feed = [&](const char* d, size_t n) {
        pending.append(d, n);
        size_t pos = 0;
        while (!stop) {
            const size_t nl = pending.find('\n', pos);
            if (nl == std::string::npos) break;
            if (line_no % 4 == 1) {
                rec_seq.assign(pending, pos, nl - pos);
                while (!rec_seq.empty() && rec_seq.back() == '\r') rec_seq.pop_back();
            }
            if (line_no % 4 == 3) {   // the quality line closes the record: only complete records count
                seqs.push_back(rec_seq);
                if (seqs.size() >= max_spots) stop = true;
            }
            ++line_no;
            pos = nl + 1;
        }
        pending.erase(0, pos);
    };
    size_t n;
    bool first = true;
    while (!stop && (n = std::fread(in.data(), 1, in.size(), fp)) > 0) {
        bytes += static_cast<int64_t>(n);
        if (first) {
            first = false;
            gz = n >= 2 && in[0] == 0x1f && in[1] == 0x8b;
            if (gz) { inflateInit2(&z, 31); inited = true; }
        }
        if (!gz) { feed(reinterpret_cast<const char*>(in.data()), n); continue; }
        z.next_in = in.data();
        z.avail_in = static_cast<uInt>(n);
        while (z.avail_in > 0 && !stop) {
            if (ended) { inflateReset(&z); ended = false; }   // next gzip member
            z.next_out = out.data();
            z.avail_out = static_cast<uInt>(out.size());
            const int rc = inflate(&z, Z_NO_FLUSH);
            const size_t got = out.size() - z.avail_out;
            if (got) feed(reinterpret_cast<const char*>(out.data()), got);
            if (rc == Z_STREAM_END) ended = true;
            else if (rc != Z_OK && rc != Z_BUF_ERROR) { stop = true; break; }
            else if (rc == Z_BUF_ERROR && got == 0) break;
        }
    }
    if (inited) inflateEnd(&z);
    if (piped) pclose(fp); else std::fclose(fp);
    if (seqs.empty()) err = "no FASTQ records in " + loc;
    return !seqs.empty();
}

}  // namespace cli_detail

/// Heads of several FASTQ segments, joined into spots.
inline bool head_fastq_segments(const std::vector<std::string>& locs, size_t n, std::vector<HeadSpot>& out, SourceInfo& info) {
    std::vector<std::vector<std::string>> cols(locs.size());
    if (info.bytes < 0) info.bytes = 0;
    for (size_t j = 0; j < locs.size(); ++j) {
        std::string err;
        if (!cli_detail::fastq_head(locs[j], n, cols[j], info.bytes, err)) { info.error = err; return false; }
        info.endpoint += (j ? "," : "") + cli_detail::host_of(locs[j]);
    }
    size_t m = SIZE_MAX;
    for (auto& c : cols) m = std::min(m, c.size());
    for (size_t i = 0; i < m; ++i) {
        HeadSpot s;
        for (auto& c : cols) s.seg.push_back(std::move(c[i]));
        out.push_back(std::move(s));
    }
    return true;
}

inline bool head_bam(const std::string& path, size_t n, std::vector<HeadSpot>& out, SourceInfo& info) {
    const std::string cmd = "samtools view -F 0x900 " + cli_detail::sq(path);
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) { info.error = "cannot run samtools"; return false; }
    std::string line;
    char buf[1 << 16];
    while (out.size() < n && std::fgets(buf, sizeof(buf), fp)) {
        line = buf;
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        const auto f = cli_detail::split(line, '\t');
        if (f.size() < 11) continue;
        std::string cb, ub;
        for (size_t k = 11; k < f.size(); ++k) {
            if (f[k].rfind("CR:Z:", 0) == 0 || (cb.empty() && f[k].rfind("CB:Z:", 0) == 0) || f[k].rfind("CQ:Z:", 0) == 0) { if (f[k].rfind("CQ:Z:", 0) != 0) cb = f[k].substr(5); }
            if (f[k].rfind("UR:Z:", 0) == 0 || (ub.empty() && f[k].rfind("UB:Z:", 0) == 0)) ub = f[k].substr(5);
        }
        std::string seq = f[9];
        if (std::atoi(f[1].c_str()) & 16) seq = detail_id::revcomp(seq);
        HeadSpot s;
        s.seg.push_back(cb + ub);
        s.seg.push_back(seq);
        out.push_back(std::move(s));
    }
    pclose(fp);
    info.endpoint = "local BAM";
    if (out.empty()) info.error = "no records in BAM";
    return !out.empty();
}

inline bool head_1fq(const std::string& path, size_t n, std::vector<HeadSpot>& out, SourceInfo& info) {
    try {
        Reader rd;
        rd.open(path.c_str());
        info.run_spots = static_cast<int64_t>(rd.total_reads());
        DecodedBlock blk;
        auto asc = [](const uint8_t* p, uint32_t len) {
            std::string s(len, 'N');
            for (uint32_t i = 0; i < len; ++i) s[i] = "ACGTN"[p[i] > 4 ? 4 : p[i]];
            return s;
        };
        while (out.size() < n && rd.read_block(blk))
            for (uint32_t i = 0; i < blk.n_reads && out.size() < n; ++i) {
                HeadSpot s;
                s.seg.push_back(asc(blk.r1_seq(i), blk.r1_len(i)));
                if (blk.r2_len(i)) s.seg.push_back(asc(blk.r2_seq(i), blk.r2_len(i)));
                out.push_back(std::move(s));
            }
    } catch (const std::exception& e) { info.error = e.what(); return false; }
    info.endpoint = "local .1fq";
    return !out.empty();
}

inline bool head_sra(const std::string& acc, size_t n, std::vector<HeadSpot>& out, SourceInfo& info) {
    try {
        SraReader rd;
        rd.open(acc.c_str());
        info.run_spots = rd.total_spots();
        rd.set_row_limit(rd.first_row() + static_cast<int64_t>(n));
        SraReader::RawSpot sp;
        while (rd.read_next_spot(sp)) {
            HeadSpot s;
            for (uint32_t k = 0; k < sp.n_segments; ++k) s.seg.emplace_back(sp.read + sp.seg_start[k], sp.seg_len[k]);
            out.push_back(std::move(s));
        }
    } catch (const std::exception& e) { info.error = e.what(); return false; }
    info.endpoint = "NCBI SRA archive (libncbi-vdb block access)";
    info.bytes_note = "bytes fetched are not exposed by the archive reader";
    return !out.empty();
}

/// ENA's FASTQ links for a run, from the filereport API; also the run's read count.
inline bool ena_urls(const std::string& acc, std::vector<std::string>& urls, int64_t& run_spots) {
    const std::string cmd = "curl -sf --max-time 60 'https://www.ebi.ac.uk/ena/portal/api/filereport?accession=" + acc +
                            "&result=read_run&fields=fastq_ftp,read_count&format=tsv'";
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) return false;
    std::string all;
    char b[4096];
    while (std::fgets(b, sizeof(b), fp)) all += b;
    pclose(fp);
    const auto lines = cli_detail::split(all, '\n');
    if (lines.size() < 2) return false;
    const auto h = cli_detail::split(lines[0], '\t'), v = cli_detail::split(lines[1], '\t');
    int ci = -1, ri = -1;
    for (size_t i = 0; i < h.size(); ++i) { if (h[i] == "fastq_ftp") ci = (int)i; if (h[i] == "read_count") ri = (int)i; }
    if (ci < 0 || (size_t)ci >= v.size() || v[ci].empty()) return false;
    for (auto u : cli_detail::split(v[ci], ';')) urls.push_back("https://" + u);
    if (ri >= 0 && (size_t)ri < v.size()) run_spots = std::atoll(v[ri].c_str());
    // _1 and _2 in order; a run with a single unsuffixed file stays as is
    std::sort(urls.begin(), urls.end());
    return !urls.empty();
}

// ── BAM head by HTTP range ──

/// What the BAM route needs from the caller: the barcode lists and options (for the list rates of the barcode tags) and where to put the evidence.
struct BamCtx {
    std::vector<ScreenWhitelist>* lists = nullptr;
    const Options* opt = nullptr;
    bam::Evidence* ev = nullptr;
};

namespace cli_detail {

struct RangeGet {
    bool ok = false;
    long status = 0;
    int64_t total = -1;       // object size from Content-Range
    std::vector<uint8_t> data;
    std::string err;
};

/// One ranged GET of len bytes at `from`. The transfer is cut at len bytes by the pipe whatever the server does with the Range header.
inline RangeGet range_get(const std::string& url, int64_t from, int64_t len) {
    RangeGet g;
    static int counter = 0;
    std::error_code ec;
    const std::string hdr = (std::filesystem::temp_directory_path(ec) / ("identify-hdr-" + std::to_string(::getpid()) + "-" + std::to_string(++counter))).string();
    const std::string cmd = "curl -sL --max-time 240 -r " + std::to_string(from) + "-" + std::to_string(from + len - 1) + " -D " + sq(hdr) + " " + sq(url) + " 2>/dev/null | head -c " + std::to_string(len);
    FILE* fp = popen(cmd.c_str(), "r");
    if (!fp) { g.err = "cannot run curl"; return g; }
    uint8_t b[1 << 16];
    size_t n;
    while ((n = std::fread(b, 1, sizeof(b), fp)) > 0) g.data.insert(g.data.end(), b, b + n);
    pclose(fp);
    std::ifstream h(hdr);
    std::string line;
    while (std::getline(h, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        std::string low = line;
        for (auto& c : low) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        if (low.rfind("http/", 0) == 0) { const size_t sp = low.find(' '); g.status = sp == std::string::npos ? 0 : std::atol(low.c_str() + sp + 1); g.total = -1; }
        else if (low.rfind("content-range:", 0) == 0) { const size_t sl = low.rfind('/'); if (sl != std::string::npos) g.total = std::atoll(low.c_str() + sl + 1); }
    }
    std::filesystem::remove(hdr, ec);
    if (g.data.empty() || g.status >= 400) { g.err = g.status ? "HTTP " + std::to_string(g.status) + " from " + host_of(url) : "no answer from " + host_of(url); return g; }
    if (g.status == 206 || (g.status == 200 && from == 0)) g.ok = true;
    else g.err = "HTTP " + std::to_string(g.status) + " for a range request on " + host_of(url) + (g.status == 200 ? " (the server ignored the range)" : "");
    return g;
}

inline bool looks_like_bam_url(const std::string& u) {
    if (!is_url(u)) return false;
    const size_t p = u.find(".bam");
    if (p == std::string::npos) return false;
    const std::string rest = u.substr(p + 4);   // "", "?query" or ".1" (the archive's copy suffix); ".bai" is an index, not a BAM
    if (rest.empty() || rest[0] == '?') return true;
    if (rest[0] != '.' || rest.size() < 2) return false;
    for (size_t i = 1; i < rest.size(); ++i) if (!std::isdigit(static_cast<unsigned char>(rest[i]))) return false;
    return true;
}

}  // namespace cli_detail

/// Barcode list rates of the BAM's barcode tag (raw CR, or corrected CB when CR is absent), over the primary records of every sample.
inline void bam_list_rates(bam::Evidence& ev, const std::vector<bam::Rec>& recs, std::vector<ScreenWhitelist>& lists, const Options& opt) {
    const bool use_cr = ev.barcode_tags.rfind("CR", 0) == 0;
    ev.list_rate_tag = use_cr ? "CR" : (ev.barcode_tags.rfind("CB", 0) == 0 ? "CB" : "");
    ev.list_rates.clear();
    if (ev.list_rate_tag.empty()) return;
    const size_t limit = 50000, stride = recs.size() > limit ? recs.size() / limit : 1;
    for (auto& wl : lists) {
        if (!wl.loaded()) detail_id::load_list(wl, opt.whitelist_dirs);
        bam::Evidence::ListRate lr;
        lr.file = wl.file;
        lr.family = wl.family;
        if (wl.loaded()) {
            uint64_t n = 0, ex = 0, m1 = 0;
            for (size_t i = 0; i < recs.size(); i += stride) {
                const std::string& bc = use_cr ? recs[i].cr : recs[i].cb;
                if (bc.size() < wl.bc_len) continue;
                ++n;
                uint64_t h;
                if (!detail_id::pack(bc.data(), wl.bc_len, h)) continue;
                const bool hit = screen_wl_has(wl, h);
                ex += hit;
                m1 += hit || screen_wl_match1(wl, h);
            }
            if (n) { lr.exact = double(ex) / n; lr.one_mm = double(m1) / n; }
        }
        ev.list_rates.push_back(lr);
    }
}

/// The head of a BAM by HTTP range: the first 4 MB (header and first records), then three 1 MB samples inside the file and its last MB, so that the
/// statistics do not rest on one genomic region. The BAM is never downloaded whole. The spots carry the barcode tag and UMI tag as segment 0
/// and the read, in its original orientation, as segment 1.
inline bool head_bam_ranged(const std::string& url, size_t n, std::vector<HeadSpot>& out, SourceInfo& info, BamCtx* ctx) {
    constexpr int64_t kHead = 4ll << 20, kSample = 1ll << 20;
    constexpr int kInterior = 3;
    bam::Evidence local;
    bam::Evidence& ev = ctx && ctx->ev ? *ctx->ev : local;
    ev = bam::Evidence();
    ev.url = url;
    const double t0 = cli_detail::now_s();
    std::vector<bam::Rec> recs;
    auto fail = [&](const std::string& e) {
        ev.error = e;
        ev.seconds = cli_detail::now_s() - t0;
        info.error = e;
        info.bytes = static_cast<int64_t>(ev.bytes);
        info.seconds = ev.seconds;
        info.endpoint = cli_detail::host_of(url);
        return false;
    };
    auto g = cli_detail::range_get(url, 0, kHead);
    ++ev.requests;
    if (!g.ok) return fail(g.err);
    ev.bytes += g.data.size();
    ev.file_size = g.total;
    if (!bam::ingest_head(ev, g.data.data(), g.data.size(), recs)) return fail(ev.error.empty() ? "not a BAM" : ev.error);
    if (ev.file_size > 8 * kHead) {
        std::vector<std::pair<std::string, int64_t>> places;
        for (int k = 1; k <= kInterior; ++k) places.push_back({"interior", (ev.file_size / (kInterior + 1)) * k});
        places.push_back({"tail", ev.file_size - kSample});
        for (const auto& sp : places) {
            usleep(300000);   // a few requests per second at most
            auto s = cli_detail::range_get(url, sp.second, kSample);
            ++ev.requests;
            if (!s.ok) continue;
            ev.bytes += s.data.size();
            bam::ingest_sample(ev, sp.first, static_cast<uint64_t>(sp.second), s.data.data(), s.data.size(), recs);
        }
    }
    bam::finalize(ev);
    ev.seconds = cli_detail::now_s() - t0;
    info.endpoint = cli_detail::host_of(url);
    info.bytes = static_cast<int64_t>(ev.bytes);
    info.seconds = ev.seconds;
    if (ev.file_size >= 0) info.bytes_note = std::to_string(ev.requests) + " ranged GETs on a " + std::to_string(ev.file_size) + " byte object";
    if (ctx && ctx->lists && ctx->opt) bam_list_rates(ev, recs, *ctx->lists, *ctx->opt);
    if (!ev.usable_barcodes) return fail("the BAM carries no CR or CB tag on half of its records (bam.tags_seen lists the tags)");
    const bool raw = ev.barcode_tags.rfind("CR", 0) == 0;
    const size_t stride = recs.size() > n ? recs.size() / n + 1 : 1;
    for (size_t i = 0; i < recs.size(); i += stride) {
        const auto& r = recs[i];
        const std::string& bc = raw ? r.cr : r.cb;
        if (bc.empty()) continue;
        std::string seq = r.seq;
        if (r.flag & 16) seq = bam::revcomp(seq);
        HeadSpot s;
        s.seg.push_back(bc + (raw ? r.ur : r.ub));
        s.seg.push_back(std::move(seq));
        out.push_back(std::move(s));
    }
    if (out.empty()) return fail("no record with a barcode tag in the sampled blocks");
    return true;
}

/// The head of one run (all segments), by the route the source names.
inline bool head_run(const std::string& src, const std::string& route, size_t n, std::vector<HeadSpot>& out, SourceInfo& info, BamCtx* bctx = nullptr) {
    if (src.size() > 4 && src.compare(src.size() - 4, 4, ".1fq") == 0) { info.route = "1fq"; return head_1fq(src, n, out, info); }
    if (src.rfind("bam:", 0) == 0) {
        const std::string rest = src.substr(4);
        if (cli_detail::is_url(rest)) { info.route = "bam_url"; return head_bam_ranged(rest, n, out, info, bctx); }
        info.route = "bam";
        return head_bam(rest, n, out, info);
    }
    if (cli_detail::looks_like_bam_url(src)) { info.route = "bam_url"; return head_bam_ranged(src, n, out, info, bctx); }
    if (cli_detail::is_accession(src)) {
        if (route == "ena") {
            info.route = "ena";
            std::vector<std::string> urls;
            if (!ena_urls(src, urls, info.run_spots)) { info.error = "ENA lists no FASTQ for " + src; return false; }
            return head_fastq_segments(urls, n, out, info);
        }
        info.route = "sra";
        return head_sra(src, n, out, info);
    }
    const auto locs = cli_detail::split(src, ',');
    info.route = cli_detail::is_url(locs[0]) ? "url" : "fastq";
    return head_fastq_segments(locs, n, out, info);
}

/// Layer D: align the head's cDNA reads with the bundled STAR (this binary's `star` subcommand) and read the strand and gene shares.
inline void map_head(const std::vector<HeadSpot>& spots, int seg, int seg2, size_t nreads, const std::string& genome_dir, int threads,
                     const std::string& self_exe, MapEvidence& m) {
    namespace fs = std::filesystem;
    if (seg < 0) { m.note = "no cDNA segment"; return; }
    const double t0 = cli_detail::now_s();
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec) / ("identify-" + std::to_string(::getpid()));
    fs::create_directories(base, ec);
    const std::string fq = (base / "head.fq").string(), fq2 = (base / "head2.fq").string(), pfx = (base / "star.").string();
    size_t n = 0;
    const bool pe = seg2 >= 0;   // two insert reads without a barcode read: map them as a pair for the insert size
    {
        std::ofstream o(fq), o2;
        if (pe) o2.open(fq2);
        for (const auto& s : spots) {
            if (n >= nreads) break;
            if ((size_t)seg >= s.seg.size() || s.seg[seg].size() < 30) continue;
            if (pe && ((size_t)seg2 >= s.seg.size() || s.seg[seg2].size() < 30)) continue;
            const std::string q = s.seg[seg].substr(0, 100);
            o << "@r" << n << "\n" << q << "\n+\n" << std::string(q.size(), 'I') << "\n";
            if (pe) {
                const std::string q2 = s.seg[seg2].substr(0, 100);
                o2 << "@r" << n << "\n" << q2 << "\n+\n" << std::string(q2.size(), 'I') << "\n";
            }
            ++n;
        }
    }
    if (n < 1000) { m.note = "fewer than 1000 usable cDNA reads"; fs::remove_all(base, ec); return; }
    const std::string cmd = cli_detail::sq(self_exe) + " star --runThreadN " + std::to_string(threads) + " --genomeDir " + cli_detail::sq(genome_dir) +
                            " --readFilesIn " + cli_detail::sq(fq) + (pe ? " " + cli_detail::sq(fq2) : std::string()) + " --outFileNamePrefix " + cli_detail::sq(pfx) +
                            " --outSAMtype SAM --outSAMmode NoQS --outSAMunmapped None --outFilterMultimapNmax 10 --quantMode GeneCounts > " +
                            cli_detail::sq((base / "star.stdout").string()) + " 2>&1";
    const int rc = std::system(cmd.c_str());
    static std::map<std::string, TxAnnot> annots;   // one transcript table per index, kept across the samples of a batch
    auto it = annots.find(genome_dir);
    if (it == annots.end()) { TxAnnot a; a.load(genome_dir); it = annots.emplace(genome_dir, std::move(a)).first; }
    m = parse_star_outputs(pfx, n, &it->second);
    m.paired = pe && m.paired;
    if (rc != 0 && !m.measured) m.note = "STAR failed (exit " + std::to_string(rc) + ")";
    m.seconds = cli_detail::now_s() - t0;
    fs::remove_all(base, ec);
}

inline std::string self_exe_path() {
    char buf[4096];
    const ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return len > 0 ? std::string(buf, static_cast<size_t>(len)) : std::string("singlet");
}

// â”€â”€ endpoints: where a run's reads can be read from, and which of them carry a barcode read â”€â”€

namespace cli_detail {

/// Minimal JSON reader for the SDL locate answer: strings, numbers, booleans, null, arrays, objects.
struct J {
    enum T { Null, Str, Num, Bool, Arr, Obj } t = Null;
    std::string s;
    double n = 0;
    std::vector<J> a;
    std::vector<std::pair<std::string, J>> o;
    const J* get(const std::string& k) const {
        for (const auto& kv : o) if (kv.first == k) return &kv.second;
        return nullptr;
    }
};
inline void ws(const std::string& s, size_t& i) { while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i; }
inline J parse_json(const std::string& s, size_t& i) {
    J v;
    ws(s, i);
    if (i >= s.size()) return v;
    if (s[i] == '"') {
        v.t = J::Str;
        ++i;
        while (i < s.size() && s[i] != '"') {
            if (s[i] == '\\' && i + 1 < s.size()) { ++i; v.s += s[i] == 'n' ? '\n' : s[i]; }
            else v.s += s[i];
            ++i;
        }
        ++i;
    } else if (s[i] == '{') {
        v.t = J::Obj;
        ++i;
        for (;;) {
            ws(s, i);
            if (i >= s.size() || s[i] == '}') { ++i; break; }
            if (s[i] == ',') { ++i; continue; }
            J k = parse_json(s, i);
            ws(s, i);
            if (i < s.size() && s[i] == ':') ++i;
            v.o.emplace_back(k.s, parse_json(s, i));
        }
    } else if (s[i] == '[') {
        v.t = J::Arr;
        ++i;
        for (;;) {
            ws(s, i);
            if (i >= s.size() || s[i] == ']') { ++i; break; }
            if (s[i] == ',') { ++i; continue; }
            v.a.push_back(parse_json(s, i));
        }
    } else {
        size_t j = i;
        while (j < s.size() && s[j] != ',' && s[j] != '}' && s[j] != ']') ++j;
        const std::string tok = s.substr(i, j - i);
        i = j;
        if (tok == "true" || tok == "false") { v.t = J::Bool; v.n = tok == "true"; }
        else if (tok != "null") { v.t = J::Num; v.n = std::atof(tok.c_str()); }
    }
    return v;
}
inline std::string curl_text(const std::string& url, int max_s = 60) {
    FILE* fp = popen(("curl -sfL --max-time " + std::to_string(max_s) + " " + sq(url)).c_str(), "r");
    std::string all;
    if (!fp) return all;
    char b[8192];
    size_t n;
    while ((n = std::fread(b, 1, sizeof(b), fp)) > 0) all.append(b, n);
    pclose(fp);
    return all;
}

}  // namespace cli_detail

/// ENA submitted_ftp files of a run (the files as the submitter uploaded them), as https links.
inline std::vector<std::string> ena_submitted(const std::string& acc) {
    std::vector<std::string> urls;
    const std::string all = cli_detail::curl_text("https://www.ebi.ac.uk/ena/portal/api/filereport?accession=" + acc + "&result=read_run&fields=submitted_ftp&format=tsv");
    const auto lines = cli_detail::split(all, '\n');
    if (lines.size() < 2) return urls;
    const auto h = cli_detail::split(lines[0], '\t'), v = cli_detail::split(lines[1], '\t');
    for (size_t i = 0; i < h.size() && i < v.size(); ++i)
        if (h[i] == "submitted_ftp")
            for (auto u : cli_detail::split(v[i], ';'))
                if (u.size() > 4) urls.push_back("https://" + u);
    return urls;
}

/// Original-format objects of a run on NCBI's cloud copy (what the submitter deposited, before the archive normalized it), from the SDL locate service.
struct OriginalFile { std::string type, name, link; double size = 0; };
inline std::vector<OriginalFile> sdl_original_files(const std::string& acc) {
    std::vector<OriginalFile> out;
    const std::string txt = cli_detail::curl_text("https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=" + acc + "&location=s3.us-east-1&accept-charges=aws");
    size_t i = 0;
    const cli_detail::J j = cli_detail::parse_json(txt, i);
    const auto* res = j.get("result");
    if (!res) return out;
    for (const auto& r : res->a) {
        const auto* files = r.get("files");
        if (!files) continue;
        for (const auto& f : files->a) {
            const auto* ty = f.get("type");
            const auto* nm = f.get("name");
            const auto* loc = f.get("locations");
            if (!ty || !nm || ty->s == "sra" || !loc) continue;
            OriginalFile of;
            of.type = ty->s;
            of.name = nm->s;
            if (const auto* sz = f.get("size")) of.size = sz->n;
            for (const auto& l : loc->a)
                if (const auto* lk = l.get("link")) { of.link = lk->s; break; }
            if (!of.link.empty()) out.push_back(of);
        }
    }
    return out;
}

/// One endpoint's head, measured on its own: layout, whether a barcode read is present, and what the reads call.
inline EndpointRec evaluate_endpoint(const std::string& name, bool reachable, const std::vector<HeadSpot>& spots, const SourceInfo& info,
                                     std::vector<ScreenWhitelist>& lists, const Options& opt, const std::shared_ptr<bam::Evidence>& bam_ev = nullptr,
                                     const std::string& url = std::string()) {
    EndpointRec e;
    e.name = name;
    e.url = url;
    e.barcode_source = "reads";
    if (bam_ev) {
        e.bam = bam_ev;
        e.tags = bam_ev->present ? bam_ev->barcode_tags : std::string();
        e.barcode_source = bam_ev->usable_barcodes ? "bam_tags" : "";
    }
    e.reachable = reachable && !spots.empty();
    e.bytes = info.bytes;
    e.seconds = info.seconds;
    e.note = info.error;
    if (!e.reachable) return e;
    Report r;
    measure(spots, opt, lists, r.ev);
    if (bam_ev && bam_ev->present) apply_bam_evidence(r.ev, *bam_ev);
    rank(r, opt);
    for (const auto& s : r.ev.segs) e.seg_lens.push_back(s.len_mode);
    double best = 0;
    for (const auto& h : r.ev.lists) best = std::max(best, h.one_mm);
    e.list_rate = best;
    int nonindex = 0;
    for (const auto& s : r.ev.segs) nonindex += !s.index_like;
    // a barcode read needs a read of its own: a list hit, or a barcode window, repeated prefix or poly(T) geometry on a run with an insert read besides it
    e.barcode_read = best >= kThr.wl_hit || (nonindex >= 2 && ((r.ev.window.found && !r.ev.window_on_index) || r.ev.prefix_mult >= kThr.prefix_mult || r.ev.polyt_seg >= 0));
    e.verdict = r.verdict;
    e.protocol = r.verdict_protocol;
    return e;
}

/// Survey every endpoint of an accession. Heads are read with the same spot limit as the main head.
struct BamBest {
    std::vector<HeadSpot> spots;
    std::string name, url;
    std::shared_ptr<bam::Evidence> ev;
};

inline std::vector<EndpointRec> survey_endpoints(const std::string& acc, size_t n, std::vector<ScreenWhitelist>& lists, const Options& opt,
                                                 std::vector<HeadSpot>* best = nullptr, std::string* best_name = nullptr, BamBest* bam_best = nullptr) {
    std::vector<EndpointRec> out;
    auto push_bam = [&](const std::string& nm, const std::string& url, const std::vector<HeadSpot>& sp, const SourceInfo& info, const std::shared_ptr<bam::Evidence>& ev) {
        out.push_back(evaluate_endpoint(nm, !sp.empty(), sp, info, lists, opt, ev, url));
        if (!ev->present && !info.error.empty() && out.back().note.empty()) out.back().note = info.error;
        if (best && best_name && best->empty() && out.back().barcode_read) { *best = sp; *best_name = nm; }
        if (bam_best && out.back().barcode_read && ev->usable_barcodes && (bam_best->spots.empty() || ev->seconds < bam_best->ev->seconds)) { bam_best->spots = sp; bam_best->name = nm; bam_best->url = url; bam_best->ev = ev; }
    };
    // the first endpoint that carries a barcode read keeps its head, for runs whose normalized copy is empty
    auto push_ep = [&](const std::string& nm, const std::vector<HeadSpot>& sp, const SourceInfo& info) {
        out.push_back(evaluate_endpoint(nm, !sp.empty(), sp, info, lists, opt));
        if (best && best_name && best->empty() && out.back().barcode_read) { *best = sp; *best_name = nm; }
    };
    {   // SRA normalized
        SourceInfo info;
        std::vector<HeadSpot> sp;
        const double t0 = cli_detail::now_s();
        head_sra(acc, n, sp, info);
        info.seconds = cli_detail::now_s() - t0;
        push_ep("sra_normalized", sp, info);
    }
    {   // ENA fastq_ftp
        SourceInfo info;
        std::vector<HeadSpot> sp;
        std::vector<std::string> urls;
        int64_t rs = -1;
        const double t0 = cli_detail::now_s();
        if (ena_urls(acc, urls, rs)) head_fastq_segments(urls, n, sp, info);
        else info.error = "ENA lists no FASTQ";
        info.seconds = cli_detail::now_s() - t0;
        push_ep("ena_fastq_ftp", sp, info);
    }
    {   // ENA submitted_ftp
        SourceInfo info;
        std::vector<HeadSpot> sp;
        const auto urls = ena_submitted(acc);
        const double t0 = cli_detail::now_s();
        bool was_bam = false;
        if (urls.empty()) info.error = "ENA lists no submitted files";
        else if (std::any_of(urls.begin(), urls.end(), [](const std::string& u) { return cli_detail::looks_like_bam_url(u); })) {
            const std::string bu = *std::find_if(urls.begin(), urls.end(), [](const std::string& u) { return cli_detail::looks_like_bam_url(u); });
            auto ev = std::make_shared<bam::Evidence>();
            BamCtx bc{&lists, &opt, ev.get()};
            head_bam_ranged(bu, n, sp, info, &bc);
            push_bam("ena_submitted_ftp", bu, sp, info, ev);
            was_bam = true;
        } else { std::vector<std::string> g; for (const auto& u : urls) if (u.size() < 4 || u.compare(u.size() - 4, 4, ".bai") != 0) g.push_back(u); std::sort(g.begin(), g.end()); if (g.size() > 4) g.resize(4); head_fastq_segments(g, n, sp, info); }
        if (!was_bam) {
            info.seconds = cli_detail::now_s() - t0;
            push_ep("ena_submitted_ftp", sp, info);
        }
    }
    {   // SRA original-format objects (SDL locate service)
        const auto files = sdl_original_files(acc);
        if (files.empty()) {
            EndpointRec e;
            e.name = "sra_original";
            e.note = "SDL lists no original-format object for this run";
            out.push_back(e);
        }
        std::vector<OriginalFile> fq;
        for (const auto& f : files) {
            const bool bam = f.name.size() > 4 && f.name.compare(f.name.size() - 4, 4, ".bam") == 0;
            if (bam) {
                SourceInfo info;
                std::vector<HeadSpot> sp;
                auto ev = std::make_shared<bam::Evidence>();
                BamCtx bc{&lists, &opt, ev.get()};
                head_bam_ranged(f.link, n, sp, info, &bc);
                push_bam("sra_original:" + f.name, f.link, sp, info, ev);
                out.back().note += (out.back().note.empty() ? "" : "; ") + std::string("BAM, ") + std::to_string(static_cast<int64_t>(f.size)) + " bytes";
            } else if (f.name.find(".fastq") != std::string::npos || f.name.find(".fq") != std::string::npos) fq.push_back(f);
        }
        if (!fq.empty()) {   // one lane or file set: names sorted, at most four files, one segment each
            std::sort(fq.begin(), fq.end(), [](const OriginalFile& a, const OriginalFile& b) { return a.name < b.name; });
            std::vector<std::string> urls;
            for (size_t k = 0; k < fq.size() && k < 4; ++k) urls.push_back(fq[k].link);
            SourceInfo info;
            std::vector<HeadSpot> sp;
            const double t0 = cli_detail::now_s();
            head_fastq_segments(urls, n, sp, info);
            info.seconds = cli_detail::now_s() - t0;
            push_ep("sra_original:fastq(" + std::to_string(fq.size()) + " files)", sp, info);
        }
    }
    return out;
}

inline bool& want_endpoints() { static bool b = false; return b; }
inline std::string& dump_dir() { static std::string d; return d; }   // --dump DIR: the first 40 spots of each source, for layout diagnosis

inline Report identify_source(const std::string& source, const std::string& hint_in, const std::string& route, size_t n_reads,
                              std::vector<ScreenWhitelist>& lists, Options opt, const std::string& genome_dir, size_t map_reads, int threads) {
    Report R;
    R.src.source = source;
    std::vector<HeadSpot> spots;
    const double t0 = cli_detail::now_s();
    int64_t run_total = 0;
    bool any = false;
    BamCtx bctx{&lists, &opt, &R.bam};
    for (const auto& run : cli_detail::split(source, ';')) {
        if (run.empty()) continue;
        SourceInfo one;
        const bool ok = head_run(run, route, n_reads, spots, one, &bctx);
        if (R.src.route.empty()) R.src.route = one.route;
        R.src.endpoint += (R.src.endpoint.empty() ? "" : "; ") + one.endpoint;
        if (one.bytes >= 0) R.src.bytes = std::max<int64_t>(R.src.bytes, 0) + one.bytes;
        if (!one.bytes_note.empty()) R.src.bytes_note = one.bytes_note;
        if (one.run_spots >= 0) run_total += one.run_spots;
        if (!ok) R.src.error += (R.src.error.empty() ? "" : "; ") + one.error;
        any = any || ok;
    }
    R.src.seconds = cli_detail::now_s() - t0;
    R.src.spots = spots.size();
    R.spots = spots.size();
    R.src.run_spots = run_total > 0 ? run_total : -1;
    opt.hint = hint_in;
    opt.run_spots = R.src.run_spots;
    if ((!any || spots.empty()) && want_endpoints() && cli_detail::is_accession(source)) {
        // the normalized run gave nothing: take the head from another endpoint that carries a barcode read
        std::vector<HeadSpot> bs;
        std::string bn;
        BamBest bb;
        auto eps = survey_endpoints(source, n_reads, lists, opt, &bs, &bn, &bb);
        R.endpoints = eps;
        bool bn_is_bam = false;
        for (const auto& e : eps) if (e.name == bn && e.bam) bn_is_bam = true;
        if (bn_is_bam && bb.ev) { bs = bb.spots; bn = bb.name; }   // two copies of one BAM: the quicker one
        if (!bs.empty()) {
            spots = std::move(bs);
            R.src.route = "endpoint:" + bn;
            R.src.endpoint = bn;
            if (bb.ev && bb.name == bn) {   // the head is a BAM's: its tags are the barcode source
                R.bam = *bb.ev;
                R.barcode_source = "bam_tags";
                R.src.endpoint = bb.url;
                R.src.bytes = static_cast<int64_t>(bb.ev->bytes);
                R.src.seconds = bb.ev->seconds;
            }
            R.src.spots = spots.size();
            R.spots = spots.size();
            R.src.error += std::string(R.src.error.empty() ? "" : "; ") + "normalized run gave no reads; the head comes from " + bn;
            any = true;
        }
    }
    if (!any || spots.empty()) {
        Interp u; u.family = "unknown"; u.protocol = "none"; u.modality = "none"; u.support = 0;
        u.reasons.push_back("no reads could be read: " + R.src.error);
        R.ranked.push_back(u);
        R.verdict = "unknown"; R.verdict_protocol = "none";
        R.action = "refuse"; R.action_note = "head could not be read";
        return R;
    }
    if (!dump_dir().empty()) {
        std::string id = source.substr(0, 40);
        for (auto& c : id) if (!std::isalnum(static_cast<unsigned char>(c))) c = '_';
        std::ofstream d(dump_dir() + "/" + id + ".spots.txt");
        for (size_t i = 0; i < std::min<size_t>(spots.size(), 40); ++i) {
            for (size_t k = 0; k < spots[i].seg.size(); ++k) d << (k ? "|" : "") << spots[i].seg[k];
            d << std::endl;
        }
    }
    if (R.bam.present && R.bam.usable_barcodes) R.barcode_source = "bam_tags";
    measure(spots, opt, lists, R.ev);
    if (R.bam.present) apply_bam_evidence(R.ev, R.bam);
    if (!genome_dir.empty()) map_head(spots, R.ev.cdna_seg, R.ev.mate_seg, map_reads, genome_dir, threads, self_exe_path(), R.ev.map);
    rank(R, opt);
    attach_species(R, spots, opt);
    if (want_endpoints() && cli_detail::is_accession(source)) {
        BamBest bb;
        if (R.endpoints.empty()) R.endpoints = survey_endpoints(source, n_reads, lists, opt, nullptr, nullptr, &bb);
        std::string via;
        for (const auto& e : R.endpoints)
            if (e.barcode_read && e.name != "sra_normalized") via += (via.empty() ? "" : ", ") + e.name;
        if (R.verdict == "barcode_read_missing" && !R.ranked.empty())
            R.ranked[0].reasons.push_back(via.empty() ? "no other endpoint of the run carries a barcode read" : "a barcode read is present in: " + via);
        // the run's own reads hold no barcode read but a BAM of the run does: classify the BAM's tags and reads, with the barcode source named
        const std::string v0 = R.verdict;
        const bool no_barcode = v0 == "barcode_read_missing" || v0 == "rna_no_barcode" || v0 == "bulk_rna" || v0 == "unknown" || v0 == "genomic_dna" || v0 == "smartseq" || v0 == "ambiguous";
        if (no_barcode && !bb.spots.empty() && bb.ev) {
            Report P;
            P.src = R.src;
            P.src.route = "endpoint:bam";
            P.src.endpoint = bb.url;
            P.src.bytes = static_cast<int64_t>(bb.ev->bytes);
            P.src.bytes_note = bb.ev->file_size >= 0 ? std::to_string(bb.ev->requests) + " ranged GETs on a " + std::to_string(bb.ev->file_size) + " byte object" : "";
            P.src.seconds = bb.ev->seconds;
            P.src.spots = bb.spots.size();
            P.spots = bb.spots.size();
            P.bam = *bb.ev;
            P.barcode_source = "bam_tags";
            measure(bb.spots, opt, lists, P.ev);
            apply_bam_evidence(P.ev, P.bam);
            if (!genome_dir.empty()) map_head(bb.spots, P.ev.cdna_seg, P.ev.mate_seg, map_reads, genome_dir, threads, self_exe_path(), P.ev.map);
            rank(P, opt);
            attach_species(P, bb.spots, opt);
            P.endpoints = R.endpoints;
            if (!P.ranked.empty())
                P.ranked[0].reasons.push_back("barcode read missing from the run's own reads (verdict there: " + v0 + "); cell barcode and UMI taken from the " + P.bam.barcode_tags +
                                              " tags of " + bb.name + ", a " + std::to_string(P.bam.file_size) + " byte BAM read by " + std::to_string(P.bam.requests) + " ranged GETs");
            R = std::move(P);
        }
    }
    return R;
}

inline int identify_main(int argc, char** argv) {
    std::vector<std::string> sources, wl_dirs;
    std::string json_out, hint, route = "auto", genome_dir, list_path, out_dir, route_status_path, species_dir, species_hint;
    size_t n_reads = 200000, map_reads = 50000;
    int threads = 8;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--reads") n_reads = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--map-reads") map_reads = std::strtoull(next().c_str(), nullptr, 10);
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--genome-dir") genome_dir = next();
        else if (a == "--whitelist-dir") wl_dirs.push_back(next());
        else if (a == "--hint") hint = next();
        else if (a == "--json") json_out = next();
        else if (a == "--route") route = next();
        else if (a == "--list") list_path = next();
        else if (a == "--dump") dump_dir() = next();
        else if (a == "--endpoints") want_endpoints() = true;
        else if (a == "--out-dir") out_dir = next();
        else if (a == "--route-status") route_status_path = next();
        else if (a == "--species-filters") species_dir = next();
        else if (a == "--species-hint") species_hint = next();
        else if (a == "-h" || a == "--help") {
            std::cout << "usage: singlet identify SOURCE [--reads N] [--genome-dir DIR] [--whitelist-dir DIR] [--hint assay=X] [--json OUT] [--route auto|sra|ena] [--route-status FILE]\n               [--species-filters DIR [--species-hint ORGANISM]]\n"
                         "       singlet identify --list samples.tsv --out-dir DIR [options]\n"
                         "SOURCE: SRR accession | URL[,URL] | FASTQ[,FASTQ] | bam:FILE | bam:URL (or a URL ending .bam: head by HTTP range) | FILE.1fq ; runs separated by ';'\n"
                         "       --endpoints  also read the run's other endpoints; a BAM with CR/UR or CB/UB tags that holds the barcode read replaces a run without one\n";
            return 0;
        } else sources.push_back(a);
    }
    if (sources.empty() && list_path.empty()) { std::cerr << "singlet identify: give a SOURCE or --list (see --help)\n"; return 2; }
    for (const auto& d : default_whitelist_dirs()) wl_dirs.push_back(d);
    Options opt;
    opt.whitelist_dirs = wl_dirs;
    opt.species_filter_dir = species_dir;
    opt.species_hint = species_hint;
    if (!route_status_path.empty()) {
        std::ifstream rs(route_status_path);
        std::stringstream buf;
        buf << rs.rdbuf();
        if (!rs || !parse_route_status(buf.str(), opt.route_status)) { std::cerr << "singlet identify: cannot read route status from " << route_status_path << "\n"; return 2; }
        opt.use_route_status = true;
    }
    auto lists = identify_lists();

    auto line_of = [](const std::string& id, const Report& r) {
        std::string top = r.ranked.empty() ? "" : r.ranked[0].family;
        std::ostringstream o;
        o << id << "\t" << r.verdict << "\t" << r.verdict_protocol << "\t" << r.protocol_tag << "\t" << r.action << "\t" << top << "\t"
          << (r.ranked.empty() ? 0.0 : r.ranked[0].support) << "\t" << r.src.spots << "\t" << r.src.route << "\t" << r.src.bytes << "\t" << r.src.seconds << "\t"
          << r.ambiguous_with << "\t" << r.src.error << "\t" << r.barcode_source << "\t" << (r.hint_used ? 1 : 0);
        return o.str();
    };
    if (!list_path.empty()) {
        std::ifstream in(list_path);
        if (!in) { std::cerr << "cannot read " << list_path << "\n"; return 1; }
        std::filesystem::create_directories(out_dir.empty() ? "." : out_dir);
        std::ofstream sum((out_dir.empty() ? std::string(".") : out_dir) + "/summary.tsv");
        sum << "id\tverdict\tprotocol\tprotocol_tag\taction\ttop_family\ttop_support\tspots\troute\tbytes\tseconds\tambiguous_with\terror\tbarcode_source\thint_used\n";
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            const auto f = cli_detail::split(line, '\t');
            if (f.size() < 2 || f[0] == "id") continue;
            const std::string row_genome = (f.size() > 3 && f[3] != "-" && !f[3].empty()) ? f[3] : genome_dir;   // optional fourth column: genome per row
            const Report r = identify_source(f[1], f.size() > 2 && !f[2].empty() ? f[2] : hint, route, n_reads, lists, opt, row_genome, map_reads, threads);
            std::ofstream j((out_dir.empty() ? std::string(".") : out_dir) + "/" + f[0] + ".json");
            j << report_json(r);
            sum << line_of(f[0], r) << "\n";
            sum.flush();
            std::cerr << "[identify] " << f[0] << ": " << r.verdict << " " << r.verdict_protocol << " -> " << r.action << "\n";
        }
        return 0;
    }
    const Report r = identify_source(sources[0], hint, route, n_reads, lists, opt, genome_dir, map_reads, threads);
    if (!json_out.empty()) { std::ofstream j(json_out); j << report_json(r); }
    std::cout << "verdict:  " << r.verdict << (r.ambiguous_with.empty() ? "" : " (" + r.ambiguous_with + ")") << "\nprotocol: " << r.verdict_protocol
              << "\naction:   " << r.action << " (" << r.action_note << ")\nspots:    " << r.src.spots << " via " << r.src.route << ", " << r.src.seconds << " s\n";
    if (!r.src.error.empty()) std::cout << "error:    " << r.src.error << "\n";
    for (size_t i = 0; i < r.ranked.size() && i < 4; ++i)
        std::cout << "  " << i + 1 << ". " << r.ranked[i].family << " / " << r.ranked[i].protocol << "  support " << r.ranked[i].support << "  usefulness "
                  << r.ranked[i].usefulness << "\n";
    if (json_out.empty()) std::cout << report_json(r);
    return 0;
}

}  // namespace singlet::fq::identify
