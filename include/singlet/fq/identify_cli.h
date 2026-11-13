// SPDX-License-Identifier: MIT
// fq/identify_cli.h — head readers and the `singlet identify` subcommand
//
//   singlet identify SOURCE [--reads N] [--genome-dir DIR] [--whitelist-dir DIR] [--hint assay=...] [--json OUT] [--route auto|sra|ena]
//   singlet identify --list samples.tsv --out-dir DIR [same options]      (rows: id TAB source [TAB hint]; lists stay loaded)
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

/// The head of one run (all segments), by the route the source names.
inline bool head_run(const std::string& src, const std::string& route, size_t n, std::vector<HeadSpot>& out, SourceInfo& info) {
    if (src.size() > 4 && src.compare(src.size() - 4, 4, ".1fq") == 0) { info.route = "1fq"; return head_1fq(src, n, out, info); }
    if (src.rfind("bam:", 0) == 0) { info.route = "bam"; return head_bam(src.substr(4), n, out, info); }
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
inline void map_head(const std::vector<HeadSpot>& spots, int seg, size_t nreads, const std::string& genome_dir, int threads,
                     const std::string& self_exe, MapEvidence& m) {
    namespace fs = std::filesystem;
    if (seg < 0) { m.note = "no cDNA segment"; return; }
    const double t0 = cli_detail::now_s();
    std::error_code ec;
    const fs::path base = fs::temp_directory_path(ec) / ("identify-" + std::to_string(::getpid()));
    fs::create_directories(base, ec);
    const std::string fq = (base / "head.fq").string(), pfx = (base / "star.").string();
    size_t n = 0;
    {
        std::ofstream o(fq);
        for (const auto& s : spots) {
            if (n >= nreads) break;
            if ((size_t)seg >= s.seg.size() || s.seg[seg].size() < 30) continue;
            const std::string q = s.seg[seg].substr(0, 100);
            o << "@r" << n << "\n" << q << "\n+\n" << std::string(q.size(), 'I') << "\n";
            ++n;
        }
    }
    if (n < 1000) { m.note = "fewer than 1000 usable cDNA reads"; fs::remove_all(base, ec); return; }
    const std::string cmd = cli_detail::sq(self_exe) + " star --runThreadN " + std::to_string(threads) + " --genomeDir " + cli_detail::sq(genome_dir) +
                            " --readFilesIn " + cli_detail::sq(fq) + " --outFileNamePrefix " + cli_detail::sq(pfx) +
                            " --outSAMtype SAM --outSAMmode NoQS --outSAMunmapped None --outFilterMultimapNmax 10 --quantMode GeneCounts > " +
                            cli_detail::sq((base / "star.stdout").string()) + " 2>&1";
    const int rc = std::system(cmd.c_str());
    m = parse_star_outputs(pfx, n);
    if (rc != 0 && !m.measured) m.note = "STAR failed (exit " + std::to_string(rc) + ")";
    m.seconds = cli_detail::now_s() - t0;
    fs::remove_all(base, ec);
}

inline std::string self_exe_path() {
    char buf[4096];
    const ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    return len > 0 ? std::string(buf, static_cast<size_t>(len)) : std::string("singlet");
}

inline Report identify_source(const std::string& source, const std::string& hint_in, const std::string& route, size_t n_reads,
                              std::vector<ScreenWhitelist>& lists, Options opt, const std::string& genome_dir, size_t map_reads, int threads) {
    Report R;
    R.src.source = source;
    std::vector<HeadSpot> spots;
    const double t0 = cli_detail::now_s();
    int64_t run_total = 0;
    bool any = false;
    for (const auto& run : cli_detail::split(source, ';')) {
        if (run.empty()) continue;
        SourceInfo one;
        const bool ok = head_run(run, route, n_reads, spots, one);
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
    if (!any || spots.empty()) {
        Interp u; u.family = "unknown"; u.protocol = "none"; u.modality = "none"; u.support = 0;
        u.reasons.push_back("no reads could be read: " + R.src.error);
        R.ranked.push_back(u);
        R.verdict = "unknown"; R.verdict_protocol = "none";
        R.action = "refuse"; R.action_note = "head could not be read";
        return R;
    }
    measure(spots, opt, lists, R.ev);
    if (!genome_dir.empty()) map_head(spots, R.ev.cdna_seg, map_reads, genome_dir, threads, self_exe_path(), R.ev.map);
    rank(R, opt);
    return R;
}

inline int identify_main(int argc, char** argv) {
    std::vector<std::string> sources, wl_dirs;
    std::string json_out, hint, route = "auto", genome_dir, list_path, out_dir;
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
        else if (a == "--out-dir") out_dir = next();
        else if (a == "-h" || a == "--help") {
            std::cout << "usage: singlet identify SOURCE [--reads N] [--genome-dir DIR] [--whitelist-dir DIR] [--hint assay=X] [--json OUT] [--route auto|sra|ena]\n"
                         "       singlet identify --list samples.tsv --out-dir DIR [options]\n"
                         "SOURCE: SRR accession | URL[,URL] | FASTQ[,FASTQ] | bam:FILE | FILE.1fq ; runs separated by ';'\n";
            return 0;
        } else sources.push_back(a);
    }
    if (sources.empty() && list_path.empty()) { std::cerr << "singlet identify: give a SOURCE or --list (see --help)\n"; return 2; }
    for (const auto& d : default_whitelist_dirs()) wl_dirs.push_back(d);
    Options opt;
    opt.whitelist_dirs = wl_dirs;
    auto lists = identify_lists();

    auto line_of = [](const std::string& id, const Report& r) {
        std::string top = r.ranked.empty() ? "" : r.ranked[0].family;
        std::ostringstream o;
        o << id << "\t" << r.verdict << "\t" << r.verdict_protocol << "\t" << r.action << "\t" << top << "\t"
          << (r.ranked.empty() ? 0.0 : r.ranked[0].support) << "\t" << r.src.spots << "\t" << r.src.route << "\t" << r.src.bytes << "\t" << r.src.seconds << "\t"
          << r.ambiguous_with << "\t" << r.src.error;
        return o.str();
    };
    if (!list_path.empty()) {
        std::ifstream in(list_path);
        if (!in) { std::cerr << "cannot read " << list_path << "\n"; return 1; }
        std::filesystem::create_directories(out_dir.empty() ? "." : out_dir);
        std::ofstream sum((out_dir.empty() ? std::string(".") : out_dir) + "/summary.tsv");
        sum << "id\tverdict\tprotocol\taction\ttop_family\ttop_support\tspots\troute\tbytes\tseconds\tambiguous_with\terror\n";
        std::string line;
        while (std::getline(in, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            if (line.empty() || line[0] == '#') continue;
            const auto f = cli_detail::split(line, '\t');
            if (f.size() < 2 || f[0] == "id") continue;
            const Report r = identify_source(f[1], f.size() > 2 ? f[2] : hint, route, n_reads, lists, opt, genome_dir, map_reads, threads);
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
