// SPDX-License-Identifier: MIT
// fq/spot_source.h — FASTQ and BAM inputs presented as SRA spots
//
// `singlet download` encodes an SRA run through SraReader, which hands the encoder one spot at a time: the concatenated bases and
// raw phred qualities of all of the spot's segments, with each segment's start and length. SpotSource builds the same spots from
// files, so FASTQ and BAM inputs reach the SRA encoder unchanged and get every protocol, orientation and whitelist rule it has.
// A run staged from FASTQ that holds the SRA run's own reads gives the .1fq that the SRA path writes.
//
// A source is a list of segments in spot order. A segment is the concatenation, record by record, of one or more streams, and a
// stream is a list of files read one after another (lanes, chunks) with an optional interleave: every k-th record, starting at
// record p (10x GemCode `read-RA` files hold Read 1 and Read 2 interleaved). A BAM source has two segments built from each primary
// record: the cell barcode and UMI tags (CR+UR with CY+UY, or CQ+UQ in Cell Ranger 1.x BAMs) and the read, reverse-complemented
// back to the sequenced strand when it aligned to the reverse strand.
//
// Spec strings (SraReader::open takes them with the prefix "fastq:" or "bam:"):
//   fastq:SEG|SEG|...        SEG    = STREAM+STREAM+...   (parts joined base by base into one segment)
//                            STREAM = FILE,FILE,...[@k/p] (files read in order; @k/p keeps records p, p+k, p+2k, ...; p from 0)
//   bam:FILE                 Cell Ranger style BAM (needs samtools on PATH)
// Gzipped files (.gz) are read through `pigz -dc` (or `gzip -dc`), other files directly.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <map>
#include <regex>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace singlet::fq {

class SpotSource {
   public:
    struct Stream {
        std::vector<std::string> files;
        uint32_t every = 1, phase = 0;
    };
    struct Segment {
        std::vector<Stream> parts;
    };

    static bool is_spec(const std::string& s) { return s.rfind("fastq:", 0) == 0 || s.rfind("bam:", 0) == 0; }

    static std::vector<std::string> split(const std::string& s, char sep) {
        std::vector<std::string> out;
        size_t p = 0;
        while (p <= s.size()) {
            size_t q = s.find(sep, p);
            if (q == std::string::npos) q = s.size();
            out.push_back(s.substr(p, q - p));
            p = q + 1;
        }
        return out;
    }

    // Parse a "fastq:" spec into segments.
    static std::vector<Segment> parse_fastq_spec(const std::string& body) {
        std::vector<Segment> segs;
        for (const auto& sg : split(body, '|')) {
            if (sg.empty()) throw std::runtime_error("fastq spec: empty segment in '" + body + "'");
            Segment seg;
            for (const auto& st : split(sg, '+')) {
                Stream s;
                std::string files = st;
                const size_t at = st.rfind('@');
                if (at != std::string::npos) {
                    files = st.substr(0, at);
                    const auto kp = split(st.substr(at + 1), '/');
                    if (kp.size() != 2) throw std::runtime_error("fastq spec: interleave must be @k/p in '" + st + "'");
                    s.every = static_cast<uint32_t>(std::strtoul(kp[0].c_str(), nullptr, 10));
                    s.phase = static_cast<uint32_t>(std::strtoul(kp[1].c_str(), nullptr, 10));
                    if (s.every == 0 || s.phase >= s.every) throw std::runtime_error("fastq spec: bad interleave in '" + st + "'");
                }
                for (const auto& f : split(files, ','))
                    if (!f.empty()) s.files.push_back(f);
                if (s.files.empty()) throw std::runtime_error("fastq spec: stream without files in '" + sg + "'");
                seg.parts.push_back(std::move(s));
            }
            segs.push_back(std::move(seg));
        }
        return segs;
    }

    static std::string to_spec(const std::vector<Segment>& segs) {
        std::string out = "fastq:";
        for (size_t i = 0; i < segs.size(); ++i) {
            if (i) out += '|';
            for (size_t j = 0; j < segs[i].parts.size(); ++j) {
                if (j) out += '+';
                const auto& s = segs[i].parts[j];
                for (size_t k = 0; k < s.files.size(); ++k) out += (k ? "," : "") + s.files[k];
                if (s.every > 1) out += "@" + std::to_string(s.every) + "/" + std::to_string(s.phase);
            }
        }
        return out;
    }

    SpotSource() = default;
    ~SpotSource() { close_all(); }
    SpotSource(const SpotSource&) = delete;
    SpotSource& operator=(const SpotSource&) = delete;

    // Open a spec. Counts the spots with one pass over the first stream (or `samtools view -c`) unless n_spots_hint > 0.
    void open(const std::string& spec, uint64_t n_spots_hint = 0) {
        spec_ = spec;
        if (spec.rfind("bam:", 0) == 0) {
            bam_ = spec.substr(4);
            if (bam_.empty()) throw std::runtime_error("bam spec without a file");
        } else if (spec.rfind("fastq:", 0) == 0) {
            segs_ = parse_fastq_spec(spec.substr(6));
        } else {
            throw std::runtime_error("not a spot spec: " + spec);
        }
        // The encoder opens several readers on one input (probe, late probe, samplers): count once per process.
        static std::mutex mu;
        static std::map<std::string, uint64_t> counted;
        {
            std::lock_guard<std::mutex> lk(mu);
            auto it = counted.find(spec);
            if (n_spots_hint == 0 && it != counted.end()) n_spots_hint = it->second;
        }
        n_spots_ = n_spots_hint > 0 ? n_spots_hint : count_spots();
        {
            std::lock_guard<std::mutex> lk(mu);
            counted[spec] = n_spots_;
        }
        rewind();
    }

    uint64_t n_spots() const { return n_spots_; }
    uint64_t next_index() const { return next_; }   // index of the spot the next read() returns
    const std::string& spec() const { return spec_; }

    // Position so that the next read() returns spot `idx` (0-based). Moving back reopens the files.
    void seek(uint64_t idx) {
        if (idx < next_) rewind();
        while (next_ < idx) {
            if (!read()) throw std::runtime_error("spot source: seek past the end (" + std::to_string(idx) + ")");
        }
    }

    // Read the next spot into the buffers below. Returns false at the end of the input.
    bool read() {
        if (!bam_.empty()) return read_bam();
        seq_.clear();
        qual_.clear();
        seg_len_.clear();
        seg_start_.clear();
        for (size_t s = 0; s < segs_.size(); ++s) {
            const int32_t start = static_cast<int32_t>(seq_.size());
            for (size_t p = 0; p < segs_[s].parts.size(); ++p) {
                if (!readers_[s][p]->next(seq_, qual_)) {
                    if (s == 0 && p == 0) return false;
                    throw std::runtime_error("spot source: segment " + std::to_string(s + 1) + " ended before segment 1 (" +
                                             std::to_string(next_) + " spots read)");
                }
            }
            seg_start_.push_back(start);
            seg_len_.push_back(static_cast<uint32_t>(seq_.size() - start));
        }
        ++next_;
        return true;
    }

    const char* seq() const { return seq_.data(); }
    const uint8_t* qual() const { return qual_.data(); }
    const uint32_t* seg_len() const { return seg_len_.data(); }
    const int32_t* seg_start() const { return seg_start_.data(); }
    uint32_t n_segments() const { return static_cast<uint32_t>(seg_len_.size()); }
    uint32_t total_bases() const { return static_cast<uint32_t>(seq_.size()); }
    uint64_t bam_skipped() const { return bam_skipped_; }

   private:
    // One stream: its files in order, every k-th FASTQ record from phase p.
    class StreamReader {
       public:
        explicit StreamReader(const Stream& s) : s_(s) {}
        ~StreamReader() { close(); }
        // Append the next kept record's bases and phred qualities. False at the end of the last file.
        bool next(std::string& seq, std::vector<uint8_t>& qual) {
            for (;;) {
                if (!fp_ && !open_next()) return false;
                if (!read_record()) {   // end of this file
                    close();
                    continue;
                }
                const bool keep = (rec_ % s_.every) == s_.phase;
                ++rec_;
                if (!keep) continue;
                const size_t n = line_seq_.size();
                if (line_qual_.size() != n) throw std::runtime_error("FASTQ record with sequence and quality of different lengths in " + cur_);
                seq.append(line_seq_);
                for (size_t i = 0; i < n; ++i) qual.push_back(static_cast<uint8_t>(line_qual_[i] - 33));
                return true;
            }
        }

       private:
        bool open_next() {
            if (file_ >= s_.files.size()) return false;
            cur_ = s_.files[file_++];
            rec_ = 0;   // the interleave restarts with each file (each file holds whole spots)
            fp_ = open_text(cur_, piped_);
            if (!fp_) throw std::runtime_error("cannot open " + cur_);
            return true;
        }
        void close() {
            if (fp_) { piped_ ? pclose(fp_) : std::fclose(fp_); fp_ = nullptr; }
        }
        bool getline(std::string& out) {
            out.clear();
            char buf[4096];
            for (;;) {
                if (!std::fgets(buf, sizeof(buf), fp_)) return !out.empty();
                out.append(buf);
                if (!out.empty() && out.back() == '\n') break;
            }
            while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
            return true;
        }
        bool read_record() {
            if (!getline(line_name_)) return false;
            if (line_name_.empty() || line_name_[0] != '@') throw std::runtime_error("FASTQ record without '@' header in " + cur_);
            if (!getline(line_seq_) || !getline(line_plus_) || !getline(line_qual_))
                throw std::runtime_error("truncated FASTQ record in " + cur_);
            return true;
        }
        const Stream& s_;
        size_t file_ = 0;
        uint64_t rec_ = 0;
        FILE* fp_ = nullptr;
        bool piped_ = false;
        std::string cur_, line_name_, line_seq_, line_plus_, line_qual_;
    };

    static std::string shell_quote(const std::string& s) {
        std::string o = "'";
        for (char c : s) o += (c == '\'') ? std::string("'\\''") : std::string(1, c);
        return o + "'";
    }
    static bool ends_with(const std::string& s, const char* suf) {
        const size_t n = std::strlen(suf);
        return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
    }
    static FILE* open_text(const std::string& path, bool& piped) {
        piped = false;
        if (ends_with(path, ".gz")) {
            piped = true;
            const std::string cmd = "(command -v pigz >/dev/null 2>&1 && pigz -dc -p 2 " + shell_quote(path) + " || gzip -dc " +
                                    shell_quote(path) + ")";
            return popen(cmd.c_str(), "r");
        }
        return std::fopen(path.c_str(), "r");
    }

    void close_all() {
        readers_.clear();
        if (bam_fp_) { pclose(bam_fp_); bam_fp_ = nullptr; }
    }
    void rewind() {
        close_all();
        next_ = 0;
        bam_skipped_ = 0;
        if (!bam_.empty()) {
            const std::string cmd = "samtools view -F 0x900 " + shell_quote(bam_);
            bam_fp_ = popen(cmd.c_str(), "r");
            if (!bam_fp_) throw std::runtime_error("cannot run samtools on " + bam_);
            return;
        }
        readers_.resize(segs_.size());
        for (size_t s = 0; s < segs_.size(); ++s)
            for (const auto& st : segs_[s].parts) readers_[s].push_back(std::make_unique<StreamReader>(st));
    }
    uint64_t count_spots() {
        if (!bam_.empty()) {   // one pass with the same filter as read_bam(), so the count is the spots the encoder gets
            rewind();
            uint64_t n = 0;
            while (read_bam()) ++n;
            return n;
        }
        // Count the first stream's kept records.
        StreamReader r(segs_.at(0).parts.at(0));
        std::string s;
        std::vector<uint8_t> q;
        uint64_t n = 0;
        while (r.next(s, q)) {
            ++n;
            s.clear();
            q.clear();
        }
        return n;
    }

    // One primary BAM record -> segment 1 = CR+UR (CY+UY or CQ+UQ), segment 2 = the read in sequencing orientation.
    bool read_bam() {
        for (;;) {
            if (!std::fgets(bam_buf_, sizeof(bam_buf_), bam_fp_)) return false;
            size_t len = std::strlen(bam_buf_);
            while (len > 0 && (bam_buf_[len - 1] == '\n' || bam_buf_[len - 1] == '\r')) bam_buf_[--len] = '\0';
            // Fields: QNAME FLAG RNAME POS MAPQ CIGAR RNEXT PNEXT TLEN SEQ QUAL TAGS...
            char* f[12] = {};
            int nf = 0;
            char* p = bam_buf_;
            while (nf < 11) {
                f[nf++] = p;
                char* t = std::strchr(p, '\t');
                if (!t) break;
                *t = '\0';
                p = t + 1;
            }
            if (nf < 11) continue;
            const char* tags = (nf == 11 && p != f[10]) ? p : "";
            const unsigned flag = static_cast<unsigned>(std::strtoul(f[1], nullptr, 10));
            std::string cr, cy, ur, uy;
            for (const char* t = tags; *t;) {
                const char* e = std::strchr(t, '\t');
                const size_t n = e ? static_cast<size_t>(e - t) : std::strlen(t);
                if (n > 5 && t[2] == ':' && t[3] == 'Z' && t[4] == ':') {
                    const std::string v(t + 5, n - 5);
                    if (t[0] == 'C' && t[1] == 'R') cr = v;
                    else if (t[0] == 'C' && (t[1] == 'Y' || t[1] == 'Q')) cy = v;
                    else if (t[0] == 'U' && t[1] == 'R') ur = v;
                    else if (t[0] == 'U' && (t[1] == 'Y' || t[1] == 'Q')) uy = v;
                }
                t += n;
                if (*t == '\t') ++t;
            }
            std::string sq = f[9], ql = f[10];
            if (cr.empty() || ur.empty() || sq == "*" || cy.size() != cr.size() || uy.size() != ur.size()) {
                ++bam_skipped_;
                continue;
            }
            if (ql == "*") ql.assign(sq.size(), 'I');
            if (flag & 0x10) {   // aligned to the reverse strand: BAM stores the reverse complement
                std::string rs(sq.rbegin(), sq.rend());
                for (char& c : rs) {
                    switch (c) {
                        case 'A': c = 'T'; break;
                        case 'C': c = 'G'; break;
                        case 'G': c = 'C'; break;
                        case 'T': c = 'A'; break;
                        default: c = 'N';
                    }
                }
                sq.swap(rs);
                ql.assign(ql.rbegin(), ql.rend());
            }
            seq_ = cr + ur + sq;
            qual_.clear();
            for (char c : cy + uy + ql) qual_.push_back(static_cast<uint8_t>(c - 33));
            seg_start_ = {0, static_cast<int32_t>(cr.size() + ur.size())};
            seg_len_ = {static_cast<uint32_t>(cr.size() + ur.size()), static_cast<uint32_t>(sq.size())};
            ++next_;
            return true;
        }
    }

    std::string spec_, bam_;
    std::vector<Segment> segs_;
    std::vector<std::vector<std::unique_ptr<StreamReader>>> readers_;
    FILE* bam_fp_ = nullptr;
    char bam_buf_[1 << 16];
    uint64_t n_spots_ = 0, next_ = 0, bam_skipped_ = 0;
    std::string seq_;
    std::vector<uint8_t> qual_;
    std::vector<uint32_t> seg_len_;
    std::vector<int32_t> seg_start_;
};


// ── FASTQ folder layouts ──
// Finds the reads of one library in a folder (searched recursively, as a 10x FASTQ tar unpacks) and returns its spot spec.
//   bcl2fastq / Cell Ranger names  NAME_S1_L001_R1_001.fastq.gz (also without _L00N): spot = R1|R2 (R3 when present,
//                                  index reads only with include_index)
//   Cell Ranger 1.x (GemCode, 10x 3' v1)  read-RA_si-SEQ_lane-001-chunk-001.fastq.gz with read-I1 / read-I2: RA holds Read 1
//                                  (cDNA) and Read 2 (UMI) interleaved, and the 14-nt cell barcode is the index read of that
//                                  length. Spot = barcode + UMI | cDNA, the 24-nt barcode read of the 10x-3p-v1 spec.
// `sample` keeps files whose library name contains it (several libraries in one folder need it). Files are taken in
// lane, then chunk order (GemCode: every sample-index oligo of the set, which together make one library).
struct FastqLayout {
    std::string spec;          // "fastq:..." for SpotSource
    std::string kind;          // "illumina" or "gemcode"
    std::string library;       // library name of the files used
    std::vector<std::string> notes;
};

inline uint32_t fastq_first_read_length(const std::string& path) {
    const bool gz = path.size() > 3 && path.compare(path.size() - 3, 3, ".gz") == 0;
    std::string cmd = (gz ? "gzip -dc '" : "cat '") + path + "' 2>/dev/null | head -2 | tail -1";
    FILE* p = popen(cmd.c_str(), "r");
    if (!p) return 0;
    char buf[4096] = {};
    const char* got = std::fgets(buf, sizeof(buf), p);
    pclose(p);
    if (!got) return 0;
    size_t n = std::strlen(buf);
    while (n > 0 && (buf[n - 1] == '\n' || buf[n - 1] == '\r')) --n;
    return static_cast<uint32_t>(n);
}

inline FastqLayout discover_fastq_layout(const std::string& dir, const std::string& sample = "", bool include_index = false) {
    namespace fs = std::filesystem;
    static const std::regex illumina(R"(^(.+?)_S\d+(?:_L(\d{3}))?_(R1|R2|R3|I1|I2)_(\d{3})\.f(?:ast)?q(?:\.gz)?$)");
    static const std::regex gemcode(R"(^read-(RA|I1|I2)_si-([ACGTN]+)_lane-(\d+)-chunk-(\d+)\.f(?:ast)?q(?:\.gz)?$)");
    struct F { std::string path, lib, read, lane, chunk, si; };
    std::vector<F> ill, gem;
    for (const auto& e : fs::recursive_directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        const std::string name = e.path().filename().string();
        std::smatch m;
        if (std::regex_match(name, m, illumina)) {
            ill.push_back({e.path().string(), m[1].str(), m[3].str(), m[2].matched ? m[2].str() : "000", m[4].str(), ""});
        } else if (std::regex_match(name, m, gemcode)) {
            // GemCode folders hold one library per folder; the library is the folder name
            gem.push_back({e.path().string(), e.path().parent_path().filename().string(), m[1].str(), m[3].str(), m[4].str(), m[2].str()});
        }
    }
    auto keep = [&](std::vector<F>& v) {
        if (!sample.empty())
            v.erase(std::remove_if(v.begin(), v.end(), [&](const F& f) { return f.lib.find(sample) == std::string::npos; }), v.end());
        std::sort(v.begin(), v.end(), [](const F& a, const F& b) {
            return std::tie(a.lib, a.lane, a.si, a.chunk, a.path) < std::tie(b.lib, b.lane, b.si, b.chunk, b.path);
        });
    };
    keep(ill);
    keep(gem);
    if (ill.empty() && gem.empty())
        throw std::runtime_error("no FASTQ files with Illumina or Cell Ranger 1.x names under " + dir +
                                 (sample.empty() ? "" : " for library '" + sample + "'"));
    if (!ill.empty() && !gem.empty()) throw std::runtime_error("both Illumina and GemCode FASTQ names under " + dir + ": pass --fastq-sample");

    std::vector<F>& v = ill.empty() ? gem : ill;
    std::vector<std::string> libs;
    for (const auto& f : v)
        if (std::find(libs.begin(), libs.end(), f.lib) == libs.end()) libs.push_back(f.lib);
    if (libs.size() > 1) {
        std::string all;
        for (const auto& l : libs) all += " " + l;
        throw std::runtime_error("several libraries under " + dir + ":" + all + " (choose one with --fastq-sample)");
    }
    auto files_of = [&](const std::string& read) {
        std::vector<std::string> out;
        for (const auto& f : v)
            if (f.read == read) out.push_back(f.path);
        return out;
    };
    auto join = [](const std::vector<std::string>& fs_) {
        std::string o;
        for (size_t i = 0; i < fs_.size(); ++i) o += (i ? "," : "") + fs_[i];
        return o;
    };
    FastqLayout L;
    L.library = libs[0];
    if (!ill.empty()) {
        L.kind = "illumina";
        const auto r1 = files_of("R1"), r2 = files_of("R2"), r3 = files_of("R3"), i1 = files_of("I1"), i2 = files_of("I2");
        if (r1.empty() || r2.empty()) throw std::runtime_error("library " + L.library + " has no R1/R2 pair");
        if (r2.size() != r1.size()) throw std::runtime_error("library " + L.library + ": R1 and R2 have different file counts");
        std::vector<std::vector<std::string>> segs;
        if (include_index && !i1.empty()) segs.push_back(i1);
        if (include_index && !i2.empty()) segs.push_back(i2);
        segs.push_back(r1);
        segs.push_back(r2);
        if (!r3.empty()) segs.push_back(r3);
        L.spec = "fastq:";
        for (size_t i = 0; i < segs.size(); ++i) L.spec += (i ? "|" : "") + join(segs[i]);
        L.notes.push_back(std::to_string(r1.size()) + " R1/R2 file pairs");
        return L;
    }
    // GemCode: the 14-nt index read is the cell barcode, the other (8 nt) the sample index
    L.kind = "gemcode";
    const auto ra = files_of("RA"), i1 = files_of("I1"), i2 = files_of("I2");
    if (ra.empty()) throw std::runtime_error("GemCode library " + L.library + " has no read-RA files");
    std::vector<std::string> bc;
    for (const auto* cand : {&i1, &i2})
        if (!cand->empty() && fastq_first_read_length(cand->front()) == 14) bc = *cand;
    if (bc.empty()) throw std::runtime_error("GemCode library " + L.library + ": no 14-nt barcode index read (read-I1/read-I2)");
    if (bc.size() != ra.size()) throw std::runtime_error("GemCode library " + L.library + ": barcode and RA file counts differ");
    // Interleave: record 0 of each RA pair is Read 1 (cDNA), record 1 is Read 2 (UMI)
    L.spec = "fastq:" + join(bc) + "+" + join(ra) + "@2/1|" + join(ra) + "@2/0";
    L.notes.push_back(std::to_string(ra.size()) + " RA files; barcode read " + (bc == i1 ? "read-I1" : "read-I2"));
    return L;
}

}  // namespace singlet::fq
