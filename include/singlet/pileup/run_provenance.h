// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: run_provenance.h
// What every bundle must say about the engine that wrote it (docs/atlas-rerun-plan.md §6 item 5): the engine git
// commit (compiled in), the sha256 of the running binary, the reference build, the cell-call method and parameters,
// the runs used and the thread count. Written as the "provenance" object of pileup_stats.json.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#ifndef SINGLET_GIT_COMMIT
#define SINGLET_GIT_COMMIT "unknown"
#endif

namespace singlet {

/// Portable SHA-256 (FIPS 180-4) of a byte stream.
class Sha256Stream {
public:
    Sha256Stream() {
        static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::memcpy(h_, iv, sizeof(h_));
    }
    void update(const uint8_t* p, size_t n) {
        len_ += n;
        while (n) {
            size_t take = std::min(n, size_t(64) - fill_);
            std::memcpy(buf_ + fill_, p, take);
            fill_ += take;
            p += take;
            n -= take;
            if (fill_ == 64) {
                block(buf_);
                fill_ = 0;
            }
        }
    }
    std::string hex() {
        uint64_t bits = len_ * 8;
        uint8_t pad = 0x80;
        update(&pad, 1);
        uint8_t zero = 0;
        while (fill_ != 56) update(&zero, 1);
        uint8_t lb[8];
        for (int i = 0; i < 8; ++i) lb[i] = static_cast<uint8_t>(bits >> (56 - 8 * i));
        update(lb, 8);
        static const char* d = "0123456789abcdef";
        std::string s;
        for (int i = 0; i < 8; ++i)
            for (int b = 3; b >= 0; --b) {
                uint8_t v = static_cast<uint8_t>(h_[i] >> (8 * b));
                s.push_back(d[v >> 4]);
                s.push_back(d[v & 15]);
            }
        return s;
    }

private:
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t fill_ = 0;
    uint64_t len_ = 0;
    static uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    void block(const uint8_t* p) {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        uint32_t w[64];
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) |
                   uint32_t(p[4 * i + 3]);
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }
};

/// sha256 of a file ("" if unreadable).
inline std::string sha256_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "";
    Sha256Stream s;
    std::vector<uint8_t> buf(1 << 20);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), f)) > 0) s.update(buf.data(), n);
    std::fclose(f);
    return s.hex();
}

/// sha256 of the running binary (/proc/self/exe on Linux).
inline std::string self_binary_sha256() { return sha256_file("/proc/self/exe"); }

inline std::string json_str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        if (c == '"' || c == '\\') {
            o.push_back('\\');
            o.push_back(c);
        } else if (static_cast<unsigned char>(c) < 0x20) {
            char b[8];
            std::snprintf(b, sizeof(b), "\\u%04x", c);
            o += b;
        } else {
            o.push_back(c);
        }
    }
    return o + "\"";
}

struct RunProvenance {
    std::string engine_version;
    std::string engine_commit = SINGLET_GIT_COMMIT;
    std::string binary_sha256;
    std::string reference_build;
    std::string genome_dir;
    std::string gtf_path;
    int threads = 0;
    std::vector<std::string> runs_used;   ///< --runs-used (e.g. the runs_kept of `singlet screen-runs`)
    std::string run_screen_json;          ///< --run-screen path, if given
    std::string cell_caller;
    std::string cell_call_json;           ///< knee + called_v2 parameters
    bool reverse_strand = false;
    std::string strand_source;            ///< "header", "probe", "retry", "user", "default"

    /// The "provenance" object (no trailing comma).
    std::string json(const std::string& ind = "  ") const {
        std::ostringstream o;
        o << "{\n"
          << ind << "  \"engine_version\": " << json_str(engine_version) << ",\n"
          << ind << "  \"engine_commit\": " << json_str(engine_commit) << ",\n"
          << ind << "  \"binary_sha256\": " << json_str(binary_sha256) << ",\n"
          << ind << "  \"reference_build\": " << json_str(reference_build) << ",\n"
          << ind << "  \"genome_dir\": " << json_str(genome_dir) << ",\n"
          << ind << "  \"gtf\": " << json_str(gtf_path) << ",\n"
          << ind << "  \"threads\": " << threads << ",\n"
          << ind << "  \"runs_used\": [";
        for (size_t i = 0; i < runs_used.size(); ++i) o << (i ? ", " : "") << json_str(runs_used[i]);
        o << "],\n"
          << ind << "  \"run_screen\": " << json_str(run_screen_json) << ",\n"
          << ind << "  \"reverse_strand\": " << (reverse_strand ? "true" : "false") << ",\n"
          << ind << "  \"strand_source\": " << json_str(strand_source) << ",\n"
          << ind << "  \"cell_caller\": " << json_str(cell_caller) << ",\n"
          << ind << "  \"cell_call\": " << (cell_call_json.empty() ? std::string("null") : cell_call_json) << "\n"
          << ind << "}";
        return o.str();
    }
};

/// Split "a,b,c" (whitespace trimmed, empties dropped).
inline std::vector<std::string> split_list(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    auto flush = [&]() {
        size_t a = cur.find_first_not_of(" \t\r\n"), b = cur.find_last_not_of(" \t\r\n");
        if (a != std::string::npos) out.push_back(cur.substr(a, b - a + 1));
        cur.clear();
    };
    for (char c : s) {
        if (c == ',')
            flush();
        else
            cur.push_back(c);
    }
    flush();
    return out;
}

/// runs_kept from a `singlet screen-runs --json` report (minimal parse: the string array after "runs_kept").
inline std::vector<std::string> runs_kept_from_screen_json(const std::string& path) {
    std::vector<std::string> out;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    std::string s;
    char buf[65536];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    size_t p = s.find("\"runs_kept\"");
    if (p == std::string::npos) return out;
    p = s.find('[', p);
    size_t e = p == std::string::npos ? p : s.find(']', p);
    if (p == std::string::npos || e == std::string::npos) return out;
    std::string arr = s.substr(p + 1, e - p - 1);
    size_t i = 0;
    while ((i = arr.find('"', i)) != std::string::npos) {
        size_t j = arr.find('"', i + 1);
        if (j == std::string::npos) break;
        out.push_back(arr.substr(i + 1, j - i - 1));
        i = j + 1;
    }
    return out;
}

}  // namespace singlet
