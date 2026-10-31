// SPDX-License-Identifier: MIT
// fg2_source.h — the staged .1fg2 (Fg2RunSet, fg2_runset.h) behind Reader's / RunSet's interface (reader.h StagedSource)
//
// register_fg2() makes Reader::open() and RunSet::open() accept a .1fg2 path. Call it once at the start of main().
// Whitelist: the file named in the .1fg2 header is looked up in $SINGLET_WHITELIST_DIR, then <binary dir>/../whitelists.
#pragma once

#include <fcntl.h>
#include <unistd.h>

#include <climits>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include <zstd.h>

#include "fg2_runset.h"
#include "reader.h"

namespace singlet::fq {

class Fg2Source : public StagedSource {
   public:
    explicit Fg2Source(const char* path) {
        std::vector<std::string> dirs;
        if (const char* e = std::getenv("SINGLET_WHITELIST_DIR")) dirs.push_back(e);
        char exe[PATH_MAX];
        const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
        if (n > 0) {
            exe[n] = 0;
            std::string d(exe);
            const auto sl = d.rfind('/');
            if (sl != std::string::npos) dirs.push_back(d.substr(0, sl) + "/../whitelists");
        }
        rs_.open({path}, dirs, /*verify_crc=*/false);
        hdr_ = rs_.header();
        // the source .1fq's metadata JSON (soloCBlen, protocol tag, ...) lives zstd-compressed at meta_offset
        const auto& fh = rs_.fg2_header();
        if (fh.meta_comp_size > 0 && fh.meta_raw_size > 0) {
            std::vector<uint8_t> comp(fh.meta_comp_size);
            const int fd = ::open(path, O_RDONLY);
            if (fd >= 0) {
                if (::pread(fd, comp.data(), comp.size(), static_cast<off_t>(fh.meta_offset)) == static_cast<ssize_t>(comp.size())) {
                    std::string raw(fh.meta_raw_size, '\0');
                    const size_t r = ZSTD_decompress(&raw[0], raw.size(), comp.data(), comp.size());
                    if (!ZSTD_isError(r)) { raw.resize(r); meta_ = raw; }
                }
                ::close(fd);
            }
            hdr_.meta_size = meta_.empty() ? 0 : static_cast<uint32_t>(meta_.size());
        }
    }
    const Header& header() const override { return hdr_; }
    uint32_t block_count() const override { return rs_.block_count(); }
    uint64_t stored_reads() const override { return rs_.stored_reads(); }
    uint64_t block_read_offset(uint32_t g) const override { return rs_.block_read_offset(g); }
    uint16_t bc_length() const override { return rs_.bc_length(); }
    void decode(uint32_t g, DecodedBlock& blk) const override { rs_.decode(g, blk); }
    std::string metadata() const override { return meta_; }
    const Fg2RunSet& runset() const { return rs_; }

   private:
    Fg2RunSet rs_;
    Header hdr_{};
    std::string meta_;
};

inline std::shared_ptr<StagedSource> open_fg2_source(const char* path) { return std::make_shared<Fg2Source>(path); }
inline void register_fg2() { staged_opener() = &open_fg2_source; }

}  // namespace singlet::fq
