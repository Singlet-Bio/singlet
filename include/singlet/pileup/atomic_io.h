#pragma once
// singlet-pileup: atomic_io.h
// Atomic file I/O helpers for crash-safe output writing.
// All files are written to a .tmp path first, then renamed atomically.
// If the process crashes mid-write, no corrupt partial files remain.

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "pz_writer.h"

namespace singlet {

/// Atomically rename a temporary file to its final path.
/// On rename failure, removes the temporary file.
/// @return true on success, false on failure (tmp file cleaned up).
inline bool atomic_rename(const std::string& tmp_path, const std::string& final_path) {
    std::error_code ec;
    std::filesystem::rename(tmp_path, final_path, ec);
    if (ec) {
        std::filesystem::remove(tmp_path, ec);
        return false;
    }
    return true;
}

/// RAII wrapper for atomic file writes.
///
/// Usage:
///   { AtomicOfstream f("output.tsv");
///     f << "data\n";
///   }  // destructor renames .tmp → output.tsv
///
/// On destruction, renames .tmp → final if stream is in a good state.
/// If the stream had errors, the .tmp file is removed.
class AtomicOfstream {
    std::string   final_path_;
    std::string   tmp_path_;
    std::ofstream stream_;
    bool          committed_ = false;

public:
    explicit AtomicOfstream(const std::string& path)
        : final_path_(path), tmp_path_(path + ".tmp"), stream_(tmp_path_) {}

    ~AtomicOfstream() {
        if (!committed_ && stream_.is_open()) {
            stream_.close();
            if (stream_.good()) {
                atomic_rename(tmp_path_, final_path_);
            } else {
                std::error_code ec;
                std::filesystem::remove(tmp_path_, ec);
            }
        }
    }

    // Non-copyable, non-movable (RAII scope guard)
    AtomicOfstream(const AtomicOfstream&) = delete;
    AtomicOfstream& operator=(const AtomicOfstream&) = delete;

    /// Access the underlying stream.
    std::ofstream& stream() { return stream_; }
    bool is_open() const { return stream_.is_open(); }

    /// Explicit commit: close and rename immediately.
    /// @return true if the file was written and renamed successfully.
    bool commit() {
        stream_.close();
        committed_ = true;
        return stream_.good() && atomic_rename(tmp_path_, final_path_);
    }

    /// Stream insertion operator (convenience).
    template <typename T>
    AtomicOfstream& operator<<(const T& v) { stream_ << v; return *this; }
};

/// Write a .1pz file atomically: write to .tmp, then rename.
/// Wraps pz::write_1pz with atomic rename.
/// @return true if write + rename succeeded.
template <typename... Args>
inline bool atomic_write_1pz(const std::string& path, Args&&... args) {
    const std::string tmp = path + ".tmp";
    pz::write_1pz(tmp, std::forward<Args>(args)...);
    // Check that the file was created and has size > 0
    std::error_code ec;
    auto sz = std::filesystem::file_size(tmp, ec);
    if (ec || sz == 0) {
        std::filesystem::remove(tmp, ec);
        return false;
    }
    return atomic_rename(tmp, path);
}

}  // namespace singlet
