#pragma once
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <map>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace singlet {

enum class LogLevel { QUIET = 0,
                      NORMAL = 1,
                      VERBOSE = 2,
                      DEBUG = 3 };

class SinglifyLogger {
   public:
    static SinglifyLogger& instance() {
        static SinglifyLogger inst;
        return inst;
    }

    void set_level(LogLevel level) { level_ = level; }
    void set_stderr(bool enabled) { stderr_enabled_ = enabled; }

    void set_log_file(const std::string& path) {
        std::lock_guard<std::mutex> lk(mtx_);
        logfile_.open(path, std::ios::out | std::ios::trunc);
    }

    void info(const char* fmt, ...) {
        if (level_ < LogLevel::NORMAL) return;
        char buf[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        write_stderr(LogLevel::NORMAL, buf);
        write_jsonl("info", buf);
    }

    void verbose(const char* fmt, ...) {
        if (level_ < LogLevel::VERBOSE) return;
        char buf[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        write_stderr(LogLevel::VERBOSE, buf);
        write_jsonl("verbose", buf);
    }

    void debug(const char* fmt, ...) {
        if (level_ < LogLevel::DEBUG) return;
        char buf[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        write_stderr(LogLevel::DEBUG, buf);
        write_jsonl("debug", buf);
    }

    void error(const char* fmt, ...) {
        if (level_ == LogLevel::QUIET) return;
        char buf[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        write_stderr(LogLevel::NORMAL, buf);
        write_jsonl("error", buf);
    }

    void warn(const char* fmt, ...) {
        if (level_ == LogLevel::QUIET) return;
        char buf[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        write_stderr(LogLevel::NORMAL, buf);
        write_jsonl("warning", buf);
    }

    void event(const char* type, const char* fmt, ...) {
        char buf[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        if (level_ >= LogLevel::NORMAL) {
            write_stderr(LogLevel::NORMAL, buf);
        }
        write_jsonl(type, buf);
    }

    void progress(float fraction, const char* fmt, ...) {
        auto now = std::chrono::steady_clock::now();
        {
            std::lock_guard<std::mutex> lk(mtx_);
            float elapsed = std::chrono::duration<float>(now - last_progress_).count();
            if (elapsed < throttle_interval_ && fraction < 1.0f) return;
            last_progress_ = now;
        }
        if (level_ == LogLevel::QUIET) return;
        char buf[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(buf, sizeof(buf), fmt, ap);
        va_end(ap);
        char full[4128];
        snprintf(full, sizeof(full), "[%.0f%%] %s", fraction * 100.f, buf);
        write_stderr(LogLevel::NORMAL, full);
        write_jsonl("progress", full);
    }

    void stat(const char* key, double value) {
        char buf[256];
        snprintf(buf, sizeof(buf), "%s=%.6g", key, value);
        write_jsonl("stat", buf);
    }

    void stat(const char* key, const char* value) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s=%s", key, value);
        write_jsonl("stat", buf);
    }

    void close() {
        std::lock_guard<std::mutex> lk(mtx_);
        if (logfile_.is_open()) logfile_.close();
    }

    LogLevel level() const { return level_; }

   private:
    SinglifyLogger() {
        last_progress_ = std::chrono::steady_clock::now() -
                         std::chrono::seconds(100);
    }

    LogLevel level_ = LogLevel::NORMAL;
    std::ofstream logfile_;
    std::mutex mtx_;
    bool stderr_enabled_ = true;
    std::chrono::steady_clock::time_point last_progress_;
    float throttle_interval_ = 2.0f;

    static std::string iso8601_now() {
        auto now = std::chrono::system_clock::now();
        std::time_t t = std::chrono::system_clock::to_time_t(now);
        struct tm tm_buf;
        gmtime_r(&t, &tm_buf);
        char ts[32];
        strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%SZ", &tm_buf);
        return ts;
    }

    // Escape double-quotes and backslashes for embedding in JSON string
    static std::string json_escape(const char* s) {
        std::string out;
        out.reserve(128);
        for (const char* p = s; *p; ++p) {
            if (*p == '"') {
                out += "\\\"";
            } else if (*p == '\\') {
                out += "\\\\";
            } else if (*p == '\n') {
                out += "\\n";
            } else if (*p == '\r') {
                out += "\\r";
            } else if (*p == '\t') {
                out += "\\t";
            } else {
                out += *p;
            }
        }
        return out;
    }

    void write_stderr(LogLevel /*msg_level*/, const char* formatted) {
        if (!stderr_enabled_) return;
        fprintf(stderr, "%s\n", formatted);
    }

    void write_jsonl(const char* type, const char* msg) {
        std::lock_guard<std::mutex> lk(mtx_);
        if (!logfile_.is_open()) return;
        logfile_ << "{\"ts\":\"" << iso8601_now()
                 << "\",\"type\":\"" << type
                 << "\",\"msg\":\"" << json_escape(msg) << "\"}\n";
        logfile_.flush();
    }
};

// Convenience macros
#define SLOG(...) singlet::SinglifyLogger::instance().info(__VA_ARGS__)
#define SLOG_VERBOSE(...) singlet::SinglifyLogger::instance().verbose(__VA_ARGS__)
#define SLOG_DEBUG(...) singlet::SinglifyLogger::instance().debug(__VA_ARGS__)
#define SLOG_ERROR(...) singlet::SinglifyLogger::instance().error(__VA_ARGS__)
#define SLOG_WARN(...) singlet::SinglifyLogger::instance().warn(__VA_ARGS__)
#define SLOG_EVENT(type, ...) singlet::SinglifyLogger::instance().event(type, __VA_ARGS__)

}  // namespace singlet
