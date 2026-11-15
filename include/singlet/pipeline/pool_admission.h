// pool_admission.h: the memory-aware start rule of `singlet gpu-pool` (v1.4.0).
//
// Each queued sample gets a projected peak host memory from its depth and chemistry:
//     peak_gb = a + b * million read pairs        (b = b5 for 10x 5' libraries, b3 for the rest)
// A sample starts only when the projected peaks of every sample in flight in this job (all pools of the job's cgroup, through a small
// ledger in /dev/shm), plus its own, fit the job's memory budget. A sample whose projection exceeds the whole budget waits until nothing is
// in flight and then runs alone. The sample at the head of a pool's queue reserves its projection: later, smaller samples may start beside
// the samples in flight (backfill) only when they leave room for every older reservation, so a deep sample is never starved.
//
// Calibration (v1.4.0 integration runs and the second dry run; peak = ru_maxrss of the sample's process, which includes the pool's
// copy-on-write genome pages, so the rule errs on the safe side):
//   3'  d300_3p  300 M pairs  88.0 GB    d1000_3p 1.00 B pairs 266.3 GB    d2600_3p 1.30 B pairs 302.3 GB   ->  a = 15, b3 = 0.26
//   5'  435-516 M pairs held 130-190 GB each when the dry-run job was OOM-killed (not yet at peak)              ->  b5 = 0.42
// The model only grows during a run: a sample that ends above its projection raises the slope of its chemistry (learn), and a sample in
// flight whose resident memory already exceeds its projection counts at its measured size (live).
//
// Switches (environment of the pool):
//   SINGLET_POOL_MEM_RULE=0              off: samples start as soon as a worker is free (v1.3.9 behaviour)
//   SINGLET_POOL_MEM_BUDGET_GB=G         the memory for samples (default: 0.9 x the cgroup's memory.max, else MemTotal, minus the base)
//   SINGLET_POOL_MEM_BASE_GB=G           the pools' own memory (genome copies, non-host index); default: measured (cgroup anon + shmem
//                                        while no sample is in flight), else 45
//   SINGLET_POOL_MEM_A_GB, _B3_GB, _B5_GB  the model's coefficients (B in GB per million read pairs)
//   SINGLET_POOL_MEM_BACKFILL=0          strict queue order (no backfill)
//   SINGLET_POOL_LEDGER_DIR=DIR          where the pools of one job meet (default /dev/shm/singlet-pool-<uid>-<cgroup hash>)
#pragma once

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iterator>
#include <signal.h>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#include "singlet/fq/types.h"

namespace singlet {
namespace pool_admission {

inline double env_or(const char* name, double dflt) {
    const char* e = std::getenv(name);
    return (e != nullptr && e[0] != 0) ? std::atof(e) : dflt;
}
inline bool rule_on() { const char* e = std::getenv("SINGLET_POOL_MEM_RULE"); return !(e != nullptr && e[0] == '0'); }
inline bool backfill_on() { const char* e = std::getenv("SINGLET_POOL_MEM_BACKFILL"); return !(e != nullptr && e[0] == '0'); }

struct Model {
    double a_gb = 15.0;    // per-sample fixed part
    double b3_gb = 0.26;   // GB per million read pairs: 3' and every other library
    double b5_gb = 0.42;   // GB per million read pairs: 10x 5'
    static Model from_env() {
        Model m;
        m.a_gb = env_or("SINGLET_POOL_MEM_A_GB", m.a_gb);
        m.b3_gb = env_or("SINGLET_POOL_MEM_B3_GB", m.b3_gb);
        m.b5_gb = env_or("SINGLET_POOL_MEM_B5_GB", m.b5_gb);
        return m;
    }
};

inline double projected_peak_gb(double mpairs, bool five_prime, const Model& m) {
    return m.a_gb + (five_prime ? m.b5_gb : m.b3_gb) * std::max(0.0, mpairs);
}

// After a sample ends: raise its chemistry's slope when the measured peak exceeded the projection (the model never shrinks). Samples under
// 50 M pairs say little about the slope and are not used. Returns true when the model changed.
inline bool learn(Model& m, double mpairs, bool five_prime, double peak_gb) {
    if (mpairs < 50.0 || peak_gb <= projected_peak_gb(mpairs, five_prime, m)) return false;
    double& b = five_prime ? m.b5_gb : m.b3_gb;
    b = (peak_gb - m.a_gb) / mpairs;
    return true;
}

// 10x 5' protocol ids of the .1fq header (fq/protocol.h known_protocols(): 10x-5p-v3 = 4, 10x-5p-v2 = 5, 10x-5p-v4 = 14; the unit test
// checks this list against the registry)
inline bool is_10x_5p(uint8_t protocol_id) { return protocol_id == 4 || protocol_id == 5 || protocol_id == 14; }

// Depth and chemistry of a queued sample from its .1fq header (96 bytes). pairs = original spots (n_original), else the stored count.
struct SampleInfo { bool known = false; double mpairs = 0; bool five_prime = false; uint8_t protocol_id = 0; };
inline SampleInfo read_1fq_info(const std::string& path) {
    SampleInfo s;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) return s;
    singlet::fq::Header h{};
    const bool ok = std::fread(&h, sizeof h, 1, f) == 1;
    std::fclose(f);
    if (!ok || !h.valid_magic()) return s;
    const uint64_t n = h.n_original ? h.n_original : h.n_unique;
    if (n == 0) return s;
    s.known = true;
    s.mpairs = (double) n / 1e6;
    s.protocol_id = h.protocol_id;
    s.five_prime = is_10x_5p(h.protocol_id);
    return s;
}

// The decision for one candidate.
//   budget_gb    memory for samples in this job
//   inflight_gb  the projected peaks of every sample in flight in the job (all pools)
//   n_inflight   how many samples are in flight in the job
//   reserved_gb  reservations the candidate must leave room for (older waiting heads of any pool; for a backfill candidate also its own
//                pool's head), each capped at the budget
enum class Verdict { Start, StartAlone, Wait };
inline Verdict decide(double budget_gb, double inflight_gb, size_t n_inflight, double reserved_gb, double cand_gb) {
    if (n_inflight == 0 && reserved_gb <= 0.0) return cand_gb > budget_gb ? Verdict::StartAlone : Verdict::Start;
    if (inflight_gb + reserved_gb + cand_gb <= budget_gb) return Verdict::Start;
    return Verdict::Wait;
}

// ---- the ledger: one file per pool of the job, rewritten under a lock by its pool.
//   I <projected gb>                  a sample in flight (live size when it exceeds the projection)
//   H <projected gb> <since unix>     the pool's waiting head (a reservation)
//   B <base gb>                       the base memory this pool measured
struct PoolEntry {
    pid_t pid = 0;
    std::vector<double> items;
    double head_gb = 0, head_since = 0;   // head_gb 0: no waiting head
    double base_gb = 0;
};

inline std::string cgroup_dir() {
    std::ifstream cl("/proc/self/cgroup");
    std::string line, path;
    while (std::getline(cl, line)) {
        if (line.rfind("0::", 0) == 0) { path = line.substr(3); break; }   // cgroup v2
    }
    if (path.empty()) return std::string();
    // the job's cgroup, not a step's or a task's (slurm: .../job_<id>/step_<x>/...)
    const size_t j = path.find("/job_");
    if (j != std::string::npos) { const size_t e = path.find('/', j + 1); if (e != std::string::npos) path = path.substr(0, e); }
    return "/sys/fs/cgroup" + path;
}
inline std::string default_ledger_dir() {
    if (const char* d = std::getenv("SINGLET_POOL_LEDGER_DIR"); d != nullptr && d[0] != 0) return d;
    const std::string cg = cgroup_dir();
    uint64_t h = 1469598103934665603ull;
    for (char c : cg) { h ^= (unsigned char) c; h *= 1099511628211ull; }
    char hx[24];
    std::snprintf(hx, sizeof hx, "%016llx", (unsigned long long) h);
    return "/dev/shm/singlet-pool-" + std::to_string((unsigned) getuid()) + "-" + hx;
}

inline bool pid_alive(pid_t p) { return p > 0 && (kill(p, 0) == 0 || errno == EPERM); }

inline std::vector<PoolEntry> read_ledger(const std::string& dir) {
    std::vector<PoolEntry> out;
    DIR* d = opendir(dir.c_str());
    if (d == nullptr) return out;
    while (struct dirent* e = readdir(d)) {
        const std::string n = e->d_name;
        if (n.size() < 5 || n.compare(n.size() - 4, 4, ".tsv") != 0) continue;
        const pid_t p = (pid_t) std::atol(n.c_str());
        if (!pid_alive(p)) { unlink((dir + "/" + n).c_str()); continue; }   // a pool that ended without cleaning up
        PoolEntry pe;
        pe.pid = p;
        std::ifstream f(dir + "/" + n);
        std::string k;
        while (f >> k) {
            if (k == "I") { double g = 0; f >> g; pe.items.push_back(g); }
            else if (k == "H") f >> pe.head_gb >> pe.head_since;
            else if (k == "B") f >> pe.base_gb;
            else { std::string rest; std::getline(f, rest); }
        }
        out.push_back(pe);
    }
    closedir(d);
    return out;
}
inline bool write_ledger(const std::string& dir, const PoolEntry& pe) {
    const std::string fn = dir + "/" + std::to_string((long) pe.pid) + ".tsv", tmp = fn + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (f == nullptr) return false;
    for (double g : pe.items) std::fprintf(f, "I %.3f\n", g);
    if (pe.head_gb > 0) std::fprintf(f, "H %.3f %.3f\n", pe.head_gb, pe.head_since);
    if (pe.base_gb > 0) std::fprintf(f, "B %.3f\n", pe.base_gb);
    std::fclose(f);
    return std::rename(tmp.c_str(), fn.c_str()) == 0;
}

// flock on <dir>/lock for the life of the object; ok() false when the ledger cannot be used (the pool then counts only its own samples)
class LedgerLock {
public:
    explicit LedgerLock(const std::string& dir) {
        mkdir(dir.c_str(), 0700);
        fd_ = open((dir + "/lock").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (fd_ >= 0 && flock(fd_, LOCK_EX) != 0) { close(fd_); fd_ = -1; }
    }
    ~LedgerLock() { if (fd_ >= 0) { flock(fd_, LOCK_UN); close(fd_); } }
    bool ok() const { return fd_ >= 0; }
    LedgerLock(const LedgerLock&) = delete;
    LedgerLock& operator=(const LedgerLock&) = delete;
private:
    int fd_ = -1;
};

// The job's memory: the cgroup's limit (memory.max) and its anon + shmem now (memory.stat); MemTotal / MemTotal - MemAvailable without one.
inline double limit_gb() {
    const std::string cg = cgroup_dir();
    if (!cg.empty()) {
        std::ifstream mx(cg + "/memory.max");
        std::string m;
        if (mx >> m && !m.empty() && std::isdigit((unsigned char) m[0])) return std::atof(m.c_str()) / 1073741824.0;
    }
    std::ifstream mi("/proc/meminfo");
    std::string k, u;
    double v = 0;
    while (mi >> k >> v >> u) if (k == "MemTotal:") return v / 1048576.0;
    return 0;
}
inline double used_gb() {
    const std::string cg = cgroup_dir();
    if (!cg.empty()) {
        std::ifstream ms(cg + "/memory.stat");
        std::string k;
        double v = 0, t = 0;
        bool any = false;
        while (ms >> k >> v) if (k == "anon" || k == "shmem") { t += v; any = true; }
        if (any) return t / 1073741824.0;
    }
    std::ifstream mi("/proc/meminfo");
    std::string k, u;
    double v = 0, tot = 0, av = 0;
    while (mi >> k >> v >> u) { if (k == "MemTotal:") tot = v; else if (k == "MemAvailable:") av = v; }
    return (tot - av) / 1048576.0;
}

// The largest resident set (VmRSS, GB) among the processes of process group `pgid`: the same measure as ru_maxrss, read live.
inline double group_rss_gb(pid_t pgid) {
    double best = 0;
    DIR* d = opendir("/proc");
    if (d == nullptr) return 0;
    while (struct dirent* e = readdir(d)) {
        if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
        std::ifstream f(std::string("/proc/") + e->d_name + "/stat");
        std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        const size_t r = s.rfind(')');
        if (r == std::string::npos || r + 2 >= s.size()) continue;
        std::istringstream in(s.substr(r + 2));
        std::string st, ppid, pg;
        in >> st >> ppid >> pg;
        if ((pid_t) std::atol(pg.c_str()) != pgid) continue;
        std::ifstream sf(std::string("/proc/") + e->d_name + "/status");
        std::string l;
        while (std::getline(sf, l))
            if (l.rfind("VmRSS:", 0) == 0) { best = std::max(best, std::atof(l.c_str() + 6) / 1048576.0); break; }
    }
    closedir(d);
    return best;
}

}  // namespace pool_admission
}  // namespace singlet
