// singlet E21 (docs/gpu-engine/fable/E21-pool.md): `singlet gpu-pool`, one resident process per GPU that runs a queue of samples on the GPU
// path with the per-sample fixed costs paid once.
//
//   singlet gpu-pool --device D --queue FILE [--slots 2] [--workers 8] [--threads 7] [--args "<pipeline arguments>"] [--report DIR]
//                    [--nonhost-server on|off] [--genome-preload on|off] [--shared-index on|off] [--timeout S]
//
// The queue file has one sample per line: `<sample.1fq> <out dir> [extra pipeline arguments]` (blank lines and # comments skipped; quotes
// group words). Sample i runs `singlet <sample.1fq> <--args ...> --out-prefix <out dir> --threads T <extra ...>` exactly as on its own,
// with SINGLET_ENGINE_GPU_SLOTS=slots and SINGLET_ENGINE_GPU_DEVICE=D, in a process forked from the pool.
//
// Loaded once, before any sample (and shared by every sample's process through fork, copy-on-write, nothing written):
//   - each STAR genome named by --genome-dir in --args or in a queue line (E14's preload, star_genome_preload_resident: Genome, SA and
//     SAindex in STAR's layout on transparent hugepages). A sample's process selects its genome (star_genome_preload_select) and its STAR
//     child takes the arrays as from E14's per-sample preload; no sample reads or copies the 16.8 GB (GRCh38) itself.
//   - the non-host screen's three indexes and the host Bloom filter, in a server process the pool forks first (nonhostServe in
//     src/engine_gpu/nonhost_gpu.cu: page-locked host memory, its own CUDA context on device D). A sample's non-host step sends its reads
//     to it (SINGLET_NONHOST_SERVER) while it holds its device slot, and gets the same arrays back; on any failure it loads the index
//     itself. The pool's own process never initialises CUDA (a forked child could not use a context of its parent's).
// Then a stream: up to --workers samples in flight, the next started as soon as one ends. Each sample's stdout and stderr go to
// <report>/<i>.log with a unix-time prefix per line; on its end the pool records wall, user + system core-s and peak RSS (wait4: the
// sample's process and every descendant it waited for) and the exit status or signal in <report>/samples.tsv and <report>/<i>.time
// (the fields of /usr/bin/time -v). A sample that crashes or is killed is recorded and the stream goes on; the pool exits when the queue
// is drained, with status 0 if every sample exited 0, else 1, after a summary: per sample wall, core-s, peak RSS, and the pool's pairs/s.
// <report>/pids.tsv lists pid, label and start time of the pool, the server and each sample's process (for a sampler).
// --shared-index on (the default) exports SINGLET_GPU_SHARED_INDEX=1 to every sample unless the variable is already set: one holder per
// device and genome keeps G, SA and SAi on the device and every slot maps them, so no sample uploads the index (outputs are the same).
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <csignal>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <poll.h>
#include <sstream>
#include <string>
#include <sys/file.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "star_api.h"
#include "singlet/fq/read_source.h"   // cggate: RunSet, the first block of a .1fq for the peak model
#include "singlet/gpu_pool_peak.h"
namespace poolpeak = singlet::poolpeak;
#ifdef SINGLET_HAVE_ENGINE_GPU
#include "singlet/engine_gpu.h"
#endif

namespace {

double wall_now() {
    struct timespec ts{};
    clock_gettime(CLOCK_REALTIME, &ts);
    return (double) ts.tv_sec + (double) ts.tv_nsec * 1e-9;
}

double env_num(const char* name, double dflt) {
    const char* e = std::getenv(name);
    return (e != nullptr && e[0] != 0) ? std::atof(e) : dflt;
}
// GB the node (and this job's cgroup, when it has a limit) can still hand out without evicting anything but page cache
double avail_gb() {
    double a = 1e9;
    std::ifstream mi("/proc/meminfo");
    std::string k, u;
    double v = 0;
    double mfree = -1, cached = 0, shm = 0;   // MemFree plus half of the file cache: the cache of other tenants is reclaimed slowly under load (g051, 815877)
    while (mi >> k >> v >> u) {
        if (k == "MemFree:") mfree = v; else if (k == "Cached:") cached = v; else if (k == "Shmem:") shm = v;
    }
    if (mfree >= 0) a = (mfree + 0.5 * std::max(0.0, cached - shm)) / 1048576.0;
    // The job's limit sits on an ancestor of this process's cgroup: Slurm puts it on job_<id> and the process lives in
    // .../job_<id>/step_<n>/user/task_<k>, whose own memory.max is "max".  Every ancestor from the leaf up is read and the
    // smallest headroom wins (adaptive-scheduler live pilot 2026-11-18: reading only the leaf saw 510 GB on a 365 GB h100x1 job,
    // admitted three deep samples at once, and the cgroup OOM ended one of them).  Headroom is limit - usage + half of the
    // cgroup's file cache (less shmem), the same rule as the node figure.  cgroup v2 (the "0::" line) and v1 (the memory controller).
    std::ifstream cl("/proc/self/cgroup");
    std::string line, path, root, maxF, curF;
    while (std::getline(cl, line)) {
        if (line.rfind("0::", 0) == 0 && path.empty()) { path = line.substr(3); root = "/sys/fs/cgroup"; maxF = "memory.max"; curF = "memory.current"; }
        const size_t c1 = line.find(':'), c2 = c1 == std::string::npos ? c1 : line.find(':', c1 + 1);
        if (c2 != std::string::npos) {
            const std::string ctl = "," + line.substr(c1 + 1, c2 - c1 - 1) + ",";
            if (ctl.find(",memory,") != std::string::npos) { path = line.substr(c2 + 1); root = "/sys/fs/cgroup/memory"; maxF = "memory.limit_in_bytes"; curF = "memory.usage_in_bytes"; break; }
        }
    }
    for (std::string p = path; !root.empty();) {
        const std::string base = root + (p == "/" ? std::string() : p);
        std::ifstream mx(base + "/" + maxF), cu(base + "/" + curF), stf(base + "/memory.stat");
        std::string m;
        double cur = 0;
        if (mx >> m && cu >> cur && !m.empty() && std::isdigit((unsigned char) m[0])) {
            const double lim = std::atof(m.c_str());
            if (lim < 1e18) {                      // v1 writes a huge number for "no limit"
                double file = 0, shmem = 0, sv = 0;
                std::string sk;
                while (stf >> sk >> sv) {
                    if (sk == "file" || sk == "total_cache") file = sv; else if (sk == "shmem" || sk == "total_shmem") shmem = sv;
                }
                a = std::min(a, (lim - cur + 0.5 * std::max(0.0, file - shmem)) / 1073741824.0);
            }
        }
        if (p.empty() || p == "/") break;
        const size_t s = p.rfind('/');
        p = s == 0 || s == std::string::npos ? std::string("/") : p.substr(0, s);
    }
    return a;
}

FILE* g_plog = nullptr;
void plog(const char* fmt, ...) {
    char msg[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    char ts[32];
    const time_t t = time(nullptr);
    struct tm tmv{};
    localtime_r(&t, &tmv);
    strftime(ts, sizeof ts, "%Y-%m-%dT%H:%M:%S", &tmv);
    fprintf(stderr, "[gpu-pool %s] %s\n", ts, msg);
    if (g_plog) { fprintf(g_plog, "%ld [gpu-pool %s] %s\n", (long) t, ts, msg); fflush(g_plog); }
}

// words of a line: whitespace-separated, '...' and "..." group, backslash escapes outside single quotes
std::vector<std::string> split_args(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    bool have = false;
    char q = 0;
    for (size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (q == '\'') { if (c == '\'') q = 0; else cur += c; continue; }
        if (c == '\\' && i + 1 < s.size()) { cur += s[++i]; have = true; continue; }
        if (q == '"') { if (c == '"') q = 0; else cur += c; continue; }
        if (c == '\'' || c == '"') { q = c; have = true; continue; }
        if (std::isspace((unsigned char) c)) { if (have) { out.push_back(cur); cur.clear(); have = false; } continue; }
        cur += c; have = true;
    }
    if (have) out.push_back(cur);
    return out;
}

// the value of the last `key value` in v ("" if none)
std::string arg_value(const std::vector<std::string>& v, const std::string& key) {
    std::string r;
    for (size_t i = 0; i + 1 < v.size(); ++i) if (v[i] == key) r = v[i + 1];
    return r;
}
bool has_flag(const std::vector<std::string>& v, const std::string& key) { return std::find(v.begin(), v.end(), key) != v.end(); }

void mkdir_p(const std::string& path) {
    std::string p;
    for (size_t i = 0; i < path.size(); ++i) {
        p += path[i];
        if ((path[i] == '/' && i > 0) || i + 1 == path.size()) mkdir(p.c_str(), 0755);
    }
}
std::string parent_dir(std::string p) {
    while (p.size() > 1 && p.back() == '/') p.pop_back();
    const size_t s = p.rfind('/');
    if (s == std::string::npos) return ".";
    return s == 0 ? "/" : p.substr(0, s);
}
bool file_exists(const std::string& p) { struct stat st{}; return !p.empty() && stat(p.c_str(), &st) == 0; }
std::string absolute(const std::string& p) {   // a sample's process works in its out dir's parent (as E18's runs did)
    if (p.empty() || p[0] == '/') return p;
    char cwd[4096];
    return getcwd(cwd, sizeof cwd) ? std::string(cwd) + "/" + p : p;
}

// a small /proc file as a string; "" when the process is gone (an istreambuf_iterator over such a file throws std::ios_base::failure: the pool died of it, 816440)
std::string slurp(const std::string& path) {
    std::string s;
    if (FILE* f = fopen(path.c_str(), "r")) {
        char b[4096];
        size_t n;
        while ((n = fread(b, 1, sizeof b, f)) > 0) s.append(b, n);
        fclose(f);
    }
    return s;
}

// /proc/<pid>/stat field 22 (start time in clock ticks since boot): with the pid, a process's identity for a sampler
unsigned long long proc_starttime(pid_t pid) {
    std::string s = slurp("/proc/" + std::to_string(pid) + "/stat");
    const size_t r = s.rfind(')');
    if (r == std::string::npos) return 0;
    std::istringstream in(s.substr(r + 2));
    std::string tok;
    for (int i = 3; i <= 22 && (in >> tok); ++i)
        if (i == 22) return std::strtoull(tok.c_str(), nullptr, 10);
    return 0;
}

// rt/hostmem sampler (SINGLET_POOL_MEMSAMPLE=<period s>): every period one line per process of the pool, its server and each sample's process
// group (the sample, its STAR child and probes): `unix-time item pid comm RssAnon RssFile RssShmem` (kB; item -1: the pool, -2: its server).
// comm is the main thread's name, which the pipeline sets per phase. The counters are /proc/<pid>/status's (no page-table walk).
void memsample_write(FILE* out, pid_t self, pid_t srv, const std::map<pid_t, int>& byPid) {
    const long now = (long) time(nullptr);
    if (DIR* d = opendir("/proc")) {
        while (struct dirent* e = readdir(d)) {
            if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
            const pid_t p = (pid_t) std::atoi(e->d_name);
            int item = -9;
            if (p == self) item = -1;
            else if (p == srv) item = -2;
            else {
                std::string s = slurp("/proc/" + std::string(e->d_name) + "/stat");
                const size_t r = s.rfind(')');
                if (r == std::string::npos) continue;
                std::istringstream in(s.substr(r + 2));
                std::string st, ppid, pg;
                in >> st >> ppid >> pg;
                auto it = byPid.find((pid_t) std::atoi(pg.c_str()));
                if (it != byPid.end()) item = it->second;
            }
            if (item == -9) continue;
            std::ifstream f("/proc/" + std::string(e->d_name) + "/status");
            std::string l, comm;
            long an = 0, fi = 0, sh = 0;
            while (std::getline(f, l)) {
                if (l.rfind("Name:", 0) == 0) { comm = l.substr(5); comm.erase(0, comm.find_first_not_of(" \t")); }
                else if (l.rfind("RssAnon:", 0) == 0) an = std::atol(l.c_str() + 8);
                else if (l.rfind("RssFile:", 0) == 0) fi = std::atol(l.c_str() + 8);
                else if (l.rfind("RssShmem:", 0) == 0) sh = std::atol(l.c_str() + 9);
            }
            if (an + fi + sh > 262144) fprintf(out, "%ld\t%d\t%d\t%s\t%ld\t%ld\t%ld\n", now, item, (int) p, comm.c_str(), an, fi, sh);
        }
        closedir(d);
    }
    fflush(out);
}

// "Number of input reads |	66601652" of STAR's Log.final.out in the sample's output directory (0 if absent)
unsigned long long read_pairs(const std::string& out) {
    for (const char* name : {"/star_Log.final.out", "/Log.final.out"}) {
        std::ifstream f(out + name);
        std::string l;
        while (std::getline(f, l))
            if (l.find("Number of input reads |") != std::string::npos) return std::strtoull(l.substr(l.find('|') + 1).c_str(), nullptr, 10);
    }
    return 0;
}

std::string fmt(const char* f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}

// ---- cggate (SINGLET_POOL_PEAKADMIT=1, docs/gpu-engine/runtime/T2-cggate.md): helpers of the peak-model gate
// The job-level cgroup as avail_gb() finds it: the ancestor with a finite memory.max and the smallest headroom (limit - usage + half of file - shmem). cgroup v2 only.
struct JobCg { bool ok = false; std::string dir; double lim = 0; };
JobCg find_job_cgroup() {
    JobCg best;
    double bestHead = 1e30;
    std::ifstream cl("/proc/self/cgroup");
    std::string line, path;
    while (std::getline(cl, line)) if (line.rfind("0::", 0) == 0) { path = line.substr(3); break; }
    if (path.empty()) return best;
    for (std::string p = path;;) {
        const std::string base = "/sys/fs/cgroup" + (p == "/" ? std::string() : p);
        std::ifstream mx(base + "/memory.max"), cu(base + "/memory.current"), stf(base + "/memory.stat");
        std::string m;
        double cur = 0;
        if (mx >> m && cu >> cur && !m.empty() && std::isdigit((unsigned char) m[0])) {
            const double lim = std::atof(m.c_str());
            if (lim < 1e18) {
                double file = 0, shmem = 0, sv = 0;
                std::string sk;
                while (stf >> sk >> sv) { if (sk == "file") file = sv; else if (sk == "shmem") shmem = sv; }
                const double head = lim - cur + 0.5 * std::max(0.0, file - shmem);
                if (head < bestHead) { bestHead = head; best.ok = true; best.dir = base; best.lim = lim; }
            }
        }
        if (p.empty() || p == "/") break;
        const size_t s = p.rfind('/');
        p = s == 0 || s == std::string::npos ? std::string("/") : p.substr(0, s);
    }
    return best;
}
// GiB of the job cgroup: memory.current, and from memory.stat anon, shmem, file
struct CgNow { bool ok = false; double cur = 0, anon = 0, shmem = 0, file = 0, lim = 0; double as() const { return anon + shmem; } };
CgNow read_job_cgroup(const JobCg& j) {
    CgNow r;
    if (!j.ok) return r;
    std::ifstream cu(j.dir + "/memory.current"), stf(j.dir + "/memory.stat");
    double cur = 0, sv = 0;
    std::string sk;
    bool gotAnon = false;
    if (!(cu >> cur)) return r;
    while (stf >> sk >> sv) {
        if (sk == "anon") { r.anon = sv; gotAnon = true; } else if (sk == "shmem") r.shmem = sv; else if (sk == "file") r.file = sv;
    }
    if (!gotAnon) return r;
    const double G = 1073741824.0;
    r.cur = cur / G; r.anon /= G; r.shmem /= G; r.file /= G; r.lim = j.lim / G;
    r.ok = true;
    return r;
}
// the first block's cDNA bases per read of a .1fq (the pipeline's head scan on block 0); cached per path
struct FqBpr { double bpr = -1; std::string why; };
FqBpr fq_first_block_bpr(const std::string& path) {
    static std::map<std::string, FqBpr> cache;
    auto f = cache.find(path);
    if (f != cache.end()) return f->second;
    FqBpr r;
    try {
        singlet::fq::RunSet rs;
        rs.open(std::vector<std::string>{path});
        std::string w;
        if (rs.block_count() == 0) r.why = "no blocks";
        else if (!rs.head_scan_supported(&w)) r.why = "the head scan does not support this .1fq (" + w + ")";
        else {
            singlet::fq::RunSet::HeadScanOptions o;
            singlet::fq::RunSet::BlockHead h;
            rs.scan_head(0, o, h);
            if (h.n_out > 0 && h.bases_out > 0) r.bpr = (double) h.bases_out / (double) h.n_out; else r.why = "the first block has no reads for the mapper";
        }
    } catch (const std::exception& e) { r.why = std::string("the first block could not be read: ") + e.what(); }
    cache[path] = r;
    return r;
}

volatile sig_atomic_t g_stop = 0;
void on_stop(int) { g_stop = 1; }

struct Item { int idx = 0; std::string fq, out, genome; std::vector<std::string> extra; };
struct Run {
    int idx = 0, slot = 0;
    pid_t pid = -1;
    int logFd = -1;
    FILE* log = nullptr;
    std::string partial;
    double t0 = 0, t1 = 0, killAt = 0;
    bool done = false, killed = false;
    bool front = true;   // rt/hostmem: in the front half (from the start until the engine's "[engine_gpu] level" line: the packed planes and aux are freed by then)
    bool plateau = false;   // rt/phaseadmit: the all-site pass has been joined ("[allsite] async: waited"): host memory is on its tail plateau and no longer grows with depth
    mutable double charge = 0;      // rt/phaseadmit: the charge (GB, whole item) last computed for this item, for the phase-drop log line
    bool decoded = false, dropped = false;   // rt/hostmem: the engine decoded the .1fq into the packed buffers (its page cache can go)
    // cggate (SINGLET_POOL_PEAKADMIT=1): the item's peak figure (GiB above the job's idle baseline) and what the engine printed
    bool pkChim = false;                        // the item's own argument list has --chimeric: the second pass' floor on every figure
    // T2 chimshare: pkChimShared = the pool established that the item's pass takes the pool's resident genome (floor chimSharedGb, else chimGb); keptGenome = the item's parent keeps its mapping of
    // that genome from the first STAR fork ("[singlet] chimeric: genome kept for the pass") until the pass's fork or release (its group then maps it twice); keptSeen: the kept line came;
    // demoted: the item reached its engine level line without the kept line, so its floor is chimGb from then on
    bool pkChimShared = false, keptGenome = false, keptSeen = false, demoted = false, demotedLogged = false;
    mutable double heldPeak = 0, heldPeakStar = 0;   // the gate's held figure (GiB): the maximum over the item, and after its engine level line
    bool pk = false, pkZone = false;            // the figure comes from the peak model (else the old estimate), the item is in the zone of a possible fallback
    double pkN = 0, pkBases = 0;                // M reads, G bases (from the .1fq headers and their first blocks)
    double pkP = 0, pkFig = 0, pkFloor = 0;     // P now (no margin), the figure now (with margin), a raise of the figure that holds for pkFloorPhase
    int pkFloorPhase = -1;
    bool pm1 = false, pm2 = false, pm3 = false, pmFallback = false, pm1Done = false, pm3Done = false, pmFbDone = false;
    double pmN = 0, pmD = 0, pmBases = 0, pmRc = 0, pmFb = 0, pmRegion = 0, pmUnmapped = 0, pmLate = 0, pmKeep = 0;
    int pkPhase = 0;
};

// cggate: "[engine_gpu] peakmodel: key=value ..." lines of the engine (SINGLET_ENGINE_GPU_PEAKMODEL=1): the first (n nw D rc bases budget_gib need_est_gib fb), the second (records aln ops
// seqbytes region_bytes hostrecs_bytes), the third (unmapped late_bytes solokeep_bytes), and a late "fallback=1 D=" when E8's records were not built
bool g_peakParse = false;
const char kPeakTag[] = "[engine_gpu] peakmodel: ";
void peak_parse_line(Run& r, const std::string& body) {
    std::map<std::string, double> kv;
    std::istringstream in(body);
    std::string tok;
    while (in >> tok) {
        const size_t q = tok.find('=');
        if (q != std::string::npos) kv[tok.substr(0, q)] = std::atof(tok.c_str() + q + 1);
    }
    if (kv.count("fallback")) { r.pmFallback = true; if (kv.count("D")) r.pmD = kv["D"]; return; }
    if (kv.count("rc") && kv.count("D") && kv.count("n")) {
        r.pm1 = true; r.pmN = kv["n"]; r.pmD = kv["D"]; r.pmRc = kv["rc"]; r.pmBases = kv.count("bases") ? kv["bases"] : 0; r.pmFb = kv.count("fb") ? kv["fb"] : 0;
    } else if (kv.count("region_bytes")) {
        r.pm2 = true; r.pmRegion = kv["region_bytes"];
    } else if (kv.count("late_bytes")) {
        r.pm3 = true; r.pmUnmapped = kv.count("unmapped") ? kv["unmapped"] : 0; r.pmLate = kv["late_bytes"]; r.pmKeep = kv.count("solokeep_bytes") ? kv["solokeep_bytes"] : 0;
    }
}

// the sample's log: each complete line with the unix time it arrived
void pump(Run& r, bool eof) {
    char buf[65536];
    for (;;) {
        const ssize_t n = read(r.logFd, buf, sizeof buf);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) eof = true;
            break;
        }
        r.partial.append(buf, (size_t) n);
        size_t s = 0, e;
        const long t = (long) time(nullptr);
        while ((e = r.partial.find('\n', s)) != std::string::npos) {
            if (r.log) fprintf(r.log, "%ld %.*s\n", t, (int) (e - s), r.partial.data() + s);
            if (r.front && r.partial.compare(s, 19, "[engine_gpu] level ") == 0) { r.front = false; if (r.pkChim && r.pkChimShared && !r.keptSeen) { r.pkChimShared = false; r.demoted = true; } }
            if (r.partial.compare(s, 45, "[singlet] chimeric: genome kept for the pass ") == 0) { r.keptGenome = true; r.keptSeen = true; }
            else if (r.keptGenome && (r.partial.compare(s, 49, "[singlet] chimeric: genome shared with the pool (") == 0 || r.partial.compare(s, 45, "[singlet] chimeric: genome released (pass not") == 0)) r.keptGenome = false;
            if (!r.plateau && r.partial.compare(s, 24, "[allsite] async: waited ") == 0) r.plateau = true;
            if (!r.decoded) { const size_t dp = r.partial.find("decoded into the packed buffers", s); if (dp != std::string::npos && dp < e) r.decoded = true; }
            if (g_peakParse && r.partial.compare(s, sizeof kPeakTag - 1, kPeakTag) == 0) peak_parse_line(r, r.partial.substr(s + sizeof kPeakTag - 1, e - s - (sizeof kPeakTag - 1)));
            s = e + 1;
        }
        r.partial.erase(0, s);
    }
    if (eof) {
        if (r.log) {
            if (!r.partial.empty()) fprintf(r.log, "%ld %s\n", (long) time(nullptr), r.partial.c_str());
            fclose(r.log);
            r.log = nullptr;
        }
        r.partial.clear();
        if (r.logFd >= 0) close(r.logFd);
        r.logFd = -1;
    } else if (r.log) fflush(r.log);
}

// the non-host screen's files as the pipeline resolves them by default (its NONHOST-AUTO-DISCOVER: explicit --nonhost-db / --bacterial-db /
// --fungal-db with discovery in the same directory, else /dev/shm/singlet_nonhost/ or $SINGLET_REF_BASE/nonhost/; the Bloom filter
// host_21mer.bloom beside the first index, else in $SINGLET_REF_BASE/nonhost/). Only a warm start: a sample that resolves other files is
// served from what its first request names, or declined and loads its own.
#ifdef SINGLET_HAVE_ENGINE_GPU
singlet_engine_gpu::NonhostPaths warm_paths(const std::vector<std::string>& a) {
    singlet_engine_gpu::NonhostPaths p;
    const bool noV = has_flag(a, "--no-viral-screen"), noM = has_flag(a, "--no-microbial-screen");
    std::string v = noV ? "" : arg_value(a, "--nonhost-db"), b = noM ? "" : arg_value(a, "--bacterial-db"), f = noM ? "" : arg_value(a, "--fungal-db");
    if (!v.empty() || !b.empty() || !f.empty()) {
        const std::string& anchor = !v.empty() ? v : !b.empty() ? b : f;
        const std::string dir = anchor.substr(0, anchor.rfind('/') + 1);
        if (v.empty() && !noV && file_exists(dir + "viral.snhskidx")) v = dir + "viral.snhskidx";
        if (b.empty() && !noM && file_exists(dir + "bacterial.snhskidx")) b = dir + "bacterial.snhskidx";
        if (f.empty() && !noM && file_exists(dir + "fungal.snhskidx")) f = dir + "fungal.snhskidx";
    } else {
        const std::string shm = "/dev/shm/singlet_nonhost/";
        const char* rb = std::getenv("SINGLET_REF_BASE");
        const std::string nfs = rb ? std::string(rb) + "/nonhost/" : "";
        auto res = [&](const std::string& name) -> std::string {
            if (file_exists(shm + name)) return shm + name;
            if (!nfs.empty() && file_exists(nfs + name)) return nfs + name;
            return "";
        };
        if (!noV) v = res("viral.snhskidx");
        if (!noM) { b = res("bacterial.snhskidx"); f = res("fungal.snhskidx"); }
    }
    if (v.empty() || b.empty() || f.empty()) return p;   // the GPU screen needs all three (the pipeline's use_gpu)
    const std::string dir = v.substr(0, v.rfind('/') + 1);
    std::string bloom = dir + "host_21mer.bloom";
    if (!file_exists(bloom)) {
        const char* rb = std::getenv("SINGLET_REF_BASE");
        bloom = rb ? std::string(rb) + "/nonhost/host_21mer.bloom" : "";
    }
    if (!file_exists(bloom)) return p;
    p.viral = v; p.bacterial = b; p.fungal = f; p.bloom = bloom;
    return p;
}
#endif

void close_fds_from(int lo, int keep = -1) {
    std::vector<int> fds;
    if (DIR* d = opendir("/proc/self/fd")) {
        const int dfd = dirfd(d);
        while (struct dirent* e = readdir(d)) {
            const int fd = std::atoi(e->d_name);
            if (e->d_name[0] != '.' && fd >= lo && fd != dfd && fd != keep) fds.push_back(fd);
        }
        closedir(d);
    }
    for (int fd : fds) close(fd);
}

}   // namespace

// The pool. argv: the arguments after `gpu-pool`. pipeline: the `singlet` command's main (singlet.cpp), run in each sample's process.
// preload_refs (singlet.cpp, may be null): parses the pileup references of the samples' arguments once, in the pool, before any fork.
int singlet_gpu_pool(int argc, char** argv, int (*pipeline)(int, char**),
                     std::string (*preload_refs)(const std::vector<std::vector<std::string>>&)) {
    int device = -1, slots = 2, workers = 8, threads = 7, preloadThreads = 8;
    double timeout = 0;
    std::string queue, args, report;
    bool server = true, preload = true, sharedIdx = true;
    for (int i = 0; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "gpu-pool: " << a << " needs a value\n"; exit(2); }
            return argv[++i];
        };
        if (a == "--device") device = std::atoi(val().c_str());
        else if (a == "--slots") slots = std::atoi(val().c_str());
        else if (a == "--workers") workers = std::atoi(val().c_str());
        else if (a == "--threads") threads = std::atoi(val().c_str());
        else if (a == "--queue") queue = val();
        else if (a == "--args") args = val();
        else if (a == "--report") report = val();
        else if (a == "--timeout") timeout = std::atof(val().c_str());
        else if (a == "--preload-threads") preloadThreads = std::atoi(val().c_str());
        else if (a == "--nonhost-server") server = val() != "off";
        else if (a == "--genome-preload") preload = val() != "off";
        else if (a == "--shared-index") sharedIdx = val() != "off";
        else if (a == "-h" || a == "--help") {
            std::cout << "singlet gpu-pool --device D --queue FILE [--slots 2] [--workers 8] [--threads 7] [--args \"<pipeline arguments>\"]\n"
                         "                 [--report DIR] [--nonhost-server on|off] [--genome-preload on|off] [--shared-index on|off]\n"
                         "                 [--timeout S] [--preload-threads 8]\n"
                         "Queue lines: <sample.1fq> <out dir> [extra pipeline arguments]. docs/gpu-engine/fable/GPU-PATH.md, E21-pool.md.\n";
            return 0;
        } else { std::cerr << "gpu-pool: unknown option " << a << " (--help)\n"; return 2; }
    }
    if (device < 0 || queue.empty() || workers < 1 || threads < 1 || slots < 1) {
        std::cerr << "gpu-pool: --device D and --queue FILE are required (--help)\n";
        return 2;
    }
    const std::vector<std::string> base = split_args(args);
    std::vector<Item> items;
    {
        std::ifstream q(queue);
        if (!q) { std::cerr << "gpu-pool: cannot read the queue " << queue << "\n"; return 2; }
        std::string line;
        int ln = 0;
        while (std::getline(q, line)) {
            ++ln;
            const size_t h = line.find_first_not_of(" \t\r");
            if (h == std::string::npos || line[h] == '#') continue;
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
            std::vector<std::string> w = split_args(line);
            if (w.size() < 2) { std::cerr << "gpu-pool: queue line " << ln << ": needs <sample.1fq> <out dir>\n"; return 2; }
            Item it;
            it.idx = (int) items.size();
            it.fq = absolute(w[0]); it.out = absolute(w[1]);
            it.extra.assign(w.begin() + 2, w.end());
            std::vector<std::string> all = base;
            all.insert(all.end(), it.extra.begin(), it.extra.end());
            it.genome = arg_value(all, "--genome-dir");
            items.push_back(it);
        }
    }
    const pid_t self = getpid();
    if (report.empty()) report = "gpu-pool-d" + std::to_string(device) + "-" + std::to_string(self);
    mkdir_p(report);
    g_plog = fopen((report + "/pool.log").c_str(), "a");
    FILE* pids = fopen((report + "/pids.tsv").c_str(), "a");
    FILE* stsv = fopen((report + "/samples.tsv").c_str(), "a");
    if (!g_plog || !pids || !stsv) { std::cerr << "gpu-pool: cannot write the report directory " << report << "\n"; return 2; }
    fprintf(stsv, "item\tslot\tsample\tout\tstart\tend\twall_s\tuser_s\tsys_s\tcore_s\tmaxrss_gb\texit\tsignal\tpairs\tpid\n");
    fflush(stsv);
    auto pid_line = [&](pid_t p, const std::string& label) { fprintf(pids, "%d\t%s\t%llu\n", (int) p, label.c_str(), proc_starttime(p)); fflush(pids); };
    pid_line(self, "pool" + std::to_string(device));
    plog("pool pid %d on device %d: %zu samples, %d in flight, --threads %d, SINGLET_ENGINE_GPU_SLOTS=%d; report %s; args: %s", (int) self, device,
         items.size(), workers, threads, slots, report.c_str(), args.c_str());
    // ---- cggate: SINGLET_POOL_PEAKADMIT=1 (default 0: nothing below runs, nothing is written, the gate is as before). The gate then accounts on the job's cgroup:
    // see docs/gpu-engine/runtime/T2-cggate.md. Decided here, before the server and the genome load, because the pool announces itself to the other pools of the job at once.
    const bool peakReq = env_num("SINGLET_POOL_PEAKADMIT", 0) != 0 && env_num("SINGLET_POOL_MEMADMIT", 1) != 0;
    JobCg jobCg;
    bool peakMode = false;
    if (peakReq) {
        jobCg = find_job_cgroup();
        const CgNow c0 = read_job_cgroup(jobCg);
        if (!jobCg.ok || !c0.ok) plog("peak admission (SINGLET_POOL_PEAKADMIT=1) is not available: this process has no cgroup v2 ancestor with a memory limit (or its memory.stat has no anon); the current gate is used");
        else {
            peakMode = true;
            g_peakParse = true;
            setenv("SINGLET_ENGINE_GPU_PEAKMODEL", "1", 1);   // the items print their peakmodel lines (only in this mode)
            plog("peak admission on (SINGLET_POOL_PEAKADMIT=1): job cgroup %s, memory.max %.1f GiB, anon+shmem now %.1f GiB; the items print the engine's peakmodel lines (SINGLET_ENGINE_GPU_PEAKMODEL=1)",
                 jobCg.dir.c_str(), c0.lim, c0.as());
        }
    }
    const poolpeak::Consts pkK = poolpeak::fromEnv();
    const double cgShare = env_num("SINGLET_POOL_CG_CACHE_SHARE", 0.5), pkSettle = env_num("SINGLET_POOL_PEAK_SETTLE_S", 30), pkStableS = env_num("SINGLET_POOL_PEAK_STABLE_S", 10),
                 pkRangeGb = env_num("SINGLET_POOL_PEAK_B_RANGE_GB", 0.5), pkBWait = env_num("SINGLET_POOL_PEAK_B_WAIT_S", 120), pkNPools = env_num("SINGLET_POOL_PEAK_NPOOLS", 0);
    const char* gjob = std::getenv("SLURM_JOB_ID");
    const std::string gateDir = "/dev/shm/singlet-pool-mem-" + std::string(gjob ? gjob : "node") + "-" + std::to_string((unsigned) getuid());
    const std::string gateMine = gateDir + "/" + std::to_string((int) self);
    // the pools' files, mode 1: `t rem act waitEst waitSince figSum ready lastEvt` (the first five as before; rem = figSum), written whole and renamed
    auto pkWriteFile = [&](double figSum, int act, double waitFig, double waitSince, int ready, double lastEvt) {
        const std::string tmp = gateMine + ".tmp";
        if (FILE* f = fopen(tmp.c_str(), "w")) {
            fprintf(f, "%.3f %.3f %d %.3f %.3f %.3f %d %.3f\n", wall_now(), figSum, act, waitFig, waitSince, figSum, ready, lastEvt);
            fclose(f);
            rename(tmp.c_str(), gateMine.c_str());
        }
    };
    // until the stream starts the pool's start-up (server, genome, references) blocks for a minute or more: a thread keeps the pool's file fresh, `ready` 0, so that a pool that
    // measures the idle baseline sees this pool (its genome and server are not in the baseline yet) and does not take the baseline too low
    std::atomic<bool> pkHbStop{false};
    std::thread pkHb;
    struct PkHbGuard { std::atomic<bool>& s; std::thread& t; ~PkHbGuard() { s = true; if (t.joinable()) t.join(); } } pkHbGuard{pkHbStop, pkHb};
    if (peakMode) {
        mkdir(gateDir.c_str(), 0700);
        pkWriteFile(0, 0, 0, 0, 0, 0);
        pkHb = std::thread([&]() { while (!pkHbStop.load()) { pkWriteFile(0, 0, 0, 0, 0, 0); for (int i = 0; i < 20 && !pkHbStop.load(); ++i) usleep(100000); } });
    }
    setenv("SINGLET_ENGINE_GPU_SLOTS", std::to_string(slots).c_str(), 1);
    setenv("SINGLET_ENGINE_GPU_DEVICE", std::to_string(device).c_str(), 1);
    if (sharedIdx && std::getenv("SINGLET_GPU_SHARED_INDEX") == nullptr) setenv("SINGLET_GPU_SHARED_INDEX", "1", 1);
    { const char* si = std::getenv("SINGLET_GPU_SHARED_INDEX"); plog("shared device index: SINGLET_GPU_SHARED_INDEX=%s", si ? si : "(unset)"); }
    unsetenv("SINGLET_NONHOST_SERVER");
    struct sigaction sa{};
    sa.sa_handler = on_stop;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGTERM, &sa, nullptr);
    sigaction(SIGINT, &sa, nullptr);
    signal(SIGPIPE, SIG_IGN);
    const double tStart = wall_now();

    // ---- the non-host server, forked first (before the pool holds the genome, and the pool never initialises CUDA)
    pid_t srvPid = -1;
    std::string sock;
#ifdef SINGLET_HAVE_ENGINE_GPU
    if (server) {
        sock = "/tmp/singlet-gpu-pool-" + std::to_string(self) + "-d" + std::to_string(device) + ".sock";
        const singlet_engine_gpu::NonhostPaths warm = warm_paths(base);
        plog("non-host server: %s", warm.bloom.empty() ? "no default index files found; the first request's files are loaded"
                                                          : ("warm start with " + warm.viral + ", " + warm.bacterial + ", " + warm.fungal + ", " + warm.bloom).c_str());
        int rp[2];
        if (pipe(rp) == 0) {
            fflush(nullptr);
            std::cout.flush();
            srvPid = fork();
            if (srvPid == 0) {
                prctl(PR_SET_PDEATHSIG, SIGTERM);
                close(rp[0]);
                const int lf = open((report + "/nhsrv.log").c_str(), O_WRONLY | O_CREAT | O_APPEND, 0644);
                const int dn = open("/dev/null", O_RDONLY);
                if (dn >= 0) { dup2(dn, 0); close(dn); }
                if (lf >= 0) { dup2(lf, 1); dup2(lf, 2); close(lf); }
                close_fds_from(3, rp[1]);
                struct sigaction d{};
                d.sa_handler = SIG_DFL;
                sigaction(SIGTERM, &d, nullptr);
                sigaction(SIGINT, &d, nullptr);
                _exit(singlet_engine_gpu::nonhostServe(sock, device, std::max(4, std::min(12, threads)), warm, rp[1]));
            }
            close(rp[1]);
            char c = 0;
            struct pollfd pf = {rp[0], POLLIN, 0};
            const bool up = srvPid > 0 && poll(&pf, 1, 60000) > 0 && read(rp[0], &c, 1) == 1;
            close(rp[0]);
            if (up) {
                setenv("SINGLET_NONHOST_SERVER", sock.c_str(), 1);
                pid_line(srvPid, "nhsrv" + std::to_string(device));
                plog("non-host server pid %d listening on %s (its log: %s/nhsrv.log)", (int) srvPid, sock.c_str(), report.c_str());
            } else {
                plog("non-host server did not start: every sample loads the index itself");
                if (srvPid > 0) { kill(srvPid, SIGKILL); waitpid(srvPid, nullptr, 0); }
                srvPid = -1;
            }
        }
    }
#else
    if (server) plog("non-host server: not built (no CUDA engine)");
#endif

    // ---- the pileup references (each genome's GTF and SNP panel), once, on a thread beside the genome preload
    std::thread refsTh;
    std::string refsLog;
    double refsSec = 0;
    if (preload_refs) {
        std::vector<std::vector<std::string>> itemArgs;
        for (const Item& it : items) {
            std::vector<std::string> all = base;
            all.insert(all.end(), it.extra.begin(), it.extra.end());
            itemArgs.push_back(std::move(all));
        }
        refsTh = std::thread([&refsLog, &refsSec, preload_refs, itemArgs = std::move(itemArgs)]() {
            const double t0 = wall_now();
            refsLog = preload_refs(itemArgs);
            refsSec = wall_now() - t0;
        });
    }

    // ---- the genomes, once
    std::vector<std::pair<dev_t, ino_t>> residentDirs;   // device and inode of every index directory whose resident load succeeded
    const char* gpe = std::getenv("SINGLET_GENOME_PRELOAD");
    const std::string gload = arg_value(base, "--genome-load");
    if (preload && !(gpe && gpe[0] == '0') && (gload.empty() || gload == "NoSharedMemory")) {
        std::vector<std::pair<dev_t, ino_t>> seen;
        for (const Item& it : items) {
            struct stat st{};
            if (it.genome.empty() || stat(it.genome.c_str(), &st) != 0) continue;
            if (std::find(seen.begin(), seen.end(), std::make_pair(st.st_dev, st.st_ino)) != seen.end()) continue;
            seen.emplace_back(st.st_dev, st.st_ino);
            std::string lg;
            const bool ok = star_genome_preload_resident(it.genome, preloadThreads, lg);
            plog("genome %s: %s%s", it.genome.c_str(), ok ? "resident, " : "not preloaded: ", lg.c_str());
            if (ok) residentDirs.emplace_back(st.st_dev, st.st_ino);   // T2 chimshare: the directories this pool holds resident
        }
    } else plog("genome preload off: every sample reads its own");
    if (refsTh.joinable()) {
        refsTh.join();
        plog("pileup references (%.1f s): %s", refsSec, refsLog.c_str());
    }
    const double tReady = wall_now();
    plog("ready in %.2f s", tReady - tStart);

    // ---- the stream
    std::vector<Run> runs(items.size());
    std::vector<int> freeSlots;
    for (int s = workers - 1; s >= 0; --s) freeSlots.push_back(s);
    size_t next = 0, active = 0, failed = 0;
    std::map<pid_t, int> byPid;
    unsigned long long pairsAll = 0;
    double firstStart = 0, lastEnd = 0;
    double pkLastEvt = 0;   // cggate: the last start or end of an item of this pool (the idle baseline is not measured just after one)
    auto start = [&](Item& it) {
        Run& r = runs[it.idx];
        r.idx = it.idx;
        r.slot = freeSlots.back();
        freeSlots.pop_back();
        mkdir_p(it.out);
        int lp[2];
        if (pipe(lp) != 0) { plog("item %d: pipe: %s", it.idx, strerror(errno)); lp[0] = lp[1] = -1; }
        if (lp[0] >= 0) fcntl(lp[0], F_SETPIPE_SZ, 1 << 20);
        r.log = fopen((report + "/" + std::to_string(it.idx) + ".log").c_str(), "w");
        fflush(nullptr);
        std::cout.flush();
        r.t0 = wall_now();
        const pid_t pid = fork();
        if (pid == 0) {   // ---- the sample's process
            setpgid(0, 0);
            prctl(PR_SET_PDEATHSIG, SIGKILL);
            if (getppid() != self) _exit(125);
            struct sigaction d{};
            d.sa_handler = SIG_DFL;
            for (int s : {SIGTERM, SIGINT, SIGPIPE, SIGCHLD, SIGHUP}) sigaction(s, &d, nullptr);
            sigset_t none;
            sigemptyset(&none);
            sigprocmask(SIG_SETMASK, &none, nullptr);
            const int dn = open("/dev/null", O_RDONLY);
            if (dn >= 0) { dup2(dn, 0); close(dn); }
            if (lp[1] >= 0) { dup2(lp[1], 1); dup2(lp[1], 2); }
            close_fds_from(3);
            g_plog = nullptr;
            if (FILE* o = fopen("/proc/self/oom_score_adj", "w")) { fputs("500", o); fclose(o); }   // under memory pressure a sample goes first, not the pool
            if (chdir(parent_dir(it.out).c_str()) != 0) { }
            setenv("SINGLET_GPU_POOL_ITEM", std::to_string(it.idx).c_str(), 1);
            if (srvPid <= 0) unsetenv("SINGLET_NONHOST_SERVER");
            std::vector<std::string> av;
            av.push_back("singlet");
            av.push_back(it.fq);
            av.insert(av.end(), base.begin(), base.end());
            av.push_back("--out-prefix"); av.push_back(it.out);
            av.push_back("--threads"); av.push_back(std::to_string(threads));
            av.insert(av.end(), it.extra.begin(), it.extra.end());
            std::string cmd;
            for (const auto& w : av) cmd += (cmd.empty() ? "" : " ") + w;
            std::cerr << "[gpu-pool] item " << it.idx << " (pool pid " << self << ", device " << device << ", slots " << slots << "): " << cmd << "\n";
            if (!it.genome.empty()) {
                std::string lg;
                const bool ok = star_genome_preload_select(it.genome, lg);
                std::cerr << "[gpu-pool] genome " << it.genome << ": " << (ok ? "" : "read by this sample: ") << lg << "\n";
            }
            std::vector<char*> cv;
            for (auto& w : av) cv.push_back(&w[0]);
            cv.push_back(nullptr);
            const int rc = pipeline((int) av.size(), cv.data());
            std::cout.flush();
            std::cerr.flush();
            fflush(nullptr);
            exit(rc);
        }
        if (lp[1] >= 0) close(lp[1]);
        if (pid < 0) {
            plog("item %d: fork: %s", it.idx, strerror(errno));
            if (lp[0] >= 0) close(lp[0]);
            if (r.log) fclose(r.log);
            r.log = nullptr;
            r.done = true;
            ++failed;
            freeSlots.push_back(r.slot);
            return;
        }
        r.pid = pid;
        r.logFd = lp[0];
        if (r.logFd >= 0) fcntl(r.logFd, F_SETFL, fcntl(r.logFd, F_GETFL) | O_NONBLOCK);
        byPid[pid] = it.idx;
        if (peakMode) pkLastEvt = wall_now();
        ++active;
        if (firstStart == 0) firstStart = r.t0;
        pid_line(pid, "c" + std::to_string(it.idx));
        plog("start item %d (slot %d, pid %d): %s -> %s", it.idx, r.slot, (int) pid, it.fq.c_str(), it.out.c_str());
    };
    auto finish = [&](pid_t pid, int st, const struct rusage& ru) {
        auto f = byPid.find(pid);
        if (f == byPid.end()) return;
        Run& r = runs[f->second];
        const Item& it = items[f->second];
        byPid.erase(f);
        r.t1 = wall_now();
        r.done = true;
        r.keptGenome = false;
        if (r.heldPeak > 0) plog("item %d: held peak %.1f GiB (the gate's figure: process group RssAnon + RssShmem minus the pool's genome once per mapping process), %.1f GiB after the engine level line", r.idx, r.heldPeak, r.heldPeakStar);
        --active;
        if (peakMode) pkLastEvt = wall_now();
        freeSlots.push_back(r.slot);
        const double us = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6, ss = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
        const int ex = WIFEXITED(st) ? WEXITSTATUS(st) : -1, sg = WIFSIGNALED(st) ? WTERMSIG(st) : 0;
        const unsigned long long pairs = read_pairs(it.out);
        pairsAll += pairs;
        lastEnd = std::max(lastEnd, r.t1);
        if (ex != 0) {   // the sample's other processes (its STAR child, probes, ...) end with it: nothing it left keeps running or holds a device
            ++failed;
            kill(-pid, SIGKILL);
        }
        fprintf(stsv, "%d\t%d\t%s\t%s\t%.3f\t%.3f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\t%llu\t%d\n", r.idx, r.slot, it.fq.c_str(), it.out.c_str(), r.t0, r.t1,
                r.t1 - r.t0, us, ss, us + ss, ru.ru_maxrss / 1048576.0, ex, sg, pairs, (int) pid);
        fflush(stsv);
        if (FILE* tf = fopen((report + "/" + std::to_string(r.idx) + ".time").c_str(), "w")) {
            const double w = r.t1 - r.t0;
            fprintf(tf, "\tCommand being timed: \"singlet gpu-pool item %d: %s\"\n\tUser time (seconds): %.2f\n\tSystem time (seconds): %.2f\n"
                        "\tElapsed (wall clock) time (h:mm:ss or m:ss): %d:%05.2f\n\tMaximum resident set size (kbytes): %ld\n\tExit status: %d\n%s",
                    r.idx, it.fq.c_str(), us, ss, (int) (w / 60), std::fmod(w, 60.0), ru.ru_maxrss, ex,
                    sg ? fmt("\tCommand terminated by signal %d\n", sg).c_str() : "");
            fclose(tf);
        }
        plog("end item %d (pid %d): %s, wall %.1f s, core-s %.1f, peak RSS %.1f GB, %.2f M pairs%s", r.idx, (int) pid,
             sg ? fmt("killed by signal %d (%s)", sg, strsignal(sg)).c_str() : fmt("exit %d", ex).c_str(), r.t1 - r.t0, us + ss,
             ru.ru_maxrss / 1048576.0, pairs / 1e6, r.killed ? " (timed out)" : "");
    };
    double lastStartAt = 0, lastSample = 0, lastStartEst = 0;   // lastStartEst: the estimate (GB) of the item started last: the stagger is for items of 50 GB and more
    bool admitNoted = false;
    const double sampPeriod = env_num("SINGLET_POOL_MEMSAMPLE", 0);
    FILE* msf = sampPeriod > 0 ? fopen((report + "/memsamp.tsv").c_str(), "a") : nullptr;
    const double stagger = env_num("SINGLET_POOL_STAGGER_S", 40), memFloor = env_num("SINGLET_POOL_MEM_FLOOR_GB", 0), frontMax = env_num("SINGLET_POOL_FRONT_MAX", 0);
    // rt/hostmem SINGLET_POOL_MEMADMIT (default on, =0 off): node-wide admission by a per-item host-memory estimate. The pools of a node (one per GPU)
    // publish, once a second, what their items in flight can still allocate (estimate minus what they hold) in /dev/shm/singlet-pool-mem-<job>-<uid>/<pid>;
    // the decision and the registration of a start are made under a flock, so two pools cannot both take the last room. An item starts when
    //   free now (MemFree + SINGLET_POOL_CACHE_SHARE (0.25) of the file cache, which other tenants' load makes slow to reclaim)
    //   - what every pool's items can still allocate - the item's estimate - the estimate of a waiting item of another pool that waited longer
    //   >= SINGLET_POOL_MEM_RESERVE_GB (40: MemFree min was 29 GB at pressure 0 % with THP auto, 816788).
    // Estimate (SINGLET_POOL_ITEM_GB overrides): 11.5 GB per GB of .1fq + 8: d300_3p (6.0 GB file) peaks at 77 GB private (planes, aux, region, per-read
    // tables), d1000_3p (24.9 GB) at 285 GB (rt-deep 815819). An item in its tail (after the engine's "[engine_gpu] level" line) is counted at
    // 40 % of the estimate. An item with nothing else running on the node always starts, whatever its size (it runs alone). The wait is bounded
    // (SINGLET_POOL_MEMADMIT_MAXWAIT_S, 1800): then it starts, with a log line. The page cache of an input is dropped when the item has decoded
    // it (SINGLET_POOL_DROP_INPUT_CACHE, default on) and, for the clean pages, at pool start.
    const bool memAdmit = env_num("SINGLET_POOL_MEMADMIT", 1) != 0;
    const double itemGbEnv = env_num("SINGLET_POOL_ITEM_GB", 0), cacheShare = env_num("SINGLET_POOL_CACHE_SHARE", 0.25),
                 memReserve = env_num("SINGLET_POOL_MEM_RESERVE_GB", 40), maxWait = env_num("SINGLET_POOL_MEMADMIT_MAXWAIT_S", 1800);
    const bool dropInput = env_num("SINGLET_POOL_DROP_INPUT_CACHE", 1) != 0;
    // rt/phaseadmit SINGLET_POOL_PHASE_ADMIT (default on, =0 restores the admission above exactly): an item whose all-site pass is joined (the "[allsite] async: waited" line;
    // the STAR child has exited, the planes and the hand-off pass are long gone) sits on a tail plateau that does not grow with depth: its charge drops from 40 % of its
    // estimate to min(that, SINGLET_POOL_TAIL_GB). Fitted on memsamp of mix4 817054 (anon+shmem of the item's process group, 10 s after the line): d2600 38, d1000 65,
    // d1000s 61, d700s 49, d500s 38, d300 17 GB, so 80 = the largest plus a margin of 15. What the item holds is never charged less (the charge is on top of what MemFree already lost).
    // The head item is re-tested at every loop turn (200 ms) while a slot is free, with the pool's own charges recomputed from /proc then, so a drop admits it at once.
    const bool phaseAdmit = env_num("SINGLET_POOL_PHASE_ADMIT", 1) != 0;
    const double tailGb = env_num("SINGLET_POOL_TAIL_GB", 80);
    // SINGLET_POOL_SKIP_AHEAD (16, 0 = strict queue order): while the head of the queue waits for room, an item among the next 16 that fits starts instead,
    // for at most SINGLET_POOL_SKIP_GRACE_S (900) of the head's wait; after that only the head is tried, and another pool's head that waited that long is served first
    const double skipAhead = env_num("SINGLET_POOL_SKIP_AHEAD", 16), skipGrace = env_num("SINGLET_POOL_SKIP_GRACE_S", 900), prioAfter = skipAhead > 0 ? skipGrace : 0;
    std::vector<char> taken(items.size(), 0);
    std::vector<double> itemEst(items.size(), 0);
    std::vector<char> itemChimShared(items.size(), 0);   // T2 chimshare: the pool established that this --chimeric item's pass takes the pool's resident genome
    std::vector<char> itemPk(items.size(), 0), itemPkChim(items.size(), 0);   // cggate: the peak model applies to the item
    std::vector<double> itemFig0(items.size(), 0), itemPkN(items.size(), 0), itemPkBases(items.size(), 0), itemPkP(items.size(), 0);
    for (const Item& it : items) {
        struct stat st{};
        // rt/hostmem2: the item's host peak (GB of its process group's RssAnon + RssShmem, minus the pool's genome that its STAR child maps), fitted on the 110 items
        // of mix3 816788 (peak of the per-item sum in memsamp.tsv against the reads in the .1fq header): peak = a + b x reads (M). Least squares gave
        // 48.2 + 0.2235 x (rmse 13.6 GB, worst under-prediction 30.3 GB: d500s_3p), the margin adds 30 GB, the genome comes off (-16.4): a = 61.8, b = 0.2235.
        // The reads are the u32 at byte 20 of the .1fq header ("1FQ\0" magic); without a header, the file size: 11.6 GB per GB + 43.6 (envelope of the 16 samples).
        // A sample of several runs has one input per run: the queue line's first file, then every other run on the line (a positional .1fq / .1fg2, or the
        // comma list of --1fq, as `singlet` itself reads them; relative paths are the sample process's, from its out dir's parent). All of them count:
        // reading the first alone charged a 4-run 337 M-pair sample 85 GB, it peaked at 154 GB and the job's cgroup ended it (Clipper 818032).
        std::vector<std::string> ins{it.fq};
        for (size_t k = 0; k < it.extra.size(); ++k) {
            const std::string& w = it.extra[k];
            auto ends = [&w](const char* x) { const size_t n = std::strlen(x); return w.size() > n && w.compare(w.size() - n, n, x) == 0; };
            if (w == "--1fq" && k + 1 < it.extra.size()) {
                std::istringstream iss(it.extra[++k]);
                for (std::string f; std::getline(iss, f, ',');) if (!f.empty()) ins.push_back(f);
            } else if (!w.empty() && w[0] != '-' && (ends(".1fq") || ends(".1fg2"))) ins.push_back(w);
        }
        double reads = 0, unheadedGb = 0;
        std::vector<std::pair<std::string, double>> inList;   // cggate: each input and its reads
        for (std::string p : ins) {
            if (!p.empty() && p[0] != '/') p = parent_dir(it.out) + "/" + p;
            double r = 0;
            if (FILE* hf = fopen(p.c_str(), "rb")) { unsigned char h[24]; if (fread(h, 1, 24, hf) == 24 && h[0] == '1' && h[1] == 'F' && h[2] == 'Q' && h[3] == 0) r = (double) (h[20] | (h[21] << 8) | (h[22] << 16) | ((unsigned) h[23] << 24)); fclose(hf); }
            inList.emplace_back(p, r);
            if (r > 0) reads += r;
            else unheadedGb += stat(p.c_str(), &st) == 0 ? (double) st.st_size / 1073741824.0 : 6.0;
        }
        const double estA = env_num("SINGLET_POOL_EST_BASE_GB", 61.8), estB = env_num("SINGLET_POOL_EST_GB_PER_MPAIR", 0.2235);
        itemEst[it.idx] = itemGbEnv > 0 ? itemGbEnv : reads > 0 ? estA + estB * reads / 1e6 + unheadedGb * 11.6 : unheadedGb * 11.6 + 43.6;
        if (ins.size() > 1) plog("item %d: %zu inputs, %.1f M pairs in their headers, estimate %.0f GB", it.idx, ins.size(), reads / 1e6, itemEst[it.idx]);
        if (peakMode) {   // cggate: the item's figure from the peak model, or the old estimate with the reason
            std::string why;
            double bases = 0;
            if (itemGbEnv > 0) why = "SINGLET_POOL_ITEM_GB is set";
            else if (reads <= 0 || unheadedGb > 0) why = "an input has no readable .1fq header";
            else
                for (const auto& ir : inList) {
                    const FqBpr b = fq_first_block_bpr(ir.first);
                    if (b.bpr <= 0) { why = "the bases cannot be read: " + b.why; break; }
                    bases += ir.second * b.bpr;
                }
            if (why.empty()) {
                bool hcOff = false;
                for (const char* sw : {"SINGLET_ENGINE_GPU_LEAN_DROP", "SINGLET_ENGINE_GPU_CLIP16", "SINGLET_ENGINE_GPU_READREP_FREE", "SINGLET_ENGINE_GPU_VERIFY_FREE", "SINGLET_ENGINE_GPU_AUX_COMPACT", "SINGLET_ENGINE_GPU_PRE_RELEASE"}) {
                    const char* e = std::getenv(sw);
                    if (e != nullptr && e[0] != 0 && !(e[0] == '1' && e[1] == 0)) hcOff = true;
                }
                const char* mg = std::getenv("SINGLET_ENGINE_GPU_MEM_GB");
                std::string w2;
                if (!poolpeak::fitValid(reads / 1e6, mg != nullptr && mg[0] != 0 ? std::atof(mg) : -1.0, env_num("SINGLET_ENGINE_GPU_SLOTS", -1), hcOff, pkK, w2)) why = w2;
            }
            if (why.empty()) {
                const double nM = reads / 1e6, bG = bases / 1e9, P = poolpeak::peakP(nM, bG, pkK);
                { std::vector<std::string> allArgs = base; allArgs.insert(allArgs.end(), it.extra.begin(), it.extra.end()); itemPkChim[it.idx] = has_flag(allArgs, "--chimeric") ? 1 : 0;
                  // T2 chimshare: the pass shares the genome only when the item's preload is this pool's resident copy. The pool establishes that when: the switch is not 0, the preload is not off, the
                  // genome load is NoSharedMemory, no sjdb insertion or two-pass mode is requested, and the item's --genome-dir (device and inode) is a directory whose resident load succeeded.
                  // Anything else, and whenever it cannot tell: the pass is budgeted with its private index (chimGb). A --chimeric item that reaches its engine level line without printing
                  // "genome kept for the pass" is switched to chimGb from then on (pump).
                  const char* cse = std::getenv("SINGLET_CHIM_SHARE_GENOME");
                  struct stat gs{};
                  bool sh = itemPkChim[it.idx] != 0 && !(cse && cse[0] == '0') && preload && !(gpe && gpe[0] == '0') && (gload.empty() || gload == "NoSharedMemory") && !it.genome.empty()
                            && !has_flag(allArgs, "--sjdbGTFfile") && !has_flag(allArgs, "--sjdbFileChrStartEnd") && !has_flag(allArgs, "--twopassMode") && stat(it.genome.c_str(), &gs) == 0
                            && std::find(residentDirs.begin(), residentDirs.end(), std::make_pair(gs.st_dev, gs.st_ino)) != residentDirs.end();
                  itemChimShared[it.idx] = sh ? 1 : 0; }   // the item's own arguments (pool --args and the queue line), not the environment
                itemPk[it.idx] = 1; itemPkN[it.idx] = nM; itemPkBases[it.idx] = bG; itemPkP[it.idx] = P; itemFig0[it.idx] = poolpeak::phaseFigure(P, 0, pkK, itemPkChim[it.idx] == 0 ? poolpeak::Chim::None : itemChimShared[it.idx] ? poolpeak::Chim::Shared : poolpeak::Chim::Private);
                plog("item %d: peak model: %.1f M reads, %.2f G bases, P %.1f GiB%s, figure %.1f GiB (margin %.0f%s); the old estimate is %.0f GB", it.idx, nM, bG, P, poolpeak::inZone(nM, pkK) ? " (zone of a possible fallback)" : "", itemFig0[it.idx], pkK.margin, itemPkChim[it.idx] ? fmt(", --chimeric: floor 0.5 P + %.1f + margin", pkK.wtV + (itemChimShared[it.idx] ? pkK.chimSharedGb : pkK.chimGb)).c_str() : "", itemEst[it.idx]);
            } else {
                itemFig0[it.idx] = itemEst[it.idx];
                plog("item %d: peak model not used (%s): the old estimate, %.0f GB, is the figure", it.idx, why.c_str(), itemEst[it.idx]);
            }
        }
    }
    if (peakMode)
        for (size_t i = 0; i < items.size(); ++i) {
            Run& r = runs[i];
            r.pk = itemPk[i] != 0; r.pkN = itemPkN[i]; r.pkBases = itemPkBases[i]; r.pkP = itemPkP[i]; r.pkFig = itemFig0[i]; r.pkChim = itemPkChim[i] != 0; r.pkChimShared = r.pkChim && itemChimShared[i] != 0; r.pkZone = r.pk && poolpeak::inZone(r.pkN, pkK);
        }
    if (dropInput) {   // the harness's copy of the inputs left its pages in the cache (dirty at first): written out, then dropped, on a thread beside the start
        std::vector<std::string> fqs;
        for (const Item& it : items) if (std::find(fqs.begin(), fqs.end(), it.fq) == fqs.end()) fqs.push_back(it.fq);
        std::thread([fqs]() { for (const std::string& p : fqs) { const int fd = open(p.c_str(), O_RDONLY); if (fd >= 0) { fdatasync(fd); posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); } } }).detach();
    }
    FILE* heldF = sampPeriod > 0 ? fopen((report + "/memsamp-held.tsv").c_str(), "a") : nullptr;   // t item held_new held_old(sum - shared once) nprocs m kept front
    if (heldF != nullptr) fprintf(heldF, "# t\titem\theld_GiB\theld_old_formula_GiB\tprocs\tm\tkept\tfront\n");
    double sharedGb = 0;   // what a STAR child maps of the pool's resident genome: counted in its RSS, not in the item's own growth
    { std::ifstream f("/proc/self/status"); std::string l; while (std::getline(f, l)) if (l.rfind("RssAnon:", 0) == 0) sharedGb = std::atof(l.c_str() + 8) / 1048576.0; }
    if (memAdmit) mkdir(gateDir.c_str(), 0700);   // gateDir, gateMine: defined with the peak-admission decision above
    const std::string gateLock = gateDir + "/lock";
    double gateOwnRem = 0, gateLastScan = 0;
    int gateOwnAct = 0;
    // the pools' files: `t rem act waitEst waitSince`, written whole and renamed
    auto gatePublish = [&](double waitEst, double waitSince) {
        const std::string tmp = gateMine + ".tmp";
        if (FILE* f = fopen(tmp.c_str(), "w")) {
            fprintf(f, "%.3f %.3f %d %.3f %.3f\n", wall_now(), gateOwnRem, gateOwnAct, waitEst, waitSince);
            fclose(f);
            rename(tmp.c_str(), gateMine.c_str());
        }
    };
    struct GateOthers { double rem = 0, waitEst = 0, waitSince = 0; int act = 0; };
    auto gateReadOthers = [&]() {
        GateOthers o;
        if (DIR* d = opendir(gateDir.c_str())) {
            while (struct dirent* e = readdir(d)) {
                const int pid = std::atoi(e->d_name);
                if (pid <= 0 || pid == (int) self || std::strchr(e->d_name, '.') != nullptr) continue;
                double t = 0, rem = 0, we = 0, ws = 0;
                int act = 0;
                if (FILE* f = fopen((gateDir + "/" + e->d_name).c_str(), "r")) {
                    const int n = fscanf(f, "%lf %lf %d %lf %lf", &t, &rem, &act, &we, &ws);
                    fclose(f);
                    if (n != 5 || wall_now() - t > 20 || kill(pid, 0) != 0) continue;
                    o.rem += rem; o.act += act;
                    if (we > 0 && (o.waitEst == 0 || ws < o.waitSince)) { o.waitEst = we; o.waitSince = ws; }
                }
            }
            closedir(d);
        }
        return o;
    };
    auto gateOthersActive = [&]() { return gateReadOthers().act > 0; };
    // once a second: own items' growth still to come, from /proc (RssAnon + RssShmem of each item's process group)
    auto gateRefresh = [&](bool force) {
        const double now = wall_now();
        if (!force && now - gateLastScan < 1.0) return;
        gateLastScan = now;
        std::map<int, std::vector<poolpeak::ProcRss>> cur;
        if (DIR* d = opendir("/proc")) {
            while (struct dirent* e = readdir(d)) {
                if (e->d_name[0] < '0' || e->d_name[0] > '9') continue;
                std::string s = slurp("/proc/" + std::string(e->d_name) + "/stat");
                const size_t r = s.rfind(')');
                if (r == std::string::npos) continue;
                std::istringstream in(s.substr(r + 2));
                std::string st, ppid, pg;
                in >> st >> ppid >> pg;
                auto it = byPid.find((pid_t) std::atoi(pg.c_str()));
                if (it == byPid.end()) continue;
                std::ifstream g("/proc/" + std::string(e->d_name) + "/status");
                std::string l;
                double an = 0, sh = 0;
                while (std::getline(g, l)) { if (l.rfind("RssAnon:", 0) == 0) an = std::atof(l.c_str() + 8); else if (l.rfind("RssShmem:", 0) == 0) sh = std::atof(l.c_str() + 9); }
                poolpeak::ProcRss pr; pr.anonShmem = (an + sh) / 1048576.0; pr.anon = an / 1048576.0;
                cur[it->second].push_back(pr);
            }
            closedir(d);
        }
        double rem = 0;
        int act = 0;
        for (const Run& r : runs) {
            if (r.pid <= 0 || r.done) continue;
            ++act;
            // T2 chimshare: the single place a group sum becomes an item figure (poolpeak::heldGib: the pool's genome comes off once per process that maps it; twice while the
            // item's parent keeps its mapping beside the STAR child)
            int nMap = 1;
            const double held = poolpeak::heldGib(cur[r.idx], sharedGb, r.keptGenome, &nMap);
            r.heldPeak = std::max(r.heldPeak, held);
            if (!r.front) r.heldPeakStar = std::max(r.heldPeakStar, held);
            if (heldF != nullptr) { double sum = 0; for (const poolpeak::ProcRss& p : cur[r.idx]) sum += p.anonShmem; fprintf(heldF, "%.0f\t%d\t%.3f\t%.3f\t%zu\t%d\t%d\t%d\n", now, r.idx, held, sum - sharedGb, cur[r.idx].size(), nMap, r.keptGenome ? 1 : 0, r.front ? 1 : 0); }
            double want = itemEst[r.idx] * (r.front ? 1.0 : 0.4);
            if (phaseAdmit && r.plateau) want = std::min(want, tailGb);
            if (phaseAdmit && r.charge > 0 && want < r.charge - 0.5) plog("phase drop: item %d charge %.0f -> %.0f GB (holds %.0f GB)", r.idx, r.charge, want, held);
            if (phaseAdmit) r.charge = want;
            rem += std::max(0.0, want - held);
        }
        gateOwnRem = rem; gateOwnAct = act;
    };
    auto gateRegister = [&](double est) { gateOwnRem += est; gateOwnAct += 1; gatePublish(0, 0); };
    // under the flock: does an item of `est` fit? registers it when it does
    auto gateTry = [&](double est, double mySince, double reserve, double& freeEff, double& othRem, double& othWait, double& proj) {
        const int lk = open(gateLock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lk >= 0) flock(lk, LOCK_EX);
        const GateOthers o = gateReadOthers();
        double mfree = 0, cached = 0, shm = 0, v = 0;
        { std::ifstream mi("/proc/meminfo"); std::string k, u; while (mi >> k >> v >> u) { if (k == "MemFree:") mfree = v; else if (k == "Cached:") cached = v; else if (k == "Shmem:") shm = v; } }
        freeEff = (mfree + cacheShare * std::max(0.0, cached - shm)) / 1048576.0;
        othRem = o.rem;
        othWait = (o.waitEst > 0 && wall_now() - o.waitSince > prioAfter && o.waitSince <= mySince) ? o.waitEst : 0;   // a waiting item of another pool that waited longer goes first
        proj = freeEff - gateOwnRem - o.rem - est - othWait;
        const bool alone = gateOwnAct == 0 && o.act == 0;   // nothing else on the node: it runs alone, however large
        const bool ok = alone || proj >= reserve;
        if (ok) gateRegister(est);
        if (lk >= 0) { flock(lk, LOCK_UN); close(lk); }
        return ok;
    };
    // ======== cggate (SINGLET_POOL_PEAKADMIT=1): the gate on the job's cgroup. Everything below is unused when peakMode is false.
    // Per item a figure (GiB above the job's idle baseline B that it can still reach: T2-peakmodel.md section 2, include/singlet/gpu_pool_peak.h); B = anon + shmem of the job cgroup
    // with no item in flight, measured only when it cannot be too low (see pkMeasure). What the items in flight of the whole job can still allocate =
    // max(0, sum of their figures - (anon + shmem now - B)); with B unknown nothing counts as held (the sum of the figures: too high is safe, too low is not).
    struct PkOthers { double figSum = 0, waitFig = 0, waitSince = 0, lastEvt = 0; int act = 0, pools = 0, ready = 0; };
    auto pkReadOthers = [&]() {
        PkOthers o;
        if (DIR* d = opendir(gateDir.c_str())) {
            while (struct dirent* e = readdir(d)) {
                const int pid = std::atoi(e->d_name);
                if (pid <= 0 || pid == (int) self || std::strchr(e->d_name, '.') != nullptr) continue;
                double t = 0, rem = 0, we = 0, ws = 0, fs = 0, le = 0;
                int act = 0, rdy = 0;
                if (FILE* f = fopen((gateDir + "/" + e->d_name).c_str(), "r")) {
                    const int n = fscanf(f, "%lf %lf %d %lf %lf %lf %d %lf", &t, &rem, &act, &we, &ws, &fs, &rdy, &le);
                    fclose(f);
                    if (n < 5 || wall_now() - t > 20 || kill(pid, 0) != 0) continue;
                    if (n < 8) { fs = rem; rdy = 1; le = 0; }   // a pool of the old gate in the same job: its figure is its own estimate sum
                    o.figSum += fs; o.act += act; o.pools += 1; o.ready += rdy; o.lastEvt = std::max(o.lastEvt, le);
                    if (we > 0 && (o.waitFig == 0 || ws < o.waitSince)) { o.waitFig = we; o.waitSince = ws; }
                }
            }
            closedir(d);
        }
        return o;
    };
    // the idle baseline, shared by the pools of the job through the gate directory: `B nPools t`. Valid while the pools now alive (this one included) are no more than at its measurement:
    // a pool that appeared later brought its genome and server, so B would be too low
    auto pkBaseline = [&](int poolsNow, double& B) -> bool {
        FILE* f = fopen((gateDir + "/baseline").c_str(), "r");
        if (!f) return false;
        double b = 0, t = 0;
        int n = 0;
        const int k = fscanf(f, "%lf %d %lf", &b, &n, &t);
        fclose(f);
        if (k != 3 || b <= 0 || n < poolsNow) return false;
        B = b;
        return true;
    };
    // the non-host server is loaded (its page-locked index is 24.7 GiB of the baseline) when its log says "[nhsrv] resident"; no server (or one that ended): nothing to wait for
    bool pkSrvUp = false;
    auto pkServerUp = [&]() -> bool {
        if (pkSrvUp) return true;
        if (srvPid <= 0) { pkSrvUp = true; return true; }
        if (slurp(report + "/nhsrv.log").find("[nhsrv] resident") != std::string::npos) pkSrvUp = true;
        return pkSrvUp;
    };
    bool pkReady = false;
    double pkOwnFigSum = 0, pkLastScan = 0, pkExcLast = 0, pkExcMax = 0, pkExcSecs = 0, pkBLogged = 0;
    int pkOwnAct = 0, pkLastRefused = 0;
    bool pkJobHoldNoted = false, pkBwaitNoted = false;
    std::vector<std::pair<double, double>> pkRing;   // (time, anon + shmem of the job cgroup, GiB) about once a second
    auto pkPublish = [&](double waitFig, double waitSince) { pkWriteFile(pkOwnFigSum, pkOwnAct, waitFig, waitSince, pkReady ? 1 : 0, pkLastEvt); };
    auto pkRegister = [&](double fig) { pkOwnFigSum += fig; pkOwnAct += 1; pkLastEvt = wall_now(); pkPublish(0, 0); };
    // the node term exactly as the old gate, the job term from the job cgroup (GiB)
    auto pkTerms = [&](const CgNow& cg, double& freeNode, double& freeJob) {
        double mfree = 0, cached = 0, shm = 0, v = 0;
        { std::ifstream mi("/proc/meminfo"); std::string k, u; while (mi >> k >> v >> u) { if (k == "MemFree:") mfree = v; else if (k == "Cached:") cached = v; else if (k == "Shmem:") shm = v; } }
        freeNode = (mfree + cacheShare * std::max(0.0, cached - shm)) / 1048576.0;
        freeJob = cg.ok ? cg.lim - cg.cur + cgShare * std::max(0.0, cg.file - cg.shmem) : freeNode;
    };
    // an item's figure now: the phase (before the "level" line, after it, after the "waited" line), then what the engine printed. Logs each change.
    auto pkFigure = [&](Run& r) -> double {
        if (!r.pk) {   // the old estimate, phases as the old gate
            double want = itemEst[r.idx] * (r.front ? 1.0 : 0.4);
            if (phaseAdmit && r.plateau) want = std::min(want, tailGb);
            r.pkFig = want;
            return want;
        }
        const int phase = r.plateau ? 2 : (r.front ? 0 : 1);
        std::string why;
        double P = r.pkP;
        const double nE = r.pmN > 0 ? r.pmN / 1e6 : r.pkN, bE = r.pmBases > 0 ? r.pmBases / 1e9 : r.pkBases, dE = r.pmD / 1e6;
        if (r.pm1 && !r.pm1Done) {   // first line: the zone's fallback is decided now
            r.pm1Done = true;
            if (r.pkZone) {
                if (r.pmFb > 0) { P = poolpeak::fallbackP(nE, bE, dE, pkK); why += fmt(", line 1: fallback expected, D %.1f M", dE); }
                else { P = poolpeak::pLinear(nE, bE, pkK); why += ", line 1: no fallback"; }
            } else if (r.pmFb > 0) {
                const double Pf = poolpeak::fallbackP(nE, bE, dE, pkK);
                if (Pf > P) { P = Pf; why += fmt(", line 1: fallback expected outside the zone, D %.1f M", dE); }
            }
        }
        if (r.pmFallback && !r.pmFbDone) {   // E8's records were not built: STAR's loop does the work
            r.pmFbDone = true;
            const double Pf = poolpeak::fallbackP(nE, bE, r.pmD / 1e6, pkK);
            if (Pf > P) { P = Pf; why += fmt(", fallback taken, D %.1f M", r.pmD / 1e6); }
        }
        r.pkP = P;
        if (phase != r.pkPhase) { why += phase == 2 ? ", after the waited line" : phase == 1 ? ", after the level line" : ""; r.pkPhase = phase; }
        double f = poolpeak::phaseFigure(P, phase, pkK, chimMode(r));
        if (r.pm3 && r.pm1 && r.pm2 && !r.pm3Done) {   // third line: the ledger; raise only
            r.pm3Done = true;
            const double L = poolpeak::ledgerL(nE, bE, r.pmRegion, r.pmKeep, r.pmLate, pkK);
            if (L + pkK.margin > f) { r.pkFloor = L + pkK.margin; r.pkFloorPhase = phase; r.pkP = P = std::max(P, L); why += fmt(", ledger %.1f GiB raises the figure", L); }
            else why += fmt(", ledger %.1f GiB (not above)", L);
            f = poolpeak::phaseFigure(P, phase, pkK, chimMode(r));
        }
        if (r.pkFloorPhase == phase) f = std::max(f, r.pkFloor);
        if (std::fabs(f - r.pkFig) > 0.05 || !why.empty()) plog("peak figure: item %d %.1f -> %.1f GiB (P %.1f%s)", r.idx, r.pkFig, f, r.pkP, why.c_str());
        r.pkFig = f;
        return f;
    };
    // the idle baseline: measured only when it cannot be too low. All of: this pool's genome is resident and its non-host server has loaded its index (pkReady) and so have the other
    // pools' (their `ready`); no item of any pool is in flight and none started or ended in the last SETTLE s (an ended item's memory is still being returned); the job's anon + shmem has been
    // flat (range <= B_RANGE_GB) over the last STABLE s of 1 Hz samples. B = the largest sample of that window; once stored it is only ever raised (by a later idle moment: the gpu-holder
    // the first item starts is in it), never lowered; it is invalidated when a pool appears that was not there at the measurement.
    auto pkMeasure = [&](double now) {
        if (!pkReady || pkOwnAct != 0 || (pkLastEvt > 0 && now - pkLastEvt < pkSettle)) return;
        double lo = 1e30, hi = 0;
        int cnt = 0;
        double oldest = now;
        for (const auto& s : pkRing) if (now - s.first <= pkStableS) { lo = std::min(lo, s.second); hi = std::max(hi, s.second); ++cnt; oldest = std::min(oldest, s.first); }
        if (cnt < 6 || now - oldest < pkStableS - 1.5 || hi - lo > pkRangeGb) return;
        const int lk = open(gateLock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lk >= 0) flock(lk, LOCK_EX);
        const PkOthers o = pkReadOthers();
        if (o.act == 0 && o.ready == o.pools && (o.lastEvt <= 0 || now - o.lastEvt >= pkSettle) && (pkNPools <= 0 || o.pools + 1 >= pkNPools)) {
            double B = 0;
            const bool had = pkBaseline(o.pools + 1, B);
            if (!had || hi > B + 0.05) {
                double old = 0, t = 0;
                int n = 0;
                if (FILE* f = fopen((gateDir + "/baseline").c_str(), "r")) { if (fscanf(f, "%lf %d %lf", &old, &n, &t) != 3) old = 0; fclose(f); }
                const double nb = std::max(old, hi);
                const std::string tmp = gateDir + "/baseline.tmp";
                if (FILE* f = fopen(tmp.c_str(), "w")) { fprintf(f, "%.3f %d %.3f\n", nb, o.pools + 1, now); fclose(f); rename(tmp.c_str(), (gateDir + "/baseline").c_str()); }
                if (std::fabs(nb - pkBLogged) > 0.2) { pkBLogged = nb; plog("peak admission: idle baseline B %.2f GiB (job anon+shmem %.2f to %.2f over %.0f s, %d pool(s), servers resident, no item for %.0f s)", nb, lo, hi, pkStableS, o.pools + 1, pkLastEvt > 0 ? now - pkLastEvt : -1.0); }
            }
        }
        if (lk >= 0) { flock(lk, LOCK_UN); close(lk); }
    };
    // once a second (or forced): the job cgroup into the ring, the figures of the pool's items, the baseline, the model-exceeded check
    auto chimMode = [](const Run& r) { return !r.pkChim ? poolpeak::Chim::None : r.pkChimShared ? poolpeak::Chim::Shared : poolpeak::Chim::Private; };
    auto pkRefresh = [&](bool force) {
        const double now = wall_now();
        if (!force && now - pkLastScan < 1.0) return;
        const double dt = pkLastScan > 0 ? std::min(2.0, now - pkLastScan) : 0;
        pkLastScan = now;
        const CgNow cg = read_job_cgroup(jobCg);
        if (cg.ok && (pkRing.empty() || now - pkRing.back().first >= 0.8)) {
            pkRing.emplace_back(now, cg.as());
            while (!pkRing.empty() && now - pkRing.front().first > 120) pkRing.erase(pkRing.begin());
        }
        double fs = 0;
        int act = 0;
        for (Run& r : runs) { if (r.pid <= 0 || r.done) continue; ++act; fs += pkFigure(r); }
        pkOwnFigSum = fs; pkOwnAct = act;
        if (!pkReady && pkServerUp()) pkReady = true;
        if (!cg.ok) return;
        pkMeasure(now);
        const PkOthers o = pkReadOthers();
        double B = 0;
        if (pkBaseline(o.pools + 1, B) && act + o.act > 0) {   // the model exceeded: the items hold more than the sum of what the model lets them reach
            const double held = cg.as() - B, figs = fs + o.figSum;
            if (held > figs + 0.05) {
                pkExcSecs += dt; pkExcMax = std::max(pkExcMax, held - figs);
                if (now - pkExcLast >= 30) { pkExcLast = now; plog("model exceeded: job anon+shmem %.1f - baseline %.1f = %.1f GiB held by the items, above the sum of their figures %.1f by %.1f GiB (%d items in flight in the job; largest excess so far %.1f GiB)", cg.as(), B, held, figs, held - figs, act + o.act, pkExcMax); }
            }
        }
    };
    struct PkRes { double freeNode = 0, freeJob = 0, rem = 0, figs = 0, as = 0, B = 0, othWait = 0, projNode = 0, projJob = 0; int refused = 0; bool bOk = false; };
    // under the flock: does an item of figure `fig` fit? registers it when it does. free = min(node term, job term); admit when nothing else is in flight in the job or when
    // free - what the items in flight can still allocate - fig - the older waiting item of another pool >= reserve
    auto pkTry = [&](double fig, double mySince, double reserve, PkRes& x) {
        const int lk = open(gateLock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
        if (lk >= 0) flock(lk, LOCK_EX);
        const PkOthers o = pkReadOthers();
        const CgNow cg = read_job_cgroup(jobCg);
        pkTerms(cg, x.freeNode, x.freeJob);
        double B = 0;
        x.bOk = cg.ok && pkBaseline(o.pools + 1, B);
        x.B = B; x.as = cg.as();
        x.figs = pkOwnFigSum + o.figSum;
        x.rem = std::max(0.0, x.figs - (x.bOk ? std::max(0.0, x.as - B) : 0.0));
        x.othWait = (o.waitFig > 0 && wall_now() - o.waitSince > prioAfter && o.waitSince <= mySince) ? o.waitFig : 0;
        x.projNode = x.freeNode - x.rem - fig - x.othWait;
        x.projJob = x.freeJob - x.rem - fig - x.othWait;
        x.refused = (x.projNode < reserve ? 1 : 0) | (x.projJob < reserve ? 2 : 0);
        const bool alone = pkOwnAct == 0 && o.act == 0;   // nothing else in the job: it runs alone, however large
        const bool ok = alone || x.refused == 0;
        if (ok) pkRegister(fig);
        if (lk >= 0) { flock(lk, LOCK_UN); close(lk); }
        return ok;
    };
    // the very first start of a job waits (bounded) until the idle baseline is known: it can only be measured while nothing runs
    auto pkBaselineWait = [&]() -> bool {
        if (wall_now() - tReady >= pkBWait || pkOwnAct > 0) return false;
        const PkOthers o = pkReadOthers();
        double B = 0;
        return o.act == 0 && !pkBaseline(o.pools + 1, B);
    };
    FILE* msf2 = nullptr;
    auto pkSampleWrite = [&]() {   // memsamp's neighbour: t AS B sum-of-figures node-term job-term items-in-flight baseline-valid
        if (!msf2) {
            msf2 = fopen((report + "/memsamp-cg.tsv").c_str(), "a");
            if (!msf2) return;
            fprintf(msf2, "t\tAS_GiB\tB_GiB\tfigures_GiB\tcan_still_allocate_GiB\tnode_free_GiB\tjob_free_GiB\titems_job\tB_valid\n");
        }
        const CgNow cg = read_job_cgroup(jobCg);
        const PkOthers o = pkReadOthers();
        double fn = 0, fj = 0, B = 0;
        pkTerms(cg, fn, fj);
        const bool bOk = cg.ok && pkBaseline(o.pools + 1, B);
        const double figs = pkOwnFigSum + o.figSum;
        fprintf(msf2, "%.0f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%.2f\t%d\t%d\n", wall_now(), cg.as(), B, figs, std::max(0.0, figs - (bOk ? std::max(0.0, cg.as() - B) : 0.0)), fn, fj, pkOwnAct + o.act, bOk ? 1 : 0);
        fflush(msf2);
    };
    if (peakMode) {   // the stream starts: this pool is no longer in its start-up; its own refresh keeps the file fresh from here
        pkHbStop = true;
        if (pkHb.joinable()) pkHb.join();
        if (pkReadOthers().pools == 0) unlink((gateDir + "/baseline").c_str());   // no other pool: a baseline left by an earlier run in this job is not trusted
        pkPublish(0, 0);
    }
    double memWaitSince = 0, memWaitedTotal = 0;
    bool memWaitNoted = false;
    unsigned memWaits = 0;
    while (next < items.size() || active > 0) {
        if (g_stop) {
            plog("stopping: signal received; the samples in flight are killed");
            for (auto& kv : byPid) kill(-kv.first, SIGKILL);
            next = items.size();
            g_stop = 0;
        }
        while (active < (size_t) workers && next < items.size()) {
            // rt/hostmem admission: a start waits while another item of this pool started less than SINGLET_POOL_STAGGER_S ago (the items of a first
            // wave would otherwise all reach their memory-heavy phases together) or while the node's available memory is under SINGLET_POOL_MEM_FLOOR_GB
            // or while frontMax items of this pool are in their front half (SINGLET_POOL_FRONT_MAX): decoded reads, aux and the region, waiting for or
            // holding a device slot; the items beyond it are in their tail (CPU work, a third of the host memory), where an extra sample earns its place
            size_t front = 0;
            if (frontMax > 0) for (const Run& r : runs) if (r.pid > 0 && !r.done && r.front) ++front;
            if (active > 0 && ((stagger > 0 && lastStartEst >= 100 && wall_now() - lastStartAt < stagger) || (memFloor > 0 && avail_gb() < memFloor) || (frontMax > 0 && front >= (size_t) frontMax))) {
                if (!admitNoted) { plog("admission: the next item waits (%zu in the front half of at most %.0f, stagger %.0f s, memory floor %.0f GB, available %.0f GB)", front, frontMax, stagger, memFloor, avail_gb()); admitNoted = true; }
                break;
            }
            if (memAdmit && !peakMode) {
                const double now = wall_now(), est = itemEst[items[next].idx];
                if (memWaitSince == 0) memWaitSince = now;
                gateRefresh(true);
                double proj = 0, freeEff = 0, othRem = 0, othWait = 0;
                bool ok = gateTry(est, memWaitSince, memReserve, freeEff, othRem, othWait, proj);
                if (!ok && now - memWaitSince >= maxWait) {
                    plog("memory admission: item %d starts after %.0f s (the wait is bounded; projected %.0f GB)", items[next].idx, now - memWaitSince, proj);
                    gateRegister(est);
                    ok = true;
                }
                if (!ok) {
                    if (!memWaitNoted) {
                        ++memWaits; memWaitNoted = true;
                        plog("memory admission: item %d waits (estimate %.0f GB; free + cache share %.0f GB, own items still to allocate %.0f GB, other pools %.0f GB, reserve %.0f GB; projected %.0f GB)",
                             items[next].idx, est, freeEff, gateOwnRem, othRem, memReserve, proj);
                    }
                    gatePublish(est, memWaitSince);
                    if (skipAhead > 0 && now - memWaitSince < skipGrace) {   // the head waits: an item behind it that fits goes first
                        size_t pick = items.size();
                        for (size_t k = next + 1; k < items.size() && k <= next + (size_t) skipAhead; ++k) {
                            if (taken[k]) continue;
                            double p2 = 0, f2 = 0, r2 = 0, w2 = 0;
                            if (gateTry(itemEst[items[k].idx], now, memReserve, f2, r2, w2, p2)) { pick = k; break; }
                        }
                        if (pick < items.size()) {
                            plog("memory admission: item %d starts ahead of the waiting item %d (%.0f s)", items[pick].idx, items[next].idx, now - memWaitSince);
                            taken[pick] = 1;
                            lastStartAt = wall_now(); lastStartEst = itemEst[items[pick].idx];
                            start(items[pick]);
                            continue;
                        }
                    }
                    break;
                }
                if (memWaitNoted) { memWaitedTotal += now - memWaitSince; plog("memory admission: item %d admitted after %.0f s (projected %.0f GB)", items[next].idx, now - memWaitSince, proj); }
                memWaitSince = 0; memWaitNoted = false;
            }
            else if (memAdmit && peakMode) {   // cggate: SINGLET_POOL_PEAKADMIT=1
                const double now = wall_now();
                const double fig = itemFig0[items[next].idx];
                if (memWaitSince == 0) memWaitSince = now;
                pkRefresh(true);
                if (pkBaselineWait()) {   // nothing runs and the idle baseline is not known yet: it can only be measured now
                    if (!pkBwaitNoted) { pkBwaitNoted = true; plog("peak admission: the first start waits for the idle baseline (at most %.0f s after the pool is ready): this pool's server index and the other pools' must be resident, then %.0f s of flat memory", pkBWait, pkStableS); }
                    pkPublish(0, 0);
                    break;
                }
                PkRes x;
                bool ok = pkTry(fig, memWaitSince, memReserve, x);
                if (!ok && x.refused == 1 && now - memWaitSince >= maxWait) {   // the bounded wait stays for the node term; when the job term refuses, the pool keeps waiting
                    plog("memory admission: item %d starts after %.0f s (the wait is bounded; only the node term refused; projected node %.0f GiB, job %.0f GiB)", items[next].idx, now - memWaitSince, x.projNode, x.projJob);
                    pkRegister(fig);
                    ok = true;
                }
                if (!ok) {
                    if (x.refused != pkLastRefused || !memWaitNoted) {
                        if (!memWaitNoted) { ++memWaits; memWaitNoted = true; }
                        pkLastRefused = x.refused;
                        const char* term = x.refused == 3 ? "node and job" : x.refused == 2 ? "job" : "node";
                        plog("memory admission: item %d waits (figure %.0f GiB; free: node term %.0f, job term %.0f GiB; the items in flight can still allocate %.0f GiB (figures %.0f, job anon+shmem %.0f, baseline %s%.1f), waiting item of another pool %.0f, reserve %.0f; projected node %.0f, job %.0f; refused by the %s term)",
                             items[next].idx, fig, x.freeNode, x.freeJob, x.rem, x.figs, x.as, x.bOk ? "" : "unknown, ", x.B, x.othWait, memReserve, x.projNode, x.projJob, term);
                    }
                    if ((x.refused & 2) != 0 && now - memWaitSince >= maxWait && !pkJobHoldNoted) { pkJobHoldNoted = true; plog("memory admission: item %d has waited %.0f s and the job term still refuses it: no forced start; it starts when the items in flight have ended at the latest", items[next].idx, now - memWaitSince); }
                    pkPublish(fig, memWaitSince);
                    if (skipAhead > 0 && now - memWaitSince < skipGrace) {   // the head waits: an item behind it that fits goes first
                        size_t pick = items.size();
                        for (size_t k = next + 1; k < items.size() && k <= next + (size_t) skipAhead; ++k) {
                            if (taken[k]) continue;
                            PkRes x2;
                            if (pkTry(itemFig0[items[k].idx], now, memReserve, x2)) { pick = k; break; }
                        }
                        if (pick < items.size()) {
                            plog("memory admission: item %d starts ahead of the waiting item %d (%.0f s)", items[pick].idx, items[next].idx, now - memWaitSince);
                            taken[pick] = 1;
                            lastStartAt = wall_now(); lastStartEst = itemEst[items[pick].idx];
                            start(items[pick]);
                            continue;
                        }
                    }
                    break;
                }
                if (memWaitNoted) { memWaitedTotal += now - memWaitSince; plog("memory admission: item %d admitted after %.0f s (projected node %.0f GiB, job %.0f GiB)", items[next].idx, now - memWaitSince, x.projNode, x.projJob); }
                memWaitSince = 0; memWaitNoted = false; pkLastRefused = 0; pkJobHoldNoted = false;
            }
            admitNoted = false;
            lastStartAt = wall_now(); lastStartEst = itemEst[items[next].idx];
            taken[next] = 1;
            start(items[next]);
            while (next < items.size() && taken[next]) ++next;
        }
        // logs, then the ends
        std::vector<struct pollfd> pf;
        std::vector<int> who;
        for (Run& r : runs) if (r.logFd >= 0) { pf.push_back({r.logFd, POLLIN, 0}); who.push_back(r.idx); }
        const int pr = pf.empty() ? (usleep(200000), 0) : poll(pf.data(), pf.size(), 200);
        if (pr > 0)
            for (size_t i = 0; i < pf.size(); ++i)
                if (pf[i].revents & (POLLIN | POLLHUP | POLLERR)) pump(runs[who[i]], false);
        for (;;) {
            int st = 0;
            struct rusage ru{};
            const pid_t p = wait4(-1, &st, WNOHANG, &ru);
            if (p <= 0) break;
            if (p == srvPid) {
                plog("non-host server pid %d ended (%s); the next samples load the index themselves", (int) p,
                     WIFSIGNALED(st) ? fmt("signal %d", WTERMSIG(st)).c_str() : fmt("exit %d", WEXITSTATUS(st)).c_str());
                srvPid = -1;
                unsetenv("SINGLET_NONHOST_SERVER");
                continue;
            }
            finish(p, st, ru);
        }
        for (Run& r : runs) if (r.demoted && !r.demotedLogged) { r.demotedLogged = true; plog("item %d: --chimeric item reached its engine level line without 'genome kept for the pass': its floor is the private index (chimGb %.1f GiB) from now on", r.idx, pkK.chimGb); }
        if (memAdmit && peakMode) {   // cggate
            if (wall_now() - pkLastScan >= 1.0) { pkRefresh(true); pkPublish(memWaitSince > 0 && next < items.size() ? itemFig0[items[next].idx] : 0, memWaitSince); }
        } else
        if (memAdmit && wall_now() - gateLastScan >= 1.0) {   // the pool's file for the others: what its items can still allocate, and its waiting item
            gateRefresh(true);
            gatePublish(memWaitSince > 0 && next < items.size() ? itemEst[items[next].idx] : 0, memWaitSince);
        }
        if (dropInput) for (Run& r : runs) if (r.decoded && !r.dropped) {   // the item read its input once: its page cache is of no use to it
            r.dropped = true;
            const int fd = open(items[r.idx].fq.c_str(), O_RDONLY);
            if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
        }
        if (msf != nullptr && wall_now() - lastSample >= sampPeriod) { lastSample = wall_now(); memsample_write(msf, self, srvPid, byPid); if (peakMode) pkSampleWrite(); }
        if (timeout > 0) {
            const double t = wall_now();
            for (auto& kv : byPid) {
                Run& r = runs[kv.second];
                if (!r.killed && t - r.t0 > timeout) { plog("item %d: over --timeout %.0f s: killed", r.idx, timeout); kill(-r.pid, SIGTERM); r.killed = true; r.killAt = t; }
                else if (r.killed && t - r.killAt > 30) kill(-r.pid, SIGKILL);
            }
        }
    }
    // the last log lines (a sample's descendants may hold its pipe a moment longer)
    for (int w = 0; w < 50; ++w) {
        bool open = false;
        for (Run& r : runs) if (r.logFd >= 0) { pump(r, false); open = open || r.logFd >= 0; }
        if (!open) break;
        usleep(100000);
    }
    for (Run& r : runs) if (r.logFd >= 0) pump(r, true);
    if (srvPid > 0) {
        kill(srvPid, SIGTERM);
        waitpid(srvPid, nullptr, 0);
        unlink(sock.c_str());
    }
    const double span = lastEnd > firstStart ? lastEnd - firstStart : 0;
    plog("queue drained: %zu samples, %zu failed; span %.1f s (first start to last end), %.2f M pairs, %.3f M pairs/s; start-up %.1f s",
         items.size(), failed, span, pairsAll / 1e6, span > 0 ? pairsAll / 1e6 / span : 0.0, tReady - tStart);
    if (peakMode) plog("peak admission: model exceeded in %.0f s of the refreshes (largest excess %.1f GiB)", pkExcSecs, pkExcMax);
    if (memAdmit) { plog("memory admission: %u waits, %.0f s waited in all (SINGLET_POOL_MEMADMIT)", memWaits, memWaitedTotal); unlink(gateMine.c_str()); }
    if (msf != nullptr) fclose(msf);
    if (msf2 != nullptr) fclose(msf2);
    fclose(stsv);
    fclose(pids);
    fclose(g_plog);
    g_plog = nullptr;
    return failed ? 1 : 0;
}
