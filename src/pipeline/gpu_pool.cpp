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
    std::ifstream cl("/proc/self/cgroup");
    std::string line;
    if (std::getline(cl, line)) {
        const size_t c = line.rfind(':');
        const std::string base = "/sys/fs/cgroup" + (c == std::string::npos ? std::string() : line.substr(c + 1));
        std::ifstream mx(base + "/memory.max"), cu(base + "/memory.current");
        std::string m;
        double cur = 0;
        if (mx >> m && cu >> cur && !m.empty() && std::isdigit((unsigned char) m[0])) a = std::min(a, (std::atof(m.c_str()) - cur) / 1073741824.0);
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
    bool decoded = false, dropped = false;   // rt/hostmem: the engine decoded the .1fq into the packed buffers (its page cache can go)
};

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
            if (r.front && r.partial.compare(s, 19, "[engine_gpu] level ") == 0) r.front = false;
            if (!r.decoded) { const size_t dp = r.partial.find("decoded into the packed buffers", s); if (dp != std::string::npos && dp < e) r.decoded = true; }
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
        --active;
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
    //   free now (MemFree + SINGLET_POOL_CACHE_SHARE (0) of the file cache, which other tenants' load makes slow to reclaim)
    //   - what every pool's items can still allocate - the item's estimate - the estimate of a waiting item of another pool that waited longer
    //   >= SINGLET_POOL_MEM_RESERVE_GB (100).
    // Estimate (SINGLET_POOL_ITEM_GB overrides): 11.5 GB per GB of .1fq + 8: d300_3p (6.0 GB file) peaks at 77 GB private (planes, aux, region, per-read
    // tables), d1000_3p (24.9 GB) at 285 GB (rt-deep 815819). An item in its tail (after the engine's "[engine_gpu] level" line) is counted at
    // 40 % of the estimate. An item with nothing else running on the node always starts, whatever its size (it runs alone). The wait is bounded
    // (SINGLET_POOL_MEMADMIT_MAXWAIT_S, 1800): then it starts, with a log line. The page cache of an input is dropped when the item has decoded
    // it (SINGLET_POOL_DROP_INPUT_CACHE, default on) and, for the clean pages, at pool start.
    const bool memAdmit = env_num("SINGLET_POOL_MEMADMIT", 1) != 0;
    const double itemGbEnv = env_num("SINGLET_POOL_ITEM_GB", 0), cacheShare = env_num("SINGLET_POOL_CACHE_SHARE", 0),
                 memReserve = env_num("SINGLET_POOL_MEM_RESERVE_GB", 100), maxWait = env_num("SINGLET_POOL_MEMADMIT_MAXWAIT_S", 1800);
    const bool dropInput = env_num("SINGLET_POOL_DROP_INPUT_CACHE", 1) != 0;
    // SINGLET_POOL_SKIP_AHEAD (16, 0 = strict queue order): while the head of the queue waits for room, an item among the next 16 that fits starts instead,
    // for at most SINGLET_POOL_SKIP_GRACE_S (900) of the head's wait; after that only the head is tried, and another pool's head that waited that long is served first
    const double skipAhead = env_num("SINGLET_POOL_SKIP_AHEAD", 16), skipGrace = env_num("SINGLET_POOL_SKIP_GRACE_S", 900), prioAfter = skipAhead > 0 ? skipGrace : 0;
    std::vector<char> taken(items.size(), 0);
    std::vector<double> itemEst(items.size(), 0);
    for (const Item& it : items) {
        struct stat st{};
        itemEst[it.idx] = itemGbEnv > 0 ? itemGbEnv : (stat(it.fq.c_str(), &st) == 0 ? (double) st.st_size / 1073741824.0 : 6.0) * 11.5 + 8.0;
    }
    if (dropInput) {   // the harness's copy of the inputs left its pages in the cache (dirty at first): written out, then dropped, on a thread beside the start
        std::vector<std::string> fqs;
        for (const Item& it : items) if (std::find(fqs.begin(), fqs.end(), it.fq) == fqs.end()) fqs.push_back(it.fq);
        std::thread([fqs]() { for (const std::string& p : fqs) { const int fd = open(p.c_str(), O_RDONLY); if (fd >= 0) { fdatasync(fd); posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); } } }).detach();
    }
    double sharedGb = 0;   // what a STAR child maps of the pool's resident genome: counted in its RSS, not in the item's own growth
    { std::ifstream f("/proc/self/status"); std::string l; while (std::getline(f, l)) if (l.rfind("RssAnon:", 0) == 0) sharedGb = std::atof(l.c_str() + 8) / 1048576.0; }
    const char* gjob = std::getenv("SLURM_JOB_ID");
    const std::string gateDir = "/dev/shm/singlet-pool-mem-" + std::string(gjob ? gjob : "node") + "-" + std::to_string((unsigned) getuid());
    if (memAdmit) mkdir(gateDir.c_str(), 0700);
    const std::string gateMine = gateDir + "/" + std::to_string((int) self), gateLock = gateDir + "/lock";
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
        std::map<int, double> cur;
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
                cur[it->second] += (an + sh) / 1048576.0;
            }
            closedir(d);
        }
        double rem = 0;
        int act = 0;
        for (const Run& r : runs) {
            if (r.pid <= 0 || r.done) continue;
            ++act;
            const double held = std::max(0.0, cur[r.idx] - sharedGb), want = itemEst[r.idx] * (r.front ? 1.0 : 0.4);
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
            if (active > 0 && ((stagger > 0 && lastStartEst >= 50 && wall_now() - lastStartAt < stagger) || (memFloor > 0 && avail_gb() < memFloor) || (frontMax > 0 && front >= (size_t) frontMax))) {
                if (!admitNoted) { plog("admission: the next item waits (%zu in the front half of at most %.0f, stagger %.0f s, memory floor %.0f GB, available %.0f GB)", front, frontMax, stagger, memFloor, avail_gb()); admitNoted = true; }
                break;
            }
            if (memAdmit) {
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
        if (memAdmit && wall_now() - gateLastScan >= 1.0) {   // the pool's file for the others: what its items can still allocate, and its waiting item
            gateRefresh(true);
            gatePublish(memWaitSince > 0 && next < items.size() ? itemEst[items[next].idx] : 0, memWaitSince);
        }
        if (dropInput) for (Run& r : runs) if (r.decoded && !r.dropped) {   // the item read its input once: its page cache is of no use to it
            r.dropped = true;
            const int fd = open(items[r.idx].fq.c_str(), O_RDONLY);
            if (fd >= 0) { posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
        }
        if (msf != nullptr && wall_now() - lastSample >= sampPeriod) { lastSample = wall_now(); memsample_write(msf, self, srvPid, byPid); }
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
    if (memAdmit) { plog("memory admission: %u waits, %.0f s waited in all (SINGLET_POOL_MEMADMIT)", memWaits, memWaitedTotal); unlink(gateMine.c_str()); }
    if (msf != nullptr) fclose(msf);
    fclose(stsv);
    fclose(pids);
    fclose(g_plog);
    g_plog = nullptr;
    return failed ? 1 : 0;
}
