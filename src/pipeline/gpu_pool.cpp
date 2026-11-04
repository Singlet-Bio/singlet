// singlet E21 (docs/gpu-engine/fable/E21-pool.md): `singlet gpu-pool`, one resident process per GPU that runs a queue of samples on the GPU
// path with the per-sample fixed costs paid once.
//
//   singlet gpu-pool --device D --queue FILE [--slots 2] [--workers 8] [--threads 7] [--args "<pipeline arguments>"] [--report DIR]
//                    [--nonhost-server on|off] [--genome-preload on|off] [--timeout S]
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
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
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

// /proc/<pid>/stat field 22 (start time in clock ticks since boot): with the pid, a process's identity for a sampler
unsigned long long proc_starttime(pid_t pid) {
    std::ifstream f("/proc/" + std::to_string(pid) + "/stat");
    std::string s((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    const size_t r = s.rfind(')');
    if (r == std::string::npos) return 0;
    std::istringstream in(s.substr(r + 2));
    std::string tok;
    for (int i = 3; i <= 22 && (in >> tok); ++i)
        if (i == 22) return std::strtoull(tok.c_str(), nullptr, 10);
    return 0;
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
int singlet_gpu_pool(int argc, char** argv, int (*pipeline)(int, char**)) {
    int device = -1, slots = 2, workers = 8, threads = 7, preloadThreads = 8;
    double timeout = 0;
    std::string queue, args, report;
    bool server = true, preload = true;
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
        else if (a == "-h" || a == "--help") {
            std::cout << "singlet gpu-pool --device D --queue FILE [--slots 2] [--workers 8] [--threads 7] [--args \"<pipeline arguments>\"]\n"
                         "                 [--report DIR] [--nonhost-server on|off] [--genome-preload on|off] [--timeout S] [--preload-threads 8]\n"
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
    while (next < items.size() || active > 0) {
        if (g_stop) {
            plog("stopping: signal received; the samples in flight are killed");
            for (auto& kv : byPid) kill(-kv.first, SIGKILL);
            next = items.size();
            g_stop = 0;
        }
        while (active < (size_t) workers && next < items.size()) start(items[next++]);
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
    fclose(stsv);
    fclose(pids);
    fclose(g_plog);
    g_plog = nullptr;
    return failed ? 1 : 0;
}
