// Shared resident genome index (docs/gpu-engine/bold/SHARED-INDEX.md, SINGLET_GPU_SHARED_INDEX=1).
// Included by engine_gpu.cu inside namespace singlet_engine_gpu, after deviceCount(). It uses that file's dmal helpers, now() and b2gb().
//
// One holder process per device and genome (`singlet gpu-holder`) uploads G, SA and SAi once and exports them with cudaIpcGetMemHandle on a
// Unix socket. mapDistinct imports them read only (cudaIpcOpenMemHandle) instead of its own upload. Any failure leaves the private upload as it was.
#pragma once
namespace {
static const u64 SH_PAD = 200, SH_BLK = 4096, SH_NB = 512;
static const uint32_t SH_MAGIC = 0x31495348u;
struct ShReq { uint32_t magic; uint32_t pad; u64 nG, saB, saiB; };   // allocation sizes of the client's host index
struct ShRep { uint32_t status; uint32_t pad; u64 bytes[3]; cudaIpcMemHandle_t h[3]; char msg[96]; };
static u64 shFnv(const std::string& s) { u64 h = 1469598103934665603ull; for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; } return h; }
static std::string shKey(const std::string& genomeDir, const std::string& devId) {
    char rp[4096]; const std::string gd = realpath(genomeDir.c_str(), rp) ? std::string(rp) : genomeDir;
    char b[40]; snprintf(b, sizeof b, "%016llx", (unsigned long long) shFnv(gd));
    return devId + "." + b;
}
static std::string shPath(const std::string& key, const char* ext) { return "/tmp/singlet_gpu_holder." + key + "." + ext; }
static bool shRead(int fd, void* p, size_t n) { char* c = (char*) p; while (n) { ssize_t r = read(fd, c, n); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return false; } c += r; n -= (size_t) r; } return true; }
static bool shWrite(int fd, const void* p, size_t n) { const char* c = (const char*) p; while (n) { ssize_t r = write(fd, c, n); if (r <= 0) { if (r < 0 && errno == EINTR) continue; return false; } c += r; n -= (size_t) r; } return true; }
// sampled comparison of a device buffer with the host image: SH_NB evenly spaced blocks including the first and the last (host(off, n, dst) fills dst)
static bool shVerify(const char* d, u64 len, const std::function<bool(u64, u64, char*)>& host, std::string& why) {
    std::vector<char> a(SH_BLK), b(SH_BLK);
    const u64 n = std::min<u64>(len, SH_BLK);
    for (u64 i = 0; i < SH_NB; i++) {
        const u64 off = (u64)((unsigned __int128)(len - n) * i / (SH_NB - 1));
        if (cudaMemcpy(a.data(), d + off, n, cudaMemcpyDeviceToHost) != cudaSuccess) { why = "device read failed"; return false; }
        if (!host(off, n, b.data())) { why = "host read failed"; return false; }
        if (memcmp(a.data(), b.data(), n) != 0) { why = "content differs near byte " + std::to_string(off); return false; }
    }
    return true;
}
struct ShClient { int fd = -1; void* p[3] = {nullptr, nullptr, nullptr}; u64 bytes = 0; bool on = false; };
static void shClose(ShClient& S) {
    for (int i = 0; i < 3; i++) if (S.p[i]) {
        cudaIpcCloseMemHandle(S.p[i]);
        g_ipcPtrs.erase(std::remove(g_ipcPtrs.begin(), g_ipcPtrs.end(), S.p[i]), g_ipcPtrs.end()); S.p[i] = nullptr;
    }
    if (S.fd >= 0) { close(S.fd); S.fd = -1; }
    S.on = false;
}
static int shConnect(const std::string& path) {
    int fd = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0); if (fd < 0) return -1;
    sockaddr_un a{}; a.sun_family = AF_UNIX; snprintf(a.sun_path, sizeof a.sun_path, "%s", path.c_str());
    if (connect(fd, (sockaddr*) &a, sizeof a) != 0) { close(fd); return -1; }
    return fd;
}
// Import the shared index into this process. false: no sharing (the reason is appended to `log` when the feature is on); the caller uploads its own copy.
static bool shAcquire(const std::string& genomeDir, const GenomeView& H, u64 saBytes, u64 saiBytes, int device, ShClient& S, std::string& log) {
    const char* en = std::getenv("SINGLET_GPU_SHARED_INDEX");
    if (!en || std::string(en) != "1") return false;
    auto no = [&](const std::string& w) { shClose(S); log += "engine_gpu: shared index: " + w + "; private upload of the index\n"; return false; };
    if (genomeDir.empty()) return no("no genome directory");
    const double t0 = now();
    const std::string key = shKey(genomeDir, deviceId(device)), sock = shPath(key, "sock"), slog = shPath(key, "log");
    const char* wE = std::getenv("SINGLET_GPU_SHARED_INDEX_WAIT_S");
    const double waitS = wE ? std::atof(wE) : 300.0;
    int fd = shConnect(sock);
    if (fd < 0) {   // no holder yet: one process starts it (the start lock), every process then waits for the socket, bound once the index is resident
        const int lk = open(shPath(key, "start").c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0666);
        bool have = false;
        while (lk >= 0 && now() - t0 < waitS) {
            if (flock(lk, LOCK_EX | LOCK_NB) == 0) { have = true; break; }
            usleep(200000);
            if ((fd = shConnect(sock)) >= 0) break;
        }
        pid_t child = 0; bool spawned = false;
        if (fd < 0 && have && (fd = shConnect(sock)) < 0) {
            char self[4096]; const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
            if (n <= 0) { close(lk); return no("cannot find the executable"); }
            self[n] = 0;
            const std::string dv = std::to_string(device);
            const char* argv[] = {self, "gpu-holder", "--genome-dir", genomeDir.c_str(), "--device", dv.c_str(), nullptr};
            posix_spawn_file_actions_t fa; posix_spawn_file_actions_init(&fa);
            posix_spawn_file_actions_addopen(&fa, 0, "/dev/null", O_RDONLY, 0);
            posix_spawn_file_actions_addopen(&fa, 1, slog.c_str(), O_WRONLY | O_CREAT | O_APPEND, 0666);
            posix_spawn_file_actions_adddup2(&fa, 1, 2);
            posix_spawnattr_t at; posix_spawnattr_init(&at); posix_spawnattr_setflags(&at, POSIX_SPAWN_SETSID);
            {   // the device holds one genome at a time: holders of other genomes on this device leave when they have no client (SIGUSR1)
                glob_t g{};
                if (glob(shPath(deviceId(device) + ".*", "info").c_str(), 0, nullptr, &g) == 0) {
                    for (size_t i = 0; i < g.gl_pathc; i++) {
                        if (std::string(g.gl_pathv[i]) == shPath(key, "info")) continue;
                        FILE* f = fopen(g.gl_pathv[i], "r"); if (!f) continue;
                        long pid = 0; if (fscanf(f, "%ld", &pid) == 1 && pid > 0) kill((pid_t) pid, SIGUSR1);
                        fclose(f);
                    }
                }
                globfree(&g);
            }
            const int rc = posix_spawn(&child, self, &fa, &at, (char* const*) argv, environ);
            posix_spawn_file_actions_destroy(&fa); posix_spawnattr_destroy(&at);
            if (rc != 0) { close(lk); return no(std::string("cannot start the holder: ") + strerror(rc)); }
            spawned = true;
            log += "engine_gpu: shared index: no holder for this genome and device, started one (pid " + std::to_string((int) child) + ", log " + slog + ")\n";
        }
        while (fd < 0 && now() - t0 < waitS) {
            if (spawned) { int st = 0; if (waitpid(child, &st, WNOHANG) == child) { close(lk); return no("the holder exited at start (log " + slog + ")"); } }
            usleep(250000); fd = shConnect(sock);
        }
        if (lk >= 0) close(lk);   // releases the start lock
        if (fd < 0) return no("no holder after " + std::to_string((int) waitS) + " s (log " + slog + ")");
    }
    S.fd = fd;
    timeval tv{60, 0}; setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    ShReq rq{SH_MAGIC, 0, H.nGenome + 2 * SH_PAD + 8, saBytes + 64 + 8, saiBytes + 64 + 8};
    ShRep rp{};
    if (!shWrite(fd, &rq, sizeof rq) || !shRead(fd, &rp, sizeof rp)) return no("the holder did not answer");
    if (rp.status != 0) return no(std::string("the holder refused: ") + rp.msg);
    for (int i = 0; i < 3; i++) {
        const cudaError_t e = cudaIpcOpenMemHandle(&S.p[i], rp.h[i], cudaIpcMemLazyEnablePeerAccess);
        if (e != cudaSuccess) { S.p[i] = nullptr; cudaGetLastError(); return no(std::string("cudaIpcOpenMemHandle failed: ") + cudaGetErrorString(e)); }
        g_ipcPtrs.push_back(S.p[i]);
    }
    const double tImp = now() - t0;
    std::string why;
    const char* hg = H.G - SH_PAD;
    if (!shVerify((const char*) S.p[0], H.nGenome + 2 * SH_PAD, [&](u64 o, u64 n, char* d) { memcpy(d, hg + o, n); return true; }, why)) return no("G differs from this process's genome (" + why + ")");
    if (!shVerify((const char*) S.p[1], saBytes, [&](u64 o, u64 n, char* d) { memcpy(d, (const char*) H.SA + o, n); return true; }, why)) return no("SA differs from this process's suffix array (" + why + ")");
    if (!shVerify((const char*) S.p[2], saiBytes, [&](u64 o, u64 n, char* d) { memcpy(d, (const char*) H.SAi + o, n); return true; }, why)) return no("SAi differs from this process's index (" + why + ")");
    S.bytes = budRound(rq.nG) + budRound(rq.saB) + budRound(rq.saiB); S.on = true;
    char l[300]; snprintf(l, sizeof l, "engine_gpu: shared index: imported G, SA, SAi (%.2f GB) from the holder in %.2f s, sampled content check %.2f s: no private upload\n",
                          b2gb(S.bytes), tImp, now() - t0 - tImp);
    log += l;
    return true;
}
static volatile sig_atomic_t g_shStop = 0, g_shEvict = 0;
static void shSig(int) { g_shStop = 1; }
static void shEvictSig(int) { g_shEvict = 1; }
}   // namespace

uint64_t sharedIndexReserve() {
    const char* en = std::getenv("SINGLET_GPU_SHARED_INDEX");
    if (!en || std::string(en) != "1") return 0;
    uint64_t best = 0;
    glob_t g{};
    if (glob("/tmp/singlet_gpu_holder.*.info", 0, nullptr, &g) == 0) {
        for (size_t i = 0; i < g.gl_pathc; i++) {
            FILE* f = fopen(g.gl_pathv[i], "r"); if (!f) continue;
            long pid = 0; unsigned long long b = 0;
            if (fscanf(f, "%ld %llu", &pid, &b) == 2 && pid > 0 && kill((pid_t) pid, 0) == 0) best = std::max<uint64_t>(best, b);
            fclose(f);
        }
    }
    globfree(&g);
    return best;
}

// singlet gpu-holder --genome-dir D [--device N] [--idle S]   (stays until SIGTERM, or S seconds, default 900, without a client)
// singlet gpu-holder --stop --genome-dir D [--device N]
int holderMain(int argc, char** argv) {
    std::string gd; int device = 0; double idle = 900; bool stop = false;
    if (const char* ie = std::getenv("SINGLET_GPU_SHARED_INDEX_IDLE_S")) idle = std::atof(ie);
    for (int i = 0; i < argc; i++) {
        const std::string a = argv[i];
        if (a == "--genome-dir" && i + 1 < argc) gd = argv[++i];
        else if (a == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (a == "--idle" && i + 1 < argc) idle = std::atof(argv[++i]);
        else if (a == "--stop") stop = true;
    }
    if (gd.empty()) { fprintf(stderr, "gpu-holder: --genome-dir is required\n"); return 2; }
    if (!stop) {   // nothing inherited from the starting process (pipes of its caller must reach EOF when the caller ends).
        // This runs before the first CUDA call: the driver's own descriptors (/dev/nvidia*) are opened by that call, and closing them broke every later cudaMalloc.
        std::vector<int> fds;
        if (DIR* d = opendir("/proc/self/fd")) { const int dfd = dirfd(d); while (dirent* e = readdir(d)) { const int fd = atoi(e->d_name); if (fd > 2 && fd != dfd) fds.push_back(fd); } closedir(d); }
        for (int fd : fds) close(fd);
    }
    if (cudaSetDevice(device) != cudaSuccess) { fprintf(stderr, "gpu-holder: cannot select device %d\n", device); return 1; }
    const std::string key = shKey(gd, deviceId(device)), sock = shPath(key, "sock"), info = shPath(key, "info");
    if (stop) {
        FILE* f = fopen(info.c_str(), "r"); if (!f) return 0;
        long pid = 0; if (fscanf(f, "%ld", &pid) == 1 && pid > 0) kill((pid_t) pid, SIGTERM);
        fclose(f); return 0;
    }
    signal(SIGTERM, shSig); signal(SIGINT, shSig); signal(SIGUSR1, shEvictSig); signal(SIGPIPE, SIG_IGN);
    const double t0 = now();
    const std::string dir = gd.back() == '/' ? gd : gd + "/";
    struct stat sg{}, ss{}, si{};
    if (stat((dir + "Genome").c_str(), &sg) || stat((dir + "SA").c_str(), &ss) || stat((dir + "SAindex").c_str(), &si)) { fprintf(stderr, "gpu-holder: Genome, SA or SAindex missing in %s\n", dir.c_str()); return 1; }
    const int fG = open((dir + "Genome").c_str(), O_RDONLY | O_CLOEXEC), fS = open((dir + "SA").c_str(), O_RDONLY | O_CLOEXEC), fI = open((dir + "SAindex").c_str(), O_RDONLY | O_CLOEXEC);
    u64 nb = 0;
    if (fG < 0 || fS < 0 || fI < 0 || pread(fI, &nb, 8, 0) != 8 || nb > 64) { fprintf(stderr, "gpu-holder: cannot read the index files\n"); return 1; }
    const u64 hdr = 8 + (nb + 1) * 8, nGen = (u64) sg.st_size, saB = (u64) ss.st_size, saiB = (u64) si.st_size - hdr;
    const u64 allocG = nGen + 2 * SH_PAD + 8, allocSA = saB + 64 + 8, allocSAi = saiB + 64 + 8;
    void *dG = nullptr, *dSA = nullptr, *dSAi = nullptr;
    for (int tries = 0; tries < 40; tries++) {   // another genome's holder may be leaving the device (SIGUSR1)
        if (!cudaMalloc(&dG, allocG) && !cudaMalloc(&dSA, allocSA) && !cudaMalloc(&dSAi, allocSAi)) break;
        cudaGetLastError(); cudaFree(dG); cudaFree(dSA); cudaFree(dSAi); dG = dSA = dSAi = nullptr; sleep(1);
    }
    if (!dG || !dSA || !dSAi) { size_t fr = 0, tt = 0; cudaMemGetInfo(&fr, &tt); fprintf(stderr, "gpu-holder: cannot allocate the index (%.2f GB): %s; device free %.1f of %.1f GB\n", b2gb(allocG + allocSA + allocSAi), cudaGetErrorString(cudaGetLastError()), b2gb(fr), b2gb(tt)); return 1; }
    cudaMemset(dG, 0, allocG); cudaMemset(dSA, 0, allocSA); cudaMemset(dSAi, 0, allocSAi);
    cudaMemset(dG, 5, SH_PAD); cudaMemset((char*) dG + SH_PAD + nGen, 5, SH_PAD);   // STAR's tail of the largest symbol (GENOME_spacingChar) around G
    const u64 CH = 256ull << 20; char* pin = nullptr;
    if (cudaMallocHost((void**) &pin, CH) != cudaSuccess) { fprintf(stderr, "gpu-holder: no pinned buffer\n"); return 1; }
    auto up = [&](int fd, u64 foff, char* dst, u64 len) -> bool {
        for (u64 o = 0; o < len; o += CH) {
            const u64 n = std::min(CH, len - o);
            u64 got = 0; while (got < n) { const ssize_t r = pread(fd, pin + got, n - got, (off_t) (foff + o + got)); if (r <= 0) return false; got += (u64) r; }
            if (cudaMemcpy(dst + o, pin, n, cudaMemcpyHostToDevice) != cudaSuccess) return false;
        }
        return true;
    };
    if (!up(fG, 0, (char*) dG + SH_PAD, nGen) || !up(fS, 0, (char*) dSA, saB) || !up(fI, hdr, (char*) dSAi, saiB) || cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "gpu-holder: upload failed\n"); return 1; }
    cudaFreeHost(pin);
    std::string why;
    auto rd = [&](int fd, u64 foff) { return [fd, foff](u64 o, u64 n, char* d) { return pread(fd, d, n, (off_t) (foff + o)) == (ssize_t) n; }; };
    const bool okG = shVerify((const char*) dG, nGen + 2 * SH_PAD, [&](u64 o, u64 n, char* d) {   // the padded image of G
        memset(d, 5, n);
        const u64 lo = std::max(o, SH_PAD), hi = std::min(o + n, SH_PAD + nGen);
        return hi <= lo || pread(fG, d + (lo - o), hi - lo, (off_t) (lo - SH_PAD)) == (ssize_t) (hi - lo); }, why);
    if (!okG || !shVerify((const char*) dSA, saB, rd(fS, 0), why) || !shVerify((const char*) dSAi, saiB, rd(fI, hdr), why)) { fprintf(stderr, "gpu-holder: upload check failed (%s)\n", why.c_str()); return 1; }
    ShRep rep{}; rep.status = 0; rep.bytes[0] = allocG; rep.bytes[1] = allocSA; rep.bytes[2] = allocSAi;
    if (cudaIpcGetMemHandle(&rep.h[0], dG) || cudaIpcGetMemHandle(&rep.h[1], dSA) || cudaIpcGetMemHandle(&rep.h[2], dSAi)) { fprintf(stderr, "gpu-holder: cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    const u64 held = budRound(allocG) + budRound(allocSA) + budRound(allocSAi);
    const int ls = socket(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0);
    sockaddr_un a{}; a.sun_family = AF_UNIX; snprintf(a.sun_path, sizeof a.sun_path, "%s", sock.c_str());
    unlink(sock.c_str());
    if (ls < 0 || bind(ls, (sockaddr*) &a, sizeof a) != 0 || listen(ls, 64) != 0) { fprintf(stderr, "gpu-holder: cannot listen on %s: %s\n", sock.c_str(), strerror(errno)); return 1; }
    { FILE* f = fopen((info + ".tmp").c_str(), "w"); if (f) { fprintf(f, "%ld %llu\n", (long) getpid(), (unsigned long long) held); fclose(f); rename((info + ".tmp").c_str(), info.c_str()); } }
    fprintf(stderr, "gpu-holder: pid %ld device %d genome %s resident (%.2f GB: G %llu, SA %llu, SAi %llu bytes) in %.1f s; socket %s\n", (long) getpid(), device, dir.c_str(),
            b2gb(held), (unsigned long long) nGen, (unsigned long long) saB, (unsigned long long) saiB, now() - t0, sock.c_str());
    std::vector<int> cl; double lastBusy = now(); u64 nServed = 0;
    while (!g_shStop) {
        std::vector<pollfd> v{{ls, POLLIN, 0}};
        for (int c : cl) v.push_back({c, POLLIN, 0});
        poll(v.data(), v.size(), 1000);
        for (size_t k = v.size(); k-- > 1;) if (v[k].revents) {   // a client that closes its end (or dies) is dropped
            char b; const ssize_t r = recv(v[k].fd, &b, 1, MSG_DONTWAIT);
            if (r == 0 || (r < 0 && errno != EAGAIN && errno != EINTR)) { close(v[k].fd); cl.erase(cl.begin() + (k - 1)); }
        }
        if (v[0].revents & POLLIN) {
            const int c = accept4(ls, nullptr, nullptr, SOCK_CLOEXEC);
            if (c >= 0) {
                timeval tv{5, 0}; setsockopt(c, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
                ShReq rq{}; ShRep out = rep;
                if (!shRead(c, &rq, sizeof rq) || rq.magic != SH_MAGIC) { close(c); continue; }
                if (rq.nG != allocG || rq.saB != allocSA || rq.saiB != allocSAi) {
                    out = ShRep{}; out.status = 1; snprintf(out.msg, sizeof out.msg, "sizes differ (client %llu/%llu/%llu, held %llu/%llu/%llu)", (unsigned long long) rq.nG,
                        (unsigned long long) rq.saB, (unsigned long long) rq.saiB, (unsigned long long) allocG, (unsigned long long) allocSA, (unsigned long long) allocSAi);
                }
                shWrite(c, &out, sizeof out);
                if (out.status == 0) { cl.push_back(c); nServed++; } else close(c);
            }
        }
        if (!cl.empty()) lastBusy = now();
        else if (g_shEvict) { fprintf(stderr, "gpu-holder: asked to leave the device for another genome, exiting after %llu clients\n", (unsigned long long) nServed); break; }
        else if (now() - lastBusy > idle) { fprintf(stderr, "gpu-holder: idle for %.0f s, exiting after %llu clients\n", idle, (unsigned long long) nServed); break; }
    }
    for (int c : cl) close(c);
    unlink(sock.c_str()); unlink(info.c_str());
    cudaFree(dG); cudaFree(dSA); cudaFree(dSAi);
    fprintf(stderr, "gpu-holder: stopped after %llu clients\n", (unsigned long long) nServed);
    return 0;
}
