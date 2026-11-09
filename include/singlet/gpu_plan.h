// Device plan for the GPU path (docs/gpu-engine/ANY-DEVICE.md): which of the engine's settings follow from the memory of the one GPU a run
// has, so that no per-device flag is needed. Pure functions of numbers (no CUDA here); the callers read the device through `singlet gpu-info`
// (a child process: the pipeline and the pool never initialize CUDA themselves) and the genome index size from the STAR genome directory.
//
//   fits(): can this device run the GPU path on this genome at all? The index (Genome, SA, SAindex) must sit on the device together with the
//           engine's smallest working set. Where it cannot (the 52.7 GB barnyard index on a 46 GB RTX 8000 or a 32 GB V100S), the sample
//           runs on the CPU engine, and says so in one line.
//   plan(): slots per device, the shared device index, and where the non-host screen runs. Every value a user sets (SINGLET_ENGINE_GPU_SLOTS,
//           `--slots`, SINGLET_GPU_SHARED_INDEX, `--nonhost-server`) is kept as given.
//
// The constants come from the engine's own budget line (engine_gpu.cu, "E17 budget ... estimates: lean") and from the single-device runs
// of 2026-11-09 (RTX 8000 46 GB, V100S 32 GB, H100 NVL 96 GB); see ANY-DEVICE.md for the table.
#pragma once
#include <poll.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace singlet_gpu_plan {

constexpr uint64_t kGiB = 1ull << 30;
constexpr uint64_t kMargin = 2 * kGiB;           // left to the driver and the contexts (the engine's budget takes the same off the device)
constexpr uint64_t kLeanOverhead = 5 * kGiB / 2; // beyond the index, the engine's smallest working set: context, tables, slack, smallest lanes (1.94 GB measured)
constexpr uint64_t kSlotWork = 16 * kGiB;        // one more sample's working set on a device that holds the index once (lean reads phase of a 65 M-pair sample)
constexpr uint64_t kNonhostIndex = 25 * kGiB;    // the non-host screen's index, 24.9 GB
constexpr uint64_t kNonhostWork = 5 * kGiB / 2;  // context and reads of a screen that loads the index itself
constexpr int kMaxAutoSlots = 2;                 // more slots than this did not pay on any device measured
constexpr int kDefaultMinCC = 70;                // sm_70 (V100S) and sm_75 (RTX 8000): the kernels are compiled for them

enum class Screen { Server, Private, Cpu };      // resident index on the device | each sample loads it for its screen | the CPU screen

struct Device {
    bool valid = false;
    uint64_t totalBytes = 0;
    int ccMajor = 0, ccMinor = 0;
    std::string name;
};

struct Request {
    uint64_t indexBytes = 0;   // Genome + SA + SAindex of the STAR genome
    int minCC = kDefaultMinCC; // SINGLET_ENGINE_GPU_MIN_CC, as major*10+minor
    int slots = 0;             // 0 = choose; > 0 as given
    int sharedIndex = -1;      // -1 = choose; 0 / 1 as given
    int screen = -1;           // -1 = choose; otherwise (int) Screen as given
};

struct Plan {
    bool gpu = true;           // false: run the CPU engine
    int slots = 1;
    bool sharedIndex = false;
    Screen screen = Screen::Private;
    std::string why;           // the reason for gpu == false
};

inline std::string gib(uint64_t b) {
    char s[32];
    std::snprintf(s, sizeof s, "%.1f GB", (double) b / (double) kGiB);
    return s;
}

// The smallest device memory the GPU path runs this genome in (one sample at a time, the index on the device).
inline uint64_t minDeviceBytes(uint64_t indexBytes) { return indexBytes + kLeanOverhead + kMargin; }

inline bool fits(const Device& d, uint64_t indexBytes, std::string* why = nullptr) {
    if (!d.valid || indexBytes == 0) return true;   // nothing known: leave the decision to the engine, as before
    if (d.totalBytes >= minDeviceBytes(indexBytes)) return true;
    if (why) *why = "the genome index (" + gib(indexBytes) + ") does not fit " + (d.name.empty() ? std::string("the device") : d.name) + " (" + gib(d.totalBytes) +
                    "; the GPU path needs " + gib(minDeviceBytes(indexBytes)) + ")";
    return false;
}

inline bool ccOk(const Device& d, int minCC, std::string* why = nullptr) {
    if (!d.valid || d.ccMajor * 10 + d.ccMinor >= minCC) return true;
    if (why) *why = (d.name.empty() ? std::string("the device") : d.name) + " is below compute capability " + std::to_string(minCC / 10) + "." + std::to_string(minCC % 10);
    return false;
}

inline Plan plan(const Device& d, const Request& r) {
    Plan p;
    if (r.slots > 0) p.slots = r.slots;
    if (r.sharedIndex >= 0) p.sharedIndex = r.sharedIndex == 1;
    if (r.screen >= 0) p.screen = (Screen) r.screen;
    if (!d.valid) {   // the device is not known: the settings of the 2-slot H100 node
        if (r.slots <= 0) p.slots = kMaxAutoSlots;
        if (r.sharedIndex < 0) p.sharedIndex = p.slots > 1;
        if (r.screen < 0) p.screen = Screen::Server;
        return p;
    }
    if (!ccOk(d, r.minCC, &p.why) || !fits(d, r.indexBytes, &p.why)) { p.gpu = false; p.slots = 1; p.sharedIndex = false; p.screen = Screen::Cpu; return p; }
    const uint64_t total = d.totalBytes;
    const uint64_t avail = total > kMargin ? total - kMargin : 0;
    // the screen: a resident index when the device keeps it and still holds the genome index and one sample beside it; otherwise each sample
    // loads it for its screen (the device is then free of it between screens); a device too small for that screens on the CPU
    if (r.screen < 0) {
        if (avail >= kNonhostIndex + r.indexBytes + kSlotWork) p.screen = Screen::Server;
        else if (avail >= kNonhostIndex + kNonhostWork) p.screen = Screen::Private;
        else p.screen = Screen::Cpu;
    }
    // slots: one more slot costs its working set; the index is paid once when it is shared and once per slot when it is not
    if (r.slots <= 0) {
        const uint64_t resident = p.screen == Screen::Server ? kNonhostIndex : p.screen == Screen::Private ? kNonhostIndex + kNonhostWork : 0;   // a screen that loads its index needs the room at its peak
        const uint64_t room = avail > resident + r.indexBytes ? avail - resident - r.indexBytes : 0;   // beside the shared index
        int n = (int) (room / kSlotWork);
        if (n > kMaxAutoSlots) n = kMaxAutoSlots;
        p.slots = n < 1 ? 1 : n;
    }
    if (r.sharedIndex < 0) p.sharedIndex = p.slots > 1;   // one holder per device pays off only when several samples would copy the index
    return p;
}

inline const char* screenName(Screen s) { return s == Screen::Server ? "resident server" : s == Screen::Private ? "loaded per sample" : "CPU"; }

// one log line
inline std::string describe(const Device& d, const Request& r, const Plan& p) {
    std::string s = "device " + (d.valid ? d.name + ", " + gib(d.totalBytes) + ", cc " + std::to_string(d.ccMajor) + "." + std::to_string(d.ccMinor) : std::string("unknown")) +
                    ", genome index " + gib(r.indexBytes) + ": ";
    if (!p.gpu) return s + p.why + "; using the CPU engine";
    return s + std::to_string(p.slots) + " slot(s), shared index " + (p.sharedIndex ? "on" : "off") + ", non-host screen " + screenName(p.screen);
}

// The size of the STAR genome's index files (Genome + SA + SAindex); 0 when the directory has none of them.
inline uint64_t genomeIndexBytes(const std::string& dir) {
    uint64_t t = 0;
    for (const char* f : {"/Genome", "/SA", "/SAindex"}) {
        struct stat st;
        if (stat((dir + f).c_str(), &st) == 0 && st.st_size > 0) t += (uint64_t) st.st_size;
    }
    return t;
}

// The device facts a parent passes to its children (`singlet gpu-info` output: "<total bytes> <major> <minor> <name>"), in one variable.
constexpr const char* kEnvDevice = "SINGLET_GPU_DEVICE_INFO";
inline std::string encode(const Device& d) { return std::to_string(d.totalBytes) + " " + std::to_string(d.ccMajor) + " " + std::to_string(d.ccMinor) + " " + d.name; }
inline bool decode(const std::string& s, Device& d) {
    unsigned long long t = 0; int a = 0, b = 0, n = 0;
    if (std::sscanf(s.c_str(), "%llu %d %d %n", &t, &a, &b, &n) < 3 || t == 0) return false;
    d.valid = true; d.totalBytes = t; d.ccMajor = a; d.ccMinor = b;
    d.name = n > 0 && (size_t) n <= s.size() ? s.substr((size_t) n) : std::string();
    while (!d.name.empty() && (d.name.back() == '\n' || d.name.back() == '\r' || d.name.back() == ' ')) d.name.pop_back();
    return true;
}

// The device a child sees: the parent's SINGLET_GPU_DEVICE_INFO when set (the pool asks once for all its samples), otherwise `singlet gpu-info N`
// run as a child process (exec'd at once, so this process never initializes CUDA). Invalid when there is no device or the child does not answer in 60 s.
inline bool probeDevice(int device, Device& d) {
    if (const char* e = std::getenv(kEnvDevice)) { if (decode(e, d)) return true; }
    int pp[2];
    if (pipe(pp) != 0) return false;
    const pid_t c = fork();
    if (c < 0) { close(pp[0]); close(pp[1]); return false; }
    if (c == 0) {
        dup2(pp[1], 1); close(pp[0]); close(pp[1]);
        const std::string dv = std::to_string(device < 0 ? 0 : device);
        execl("/proc/self/exe", "singlet", "gpu-info", dv.c_str(), (char*) nullptr);
        _exit(127);
    }
    close(pp[1]);
    std::string out;
    char b[512];
    for (;;) {
        struct pollfd q = {pp[0], POLLIN, 0};
        if (poll(&q, 1, 60000) <= 0) { kill(c, SIGKILL); out.clear(); break; }
        const ssize_t n = read(pp[0], b, sizeof b);
        if (n <= 0) break;
        out.append(b, (size_t) n);
    }
    close(pp[0]);
    int st = 0;
    waitpid(c, &st, 0);
    return WIFEXITED(st) && WEXITSTATUS(st) == 0 && decode(out, d);
}

}   // namespace singlet_gpu_plan
