// The device plan of the GPU path (include/singlet/gpu_plan.h): which settings follow from one GPU's memory, and when a genome index does not
// fit the device (the sample then runs on the CPU engine). The three devices of the cluster, the three genomes of the benchmarks.

#include <cstdlib>
#include <iostream>
#include <string>

#include "singlet/gpu_plan.h"

using namespace singlet_gpu_plan;

static int n_fail = 0;
#define CHECK(cond) do { if (!(cond)) { std::cout << "FAIL line " << __LINE__ << ": " #cond "\n"; ++n_fail; } } while (0)

static Device dev(const char* name, uint64_t mib, int maj, int min) {
    Device d;
    d.valid = true; d.name = name; d.totalBytes = mib * (1ull << 20); d.ccMajor = maj; d.ccMinor = min;
    return d;
}
// total memory as the driver reports it (MiB): RTX 8000 49152 (45.6 GiB usable 44.3 after the context), V100S 32768 less ECC, H100 NVL 95830
static const Device kRtx = dev("Quadro RTX 8000", 45556, 7, 5), kV100s = dev("Tesla V100S-PCIE-32GB", 32494, 7, 0), kH100 = dev("NVIDIA H100 NVL", 95830, 9, 0);
static const uint64_t kHuman = 3216071051ull + 12789324355ull + 1565873619ull;                 // GRCh38-2024-A: 16.4 GiB
static const uint64_t kMouse = 2813048951ull + 22513440027ull + 1565873619ull;                 // GRCm39-2024-A: 25.0 GiB
static const uint64_t kBarnyard = 6037373342ull + 48962556871ull + 1610612861ull;              // human + mouse: 52.7 GiB

static Plan planFor(const Device& d, uint64_t idx) { Request r; r.indexBytes = idx; return plan(d, r); }

int main() {
    // every device runs the human genome on the GPU with no setting
    for (const Device* d : {&kRtx, &kV100s, &kH100}) CHECK(planFor(*d, kHuman).gpu);
    // the 46 GB and 32 GB devices: one slot, no shared index (one holder pays only when several samples would copy the index), the screen loads its index per sample
    for (const Device* d : {&kRtx, &kV100s}) {
        const Plan p = planFor(*d, kHuman);
        CHECK(p.slots == 1); CHECK(!p.sharedIndex); CHECK(p.screen == Screen::Private);
    }
    // the 96 GB device: two slots on a shared index and a resident screen
    { const Plan p = planFor(kH100, kHuman); CHECK(p.slots == 2); CHECK(p.sharedIndex); CHECK(p.screen == Screen::Server); }
    // mouse: the 32 GB device still holds the index and one sample
    CHECK(planFor(kV100s, kMouse).gpu);
    CHECK(planFor(kV100s, kMouse).slots == 1);
    // the barnyard index (52.7 GiB) fits no device but the 96 GB one, and there it takes one slot and a screen that loads its index
    for (const Device* d : {&kRtx, &kV100s}) {
        std::string why;
        const Plan p = planFor(*d, kBarnyard);
        CHECK(!p.gpu); CHECK(p.screen == Screen::Cpu); CHECK(p.slots == 1); CHECK(!p.sharedIndex);
        CHECK(!fits(*d, kBarnyard, &why));
        CHECK(why.find("does not fit") != std::string::npos); CHECK(why.find(d->name) != std::string::npos);
    }
    { const Plan p = planFor(kH100, kBarnyard); CHECK(p.gpu); CHECK(p.slots == 1); CHECK(p.screen == Screen::Private); }
    // the limit is the index plus the engine's smallest working set plus the margin
    { Device d = kV100s; d.totalBytes = minDeviceBytes(kHuman); CHECK(fits(d, kHuman)); d.totalBytes -= 1; CHECK(!fits(d, kHuman)); }
    // a device below the compute capability floor runs on the CPU; the floor is settable
    { Device d = dev("Tesla P100", 16384, 6, 0); Request r; r.indexBytes = kHuman; CHECK(!plan(d, r).gpu); }
    { Request r; r.indexBytes = kHuman; r.minCC = 80; const Plan p = plan(kRtx, r); CHECK(!p.gpu); CHECK(p.why.find("compute capability 8.0") != std::string::npos); }
    CHECK(kDefaultMinCC == 70);   // the V100S (7.0) and RTX 8000 (7.5) pass with no variable
    // values the user sets are kept
    { Request r; r.indexBytes = kHuman; r.slots = 3; r.sharedIndex = 1; r.screen = (int) Screen::Server; const Plan p = plan(kRtx, r);
      CHECK(p.slots == 3); CHECK(p.sharedIndex); CHECK(p.screen == Screen::Server); }
    { Request r; r.indexBytes = kHuman; r.slots = 2; r.sharedIndex = 0; const Plan p = plan(kH100, r); CHECK(p.slots == 2); CHECK(!p.sharedIndex); }
    // a device too small for the screen's index screens on the CPU, and still takes the GPU path
    { const Device d = dev("24 GB card", 24000, 8, 6); const Plan p = planFor(d, 8ull << 30); CHECK(p.gpu); CHECK(p.screen == Screen::Cpu); }
    // an unknown device or genome keeps the engine's own decisions: the settings of the 2-slot node, and the GPU path is attempted
    { Device d; const Plan p = planFor(d, kHuman); CHECK(p.gpu); CHECK(p.slots == 2); CHECK(p.screen == Screen::Server); CHECK(fits(d, kBarnyard)); CHECK(fits(kRtx, 0)); }
    // the facts pass to a child process in one variable
    { Device d; CHECK(decode(encode(kRtx), d)); CHECK(d.valid); CHECK(d.totalBytes == kRtx.totalBytes); CHECK(d.ccMajor == 7 && d.ccMinor == 5); CHECK(d.name == kRtx.name); }
    { Device d; CHECK(!decode("", d)); CHECK(!decode("no CUDA device", d)); CHECK(!d.valid); }
    // the index size of a directory without a genome is 0 (the check then leaves the decision to the engine)
    CHECK(genomeIndexBytes("/nonexistent-genome-dir") == 0);
    // the log line
    { Request r; r.indexBytes = kBarnyard; const std::string l = describe(kRtx, r, plan(kRtx, r));
      CHECK(l.find("using the CPU engine") != std::string::npos); CHECK(l.find("52.7 GB") != std::string::npos); CHECK(l.find('\n') == std::string::npos); }
    if (n_fail == 0) std::cout << "test_gpu_plan: ok\n";
    return n_fail == 0 ? 0 : 1;
}
