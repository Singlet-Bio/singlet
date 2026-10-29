// Host-only run of the phase 1d kernel stage against the Python model: compiles paired_windows.cuh + stitch.cuh as plain C++ (HD is
// empty without __CUDACC__) and runs mapFinishPE on the unit cases of fixtures/cases.txt. Needs g++ only, no GPU. This checks the
// logic of the draft (the mate rules, the two-entry maxScoreMate, the per-mate length check) bit for bit against the model; it does
// not check anything that is CUDA-specific (launches, memory, the device path of paired_microbench --unit).
//   g++ -std=c++17 -O1 -I../../src/engine_gpu paired_unit_host.cpp -o paired_unit_host && ./paired_unit_host fixtures/cases.txt
#include "paired_harness_common.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: paired_unit_host cases.txt\n"); return 1; }
    std::map<std::string, std::string> chroms; std::vector<std::string> order; u64 gbits = 18; std::vector<Case> cases; std::vector<ClipCase> clips;
    if (!parseCases(argv[1], chroms, order, gbits, cases, clips)) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    HostGenome H;
    buildUnitGenome(H, chroms, order, gbits);
    auto w = std::make_unique<WorkPE<PeLane>>(); auto rb = std::make_unique<RBuf>();
    int bad = 0;
    for (auto& c : cases) {
        Params P;
        if (!caseParams(c, H.v.nGenome, gbits, P)) return 1;
        setupWork(*w, *rb, P, c);
        std::vector<uint8_t> hb(1 << 16); m5::Blob blob; blob.p = hb.data(); blob.n = 0; blob.cap = (int)hb.size(); blob.full = false;
        m5::Status st = m5::pe::mapFinishPE(P, H.v, *w, *rb, blob);
        bool ok = st == m5::Status::Ok && matches(*w, c);
        printf("%-24s %s\n", c.name.c_str(), ok ? "ok" : "DIFF");
        if (!ok) { printf("  kernel: %s\n", describe(*w).c_str()); bad++; }
    }
    printf("%d of %zu cases differ from the model\n", bad, cases.size());
    int badClip = runClipCases(clips);
    printf("%d of %zu clip cases differ from the model\n", badClip, clips.size());
    return bad + badClip != 0;
}
