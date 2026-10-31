#include "HandoffShadow.h"
#include "SeqCache.h"
#include "InOutStreams.h"
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

bool g_shadowOn = getenv("SINGLET_STAR_SHADOW") != nullptr;
const uint32 *g_handoffRecLen = nullptr;

namespace {
// counters: reads mapped unique / multi, bases, mismatches, insertions, deletions, inserted / deleted length, splices by motif, annotated splices
constexpr int NC = 8 + SJ_MOTIF_SIZE + 1;
const char *NAMES[8] = {"mappedReadsU", "mappedReadsM", "mappedBases", "mappedMismatchesN", "mappedInsN", "mappedDelN", "mappedInsL", "mappedDelL"};
struct Acc {
    uint64 c[NC] = {0};
    double portion = 0;
};
std::mutex regMutex;
std::vector<Acc *> registry;
thread_local Acc *tlAcc = nullptr;

Acc *mine() {
    if (tlAcc == nullptr) {
        tlAcc = new Acc();
        std::lock_guard<std::mutex> g(regMutex);
        registry.push_back(tlAcc);
    }
    return tlAcc;
}
}  // namespace

void shadowAddLive(const Stats &b, const Stats &a) {
    Acc *x = mine();
    x->c[0] += a.mappedReadsU - b.mappedReadsU;
    x->c[1] += a.mappedReadsM - b.mappedReadsM;
    x->c[2] += a.mappedBases - b.mappedBases;
    x->c[3] += a.mappedMismatchesN - b.mappedMismatchesN;
    x->c[4] += a.mappedInsN - b.mappedInsN;
    x->c[5] += a.mappedDelN - b.mappedDelN;
    x->c[6] += a.mappedInsL - b.mappedInsL;
    x->c[7] += a.mappedDelL - b.mappedDelL;
    for (int k = 0; k < SJ_MOTIF_SIZE; k++) x->c[8 + k] += a.splicesN[k] - b.splicesN[k];
    x->c[8 + SJ_MOTIF_SIZE] += a.splicesNsjdb - b.splicesNsjdb;
    x->portion += a.mappedPortion - b.mappedPortion;
}

// the same counters from the summaries alone: for every hand-off read with a summary, what ReadAlign::outputHandoffSummary adds to statsRA
void shadowCompare(Parameters &P) {
    if (!g_shadowOn) return;
    auto &log = P.inOut->logMain;
    if (g_handoffSum == nullptr || g_handoffReadRecord == nullptr || g_handoffRecLen == nullptr) {
        log << "SHADOW: no hand-off summaries, nothing to compare\n" << std::flush;
        return;
    }
    Acc live;
    for (Acc *r : registry) {
        for (int k = 0; k < NC; k++) live.c[k] += r->c[k];
        live.portion += r->portion;
    }
    Acc bulk;
    uint64 nReads = 0;
    for (uint64 i = 0; i < g_handoffNReads; i++) {
        const uint32 rec = g_handoffReadRecord[i];
        if (rec == 0xFFFFFFFFu || g_handoffSumOff[rec + 1] <= g_handoffSumOff[rec]) continue;
        nReads++;
        const uint8 *p = g_handoffSum + g_handoffSumOff[rec];
        uint16 n16, ng;
        memcpy(&n16, p, 2);
        const uint8 flags = p[2];
        memcpy(&ng, p + 4, 2);
        p += 6 + 4 * (uint64) ng;
        if (n16 > 1) {
            bulk.c[1]++;
            continue;
        }
        bulk.c[0]++;
        uint16 a[3];
        uint32 b[3];
        memcpy(a, p, 6);
        memcpy(b, p + 6, 12);
        const uint8 *c = p + 18;
        bulk.c[3] += a[0]; bulk.c[4] += a[1]; bulk.c[5] += a[2];
        bulk.c[6] += b[0]; bulk.c[7] += b[1];
        if (flags & 2) {
            for (int k = 0; k < SJ_MOTIF_SIZE; k++) bulk.c[8 + k] += c[k];
            bulk.c[8 + SJ_MOTIF_SIZE] += c[SJ_MOTIF_SIZE];
            bulk.c[2] += b[2];
            // the live pass divides by the read's own length: the record's (clipped) length plus the bases clipped from its ends
            uint64 L = g_handoffRecLen[rec];
            if (g_handoffClip5 != nullptr) L += g_handoffClip5[i] + g_handoffClip3[i];
            bulk.portion += double(b[2]) / double(L);
        }
    }
    int bad = 0;
    for (int k = 0; k < NC; k++) {
        if (live.c[k] != bulk.c[k]) {
            bad++;
            log << "SHADOW statsRA MISMATCH counter " << (k < 8 ? NAMES[k] : (k < 8 + SJ_MOTIF_SIZE ? "splicesN" : "splicesNsjdb"))
                << (k >= 8 && k < 8 + SJ_MOTIF_SIZE ? k - 8 : 0) << ": live " << live.c[k] << " bulk " << bulk.c[k] << "\n";
        }
    }
    char buf[200];
    snprintf(buf, sizeof buf, "live %.17g bulk %.17g (difference %.3g)", live.portion, bulk.portion, live.portion - bulk.portion);
    log << "SHADOW statsRA: " << nReads << " hand-off reads, " << NC << " integer counters " << (bad ? "DIFFER" : "equal") << "; mappedPortion " << buf
        << (live.portion == bulk.portion ? " (bit-identical)" : " (NOT bit-identical)") << "\n" << std::flush;
}
