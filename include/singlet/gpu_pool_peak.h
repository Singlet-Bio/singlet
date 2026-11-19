// SPDX-License-Identifier: MIT
// gpu_pool_peak.h: the host-memory figures of the gpu-pool's SINGLET_POOL_PEAKADMIT=1 gate (docs/gpu-engine/runtime/T2-cggate.md,
// data from T2-peakmodel.md). Pure functions and constants, no I/O except the optional environment overrides in fromEnv(), so a unit test
// can call them (tests/cpp/test_gpu_pool_peak.cpp). Sizes are GiB above the job's idle cgroup baseline (anon + shmem with no item running).
//
//   P_B(n, bases)  = a + b n[M] + c bases[G]                       outside the zone [zoneLo, zoneHi) M reads
//   P_B            = max(P_B, planes + fbC0 + fbK n[M])            inside the zone (a fallback to STAR's loop is possible and not visible before the start)
//   planes         = (24 n + bases / 2) bytes                      the .1fq decode: offsets 8, barcode read 16 B per read, cDNA and its quality 2 bits per base each
//   fallback       = planes + fbC0 + fbGibPerMD x D[M]             level 0 (E8 records not built): D distinct sequences
//   ledger         = planes + region + ledKeep x soloKeep + late + ledA + ledB n[M]     (engine path; sizes printed by the engine)
//   figure F0      = P + margin (before the engine's "level" line); F1 = min(F0, lvlU P + lvlV + margin) after it; F2 = min(F1, wtU P + wtV + margin)
//                    after the "[allsite] async: waited" line
#pragma once

#include <algorithm>
#include <cstdlib>
#include <string>

namespace singlet { namespace poolpeak {

struct Consts {
    // P_B (T2-peakmodel.md 1.5, fitted on the 14 classes outside the zone)
    double a = 6.604407719095382, b = 0.08168172883122637, c = 1.0588433857586883;
    double zoneLo = 70.0, zoneHi = 160.0;            // M reads
    double fbC0 = 6.06, fbK = 0.49, fbGibPerMD = 0.70;   // zone and fallback formulas (0.49 = 0.70 GiB per M distinct x D / n 0.70)
    double margin = 12.0;                            // SINGLET_POOL_PEAK_MARGIN_GB
    // phases (T2-peakmodel.md 2)
    double lvlU = 0.9, lvlV = 7.077160443572787, wtU = 0.5, wtV = 2.6432036829225236;
    // ledger (1.3)
    double ledA = 6.06, ledB = 0.0238, ledKeep = 0.9;
    // the fitted range and regime
    double nMaxM = 1400.0;                           // above: not fitted
    double fitMemGb = 31.0, fitMemTolGb = 1.0;       // SINGLET_ENGINE_GPU_MEM_GB the fit was made at, and how far from it the pool still trusts the figures
    double fitSlots = 2.0, fitSlotsTol = 0.0;        // SINGLET_ENGINE_GPU_SLOTS
    // the engine's estimate of the fallback (first peakmodel line): records phase need (GiB) = fbNeedA + fbNeedB x D[M]; a two-point fit to the two logged fallbacks
    // (d100s_3p: D 61.83 M, "needs about 35.07 GB"; nuclei2k_v2: D 79.81 M, 43.93 GB; GB there is GiB). Compared with the device budget.
    double fbNeedA = 4.67, fbNeedB = 0.492;
};

static const double kGiB = 1073741824.0;

inline double planesGib(double nM, double basesG) { return (24.0 * nM * 1e6 + 0.5 * basesG * 1e9) / kGiB; }
inline bool inZone(double nM, const Consts& k) { return nM >= k.zoneLo && nM < k.zoneHi; }
// the linear predictor without the zone rule
inline double pLinear(double nM, double basesG, const Consts& k) { return k.a + k.b * nM + k.c * basesG; }
// level 0 (fallback) figure with D distinct sequences (M), above the baseline, no margin
inline double fallbackP(double nM, double basesG, double DM, const Consts& k) { return planesGib(nM, basesG) + k.fbC0 + k.fbGibPerMD * DM; }
// P of an item before it starts: the zone rule applied
inline double peakP(double nM, double basesG, const Consts& k) {
    double p = pLinear(nM, basesG, k);
    if (inZone(nM, k)) p = std::max(p, planesGib(nM, basesG) + k.fbC0 + k.fbK * nM);
    return p;
}
// the figure (GiB above the baseline) an item at phase 0 (before the level line), 1 (after it), 2 (after the waited line) can still reach
inline double phaseFigure(double P, int phase, const Consts& k) {
    const double f0 = P + k.margin;
    const double f1 = std::min(f0, k.lvlU * P + k.lvlV + k.margin);
    const double f2 = std::min(f1, k.wtU * P + k.wtV + k.margin);
    return phase >= 2 ? f2 : phase == 1 ? f1 : f0;
}
// the ledger peak from the engine's printed sizes (bytes); the Solo keep figure is the logged GB (1e9) as the fit used it
inline double ledgerL(double nM, double basesG, double regionBytes, double soloKeepBytes, double lateBytes, const Consts& k) {
    return planesGib(nM, basesG) + regionBytes / kGiB + k.ledKeep * soloKeepBytes / 1e9 + lateBytes / kGiB + k.ledA + k.ledB * nM;
}
// the engine's expectation of the fallback at the dedup (first peakmodel line): the records' phase need against the device budget (both GiB)
inline bool fallbackExpected(bool rc, double DM, double budgetGib, const Consts& k) {
    return !rc && budgetGib > 0 && k.fbNeedA + k.fbNeedB * DM > budgetGib;
}
// is the fit valid for this pool's environment? memGb / slots: SINGLET_ENGINE_GPU_MEM_GB (<0: unset) and SINGLET_ENGINE_GPU_SLOTS (<0: unset); hostcutOff: any of the six hostcut
// switches set to something else than unset, empty or "1". `why` says what failed.
inline bool fitValid(double nM, double memGb, double slots, bool hostcutOff, const Consts& k, std::string& why) {
    if (nM > k.nMaxM) { why = "n above the fitted range"; return false; }
    if (hostcutOff) { why = "a hostcut switch is off"; return false; }
    if (memGb < 0) { why = "SINGLET_ENGINE_GPU_MEM_GB unset (the device budget, which sets the read-chunk and fallback regimes, is not known to the pool)"; return false; }
    if (memGb < k.fitMemGb - k.fitMemTolGb || memGb > k.fitMemGb + k.fitMemTolGb) { why = "SINGLET_ENGINE_GPU_MEM_GB differs from the fitted value"; return false; }
    if (slots < 0 || slots < k.fitSlots - k.fitSlotsTol || slots > k.fitSlots + k.fitSlotsTol) { why = "SINGLET_ENGINE_GPU_SLOTS differs from the fitted value"; return false; }
    return true;
}

inline double envd(const char* n, double d) { const char* e = std::getenv(n); return e && e[0] ? std::atof(e) : d; }
// the constants with their environment overrides: SINGLET_POOL_PEAK_<NAME>, margin SINGLET_POOL_PEAK_MARGIN_GB
inline Consts fromEnv() {
    Consts k;
    k.a = envd("SINGLET_POOL_PEAK_A", k.a); k.b = envd("SINGLET_POOL_PEAK_B", k.b); k.c = envd("SINGLET_POOL_PEAK_C", k.c);
    k.zoneLo = envd("SINGLET_POOL_PEAK_ZONE_LO_M", k.zoneLo); k.zoneHi = envd("SINGLET_POOL_PEAK_ZONE_HI_M", k.zoneHi);
    k.fbC0 = envd("SINGLET_POOL_PEAK_FB_C0", k.fbC0); k.fbK = envd("SINGLET_POOL_PEAK_FB_K", k.fbK); k.fbGibPerMD = envd("SINGLET_POOL_PEAK_FB_GIB_PER_MD", k.fbGibPerMD);
    k.margin = envd("SINGLET_POOL_PEAK_MARGIN_GB", k.margin);
    k.lvlU = envd("SINGLET_POOL_PEAK_LEVEL_U", k.lvlU); k.lvlV = envd("SINGLET_POOL_PEAK_LEVEL_V", k.lvlV);
    k.wtU = envd("SINGLET_POOL_PEAK_WAITED_U", k.wtU); k.wtV = envd("SINGLET_POOL_PEAK_WAITED_V", k.wtV);
    k.ledA = envd("SINGLET_POOL_PEAK_LEDGER_A", k.ledA); k.ledB = envd("SINGLET_POOL_PEAK_LEDGER_B", k.ledB); k.ledKeep = envd("SINGLET_POOL_PEAK_LEDGER_KEEP", k.ledKeep);
    k.nMaxM = envd("SINGLET_POOL_PEAK_NMAX_M", k.nMaxM);
    k.fitMemGb = envd("SINGLET_POOL_PEAK_FIT_MEM_GB", k.fitMemGb); k.fitMemTolGb = envd("SINGLET_POOL_PEAK_FIT_MEM_TOL_GB", k.fitMemTolGb);
    k.fitSlots = envd("SINGLET_POOL_PEAK_FIT_SLOTS", k.fitSlots); k.fitSlotsTol = envd("SINGLET_POOL_PEAK_FIT_SLOTS_TOL", k.fitSlotsTol);
    return k;
}

} }   // namespace singlet::poolpeak
