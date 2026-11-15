// test_pool_admission.cpp: the gpu-pool memory-aware start rule (include/singlet/pipeline/pool_admission.h).
//   1. the model covers the measured peaks it was calibrated on (3' 300 M, 1.0 B and 1.3 B pairs; 5' 435-516 M pairs)
//   2. decide(): empty job, oversize sample alone, fit, no fit, reservations
//   3. learn() only raises the slope of the right chemistry
//   4. the 5' protocol ids match the protocol registry
//   5. the .1fq header reader (pairs from n_original, 5' from the protocol id)
//   6. a simulated one-H100 job (375 GB, one pool) running the two deep 3' samples, two deep 5' samples and normal samples with the
//      pool's admission loop: the true peaks in flight never exceed the job's memory, every sample runs, and the 1.3 B-pair sample runs alone
//   7. two pools sharing one ledger directory see each other's samples
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unistd.h>
#include <vector>

#include "singlet/fq/protocol.h"
#include "singlet/pipeline/pool_admission.h"

namespace pa = singlet::pool_admission;

static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) \
    do { if (cond) { ++n_pass; } else { ++n_fail; std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n"; } } while (0)

static void test_model() {
    const pa::Model m;
    struct P { double mp; bool five; double peak; } pts[] = {
        {300.2, false, 88.0}, {999.8, false, 266.4}, {1303.7, false, 302.3},   // rt-node 815885, rt-deep 815819/815767 (ru_maxrss)
        {435.0, true, 190.0}, {516.0, true, 190.0},                              // dry run 2: 5' samples held 130-190 GB when OOM-killed
    };
    for (const P& p : pts)
        CHECK(pa::projected_peak_gb(p.mp, p.five, m) >= p.peak, "projection covers the measured peak at " + std::to_string(p.mp) + " M pairs");
    CHECK(pa::projected_peak_gb(300.2, false, m) < 1.15 * 88.0, "the 3' projection at 300 M pairs is within 15% of the measured peak");
    CHECK(pa::projected_peak_gb(500, true, m) > pa::projected_peak_gb(500, false, m), "5' projects above 3' at the same depth");
    CHECK(pa::projected_peak_gb(-5, false, m) == m.a_gb, "a negative depth counts as zero");
}

static void test_decide() {
    using V = pa::Verdict;
    CHECK(pa::decide(300, 0, 0, 0, 100) == V::Start, "empty job: a sample that fits starts");
    CHECK(pa::decide(300, 0, 0, 0, 400) == V::StartAlone, "empty job: a sample larger than the budget starts alone");
    CHECK(pa::decide(300, 400, 1, 0, 10) == V::Wait, "nothing starts beside a sample that runs alone");
    CHECK(pa::decide(300, 150, 1, 0, 150) == V::Start, "exact fit starts");
    CHECK(pa::decide(300, 150, 1, 0, 151) == V::Wait, "over the budget waits");
    CHECK(pa::decide(300, 100, 1, 150, 60) == V::Wait, "a reservation leaves no room");
    CHECK(pa::decide(300, 100, 1, 150, 50) == V::Start, "room beside the reservation");
    CHECK(pa::decide(300, 0, 0, 300, 10) == V::Wait, "an oversize head's reservation blocks every backfill, even in an empty job");
    CHECK(pa::decide(300, 0, 0, 100, 150) == V::Start, "empty job with a smaller reservation: a fitting sample starts");
}

static void test_learn() {
    pa::Model m;
    const double b3 = m.b3_gb, b5 = m.b5_gb;
    CHECK(!pa::learn(m, 300, false, 50), "a peak under the projection changes nothing");
    CHECK(!pa::learn(m, 20, false, 500), "a small sample changes nothing");
    CHECK(pa::learn(m, 500, true, 300), "a 5' peak above its projection raises the model");
    CHECK(m.b3_gb == b3 && m.b5_gb > b5, "only the 5' slope rose");
    CHECK(pa::projected_peak_gb(500, true, m) >= 300 - 1e-9, "the raised model covers the peak it saw");
}

static void test_5p_ids() {
    int n5 = 0;
    bool same = true;
    for (const auto& s : singlet::fq::known_protocols()) {
        const bool five = s.tag.rfind("10x-5p-", 0) == 0;
        if (five) ++n5;
        if (five != pa::is_10x_5p(static_cast<uint8_t>(s.protocol_id))) { same = false; std::cout << "  protocol " << s.tag << " id " << (int) s.protocol_id << "\n"; }
    }
    CHECK(same, "is_10x_5p agrees with the protocol registry for every protocol");
    CHECK(n5 == 3, "three 10x 5' protocols in the registry");
}

static void test_header() {
    char path[] = "/tmp/test_pool_admission_XXXXXX";
    const int fd = mkstemp(path);
    CHECK(fd >= 0, "temp file");
    if (fd < 0) return;
    singlet::fq::Header h;
    h.init();
    h.protocol_id = 4;          // 10x-5p-v3
    h.n_unique = 400000000ull;
    h.n_original = 516000000ull;
    CHECK(write(fd, &h, sizeof h) == (ssize_t) sizeof h, "header written");
    close(fd);
    const pa::SampleInfo s = pa::read_1fq_info(path);
    CHECK(s.known && s.five_prime && s.mpairs > 515.9 && s.mpairs < 516.1, "pairs from n_original and 5' from the protocol id");
    FILE* f = std::fopen(path, "wb");
    std::fputs("@read1\nACGT\n+\nIIII\n", f);
    std::fclose(f);
    CHECK(!pa::read_1fq_info(path).known, "a FASTQ is not a .1fq: unknown depth");
    unlink(path);
    CHECK(!pa::read_1fq_info("/nonexistent/x.1fq").known, "a missing file: unknown depth");
}

// The pool's admission loop (gpu_pool.cpp admitOne) on a discrete-time model of one job: each sample holds its true peak for its whole run.
struct Sim { double mp; bool five; double truePeak; double dur; };
static void test_simulated_job() {
    const double limit = 375.0, base = 45.0, budget = 0.9 * limit - base;   // the default budget of a 375 GB one-H100 job
    const pa::Model m;
    std::vector<Sim> q = {
        {999.8, false, 266.4, 1000}, {1303.7, false, 302.3, 700},         // the two deep 3' samples
        {516.0, true, 230.0, 900}, {435.0, true, 190.0, 800},             // two deep 5' samples (true peaks above what dry run 2 saw)
        {300.2, false, 88.0, 220}, {300.2, false, 88.0, 220}, {120, false, 45, 100}, {80, true, 50, 90}, {300.2, false, 88.0, 220},
        {40, false, 25, 60}, {300.2, false, 88.0, 220}, {250, true, 120, 200},
    };
    std::vector<int> pending;
    for (int i = 0; i < (int) q.size(); ++i) pending.push_back(i);
    struct Fl { int i; double end; };
    std::vector<Fl> fl;
    std::vector<int> order;
    bool aloneOk = true, everOver = false;
    double peakSeen = 0, t = 0, headSince = 0;
    int headIdx = -1;
    const int workers = 4;
    auto proj = [&](int i) { return pa::projected_peak_gb(q[i].mp, q[i].five, m); };
    for (int guard = 0; guard < 100000 && (!pending.empty() || !fl.empty()); ++guard) {
        // ends
        fl.erase(std::remove_if(fl.begin(), fl.end(), [&](const Fl& f) { return f.end <= t; }), fl.end());
        // starts
        while ((int) fl.size() < workers && !pending.empty()) {
            double infl = 0;
            for (auto& f : fl) infl += proj(f.i);
            const int head = pending.front();
            if (head != headIdx) { headIdx = head; headSince = t; }
            int pick = -1;
            pa::Verdict v = pa::decide(budget, infl, fl.size(), 0, proj(head));
            if (v != pa::Verdict::Wait) pick = 0;
            else for (size_t k = 1; k < pending.size(); ++k)
                if (pa::decide(budget, infl, fl.size(), std::min(proj(head), budget), proj(pending[k])) == pa::Verdict::Start) { pick = (int) k; break; }
            if (pick < 0) break;
            const int i = pending[pick];
            pending.erase(pending.begin() + pick);
            if (v == pa::Verdict::StartAlone && !fl.empty()) aloneOk = false;
            fl.push_back({i, t + q[i].dur});
            order.push_back(i);
        }
        double mem = base;
        for (auto& f : fl) mem += q[f.i].truePeak;
        peakSeen = std::max(peakSeen, mem);
        if (mem > limit) everOver = true;
        for (auto& f : fl) if (f.i == 1 && fl.size() > 1) aloneOk = false;   // the 1.3 B-pair sample (projection above the budget) runs alone
        t += 10;
    }
    (void) headSince;
    std::cout << "  simulated 375 GB job: budget " << budget << " GB, peak of the true sizes in flight " << peakSeen << " GB, finished at t=" << t << " s\n";
    CHECK(order.size() == q.size(), "every sample ran");
    CHECK(!everOver, "the true peaks in flight never exceed the job's memory");
    CHECK(aloneOk, "the 1.3 B-pair sample ran alone");
    CHECK(proj(1) > budget, "the 1.3 B-pair sample's projection exceeds the default budget of a 375 GB job");
    CHECK(proj(0) <= budget, "the 1.0 B-pair sample fits the budget (it runs without waiting for an empty job)");
}

static void test_ledger() {
    char tmpl[] = "/tmp/test_pool_ledger_XXXXXX";
    const char* dir = mkdtemp(tmpl);
    CHECK(dir != nullptr, "temp dir");
    if (!dir) return;
    {
        pa::LedgerLock lk(dir);
        CHECK(lk.ok(), "lock taken");
        pa::PoolEntry a; a.pid = getpid(); a.items = {93.0, 120.5}; a.head_gb = 280; a.head_since = 12.5; a.base_gb = 44;
        CHECK(pa::write_ledger(dir, a), "own entry written");
        pa::PoolEntry dead; dead.pid = 999999; dead.items = {500};   // a pool that is gone (no such pid on a test host)
        if (!pa::pid_alive(dead.pid)) CHECK(pa::write_ledger(dir, dead), "stale entry written");
        const auto all = pa::read_ledger(dir);
        bool found = false;
        for (const auto& pe : all) {
            CHECK(pe.pid != 999999, "the dead pool's entry is dropped");
            if (pe.pid == getpid()) {
                found = true;
                CHECK(pe.items.size() == 2 && pe.items[1] > 120.4 && pe.head_gb == 280 && pe.head_since == 12.5 && pe.base_gb == 44, "entry read back");
            }
        }
        CHECK(found, "own entry found");
    }
    const std::string cmd = std::string("rm -rf ") + dir;
    if (std::system(cmd.c_str()) != 0) { }
}

int main() {
    std::cout << "test_pool_admission\n";
    test_model();
    test_decide();
    test_learn();
    test_5p_ids();
    test_header();
    test_simulated_job();
    test_ledger();
    std::cout << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail ? 1 : 0;
}
