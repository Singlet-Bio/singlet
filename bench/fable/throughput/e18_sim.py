"""E18 (docs/gpu-engine/fable/E18-throughput.md §5b): a closed-queue model of k samples in flight per GPU with SINGLET_ENGINE_GPU_SLOTS=1.
k workers each run samples back to back (pinned to GPU w mod gpus); a sample is a sequence of host stretches and device-slot holds
taken from a k = 1 run (points k1t7s1f; e18_trace.py / e18_locks.py); one slot per GPU; waiters are served in random order (flock
wakes every waiter and one wins), FIFO, "late" (the sample furthest along first) or "early". Prints the steady pairs/s (completions
between the first and the third quarter) and the mean wait per phase.
Usage: e18_sim.py [random|fifo|late|early] [gpus] [e15|e16]"""
import heapq, random, sys
# pbmc1k_v3, T = 7: 44a6e9fa (job 797443, k1t7s1f) and the merged head with E16 (job 797460, k1t7s1f); seconds
PROF_E16 = [("h", 29.8), ("prefill", 22.5), ("h", 6.7), ("handoff", 2.5), ("h", 0.05), ("solo", 0.64), ("h", 0.01), ("pileup", 6.3),
            ("h", 1.6), ("nonhost", 4.2), ("h", 53.5)]
PROF = [("h", 30.9), ("prefill", 23.1), ("h", 15.7), ("solo", 0.67), ("h", 1.93), ("handoff", 3.92), ("h", 6.9), ("pileup", 9.02),
        ("h", 3.8), ("nonhost", 4.2), ("h", 52.9)]
def run(k, n_per, prof, order="random", seed=1, gpus=1):
    rnd = random.Random(seed)
    t = 0.0; ev = []  # (time, seq, worker, step)
    seq = 0
    state = {}
    done = []; waits = {}
    busy_until = [0.0] * gpus; holder = [None] * gpus; queue = [[] for _ in range(gpus)]
    for w in range(k):
        state[w] = {"n": 0, "step": 0, "start": 0.0, "g": w % gpus}
        heapq.heappush(ev, (0.0, seq, w, "next")); seq += 1
    held_total = 0.0
    while ev:
        t, _, w, kind = heapq.heappop(ev)
        st = state[w]; g = st["g"]
        if kind == "release":
            holder[g] = None
            if queue[g]:
                i = rnd.randrange(len(queue[g])) if order == "random" else (max(range(len(queue[g])), key=lambda x: (state[queue[g][x][0]]["step"], -queue[g][x][1])) if order == "late" else (min(range(len(queue[g])), key=lambda x: (state[queue[g][x][0]]["step"], queue[g][x][1])) if order == "early" else 0))
                w2, tq = queue[g].pop(i)
                ph = prof[state[w2]["step"]]
                waits.setdefault(ph[0], []).append(t - tq)
                holder[g] = w2; held_total += ph[1]
                heapq.heappush(ev, (t + ph[1], seq, w2, "release_adv")); seq += 1
            # the releasing worker advances
            st["step"] += 1
            heapq.heappush(ev, (t, seq, w, "next")); seq += 1
            continue
        if kind == "release_adv":
            # w2 finished its device segment: release the lock (handled as a release by w2)
            heapq.heappush(ev, (t, seq, w, "release")); seq += 1
            continue
        # kind == next: run the current step
        if st["step"] >= len(prof):
            done.append((st["start"], t)); st["n"] += 1
            if st["n"] >= n_per: continue
            st["step"] = 0; st["start"] = t
        name, d = prof[st["step"]]
        if name == "h":
            st["step"] += 1
            heapq.heappush(ev, (t + d, seq, w, "next")); seq += 1
        else:
            if holder[g] is None:
                holder[g] = w; held_total += d; waits.setdefault(name, []).append(0.0)
                heapq.heappush(ev, (t + d, seq, w, "release_adv")); seq += 1
            else:
                queue[g].append((w, t))
    return done, waits
def steady(k, prof, order, gpus=1, rounds=24, seeds=10):
    rates = []; W = {}
    for s in range(seeds):
        done, waits = run(k, rounds, prof, order, s, gpus)
        ends = sorted(e for b, e in done)
        n = len(ends); i, j = n // 4, 3 * n // 4
        rates.append((j - i) * 66.6 / (ends[j] - ends[i]))
        for nm, v in waits.items(): W.setdefault(nm, []).extend(v)
    return sum(rates) / len(rates), {nm: sum(v) / len(v) for nm, v in W.items()}
if __name__ == "__main__":
    order = sys.argv[1] if len(sys.argv) > 1 else "random"
    gpus = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    if len(sys.argv) > 3 and sys.argv[3] == "e16": PROF = PROF_E16
    for k in (1, 2, 3, 4, 5, 6, 8, 10, 12):
        r, w = steady(k, PROF, order, gpus)
        print(f"k={k:2d} gpus={gpus}: {r:.3f} M pairs/s; mean waits " + " ".join(f"{n} {w.get(n, 0):.1f}" for n in ("prefill", "solo", "handoff", "pileup", "nonhost")))
