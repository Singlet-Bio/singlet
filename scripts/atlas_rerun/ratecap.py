"""Global download rate cap shared by every fetch job (S9). Standard library only.

The SRA link is shared with the rest of GVSU, so the whole fetch tier must stay under one number, however many jobs run. There is no
central daemon (jobs come and go on preemptible nodes) and no locking on NFS: every fetch worker publishes a heartbeat file
`<stage>/rate/<worker>` ({"t": time, "active": bool}) every 2 s, and takes

    my_share = cap / max(1, number of workers with a fresh heartbeat that are `active` (downloading, not decoding))

enforced locally with a token bucket shared by that worker's download threads. A dead or preempted worker drops out after STALE_S;
a worker that is decoding is not `active`, so its share goes to the others. The cap is read from `<stage>/stage.json`
({"rate_cap_MBps": 600}) every few seconds, so raising it to 900 is a config edit with no restart.

Accuracy: the sum over workers is the cap within the window it takes the others to see a new worker (HB_S + one poll, about 2-4 s);
measured against the NIC receive counter in docs/research-beta/download-and-staging.md.
"""

import json
import os
import threading
import time

HB_S = 2.0
STALE_S = 10.0
DEFAULT_CAP_MBPS = 600.0


def read_cap_mbps(stage):
    try:
        return float(json.load(open(os.path.join(stage, "stage.json")))["rate_cap_MBps"])
    except (OSError, ValueError, KeyError):
        return DEFAULT_CAP_MBPS


def write_cap_mbps(stage, mbps):
    p = os.path.join(stage, "stage.json")
    try:
        d = json.load(open(p))
    except (OSError, ValueError):
        d = {}
    d["rate_cap_MBps"] = float(mbps)
    json.dump(d, open(p + ".tmp", "w"))
    os.replace(p + ".tmp", p)


class SharedRate:
    def __init__(self, stage, worker):
        self.stage, self.worker = stage, worker
        self.dir = os.path.join(stage, "rate")
        os.makedirs(self.dir, exist_ok=True)
        self.lock = threading.Lock()
        self.active, self.share_bps, self.cap = False, DEFAULT_CAP_MBPS * 1e6, DEFAULT_CAP_MBPS
        self.next_ok = time.time()
        self.stop = threading.Event()
        self._tick()
        threading.Thread(target=self._loop, daemon=True).start()

    def _tick(self):
        now = time.time()
        p = os.path.join(self.dir, self.worker)
        try:
            with open(p + ".tmp", "w") as f:
                json.dump({"t": now, "active": self.active}, f)
            os.replace(p + ".tmp", p)
        except OSError:
            pass
        n = 0
        try:
            for name in os.listdir(self.dir):
                if name.endswith(".tmp"):
                    continue
                fp = os.path.join(self.dir, name)
                try:
                    d = json.load(open(fp))
                except (OSError, ValueError):
                    continue
                if now - d.get("t", 0) > 60:
                    try:
                        os.remove(fp)  # long dead
                    except OSError:
                        pass
                elif now - d.get("t", 0) <= STALE_S and d.get("active") and name != self.worker:
                    n += 1
        except OSError:
            pass
        n += 1 if self.active else 0
        self.cap = read_cap_mbps(self.stage)
        with self.lock:
            self.share_bps = self.cap * 1e6 / max(1, n)

    def _loop(self):
        while not self.stop.wait(HB_S):
            self._tick()

    def set_active(self, active):
        self.active = active
        self._tick()

    def acquire(self, nbytes):
        """Block so that this worker's downloads average at most its current share."""
        with self.lock:
            now = time.time()
            start = max(self.next_ok, now - 0.25)  # at most 0.25 s of burst credit
            self.next_ok = start + nbytes / self.share_bps
            wait = start - now
        if wait > 0:
            time.sleep(wait)

    def close(self):
        self.stop.set()
        try:
            os.remove(os.path.join(self.dir, self.worker))
        except OSError:
            pass
