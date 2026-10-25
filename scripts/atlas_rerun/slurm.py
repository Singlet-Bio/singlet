"""Slurm adapters: the real one (sbatch/squeue/sacct/mybalance via subprocess) and a recording fake for dry runs."""

import os
import re
import subprocess
import time

JOB_PREFIX = "ar-"  # every campaign job name starts with this; squeue/sacct are filtered on it


def _run(cmd, timeout=120):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def parse_mem_gb(s):
    """'96G' / '7600M' / '96Gn' / '4000Mc' -> GB (per node; per-cpu forms are returned per cpu)."""
    if not s:
        return 0.0
    m = re.match(r"^([\d.]+)([KMGT]?)", s)
    if not m:
        return 0.0
    v = float(m.group(1))
    return v * {"K": 1e-6, "M": 1e-3, "G": 1.0, "T": 1e3, "": 1e-3}[m.group(2)]


def parse_elapsed(s):
    """[D-]HH:MM:SS -> seconds."""
    if not s:
        return 0
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(float(x)) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return days * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


class RealSlurm:
    dry = False

    def __init__(self, account):
        self.account = account

    def submit(self, script, script_args, opts):
        cmd = ["sbatch", "--parsable", "--account=" + self.account] + list(opts) + [script] + list(script_args)
        rc, out, err = _run(cmd)
        if rc != 0:
            raise RuntimeError("sbatch failed (%d): %s %s" % (rc, err.strip(), " ".join(cmd)))
        return out.strip().split(";")[0]

    def queue(self):
        """{job id (array tasks expanded as A_i, pending ranges as A_[..]): (state, name, reason)} for campaign jobs."""
        rc, out, err = _run(["squeue", "-h", "-r", "-u", os.environ.get("USER", ""), "-o", "%i|%T|%j|%r"])
        if rc != 0:
            raise RuntimeError("squeue failed: " + err)
        q = {}
        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) >= 4 and parts[2].startswith(JOB_PREFIX):
                q[parts[0]] = (parts[1], parts[2], parts[3])
        return q

    def acct(self, job_ids):
        """{job id: dict(state, exit, elapsed_s, maxrss_gb, alloc_cpus, req_mem_gb, partition, name)} (allocation rows,
        with the max RSS over the job's steps)."""
        out_map = {}
        ids = sorted(set(j for j in job_ids if j))
        for i in range(0, len(ids), 200):
            chunk = ids[i:i + 200]
            rc, out, err = _run(["sacct", "-n", "-P", "-j", ",".join(chunk), "-o",
                                 "JobID,JobName,State,ExitCode,ElapsedRaw,MaxRSS,AllocCPUS,ReqMem,Partition"])
            if rc != 0:
                continue
            for line in out.splitlines():
                f = line.split("|")
                if len(f) < 9:
                    continue
                jid = f[0]
                base = jid.split(".")[0]
                rec = out_map.setdefault(base, {"maxrss_gb": 0.0})
                if "." not in jid:
                    rec.update({"name": f[1], "state": f[2].split()[0] if f[2] else "", "exit": f[3],
                                "elapsed_s": int(f[4] or 0), "alloc_cpus": int(f[6] or 0),
                                "req_mem_gb": parse_mem_gb(f[7]), "partition": f[8]})
                if f[5]:
                    rec["maxrss_gb"] = max(rec["maxrss_gb"], parse_mem_gb(f[5]))
        return out_map

    def campaign_usage(self, since):
        """Allocation rows of every campaign job since `since` (YYYY-MM-DD), for SU accounting."""
        rc, out, err = _run(["sacct", "-n", "-P", "-X", "-u", os.environ.get("USER", ""), "-S", since, "-o",
                             "JobID,JobName,State,ElapsedRaw,AllocCPUS,ReqMem,Partition"], timeout=300)
        rows = []
        for line in out.splitlines():
            f = line.split("|")
            if len(f) >= 7 and f[1].startswith(JOB_PREFIX):
                rows.append({"job": f[0], "name": f[1], "state": f[2], "elapsed_s": int(f[3] or 0),
                             "alloc_cpus": int(f[4] or 0), "req_mem_gb": parse_mem_gb(f[5]), "partition": f[6]})
        return rows

    def balance(self, account):
        """Anvil `mybalance`: remaining SU of `account`, or None if it cannot be read."""
        rc, out, err = _run(["mybalance"])
        if rc != 0:
            return None
        for line in out.splitlines():
            f = line.split()
            if f and f[0] == account and len(f) >= 4:
                try:
                    return float(f[3].replace(",", ""))
                except ValueError:
                    return None
        return None

    def cancel(self, job_ids):
        if job_ids:
            _run(["scancel"] + list(job_ids))


class FakeSlurm:
    """Records what would be submitted; nothing leaves the process. Job ids are fake (900000000+)."""
    dry = True

    def __init__(self, account="dry"):
        self.account = account
        self.n = 900000000
        self.submitted = []

    def submit(self, script, script_args, opts):
        self.n += 1
        cmd = ["sbatch", "--parsable", "--account=" + self.account] + list(opts) + [script] + list(script_args)
        self.submitted.append({"job": str(self.n), "cmd": cmd, "t": time.time()})
        return str(self.n)

    def queue(self):
        return {}

    def acct(self, job_ids):
        return {}

    def campaign_usage(self, since):
        return []

    def balance(self, account):
        return None

    def cancel(self, job_ids):
        pass


def su_of_job(row, whole_node_partitions=("wholenode", "wide", "standard"), mem_per_core_gb=1.9, node_cores=128):
    """Anvil charge of one allocation row: whole-node partitions all 128 cores; `shared` max(cores, mem/1.9 GB);
    `highmem` 4x."""
    h = row["elapsed_s"] / 3600.0
    part = row.get("partition", "")
    if part in whole_node_partitions:
        return h * node_cores * max(1, row["alloc_cpus"] // node_cores or 1)
    cores = max(row["alloc_cpus"], row["req_mem_gb"] / mem_per_core_gb)
    return h * cores * (4.0 if part == "highmem" else 1.0)
