"""Minimal psutil stand-in (Popen, memory_info().rss, Error) for the release-gate HTO job: the adt-bench env has no psutil."""
import subprocess


class Error(Exception):
    pass


class _Mem:
    def __init__(self, rss):
        self.rss = rss


class Popen(subprocess.Popen):
    def memory_info(self):
        try:
            for line in open(f"/proc/{self.pid}/status"):
                if line.startswith("VmRSS:"):
                    return _Mem(int(line.split()[1]) * 1024)
        except OSError as e:
            raise Error(str(e))
        return _Mem(0)
