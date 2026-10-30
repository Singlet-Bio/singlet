"""A TCP proxy that limits the total rate of all its connections together, for the network tiers (netem.py).

    python -m s1bench.bwproxy LISTEN_PORT UPSTREAM_PORT BYTES_PER_S

rclone's --bwlimit lets several MiB through at full speed after an idle period, and Toxiproxy's bandwidth toxic limits
each connection on its own, so neither models a link that every request shares. Here every byte sent toward the client,
on any connection, takes tokens from one bucket that fills at BYTES_PER_S and holds at most max(64 KiB, 5 ms of the
rate) bytes. Bytes toward the server (requests) are not limited. Latency is added by Toxiproxy in front of this proxy.

The bucket may go into debt: a sender takes its tokens and then sleeps off the deficit. The event loop's timers have
millisecond resolution, so a sleep can overshoot by about a millisecond; the overshoot refills the bucket and is spent by
the next sender, which keeps the long-run rate at BYTES_PER_S. (Waiting for the tokens before sending, with a 64 KiB
bucket, lost the overshoot on every step and held a 1 Gbit/s tier to about 0.4 Gbit/s.)
"""

from __future__ import annotations

import asyncio
import sys
import time

CHUNK = 64 * 1024   # bytes read from the server, and passed on, per step


class Bucket:
    def __init__(self, rate: float):
        self.rate = float(rate)
        self.burst = max(64 * 1024, 0.005 * self.rate)   # the most that can pass at once after idle
        self.tokens, self.t = self.burst, time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self, n: int) -> None:
        async with self.lock:   # one waiter at a time: connections share the rate in turn
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.t) * self.rate) - n
            self.t = now
            if self.tokens < 0:
                await asyncio.sleep(-self.tokens / self.rate)


async def pipe(reader, writer, bucket=None) -> None:
    try:
        while True:
            b = await reader.read(CHUNK)
            if not b:
                break
            if bucket is not None:
                await bucket.take(len(b))
                writer.write(b)
                await writer.drain()
            else:
                writer.write(b)
                await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def main(listen: int, upstream: int, rate: float) -> None:
    bucket = Bucket(rate)

    async def handle(cr, cw):
        try:
            sr, sw = await asyncio.open_connection("127.0.0.1", upstream)
        except OSError:
            cw.close()
            return
        await asyncio.gather(pipe(cr, sw), pipe(sr, cw, bucket))

    srv = await asyncio.start_server(handle, "127.0.0.1", listen)
    async with srv:
        await srv.serve_forever()


if __name__ == "__main__":
    asyncio.run(main(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])))
