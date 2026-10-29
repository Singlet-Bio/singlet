"""A TCP proxy that limits the total rate of all its connections together, for the network tiers (netem.py).

    python -m s1bench.bwproxy LISTEN_PORT UPSTREAM_PORT BYTES_PER_S

rclone's --bwlimit lets several MiB through at full speed after an idle period, and Toxiproxy's bandwidth toxic limits
each connection on its own, so neither models a link that every request shares. Here every byte sent toward the client,
on any connection, takes tokens from one bucket that fills at BYTES_PER_S and holds at most BURST bytes. Bytes toward the
server (requests) are not limited. Latency is added by Toxiproxy in front of this proxy.
"""

from __future__ import annotations

import asyncio
import sys
import time

BURST = 64 * 1024   # bucket size: the most that can pass at once after idle
CHUNK = 64 * 1024   # bytes read from the server per step


class Bucket:
    def __init__(self, rate: float):
        self.rate, self.tokens, self.t = float(rate), float(BURST), time.monotonic()
        self.lock = asyncio.Lock()

    async def take(self, n: int) -> None:
        async with self.lock:   # one waiter at a time: connections share the rate in turn
            while True:
                now = time.monotonic()
                self.tokens = min(BURST, self.tokens + (now - self.t) * self.rate)
                self.t = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                await asyncio.sleep((n - self.tokens) / self.rate)


async def pipe(reader, writer, bucket=None) -> None:
    try:
        while True:
            b = await reader.read(CHUNK)
            if not b:
                break
            if bucket is not None:
                for i in range(0, len(b), BURST):
                    part = b[i:i + BURST]
                    await bucket.take(len(part))
                    writer.write(part)
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
