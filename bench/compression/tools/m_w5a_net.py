"""m_w5a: public HTTPS range reader for data.singlet.bio bundles (GET only, User-Agent set, retries, consistency checks)."""
import time
import urllib.error
import urllib.request

UA = 'singlet-m_w5a-index-audit/1.0 (zach@contigtech.com)'
BASE = 'https://data.singlet.bio/data/{g}/{g}.singlet'


class RangeSource:
    """rng(start, end_inclusive) -> bytes. Verifies 206, Content-Range total == size, and (if given) ETag == etag, so a
    bundle replaced between the HEAD and the range read is detected instead of parsed."""

    def __init__(self, gse, size, etag=''):
        self.gse, self.size, self.etag = gse, size, etag
        self.n = 0
        self.bytes_read = 0

    def rng(self, start, end):
        last = None
        for attempt in range(5):
            req = urllib.request.Request(BASE.format(g=self.gse), headers={'User-Agent': UA, 'Range': f'bytes={start}-{end}'})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    if r.status != 206:
                        raise RuntimeError(f'range ignored: HTTP {r.status}')
                    cr = r.headers.get('Content-Range', '')
                    total = int(cr.rsplit('/', 1)[1]) if '/' in cr else -1
                    if total != self.size:
                        raise RuntimeError(f'size changed: Content-Range total {total} != {self.size}')
                    et = (r.headers.get('ETag') or '').strip('"')
                    if self.etag and et and et != self.etag:
                        raise RuntimeError(f'etag changed: {et} != {self.etag}')
                    data = r.read()
                self.n += 1
                self.bytes_read += len(data)
                if len(data) != end - start + 1:
                    raise RuntimeError(f'short read {len(data)} != {end - start + 1}')
                return data
            except RuntimeError:
                raise
            except urllib.error.HTTPError as e:
                if e.code in (404, 403, 410, 416):
                    raise RuntimeError(f'HTTP {e.code}')
                last = f'HTTP {e.code}'
            except Exception as e:
                last = type(e).__name__ + ':' + str(e)[:80]
            time.sleep(1.5 * (attempt + 1))
        raise RuntimeError('range failed: ' + str(last))
