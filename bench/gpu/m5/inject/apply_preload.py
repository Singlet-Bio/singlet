"""Dev-build patch (W11 M5 end to end): STAR's sequence cache preloaded with results computed on the GPU.

Applied by inject_build.sbatch to a scratch copy of prod/campaign; src/star in git is untouched. With
SINGLET_STAR_SEQCACHE_PRELOAD=<file written by `m5 --emit`>, SeqCache::create inserts every (sequence, result) record
before mapping starts. Reads whose sequence is in the cache then take their mapping state from it (seqCacheRestore)
instead of from STAR's mapping; with SINGLET_STAR_SEQCACHE_VERIFY=1, STAR maps them anyway and counts every cached
result that differs from its own (stats line in Log.out).
"""
import sys

PRELOAD = r'''    if (const char *pre = getenv("SINGLET_STAR_SEQCACHE_PRELOAD")) {//W11 M5: results computed on the GPU (bench/gpu/m5)
        FILE *f = fopen(pre, "rb");
        char hdr[8];
        if (f == nullptr || fread(hdr, 1, 8, f) != 8 || memcmp(hdr, "M5BLOBS1", 8) != 0) {
            P.inOut->logMain << "SEQCACHE: preload file " << pre << " missing or bad\n" << flush;
            exit(1);
        };
        std::vector<char> seq(DEF_readSeqLengthMax + 1);
        std::vector<uint8> blob;
        uint64 nPre = 0, nBytes = 0;
        uint16 L; uint32 bl;
        while (fread(&L, 2, 1, f) == 1) {
            if (L > DEF_readSeqLengthMax || fread(seq.data(), 1, L, f) != L || fread(&bl, 4, 1, f) != 1) break;
            blob.resize(bl);
            if (bl > 0 && fread(blob.data(), 1, bl, f) != bl) break;
            c->insert(hashSeq(seq.data(), L), seq.data(), L, blob.data(), bl);
            nPre++; nBytes += bl;
        };
        fclose(f);
        P.inOut->logMain << "SEQCACHE: preloaded " << nPre << " sequence results (" << nBytes << " bytes) from " << pre << "\n" << flush;
    };
'''

p = sys.argv[1]
lines = open(p, encoding='utf-8').read().splitlines(keepends=True)
idx = [i for i, l in enumerate(lines) if 'SEQCACHE: on, ' in l]
assert len(idx) == 1, 'anchor not found'
lines.insert(idx[0] + 1, PRELOAD)
inc = [i for i, l in enumerate(lines) if l.startswith('#include <cstdlib>')]
assert inc, 'include anchor not found'
lines.insert(inc[0] + 1, '#include <cstdio>\n#include <vector>\n')
open(p, 'w', encoding='utf-8', newline='\n').write(''.join(lines))
print('patched', p)
