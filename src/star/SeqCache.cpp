#include "StarProfile.h"
#include "SeqCache.h"
#include "Parameters.h"
#include "Genome.h"
#include "InOutStreams.h"

#include <cstdlib>
#include <sys/stat.h>
#include <algorithm>
#include <cstring>
#include <sstream>

SeqCache *g_seqCache = nullptr;
const uint32 *g_handoffReadRecord = nullptr;
uint64 g_handoffNReads = 0;
const uint16 *g_handoffClip5 = nullptr, *g_handoffClip3 = nullptr;
const uint8 *g_cr4Info = nullptr;
uint64 g_cr4InfoN = 0;
bool g_handoffClipCheck = false;
std::atomic<uint64> g_handoffClipBad(0);
const uint8 *g_handoffRecAln = nullptr;
HandoffReadEntry *g_handoffReads = nullptr;
uint64 *g_handoffQMask = nullptr;
int g_handoffMinQ = 0;
uint32 (*g_handoffReadHook)(uint64, const char *, uint64) = nullptr;
const uint8 *g_handoffSum = nullptr;
const uint64 *g_handoffSumOff = nullptr;
#ifdef STAR_PROFILE
std::atomic<unsigned long long> g_prof[PROF_N];
std::atomic<unsigned long long> g_profOutcome[6][2];
std::atomic<unsigned long long> g_profClass[7][2];
std::atomic<unsigned long long> g_profClassDetail[7][5];
std::atomic<unsigned long long> g_profOutcomeDetail[6][12];
std::atomic<unsigned long long> g_profPath[3];
#endif

namespace {
inline uint64 mix64(uint64 x) {//splitmix64 finalizer
    x ^= x >> 30; x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27; x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}
const uint64 entHeader = 20; //uint32 entLen, uint32 L, uint64 hash, uint32 blobLen
inline uint64 packedBytes(uint64 L) {return (L+3)/4;}
inline void packSeq(const char *seq, uint64 L, uint8 *out) {
    memset(out, 0, packedBytes(L));
    for (uint64 i=0; i<L; i++)
        out[i>>2] |= (uint8) (seq[i] << ((i&3)*2));
}
}

uint64 SeqCache::hashSeq(const char *seq, uint64 L) {
    uint64 h = mix64(L + 0x9e3779b97f4a7c15ULL), w = 0;
    uint64 i = 0;
    for (; i<L; i++) {
        w = (w<<2) | (uint64) seq[i];
        if ((i&31)==31) {h = mix64(h ^ w); w = 0;}
    }
    if (i&31) h = mix64(h ^ w ^ (i&31));
    return h;
}

SeqCache *SeqCache::create(Parameters &P, uint64 nGenome) {
    const char *e = getenv("SINGLET_STAR_SEQCACHE_MB");
    uint64 mb = e ? strtoull(e, nullptr, 10) : 4096;
    const char *eg = getenv("SINGLET_ENGINE_GPU");
    if (e == nullptr && eg != nullptr && eg[0] == '1' && !P.readFilesNames.empty()) {//--engine gpu: room for every distinct sequence
        uint64 bytes = 0;                                                         //of the sample (~0.6 x the cDNA FASTQ size)
        for (const std::string &fn : P.readFilesNames[0]) {
            struct stat st;
            if (stat(fn.c_str(), &st) == 0) bytes += (uint64) st.st_size;
        };
        mb = std::max<uint64>(mb, std::min<uint64>(98304, (bytes / 10 * 6) >> 20));
    };
    if (mb == 0)
        return nullptr;
    //the cached result must depend on the read sequence only
    std::string why;
    if (P.readNmates != 1) why = "paired-end alignment";
    else if (P.pGe.gType == 101) why = "splice-graph genome";
    else if (P.pGe.transform.type != 0) why = "genome transformation";
    else if (P.pCh.segmentMin > 0) why = "chimeric detection";
    else if (P.wasp.yes) why = "WASP";
    else if (P.var.yes) why = "variation";
    else if (P.outMultimapperOrder.random) why = "random multimapper order";
    else if (P.outFilterBySJoutStage != 0) why = "outFilterType BySJout";
    if (!why.empty()) {
        P.inOut->logMain << "SEQCACHE: off (" << why << ")\n";
        return nullptr;
    }
    SeqCache *c = new SeqCache(mb << 20, (eg != nullptr && eg[0] == '1') ? 8 : 2);
    const char *v = getenv("SINGLET_STAR_SEQCACHE_VERIFY");
    c->verify = v && v[0] == '1';
    P.inOut->logMain << "SEQCACHE: on, " << mb << " MB in " << c->nShards << " shards" << (c->verify ? ", verify" : "") << "\n";
    return c;
}

SeqCache::SeqCache(uint64 totalBytes, uint64 waysIn) : verify(false), ways(waysIn) {
    nShards = 64;
    uint64 perShard = totalBytes / nShards;
    //index: one 16-byte slot per ~112 bytes of ring (entries are ~100-150 bytes)
    uint64 slots = ways;
    while (slots * 128 < perShard) slots <<= 1;
    indexMask = slots - 1;
    ringBytes = ((perShard - slots*sizeof(Slot)) / 8) * 8;
    shards = new Shard[nShards];
    for (uint64 i=0; i<nShards; i++) {
        shards[i].ring = new char[ringBytes];
        shards[i].head = 0;
        shards[i].index = new Slot[slots];
        memset(shards[i].index, 0, slots*sizeof(Slot));
    }
}

SeqCache::~SeqCache() {
    for (uint64 i=0; i<nShards; i++) {
        delete[] shards[i].ring;
        delete[] shards[i].index;
    }
    delete[] shards;
}

bool SeqCache::entryMatches(Shard &s, uint64 pos, uint64 h, const char *seq, uint64 L, uint64 &blobOff, uint64 &blobLen, uint64 &entLen) {
    if (pos == 0) return false;
    uint64 p = pos - 1;
    if (p + ringBytes < s.head) return false; //overwritten
    const char *ent = s.ring + p % ringBytes;
    uint32 eLen, eL, bLen; uint64 eh;
    memcpy(&eLen, ent, 4); memcpy(&eL, ent+4, 4); memcpy(&eh, ent+8, 8); memcpy(&bLen, ent+16, 4);
    if (eh != h || eL != L) return false;
    uint8 packed[DEF_readSeqLengthMax/4+1];
    packSeq(seq, L, packed);
    if (memcmp(packed, ent+entHeader, packedBytes(L)) != 0) return false;
    blobOff = entHeader + packedBytes(L);
    blobLen = bLen;
    entLen = eLen;
    return true;
}

uint64 SeqCache::append(Shard &s, const char *ent, uint64 entLen) {
    uint64 off = s.head % ringBytes;
    if (off + entLen > ringBytes) {//no room at the end of the ring: wrap
        s.head += ringBytes - off;
        off = 0;
    }
    memcpy(s.ring + off, ent, entLen);
    uint64 p = s.head;
    s.head += entLen;
    return p;
}

bool SeqCache::lookup(uint64 h, const char *seq, uint64 L, std::vector<uint8> &out) {
    nLookup++;
    Shard &s = shards[(h >> 40) % nShards];
    uint64 b = h & indexMask & ~(ways-1);
    std::lock_guard<std::mutex> lock(s.mtx);
    for (uint64 j=0; j<ways; j++) {
        Slot &sl = s.index[b+j];
        uint64 blobOff, blobLen, entLen;
        if (sl.hash != h || !entryMatches(s, sl.pos, h, seq, L, blobOff, blobLen, entLen))
            continue;
        const char *ent = s.ring + (sl.pos-1) % ringBytes;
        out.assign((const uint8*) ent + blobOff, (const uint8*) ent + blobOff + blobLen);
        if (s.head - (sl.pos-1) > ringBytes/2) {//second chance: move a hot entry to the head
            std::vector<char> copy(ent, ent + entLen);
            sl.pos = append(s, copy.data(), entLen) + 1;
            nRefresh++;
        }
        nHit++;
        return true;
    }
    return false;
}

void SeqCache::insert(uint64 h, const char *seq, uint64 L, const uint8 *blob, uint64 blobLen) {
    uint64 entLen = ((entHeader + packedBytes(L) + blobLen + 7) / 8) * 8;
    if (entLen > ringBytes / 64)
        return;
    std::vector<char> ent(entLen, 0);
    uint32 eLen = (uint32) entLen, eL = (uint32) L, bLen = (uint32) blobLen;
    memcpy(ent.data(), &eLen, 4); memcpy(ent.data()+4, &eL, 4); memcpy(ent.data()+8, &h, 8); memcpy(ent.data()+16, &bLen, 4);
    packSeq(seq, L, (uint8*) ent.data()+entHeader);
    memcpy(ent.data() + entHeader + packedBytes(L), blob, blobLen);

    Shard &s = shards[(h >> 40) % nShards];
    uint64 b = h & indexMask & ~(ways-1);
    std::lock_guard<std::mutex> lock(s.mtx);
    //slot: same hash, else empty or overwritten, else the oldest one (ties: the later slot)
    Slot *target = nullptr;
    for (uint64 j=0; j<ways && !target; j++)
        if (s.index[b+j].hash == h) target = &s.index[b+j];
    for (uint64 j=0; j<ways && !target; j++) {
        Slot &sl = s.index[b+j];
        if (sl.pos == 0 || sl.pos - 1 + ringBytes < s.head) target = &sl;
    }
    if (!target) {
        target = &s.index[b+ways-1];
        for (uint64 j=ways-1; j-- > 0; )
            if (s.index[b+j].pos < target->pos) target = &s.index[b+j];
    }
    target->hash = h;
    target->pos = append(s, ent.data(), entLen) + 1;
    nInsert++;
}

std::string SeqCache::stats() const {
    std::ostringstream o;
    uint64 l = nLookup, hit = nHit;
    o << "SEQCACHE: lookups=" << l << " hits=" << hit << " hit_rate=" << (l ? double(hit)/double(l) : 0.0)
      << " inserts=" << nInsert << " refreshes=" << nRefresh << " uncacheable=" << nUncacheable;
    if (verify)
        o << " verified=" << nVerified << " verify_mismatches=" << nVerifyFail;
    return o.str();
}
