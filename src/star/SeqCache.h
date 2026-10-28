#ifndef H_SeqCache
#define H_SeqCache

// Sequence cache: maps a read's (clipped, numeric) sequence to the result of aligning it, so a
// read whose sequence was already aligned skips seed search and stitching. 10x cDNA reads repeat
// a lot (PCR duplicates, highly expressed 3' ends): ~53% of PBMC 1k v3 reads repeat an earlier read.
//
// Exactness: with the settings checked in SeqCache::create (single-end alignment, no chimeric
// detection, WASP, variation, genome transformation, random multimapper order or outFilterBySJout),
// STAR's alignment of a read depends only on its numeric sequence and length: qualities, read
// name, barcode and read number are not used until output. The cache stores the full mapping
// state that output reads (nW, nTr, the selected transcripts, which one is trBest), and output is
// still produced per read with that read's own name, barcode, UMI and qualities.
// SINGLET_STAR_SEQCACHE_VERIFY=1 maps every hit anyway and counts results that differ.
//
// Structure: shards, each a ring buffer of entries (FIFO eviction, second chance: a hit on an entry
// older than half the ring copies it to the head) plus a 2-way index of (hash, position); 8-way (one
// 128-byte line) under singlet --engine gpu, where the GPU fills the cache with every distinct sequence
// up front and a 2-way bucket overflow would evict ~2.6% of them (re-mapped by STAR at full cost).
// Entries hold the 2-bit packed sequence and a varint encoding of the mapping result (~110 bytes
// with the index). SINGLET_STAR_SEQCACHE_MB sets the size (default 4096, 0 = off); ring pages are
// only touched as entries are written, so small samples use less.

#include "IncludeDefine.h"
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

class Parameters;

class SeqCache {
public:
    static SeqCache *create(Parameters &P, uint64 nGenome); //nullptr if disabled or not exact for these parameters
    ~SeqCache();

    static uint64 hashSeq(const char *seq, uint64 L); //seq: numeric bases 0..3

    //on a hit, copies the stored result into out and returns true
    bool lookup(uint64 h, const char *seq, uint64 L, std::vector<uint8> &out);
    void insert(uint64 h, const char *seq, uint64 L, const uint8 *blob, uint64 blobLen);

    std::string stats() const;

    bool verify;
    bool prefilled = false; //the GPU engine filled the cache with every distinct sequence (SeqCacheGpuPrefill.cpp)
    std::atomic<uint64> nLookup{0}, nHit{0}, nInsert{0}, nUncacheable{0}, nVerified{0}, nVerifyFail{0}, nRefresh{0};

private:
    struct Slot {uint64 hash, pos;}; //pos = absolute ring position + 1, 0 = empty
    struct Shard {
        std::mutex mtx;
        char *ring;
        uint64 head; //absolute write position
        Slot *index;
    };
    uint64 nShards, ringBytes, indexMask, ways;
    Shard *shards;

    SeqCache(uint64 totalBytes, uint64 ways);
    bool entryMatches(Shard &s, uint64 pos, uint64 h, const char *seq, uint64 L, uint64 &blobOff, uint64 &blobLen, uint64 &entLen);
    uint64 append(Shard &s, const char *ent, uint64 entLen); //returns absolute position
};

extern SeqCache *g_seqCache; //set in STAR.cpp for the main mapping pass only
//singlet GPU pileup ("on"): per read (iReadAll) its hand-off record, or 0xFFFFFFFF; reads with a record get no BAM records from STAR
//(the pileup takes their alignments from the hand-off). Set by SeqCacheGpuPrefill.cpp; nullptr otherwise.
extern const uint32 *g_handoffReadRecord;
extern uint64 g_handoffNReads;

#endif
