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
    //A read with N bases (singlet T2 cpucache, SINGLET_STAR_SEQCACHE_NKEY=0 turns it off): seq holds code 4 for N, the 2-bit packed
    //bases have 0 there, and the positions of the N are part of the key: nmask (bit i of byte i/8 set when base i is N, (L+7)/8 bytes) is
    //stored after the packed bases and compared on lookup; the entry's length field carries a flag bit. A read without N has nmask==nullptr
    //and the entry layout it always had.
    bool lookup(uint64 h, const char *seq, uint64 L, std::vector<uint8> &out, const uint8 *nmask = nullptr);
    void insert(uint64 h, const char *seq, uint64 L, const uint8 *blob, uint64 blobLen, const uint8 *nmask = nullptr);
    static bool nKeyOn(); //reads with N are cached (default)

    std::string stats() const;

    bool verify;
    bool prefilled = false; //the GPU engine filled the cache with every distinct sequence, or with hand-off summaries every unmapped one (SeqCacheGpuPrefill.cpp)
    std::atomic<uint64> nLookup{0}, nHit{0}, nInsert{0}, nUncacheable{0}, nVerified{0}, nVerifyFail{0}, nRefresh{0};

    //singlet T2 cpucache, SINGLET_STAR_SEQCACHE_SPLIT=1 (counting only, no effect on any result): the reads that are not looked up or whose
    //result is not stored, by reason x clip class (0 none, 1 5' clipped, 2 3' clipped, 3 both) x unmapType+1, and how many of them repeat
    //an earlier uncacheable read: repA by the clipped bases alone, repB by the bases plus both clip lengths and the original length.
    //Reasons: 0 zero length, 1 longer than DEF_readSeqLengthMax, 2 non-ACGT among the bases STAR maps (not cached only with SINGLET_STAR_SEQCACHE_NKEY=0), 3 (informational, a read that IS
    //looked up) non-ACGT only inside the clipped regions, 4 chimRecord, 5 mate overlap, 6 nTr>255, 7 nW==0 with trBest!=trInit,
    //8 trBest not in trMult, 9 a transcript with cigar/variation/gene fields set, 10 nExons>MAX_N_EXONS
    bool splitOn = false;
    enum {UNC_N = 11};
    std::atomic<uint64> uncCnt[UNC_N][4][5];
    std::atomic<uint64> uncRepA[UNC_N], uncRepB[UNC_N];
    void uncNote(int reason, int clipCls, int unmapType, uint64 hA, uint64 hB);
    static uint64 hashRead(const char *seq, uint64 L); //any byte codes (N included)

private:
    struct Slot {uint64 hash, pos;}; //pos = absolute ring position + 1, 0 = empty
    struct Shard {
        std::mutex mtx;
        char *ring;
        uint64 head; //absolute write position
        Slot *index;
    };
    uint64 nShards, ringBytes, indexMask, ways;
    std::atomic<uint64> *shadowA = nullptr, *shadowB = nullptr; //SINGLET_STAR_SEQCACHE_SPLIT: hash sets of the uncacheable reads seen
    uint64 shadowMask = 0;
    Shard *shards;

    SeqCache(uint64 totalBytes, uint64 ways);
    bool entryMatches(Shard &s, uint64 pos, uint64 h, const char *seq, uint64 L, uint64 &blobOff, uint64 &blobLen, uint64 &entLen,
                      const uint8 *nmask);
    uint64 append(Shard &s, const char *ent, uint64 entLen); //returns absolute position
};

extern SeqCache *g_seqCache; //set in STAR.cpp for the main mapping pass only
//singlet GPU pileup ("on"): per read (iReadAll-1) its hand-off record, or 0xFFFFFFFF; reads with a record get no BAM records from STAR
//(the pileup takes their alignments from the hand-off). Set by SeqCacheGpuPrefill.cpp; nullptr otherwise.
extern const uint32 *g_handoffReadRecord;
extern uint64 g_handoffNReads;
//per hand-off read the bases the GPU clipped from its 5' and 3' ends (engine_gpu Output clip5 / clip3); readLoad takes them instead of
//running STAR's clip for a read with a record. nullptr when clipping is off. g_handoffClipCheck (SINGLET_HANDOFF_CLIPCHECK=1): run the
//CPU clip anyway and count the reads where they differ (g_handoffClipBad)
extern const uint16 *g_handoffClip5, *g_handoffClip3;
//singlet --engine gpu with --clipAdapterType CellRanger4: per read (iReadAll-1) what ClipMate::clipChunk would have left for readLoad (the
//CellRanger4 5' clip length), computed once by the prefill; clipChunk is then skipped. nullptr otherwise.
extern const uint8 *g_cr4Info;
extern uint64 g_cr4InfoN;
extern bool g_handoffClipCheck;
extern std::atomic<uint64> g_handoffClipBad;
//per hand-off record its number of alignments (0: unmapped); STAR skips BAM output only for reads whose record has alignments
extern const uint8 *g_handoffRecAln;
//and per read what the pileup took from its BAM record: CR and UR (3 bits per base, A0 C1 G2 T3 other 4, base 0 lowest) and the
//base-quality counters of the cDNA read (bases with quality < minQ; bases where "quality < minQ" differs from "base is N"). The
//entries live in a shared file mapping (<hand-off>.reads) that the pileup reads after STAR exits; nullptr when off.
//flags: bit 0 written; for a clipped read (the record is its clipped sequence): bits 8-15 bases clipped at 5', 16-23 at 3', 24-31 A
//bases in the clipped 3' tail, bit 1 the tail starts with A, bit 5 a clip over 255 bases; bits 2-4 from g_handoffReadHook
struct HandoffReadEntry { uint64 cr3, ur3; uint16 lowQual, qualNeqN; uint32 flags; };
static_assert(sizeof(HandoffReadEntry) == 24, "HandoffReadEntry layout is shared with the pileup");
extern HandoffReadEntry *g_handoffReads;
//per hand-off read the low-quality mask of its record (g_handoffQWords words; bit i: base i of the record, 5' clip removed, quality < g_handoffMinQ), in a
//shared file mapping (<hand-off>.reads.q) next to g_handoffReads; nullptr when off. Entry flag bit 6 (64): the mask is unavailable (record over
//64 * g_handoffQWords bases). The width is singlet_qmask::words() (2 up to 128 bases, the layout before P3-ENG-14)
extern uint64 *g_handoffQMask;
extern uint32 g_handoffQWords;
extern int g_handoffMinQ;
//set by singlet: called for each clipped hand-off read (read index, the read as in the FASTQ, its length); returns entry flag bits 2-4
//(bit 2 the read is in the pileup's rRNA sample, bits 3 / 4 the rRNA detector's verdict on the read / its reverse complement)
extern uint32 (*g_handoffReadHook)(uint64 iRead, const char *seq, uint64 len);
//per hand-off record its summary (ReadAlign::seqCacheHandoffSummary) at g_handoffSum + g_handoffSumOff[rec], empty for an unmapped
//sequence (its reads take the normal path); nullptr when off
extern const uint8 *g_handoffSum;
//alignments built by ReadAlign::seqCacheLeanAln and how many differed from alignBAM's (SINGLET_HANDOFF_LEAN=verify)
extern std::atomic<unsigned long long> g_handoffLeanAln, g_handoffLeanBad;
extern const uint64 *g_handoffSumOff;

#endif
