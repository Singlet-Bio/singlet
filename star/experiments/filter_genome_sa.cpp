/*
 * filter_genome_sa.cpp
 * 
 * Standalone tool to create a filtered suffix array from STAR's genome SA.
 * Keeps only SA entries whose genome position falls within exonic regions
 * of top-N abundant genes.
 *
 * Input:
 *   - STAR genome directory (SA, SAindex, exonGeTrInfo.tab, geneInfo.tab)
 *   - Gene abundance ranking (TSV: rank, gene_id, ...)
 *   - Top N genes parameter
 *
 * Output:
 *   - SA_tx:       Filtered SA (packed array, same word width as original)
 *   - SAindex_tx:  SAi for filtered SA (with reduced gSAindexNbases)
 *   - txFirstInfo.tab: Metadata about the filtered SA
 *
 * Build:
 *   g++ -O2 -o filter_genome_sa filter_genome_sa.cpp
 *
 * Usage:
 *   ./filter_genome_sa --genomeDir /path/to/star_idx --abundance gene_abundance.tsv --topN 500
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// ============================================================================
// Packed array reader/writer (compatible with STAR's PackedArray)
// ============================================================================

class PackedArrayReader {
public:
    const char* data;
    uint64_t wordLength;
    uint64_t wordCompLength;
    uint64_t bitRecMask;
    uint64_t nEntries;

    PackedArrayReader() : data(nullptr), wordLength(0), nEntries(0) {}

    void init(const char* buf, uint64_t wl, uint64_t n) {
        data = buf;
        wordLength = wl;
        nEntries = n;
        wordCompLength = 64 - wl;
        bitRecMask = (wl < 64) ? ((1ULL << wl) - 1) : ~0ULL;
    }

    inline uint64_t operator[](uint64_t ii) const {
        uint64_t b = ii * wordLength;
        uint64_t B = b / 8;
        uint64_t S = b % 8;
        uint64_t a1;
        memcpy(&a1, data + B, 8);
        return ((a1 >> S) << wordCompLength) >> wordCompLength;
    }
};

class PackedArrayWriter {
public:
    char* data;
    uint64_t wordLength;
    uint64_t wordCompLength;
    uint64_t nEntries;
    uint64_t lengthByte;

    PackedArrayWriter() : data(nullptr), wordLength(0), nEntries(0), lengthByte(0) {}

    void init(uint64_t wl, uint64_t maxEntries) {
        wordLength = wl;
        wordCompLength = 64 - wl;
        nEntries = 0;
        lengthByte = (maxEntries * wl + 7) / 8 + 16; // +16 for safety
        data = new char[lengthByte]();
    }

    void write(uint64_t val) {
        uint64_t b = nEntries * wordLength;
        uint64_t B = b / 8;
        uint64_t S = b % 8;
        uint64_t mask = ((1ULL << wordLength) - 1) << S;
        uint64_t existing;
        memcpy(&existing, data + B, 8);
        existing = (existing & ~mask) | ((val << S) & mask);
        memcpy(data + B, &existing, 8);
        ++nEntries;
    }

    uint64_t actualBytes() const {
        return (nEntries * wordLength + 7) / 8;
    }

    ~PackedArrayWriter() { delete[] data; }
};

// ============================================================================
// Interval set for fast exon lookup
// ============================================================================

struct Interval {
    uint64_t start, end; // [start, end)
    bool operator<(const Interval& o) const { return start < o.start; }
};

class IntervalSet {
public:
    std::vector<Interval> intervals; // sorted, merged

    void addInterval(uint64_t s, uint64_t e) {
        intervals.push_back({s, e});
    }

    void build() {
        std::sort(intervals.begin(), intervals.end());
        // Merge overlapping
        std::vector<Interval> merged;
        for (auto& iv : intervals) {
            if (!merged.empty() && iv.start <= merged.back().end) {
                merged.back().end = std::max(merged.back().end, iv.end);
            } else {
                merged.push_back(iv);
            }
        }
        intervals = std::move(merged);
        std::cerr << "  Merged intervals: " << intervals.size()
                  << ", total bases: " << totalBases() << std::endl;
    }

    bool contains(uint64_t pos) const {
        // Binary search for the interval containing pos
        auto it = std::upper_bound(intervals.begin(), intervals.end(),
                                   Interval{pos, 0});
        if (it != intervals.begin()) {
            --it;
            return pos >= it->start && pos < it->end;
        }
        return false;
    }

    uint64_t totalBases() const {
        uint64_t total = 0;
        for (auto& iv : intervals) total += iv.end - iv.start;
        return total;
    }
};

// ============================================================================
// SAi builder (reproduced from STAR's genomeSAindex logic)
// ============================================================================

void buildSAi(const PackedArrayReader& sa, const char* genome,
              uint64_t nGenome, uint64_t nSA, uint64_t GstrandBit,
              uint64_t gSAindexNbases, const std::string& outPath) {
    
    uint64_t GstrandMask = ~(1ULL << GstrandBit);
    
    // Compute genomeSAindexStart
    std::vector<uint64_t> saIndexStart(gSAindexNbases + 1);
    saIndexStart[0] = 0;
    for (uint64_t L = 1; L <= gSAindexNbases; L++) {
        saIndexStart[L] = saIndexStart[L-1] + (1ULL << (2*L));
    }
    uint64_t nSAi = saIndexStart[gSAindexNbases];
    
    // SAi word width: GstrandBit + 3
    uint64_t SAiWordLength = GstrandBit + 3;
    uint64_t SAiMarkNbit = GstrandBit + 1;
    uint64_t SAiMarkAbsentBit = GstrandBit + 2;
    // uint64_t SAiMarkNmaskC = 1ULL << SAiMarkNbit;  // unused
    uint64_t SAiMarkAbsentMaskC = 1ULL << SAiMarkAbsentBit;
    
    // Allocate SAi
    PackedArrayWriter saiWriter;
    saiWriter.init(SAiWordLength, nSAi);
    
    // For each SA entry, compute its L-mer prefix for each L
    // and record the first SA index that matches
    
    // Initialize all SAi entries as "absent"
    for (uint64_t i = 0; i < nSAi; i++) {
        saiWriter.write(SAiMarkAbsentMaskC); // mark absent
    }
    
    // Actually, we need to overwrite entries. Reset.
    delete[] saiWriter.data;
    saiWriter.data = nullptr;
    
    // Build SAi by scanning SA entries
    // For each L, find the first SA index for each L-mer
    
    // Allocate output array
    std::vector<uint64_t> saiArray(nSAi, SAiMarkAbsentMaskC);
    
    // Helper: get base at genome position (2-bit encoded)
    auto getBase = [&](uint64_t gPos) -> int {
        // Genome is 2-bit packed: each byte has 4 bases
        // Wait, STAR genome is ASCII, not 2-bit packed (unless GENOME_PACKED is defined)
        // Actually, STAR stores genome as 1 byte per base: 0,1,2,3 = A,C,G,T, 4=N
        if (gPos >= nGenome) return 4; // N
        return (unsigned char)genome[gPos];
    };
    
    // For each SA entry, compute L-mer for all L levels
    std::cerr << "  Building SAi (" << nSAi << " entries, " << gSAindexNbases << " levels)..." << std::endl;
    
    for (uint64_t iSA = 0; iSA < nSA; iSA++) {
        uint64_t saVal = sa[iSA];
        bool isReverse = (saVal >> GstrandBit) != 0;
        uint64_t gPos = saVal & GstrandMask;
        
        // Compute L-mer prefix
        uint64_t lmer = 0;
        bool hasN = false;
        for (uint64_t L = 1; L <= gSAindexNbases; L++) {
            int base;
            if (!isReverse) {
                base = getBase(gPos + L - 1);
            } else {
                uint64_t revPos = nGenome - 1 - gPos - (L - 1);
                base = 3 - getBase(revPos); // complement
                if (base < 0 || base > 3) base = 4;
            }
            
            if (base >= 4) { hasN = true; break; }
            lmer = (lmer << 2) | base;
            
            uint64_t idx = saIndexStart[L-1] + lmer;
            if (idx < nSAi && saiArray[idx] == SAiMarkAbsentMaskC) {
                saiArray[idx] = iSA; // first SA index for this L-mer
            }
        }
    }
    
    if (nSA % 1000000 == 0 || true) {
        uint64_t nPresent = 0;
        for (uint64_t i = 0; i < nSAi; i++) {
            if (saiArray[i] != SAiMarkAbsentMaskC) nPresent++;
        }
        std::cerr << "  SAi: " << nPresent << "/" << nSAi << " entries present" << std::endl;
    }
    
    // Write output
    std::ofstream out(outPath, std::ios::binary);
    // Header: gSAindexNbases + genomeSAindexStart
    out.write(reinterpret_cast<const char*>(&gSAindexNbases), sizeof(gSAindexNbases));
    out.write(reinterpret_cast<const char*>(saIndexStart.data()),
              sizeof(uint64_t) * (gSAindexNbases + 1));
    
    // Write packed SAi entries
    PackedArrayWriter saiPacked;
    saiPacked.init(SAiWordLength, nSAi);
    for (uint64_t i = 0; i < nSAi; i++) {
        saiPacked.write(saiArray[i]);
    }
    out.write(saiPacked.data, saiPacked.actualBytes());
    out.close();
    
    std::cerr << "  Wrote SAi: " << outPath << " (" << saiPacked.actualBytes() << " bytes)" << std::endl;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string genomeDir, abundancePath;
    int topN = 500;
    
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--genomeDir" && i+1 < argc) genomeDir = argv[++i];
        else if (std::string(argv[i]) == "--abundance" && i+1 < argc) abundancePath = argv[++i];
        else if (std::string(argv[i]) == "--topN" && i+1 < argc) topN = atoi(argv[++i]);
    }
    
    if (genomeDir.empty() || abundancePath.empty()) {
        std::cerr << "Usage: filter_genome_sa --genomeDir <dir> --abundance <tsv> --topN <N>" << std::endl;
        return 1;
    }
    
    // ============================
    // 1. Read genomeParameters.txt
    // ============================
    uint64_t GstrandBit = 32;
    uint64_t gSAindexNbases = 14;
    uint64_t genomeSAsparseD = 2;
    uint64_t genomeFileSize = 0, saFileSize = 0;
    
    {
        std::ifstream f(genomeDir + "/genomeParameters.txt");
        std::string line;
        while (std::getline(f, line)) {
            if (line.find("GstrandBit") != std::string::npos) {
                sscanf(line.c_str(), "### GstrandBit %lu", &GstrandBit);
            }
            std::istringstream iss(line);
            std::string key; uint64_t val;
            if (iss >> key >> val) {
                if (key == "genomeSAindexNbases") gSAindexNbases = val;
                else if (key == "genomeSAsparseD") genomeSAsparseD = val;
                else if (key == "genomeFileSizes") {
                    genomeFileSize = val;
                    iss >> saFileSize;
                }
            }
        }
    }
    
    uint64_t saWordLength = GstrandBit + 1;
    uint64_t GstrandMask = ~(1ULL << GstrandBit);
    
    std::cerr << "GstrandBit=" << GstrandBit << " saWordLength=" << saWordLength
              << " gSAindexNbases=" << gSAindexNbases << " sparseD=" << genomeSAsparseD << std::endl;
    
    // ============================
    // 2. Read gene abundance ranking → get top N gene IDs
    // ============================
    std::unordered_set<std::string> topGeneIds;
    {
        std::ifstream f(abundancePath);
        std::string line;
        std::getline(f, line); // header
        int count = 0;
        while (std::getline(f, line) && count < topN) {
            std::istringstream iss(line);
            std::string rank, gid;
            iss >> rank >> gid;
            // Strip -A/-U suffix
            if (gid.size() > 2 && gid[gid.size()-2] == '-' && 
                (gid.back() == 'A' || gid.back() == 'U')) {
                gid = gid.substr(0, gid.size()-2);
            }
            // Skip unspliced entries (handled above by stripping -U)
            if (topGeneIds.find(gid) == topGeneIds.end()) {
                topGeneIds.insert(gid);
                count++;
            }
        }
    }
    std::cerr << "Top " << topN << " target genes loaded (" << topGeneIds.size() << " unique)" << std::endl;
    
    // ============================
    // 3. Read geneInfo.tab → map gene_id to gene_idx
    // ============================
    std::unordered_set<uint32_t> topGeneIdxs;
    {
        std::ifstream f(genomeDir + "/geneInfo.tab");
        std::string line;
        std::getline(f, line); // first line: number of genes
        uint32_t geneIdx = 0;
        while (std::getline(f, line)) {
            std::istringstream iss(line);
            std::string gid, gname, biotype;
            iss >> gid >> gname >> biotype;
            if (topGeneIds.count(gid)) {
                topGeneIdxs.insert(geneIdx);
            }
            geneIdx++;
        }
    }
    std::cerr << "Matched " << topGeneIdxs.size() << "/" << topGeneIds.size()
              << " genes in geneInfo.tab" << std::endl;
    
    // ============================
    // 4. Read exonGeTrInfo.tab → build exon intervals for top genes
    // ============================
    IntervalSet exonSet;
    {
        std::ifstream f(genomeDir + "/exonGeTrInfo.tab");
        std::string line;
        std::getline(f, line); // first line: number of exons
        while (std::getline(f, line)) {
            std::istringstream iss(line);
            uint64_t start, end;
            uint32_t strand, geneIdx, trIdx;
            iss >> start >> end >> strand >> geneIdx >> trIdx;
            if (topGeneIdxs.count(geneIdx)) {
                exonSet.addInterval(start, end + 1); // make end exclusive
            }
        }
    }
    std::cerr << "Raw exon intervals: " << exonSet.intervals.size() << std::endl;
    exonSet.build();
    
    // ============================
    // 5. Read genome (for SAi building later)
    // ============================
    std::cerr << "Reading genome (" << genomeFileSize << " bytes)..." << std::endl;
    std::vector<char> genome(genomeFileSize);
    {
        std::ifstream f(genomeDir + "/Genome", std::ios::binary);
        f.read(genome.data(), genomeFileSize);
    }
    uint64_t nGenome = genomeFileSize;
    
    // ============================
    // 6. Read SA and filter
    // ============================
    std::cerr << "Reading SA (" << saFileSize << " bytes, "
              << (saFileSize * 8 / saWordLength) << " entries)..." << std::endl;
    
    std::vector<char> saBuf(saFileSize + 16);
    {
        std::ifstream f(genomeDir + "/SA", std::ios::binary);
        f.read(saBuf.data(), saFileSize);
    }
    
    uint64_t nSA = saFileSize * 8 / saWordLength;
    PackedArrayReader saReader;
    saReader.init(saBuf.data(), saWordLength, nSA);
    
    std::cerr << "Filtering SA entries..." << std::endl;
    
    // Pre-allocate for filtered entries (estimate: ~0.1% of total)
    std::vector<uint64_t> filteredEntries;
    filteredEntries.reserve(nSA / 100);
    
    uint64_t nScanned = 0, nKept = 0;
    for (uint64_t i = 0; i < nSA; i++) {
        uint64_t entry = saReader[i];
        uint64_t gPos = entry & GstrandMask;
        
        if (exonSet.contains(gPos)) {
            filteredEntries.push_back(entry);
            nKept++;
        }
        
        nScanned++;
        if (nScanned % 500000000 == 0) {
            std::cerr << "  " << (nScanned / 1000000) << "M / " << (nSA / 1000000)
                      << "M scanned, " << nKept << " kept" << std::endl;
        }
    }
    
    std::cerr << "Filtered: " << nKept << " / " << nSA << " entries kept ("
              << (100.0 * nKept / nSA) << "%)" << std::endl;
    
    // ============================
    // 7. Write filtered SA
    // ============================
    std::string outDir = genomeDir;
    {
        PackedArrayWriter saWriter;
        saWriter.init(saWordLength, nKept);
        for (auto entry : filteredEntries) {
            saWriter.write(entry);
        }
        
        std::string saPath = outDir + "/SA_tx";
        std::ofstream f(saPath, std::ios::binary);
        f.write(saWriter.data, saWriter.actualBytes());
        f.close();
        std::cerr << "Wrote SA_tx: " << saPath << " (" << saWriter.actualBytes() << " bytes)" << std::endl;
    }
    
    // ============================
    // 8. Build SAi for filtered SA
    // ============================
    // Use reduced gSAindexNbases for smaller SAi
    uint64_t txSAindexNbases = (uint64_t)(log2((double)nKept) / 2 - 1);
    if (txSAindexNbases < 1) txSAindexNbases = 1;
    if (txSAindexNbases > 14) txSAindexNbases = 14;
    std::cerr << "Building SAi with gSAindexNbases=" << txSAindexNbases
              << " for " << nKept << " entries" << std::endl;
    
    // Create a temporary PackedArrayReader for filtered SA
    {
        // Need to pack the filtered entries first
        PackedArrayWriter tmpWriter;
        tmpWriter.init(saWordLength, nKept);
        for (auto entry : filteredEntries) {
            tmpWriter.write(entry);
        }
        PackedArrayReader filteredReader;
        filteredReader.init(tmpWriter.data, saWordLength, nKept);
        
        buildSAi(filteredReader, genome.data(), nGenome, nKept, GstrandBit,
                 txSAindexNbases, outDir + "/SAindex_tx");
    }
    
    // ============================  
    // 9. Write metadata
    // ============================
    {
        std::string infoPath = outDir + "/txFirstInfo.tab";
        std::ofstream f(infoPath);
        f << "topN\t" << topN << "\n";
        f << "nSA_tx\t" << nKept << "\n";
        f << "gSAindexNbases_tx\t" << txSAindexNbases << "\n";
        f << "exonic_bases\t" << exonSet.totalBases() << "\n";
        f << "n_intervals\t" << exonSet.intervals.size() << "\n";
        f.close();
        std::cerr << "Wrote metadata: " << infoPath << std::endl;
    }
    
    std::cerr << "Done!" << std::endl;
    return 0;
}
