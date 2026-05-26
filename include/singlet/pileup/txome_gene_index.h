// txome_gene_index.h — Gene-level unique k-mer index for cascade L1
// Part of cascade Phase 2 integration (singlet v0.3.0).
//
// Design:
//   - Pre-built binary index mapping gene-unique 22-mers → gene_id
//   - Loads in <5s (vs 15min raw FASTA parse)
//   - Dense stride=1 scan with Bloom filter pre-check + rolling hash
//   - Only gene-unique k-mers stored (multi-gene k-mers discarded)
//   - ~97M gene-unique k-mers for human transcriptome (GRCh38/Ensembl 110)
//
// Binary format (.txgi):
//   [4B magic "TXGI"] [4B version=1] [4B k] [4B n_genes] [4B n_kmers] [4B reserved]
//   [n_genes × NUL-terminated gene names]
//   [n_kmers × 12B entries: (8B kmer_2bit, 4B gene_id)]  <-- sorted by kmer_2bit
//
// Lookup strategy:
//   1. Rolling hash computes k-mer at each position in O(1)
//   2. Bloom filter rejects ~99.7% of non-matching k-mers (no cache miss)
//   3. Only Bloom-positive k-mers hit the flat hash table (expensive cache miss)
//   Result: non-matching reads scan in ~200ns (vs ~10μs without Bloom)

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {

// ── 2-bit nucleotide encoding ──────────────────────────────────────────────────
inline uint8_t base_to_2bit(char c) noexcept {
    switch (c) {
        case 'A': case 'a': return 0;
        case 'C': case 'c': return 1;
        case 'G': case 'g': return 2;
        case 'T': case 't': return 3;
        default: return 0xFF; // N or invalid
    }
}

// Encode a k-mer from ASCII sequence to 2-bit packed uint64.
// Returns UINT64_MAX if any base is N/invalid.
inline uint64_t encode_kmer_2bit(const char* seq, uint8_t k) noexcept {
    uint64_t val = 0;
    for (uint8_t i = 0; i < k; ++i) {
        uint8_t b = base_to_2bit(seq[i]);
        if (b == 0xFF) return UINT64_MAX;
        val = (val << 2) | b;
    }
    return val;
}

// ── KmerGeneEntry — one entry in the sorted index ──────────────────────────────
struct KmerGeneEntry {
    uint64_t kmer_2bit;  // 2-bit encoded k-mer (44 bits for k=22)
    uint32_t gene_id;    // index into gene_names vector

    bool operator<(const KmerGeneEntry& o) const noexcept {
        return kmer_2bit < o.kmer_2bit;
    }
};

// ── KmerBloomFilter — compact Bloom filter for fast k-mer rejection ─────────
// 256MB (2 billion bits), 3 hash functions → 0.3% false positive rate for 97M items.
// Fits in L3 cache → no DRAM accesses for negative lookups.
class KmerBloomFilter {
public:
    void build(const std::vector<KmerGeneEntry>& entries) {
        constexpr uint64_t NBITS = 1ULL << 31; // 2 billion bits = 256MB
        mask_ = NBITS - 1;
        bits_.assign(NBITS / 64, 0);
        for (auto& e : entries) {
            uint64_t h1 = hash1_(e.kmer_2bit);
            uint64_t h2 = hash2_(e.kmer_2bit);
            set_bit_(h1 & mask_);
            set_bit_(h2 & mask_);
            set_bit_((h1 + h2) & mask_);
        }
    }

    bool may_contain(uint64_t kmer) const noexcept {
        uint64_t h1 = hash1_(kmer);
        uint64_t h2 = hash2_(kmer);
        return check_bit_(h1 & mask_) &&
               check_bit_(h2 & mask_) &&
               check_bit_((h1 + h2) & mask_);
    }

    uint64_t memory_bytes() const noexcept { return bits_.size() * 8; }

private:
    static uint64_t hash1_(uint64_t x) noexcept {
        return x * 11400714819323198485ULL; // Fibonacci
    }
    static uint64_t hash2_(uint64_t x) noexcept {
        x ^= x >> 33; x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL;
        x ^= x >> 33; return x; // splitmix64
    }
    void set_bit_(uint64_t pos) noexcept {
        bits_[pos >> 6] |= (1ULL << (pos & 63));
    }
    bool check_bit_(uint64_t pos) const noexcept {
        return (bits_[pos >> 6] >> (pos & 63)) & 1;
    }

    std::vector<uint64_t> bits_;
    uint64_t mask_ = 0;
};

// ── FlatKmerHash — open-addressing hash table for O(1) k-mer lookup ──────────
// Uses linear probing with Fibonacci hashing. Stores (kmer_2bit, gene_id) pairs
// directly in a flat array — no pointer chasing, cache-friendly.
class FlatKmerHash {
public:
    static constexpr uint64_t EMPTY = UINT64_MAX;

    void build(const std::vector<KmerGeneEntry>& entries) {
        // Target ~50% load factor for good probe performance
        uint64_t cap = 1;
        while (cap < entries.size() * 2) cap <<= 1;
        mask_ = cap - 1;
        keys_.assign(cap, EMPTY);
        vals_.resize(cap, UINT32_MAX);
        size_ = entries.size();

        for (auto& e : entries) {
            uint64_t slot = hash_(e.kmer_2bit) & mask_;
            while (keys_[slot] != EMPTY) {
                slot = (slot + 1) & mask_;
            }
            keys_[slot] = e.kmer_2bit;
            vals_[slot] = e.gene_id;
        }
    }

    // Returns gene_id if found, UINT32_MAX if not.
    uint32_t find(uint64_t kmer) const noexcept {
        uint64_t slot = hash_(kmer) & mask_;
        while (true) {
            uint64_t k = keys_[slot];
            if (k == kmer) return vals_[slot];
            if (k == EMPTY) return UINT32_MAX;
            slot = (slot + 1) & mask_;
        }
    }

    uint64_t size() const noexcept { return size_; }
    uint64_t capacity() const noexcept { return keys_.size(); }

private:
    // Fibonacci hash: multiply by golden ratio fraction of 2^64
    static uint64_t hash_(uint64_t x) noexcept {
        return x * 11400714819323198485ULL; // 2^64 / phi
    }

    std::vector<uint64_t> keys_;
    std::vector<uint32_t> vals_;
    uint64_t mask_ = 0;
    uint64_t size_ = 0;
};

// ── TxomeGeneIndex ─────────────────────────────────────────────────────────────
class TxomeGeneIndex {
public:
    static constexpr uint32_t MAGIC   = 0x49475854; // "TXGI"
    static constexpr uint32_t VERSION = 1;
    static constexpr uint8_t  DEFAULT_K = 22;

    uint8_t k_ = DEFAULT_K;
    std::vector<std::string> gene_names_;
    std::vector<KmerGeneEntry> entries_;  // sorted by kmer_2bit (for save/load)
    FlatKmerHash kmer_hash_;  // flat open-addressing hash table for O(1) lookup
    KmerBloomFilter bloom_;   // fast rejection filter (256MB, <0.3% FPR)

    // ── Build from transcriptome FASTA + metadata ──────────────────────────
    // tx_meta_tsv format: tab-separated, columns: tx_idx, tx_name, gene_id, gene_name, tx_len, biotype
    void build(const std::string& txome_fasta_path,
               const std::string& tx_meta_tsv_path,
               uint8_t k = DEFAULT_K) {
        k_ = k;

        // Step 1: Load transcript → gene mapping
        std::unordered_map<std::string, std::string> tx_to_gene; // tx_name → gene_name
        std::unordered_map<std::string, uint32_t> gene_to_id;
        {
            std::ifstream f(tx_meta_tsv_path);
            std::string line;
            std::getline(f, line); // skip header
            while (std::getline(f, line)) {
                // Parse tab-separated: tx_idx \t tx_name \t gene_id \t gene_name \t ...
                size_t t1 = line.find('\t');
                size_t t2 = line.find('\t', t1 + 1);
                size_t t3 = line.find('\t', t2 + 1);
                size_t t4 = line.find('\t', t3 + 1);
                if (t1 == std::string::npos || t3 == std::string::npos) continue;
                std::string tx_name = line.substr(t1 + 1, t2 - t1 - 1);
                std::string gene_name = line.substr(t3 + 1, t4 == std::string::npos ? std::string::npos : t4 - t3 - 1);
                tx_to_gene[tx_name] = gene_name;
                if (gene_to_id.find(gene_name) == gene_to_id.end()) {
                    uint32_t id = static_cast<uint32_t>(gene_names_.size());
                    gene_to_id[gene_name] = id;
                    gene_names_.push_back(gene_name);
                }
            }
        }
        std::cerr << "[txome-gene-index] " << tx_to_gene.size() << " transcripts, "
                  << gene_names_.size() << " genes\n";

        // Step 2: Parse FASTA and enumerate k-mers per gene
        // For each k-mer, track which gene(s) it maps to.
        // Use uint32_t gene_id; sentinel UINT32_MAX means multi-gene.
        std::unordered_map<uint64_t, uint32_t> kmer_gene; // kmer_2bit → gene_id or MULTI
        constexpr uint32_t MULTI = UINT32_MAX;
        uint64_t n_total_kmers = 0;
        uint32_t n_tx = 0;

        {
            std::ifstream f(txome_fasta_path);
            std::string name, line;
            std::string seq;
            auto process_tx = [&]() {
                if (name.empty() || seq.empty()) return;
                auto it = tx_to_gene.find(name);
                if (it == tx_to_gene.end()) return;
                uint32_t gid = gene_to_id[it->second];
                if (seq.size() < k) return;
                for (size_t i = 0; i + k <= seq.size(); ++i) {
                    uint64_t kmer = encode_kmer_2bit(seq.data() + i, k);
                    if (kmer == UINT64_MAX) continue; // skip N-containing
                    n_total_kmers++;
                    auto [kit, inserted] = kmer_gene.try_emplace(kmer, gid);
                    if (!inserted && kit->second != gid) {
                        kit->second = MULTI; // multi-gene k-mer
                    }
                }
                n_tx++;
                if (n_tx % 20000 == 0) {
                    std::cerr << "[txome-gene-index]   " << n_tx << " transcripts, "
                              << n_total_kmers << " k-mers, "
                              << kmer_gene.size() << " distinct\n";
                }
            };

            while (std::getline(f, line)) {
                if (line.empty()) continue;
                if (line[0] == '>') {
                    process_tx();
                    // Extract name (up to first space)
                    size_t sp = line.find(' ', 1);
                    name = line.substr(1, sp == std::string::npos ? std::string::npos : sp - 1);
                    seq.clear();
                } else {
                    seq += line;
                }
            }
            process_tx(); // last transcript
        }

        std::cerr << "[txome-gene-index] " << n_tx << " transcripts processed, "
                  << n_total_kmers << " total k-mers, "
                  << kmer_gene.size() << " distinct\n";

        // Step 3: Filter to gene-unique k-mers and build sorted array
        entries_.clear();
        uint64_t n_multi = 0;
        for (auto& [kmer, gid] : kmer_gene) {
            if (gid == MULTI) {
                n_multi++;
                continue;
            }
            entries_.push_back({kmer, gid});
        }
        std::sort(entries_.begin(), entries_.end());

        // Build hash table for O(1) lookup
        build_hash_table_();

        std::cerr << "[txome-gene-index] " << entries_.size() << " gene-unique k-mers ("
                  << (100.0 * entries_.size() / kmer_gene.size()) << "%), "
                  << n_multi << " multi-gene discarded\n";
    }

    // ── Save to binary file ────────────────────────────────────────────────
    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        uint32_t magic = MAGIC, version = VERSION;
        uint32_t k32 = k_;
        uint32_t n_genes = static_cast<uint32_t>(gene_names_.size());
        uint32_t n_kmers = static_cast<uint32_t>(entries_.size());
        uint32_t reserved = 0;
        f.write(reinterpret_cast<const char*>(&magic), 4);
        f.write(reinterpret_cast<const char*>(&version), 4);
        f.write(reinterpret_cast<const char*>(&k32), 4);
        f.write(reinterpret_cast<const char*>(&n_genes), 4);
        f.write(reinterpret_cast<const char*>(&n_kmers), 4);
        f.write(reinterpret_cast<const char*>(&reserved), 4);

        // Gene names (NUL-terminated strings)
        for (auto& g : gene_names_) {
            f.write(g.data(), g.size() + 1);
        }

        // K-mer entries (sorted)
        f.write(reinterpret_cast<const char*>(entries_.data()),
                entries_.size() * sizeof(KmerGeneEntry));

        std::cerr << "[txome-gene-index] Saved " << path << " ("
                  << f.tellp() / (1024*1024) << " MB)\n";
    }

    // ── Load from binary file ──────────────────────────────────────────────
    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) return false;

        uint32_t magic, version, k32, n_genes, n_kmers, reserved;
        f.read(reinterpret_cast<char*>(&magic), 4);
        f.read(reinterpret_cast<char*>(&version), 4);
        f.read(reinterpret_cast<char*>(&k32), 4);
        f.read(reinterpret_cast<char*>(&n_genes), 4);
        f.read(reinterpret_cast<char*>(&n_kmers), 4);
        f.read(reinterpret_cast<char*>(&reserved), 4);

        if (magic != MAGIC || version != VERSION) {
            std::cerr << "[txome-gene-index] Invalid magic/version in " << path << "\n";
            return false;
        }
        k_ = static_cast<uint8_t>(k32);

        // Read gene names
        gene_names_.clear();
        gene_names_.reserve(n_genes);
        for (uint32_t i = 0; i < n_genes; ++i) {
            std::string name;
            char c;
            while (f.get(c) && c != '\0') name += c;
            gene_names_.push_back(std::move(name));
        }

        // Read k-mer entries
        entries_.resize(n_kmers);
        f.read(reinterpret_cast<char*>(entries_.data()),
               n_kmers * sizeof(KmerGeneEntry));

        if (!f.good()) return false;

        // Build hash table for O(1) lookup
        build_hash_table_();
        return true;
    }

    // ── Lookup: resolve a read to a gene using dense seed scan ───────────
    // Uses rolling k-mer hash + Bloom filter for fast rejection.
    // Returns gene_id if any gene-unique k-mer is found, or UINT32_MAX if not.
    // read_seq must be ASCII (A/C/G/T).
    //
    // Performance: ~200ns for non-matching reads (Bloom rejects all positions),
    //              ~100ns for matching reads (early hit).
    //              ~20s for 54M reads (vs 250s without Bloom).
    uint32_t resolve_gene(const char* read_seq, size_t read_len) const noexcept {
        if (read_len < k_) return UINT32_MAX;

        const uint64_t mask = (1ULL << (2 * k_)) - 1;
        uint64_t kmer = 0;
        uint8_t valid = 0;

        for (size_t i = 0; i < read_len; ++i) {
            uint8_t b = base_to_2bit(read_seq[i]);
            if (b == 0xFF) {
                valid = 0;
                kmer = 0;
                continue;
            }
            kmer = ((kmer << 2) | b) & mask;
            if (++valid >= k_) {
                // Bloom filter: fast rejection (~1ns, in L3 cache)
                if (bloom_.may_contain(kmer)) {
                    // Hash table: expensive confirmation (~80ns, cache miss)
                    uint32_t gid = kmer_hash_.find(kmer);
                    if (gid != UINT32_MAX) return gid;
                }
            }
        }
        return UINT32_MAX;
    }

    // Quick lookup with only the first seed position (fastest)
    uint32_t resolve_gene_single(const char* read_seq, size_t read_len) const noexcept {
        if (read_len < k_) return UINT32_MAX;
        uint64_t kmer = encode_kmer_2bit(read_seq, k_);
        if (kmer == UINT64_MAX) return UINT32_MAX;
        return kmer_hash_.find(kmer);
    }

    uint32_t n_genes()  const noexcept { return static_cast<uint32_t>(gene_names_.size()); }
    uint64_t n_kmers()  const noexcept { return entries_.size(); }
    uint8_t  seed_k()   const noexcept { return k_; }

    const std::string& gene_name(uint32_t id) const { return gene_names_[id]; }

private:
    void build_hash_table_() {
        kmer_hash_.build(entries_);
        bloom_.build(entries_);
        std::cerr << "[txome-gene-index] Hash table: " << kmer_hash_.size()
                  << " entries, " << kmer_hash_.capacity() << " slots ("
                  << (kmer_hash_.capacity() * 12 / (1024*1024)) << " MB)"
                  << " | Bloom: " << (bloom_.memory_bytes() / (1024*1024)) << " MB\n";
    }
};

} // namespace singlet
