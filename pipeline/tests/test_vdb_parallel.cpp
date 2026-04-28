// test_vdb_parallel.cpp — Test parallel VDB cursor reads for SRA files
// Measures if multiple independent cursors can read concurrently from the same file.
// Usage: test_vdb_parallel <sra_file> [n_threads]

#include "../src/sra_reader.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>
#include <atomic>

using Clock = std::chrono::high_resolution_clock;

struct ThreadResult {
    uint64_t spots_read = 0;
    uint64_t total_bases = 0;
    double elapsed_s = 0.0;
};

void reader_thread(const char* path, int64_t start_row, int64_t end_row,
                   ThreadResult& result) {
    SraReader reader;
    reader.open(path);

    // Seek to start position by reading ahead
    // VDB cursors support random access, but SraReader only has sequential.
    // We need a modified reader that can start at a specific row.
    // For now, just measure sequential reads from the beginning.
    // This test measures concurrent cursor throughput, not random access.

    SraReader::RawSpot spot;
    uint64_t count = 0;
    uint64_t bases = 0;
    auto t0 = Clock::now();

    // Read only our assigned range
    while (count < static_cast<uint64_t>(end_row - start_row)) {
        if (!reader.read_next_spot_fast(spot)) break;
        bases += spot.total_bases;
        count++;
    }

    auto t1 = Clock::now();
    result.spots_read = count;
    result.total_bases = bases;
    result.elapsed_s = std::chrono::duration<double>(t1 - t0).count();
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <sra_file> [n_threads]\n", argv[0]);
        return 1;
    }

    const char* sra_path = argv[1];
    int n_threads = (argc > 2) ? std::atoi(argv[2]) : 4;
    if (n_threads < 1) n_threads = 1;
    if (n_threads > 16) n_threads = 16;

    // First, get total spot count
    SraReader probe;
    probe.open(sra_path);
    int64_t total = probe.total_spots();
    probe.close();

    // Limit to 10M spots for the test
    int64_t test_spots = std::min<int64_t>(total, 10000000);
    int64_t spots_per_thread = test_spots / n_threads;

    std::fprintf(stderr, "SRA: %s\n", sra_path);
    std::fprintf(stderr, "Total spots: %lld, test spots: %lld\n",
                 (long long)total, (long long)test_spots);
    std::fprintf(stderr, "Threads: %d, spots/thread: %lld\n",
                 n_threads, (long long)spots_per_thread);

    // === Single-threaded baseline ===
    {
        ThreadResult result;
        auto t0 = Clock::now();
        reader_thread(sra_path, 0, test_spots, result);
        auto t1 = Clock::now();
        double wall = std::chrono::duration<double>(t1 - t0).count();
        std::fprintf(stderr, "\n[1-thread] %llu spots in %.3fs = %.1f Mspots/s, %.1f Mbp/s\n",
                     (unsigned long long)result.spots_read, wall,
                     result.spots_read / wall / 1e6,
                     result.total_bases / wall / 1e6);
    }

    // === Multi-threaded ===
    {
        std::vector<std::thread> threads(n_threads);
        std::vector<ThreadResult> results(n_threads);

        auto t0 = Clock::now();

        for (int i = 0; i < n_threads; i++) {
            int64_t start = i * spots_per_thread;
            int64_t end = (i == n_threads - 1) ? test_spots : (i + 1) * spots_per_thread;
            // NOTE: each thread reads from row 0 (not start), because SraReader
            // is sequential. But we read the same NUMBER of spots to measure
            // throughput. Multiple cursors reading the same rows concurrently
            // tests the VDB library's ability to handle parallel cursor reads.
            threads[i] = std::thread(reader_thread, sra_path, 0, spots_per_thread,
                                     std::ref(results[i]));
        }

        for (auto& t : threads) t.join();
        auto t1 = Clock::now();
        double wall = std::chrono::duration<double>(t1 - t0).count();

        uint64_t total_spots_read = 0;
        uint64_t total_bases_read = 0;
        double max_thread_time = 0;
        for (int i = 0; i < n_threads; i++) {
            total_spots_read += results[i].spots_read;
            total_bases_read += results[i].total_bases;
            if (results[i].elapsed_s > max_thread_time)
                max_thread_time = results[i].elapsed_s;
        }

        std::fprintf(stderr, "\n[%d-thread] %llu spots in %.3fs wall (%.3fs max-thread)\n",
                     n_threads,
                     (unsigned long long)total_spots_read, wall, max_thread_time);
        std::fprintf(stderr, "[%d-thread] Aggregate: %.1f Mspots/s, %.1f Mbp/s\n",
                     n_threads,
                     total_spots_read / wall / 1e6,
                     total_bases_read / wall / 1e6);

        double single_spots_per_sec = test_spots / max_thread_time;
        double multi_spots_per_sec = total_spots_read / wall;
        std::fprintf(stderr, "\nScaling: %.2fx (%.1f vs %.1f Mspots/s per thread equiv)\n",
                     multi_spots_per_sec / (test_spots / max_thread_time * n_threads / n_threads),
                     multi_spots_per_sec / 1e6,
                     single_spots_per_sec / 1e6);
    }

    return 0;
}
