// Test for IntervalTree
#include "singlet-pileup/interval_tree.h"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    using namespace singlet;

    // Build a small tree
    std::vector<Interval> ivs = {
        {100, 200, 0},   // exon 0: [100, 200)
        {150, 300, 1},   // exon 1: [150, 300)
        {500, 600, 2},   // exon 2: [500, 600)
        {550, 700, 3},   // exon 3: [550, 700)
        {1000, 1100, 4}, // exon 4: [1000, 1100)
    };

    IntervalTree tree;
    tree.build(std::move(ivs));

    std::vector<uint32_t> results;

    // Query: [120, 160) should hit exons 0 and 1
    tree.query(120, 160, results);
    assert(results.size() == 2);
    std::cout << "Test 1 PASS: [120,160) → " << results.size() << " hits\n";

    // Query: [250, 260) should hit exon 1 only
    results.clear();
    tree.query(250, 260, results);
    assert(results.size() == 1);
    assert(results[0] == 1);
    std::cout << "Test 2 PASS: [250,260) → exon 1\n";

    // Query: [400, 450) should hit nothing
    results.clear();
    tree.query(400, 450, results);
    assert(results.empty());
    std::cout << "Test 3 PASS: [400,450) → no hits\n";

    // Query: [550, 600) should hit exons 2 and 3
    results.clear();
    tree.query(550, 600, results);
    assert(results.size() == 2);
    std::cout << "Test 4 PASS: [550,600) → " << results.size() << " hits\n";

    // Query: [0, 2000) should hit all 5
    results.clear();
    tree.query(0, 2000, results);
    assert(results.size() == 5);
    std::cout << "Test 5 PASS: [0,2000) → all 5\n";

    // Empty tree
    IntervalTree empty;
    results.clear();
    empty.query(0, 100, results);
    assert(results.empty());
    std::cout << "Test 6 PASS: empty tree → no hits\n";

    std::cout << "All interval tree tests passed!\n";
    return 0;
}
