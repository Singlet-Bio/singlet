#include <cassert>
#include <cstdint>
#include <vector>
#include <iostream>

#include "singlet/pileup/velocity.h"

namespace singlet {

// ============================================================================
// Mock structures for testing
// ============================================================================

struct MockCSC {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint16_t> data;
    uint32_t nrows;
    uint32_t ncols;
};

struct MockGeneModel {
    uint32_t n_genes() const { return n_genes_; }
    uint32_t intron_to_gene(uint32_t intron_idx) const { 
        return mapping_[intron_idx]; 
    }
    uint32_t n_genes_;
    std::vector<uint32_t> mapping_;
};

// ============================================================================
// Test: make_empty_gene_csc - creates correct dimensions
// ============================================================================
void test_make_empty_gene_csc_dimensions() {
    std::cout << "test_make_empty_gene_csc_dimensions... ";
    
    auto csc = make_empty_gene_csc<uint32_t>(100, 50);
    
    assert(csc.nrows == 100);
    assert(csc.ncols == 50);
    assert(csc.indptr.size() == 51);  // ncols + 1
    assert(csc.indices.empty());
    assert(csc.data.empty());
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: make_empty_gene_csc - all-zero indptr
// ============================================================================
void test_make_empty_gene_csc_zero_indptr() {
    std::cout << "test_make_empty_gene_csc_zero_indptr... ";
    
    auto csc = make_empty_gene_csc<double>(25, 10);
    
    for (size_t i = 0; i < csc.indptr.size(); ++i) {
        assert(csc.indptr[i] == 0);
    }
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: make_empty_gene_csc - empty data/indices
// ============================================================================
void test_make_empty_gene_csc_empty_data() {
    std::cout << "test_make_empty_gene_csc_empty_data... ";
    
    auto csc = make_empty_gene_csc<uint16_t>(5, 3);
    
    assert(csc.data.empty());
    assert(csc.indices.empty());
    assert(csc.indptr.size() == 4);  // ncols + 1 = 3 + 1
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: collapse_intron_to_gene - empty matrix
// ============================================================================
void test_collapse_intron_to_gene_empty_matrix() {
    std::cout << "test_collapse_intron_to_gene_empty_matrix... ";
    
    MockCSC intron_csc;
    intron_csc.nrows = 100;  // 100 intron intervals
    intron_csc.ncols = 50;   // 50 cells
    intron_csc.indptr.assign(51, 0);  // All columns empty
    
    MockGeneModel gm;
    gm.n_genes_ = 20;  // 20 genes
    gm.mapping_.assign(100, 0);  // unused since no data
    
    auto out = collapse_intron_to_gene(intron_csc, gm);
    
    assert(out.nrows == 20);  // n_genes
    assert(out.ncols == 50);  // n_cells
    assert(out.indices.empty());
    assert(out.data.empty());
    assert(out.indptr.size() == 51);
    for (int32_t val : out.indptr) {
        (void)val;
        assert(val == 0);
    }
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: collapse_intron_to_gene - single cell, single intron
// ============================================================================
void test_collapse_intron_to_gene_single_cell_single_intron() {
    std::cout << "test_collapse_intron_to_gene_single_cell_single_intron... ";
    
    MockCSC intron_csc;
    intron_csc.nrows = 10;
    intron_csc.ncols = 1;
    intron_csc.indptr = {0, 1};  // 1 entry in cell 0
    intron_csc.indices = {5};    // intron 5
    intron_csc.data = {7};       // count = 7
    
    MockGeneModel gm;
    gm.n_genes_ = 5;
    gm.mapping_ = {0, 1, 2, 3, 4, 2};  // intron 5 → gene 2
    
    auto out = collapse_intron_to_gene(intron_csc, gm);
    
    assert(out.nrows == 5);
    assert(out.ncols == 1);
    assert(out.indices.size() == 1);
    assert(out.indices[0] == 2);  // gene 2
    assert(out.data[0] == 7);
    assert(out.indptr.size() == 2);
    assert(out.indptr[0] == 0);
    assert(out.indptr[1] == 1);
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: collapse_intron_to_gene - multiple introns mapping to same gene
// ============================================================================
void test_collapse_intron_to_gene_multiple_introns_same_gene() {
    std::cout << "test_collapse_intron_to_gene_multiple_introns_same_gene... ";
    
    MockCSC intron_csc;
    intron_csc.nrows = 10;
    intron_csc.ncols = 1;
    intron_csc.indptr = {0, 3};  // 3 entries in cell 0
    intron_csc.indices = {2, 4, 7};  // introns 2, 4, 7
    intron_csc.data = {3, 5, 2};     // counts
    
    MockGeneModel gm;
    gm.n_genes_ = 5;
    gm.mapping_ = {0, 1, 2, 1, 2, 3, 4, 2};  // introns 2,4,7 → genes 2,2,2
    
    auto out = collapse_intron_to_gene(intron_csc, gm);
    
    assert(out.nrows == 5);
    assert(out.ncols == 1);
    assert(out.indices.size() == 1);
    assert(out.indices[0] == 2);  // only gene 2 has nnz
    assert(out.data[0] == 10);     // 3 + 5 + 2 = 10
    assert(out.indptr[0] == 0);
    assert(out.indptr[1] == 1);
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: collapse_intron_to_gene - multiple genes in same cell
// ============================================================================
void test_collapse_intron_to_gene_multiple_genes_same_cell() {
    std::cout << "test_collapse_intron_to_gene_multiple_genes_same_cell... ";
    
    MockCSC intron_csc;
    intron_csc.nrows = 10;
    intron_csc.ncols = 1;
    intron_csc.indptr = {0, 4};  // 4 entries
    intron_csc.indices = {0, 1, 3, 5};
    intron_csc.data = {2, 3, 4, 1};
    
    MockGeneModel gm;
    gm.n_genes_ = 5;
    gm.mapping_ = {0, 1, 0, 2, 3, 4};  // introns map to genes 0,1,0,2,3,4
    
    auto out = collapse_intron_to_gene(intron_csc, gm);
    
    assert(out.nrows == 5);
    assert(out.ncols == 1);
    // Only genes with nonzero counts are output: 0, 1, 2, 4 (gene 3 has no introns)
    assert(out.indices.size() == 4);
    // Indices should be sorted: [0, 1, 2, 4]
    assert(out.indices[0] == 0);  // gene 0: introns 0,2 (but only 0,1,3,5 in csc) → just 0 → 2
    assert(out.indices[1] == 1);  // gene 1: intron 1 → 3
    assert(out.indices[2] == 2);  // gene 2: intron 3 → 4
    assert(out.indices[3] == 4);  // gene 4: intron 5 → 1
    
    assert(out.data[0] == 2);  // gene 0
    assert(out.data[1] == 3);  // gene 1
    assert(out.data[2] == 4);  // gene 2
    assert(out.data[3] == 1);  // gene 4
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: collapse_intron_to_gene - multiple cells
// ============================================================================
void test_collapse_intron_to_gene_multiple_cells() {
    std::cout << "test_collapse_intron_to_gene_multiple_cells... ";
    
    MockCSC intron_csc;
    intron_csc.nrows = 6;
    intron_csc.ncols = 3;
    // Cell 0: introns [0, 2]
    // Cell 1: introns [1]
    // Cell 2: introns [3, 4]
    intron_csc.indptr = {0, 2, 3, 5};
    intron_csc.indices = {0, 2, 1, 3, 4};
    intron_csc.data = {1, 2, 3, 4, 5};
    
    MockGeneModel gm;
    gm.n_genes_ = 4;
    gm.mapping_ = {0, 1, 0, 2, 3};  // introns 0,2→0; 1→1; 3→2; 4→3
    
    auto out = collapse_intron_to_gene(intron_csc, gm);
    
    assert(out.nrows == 4);
    assert(out.ncols == 3);
    assert(out.indptr[0] == 0);
    assert(out.indptr[1] == 1);  // cell 0 has 1 gene (both introns map to gene 0)
    assert(out.indptr[2] == 2);  // cell 1 has 1 gene (1), cumulative 2
    assert(out.indptr[3] == 4);  // cell 2 has 2 genes (2, 3), cumulative 4
    
    // Cell 0: gene 0 has introns [0,2] → 1+2=3; gene 2 has intron [none] → 0 (wait no, intron 2 maps to gene 0)
    // Actually: intron 0 → gene 0 (count 1), intron 2 → gene 0 (count 2) → gene 0 total = 3
    // So cell 0 should only have 1 entry for gene 0
    
    // Let me recalculate:
    // Cell 0 (intron_csc.indices[0:2] = [0, 2], data[0:2] = [1, 2]):
    //   intron 0 → gene 0, count 1 → acc[0] = 1
    //   intron 2 → gene 0, count 2 → acc[0] = 1+2 = 3
    //   active = [0]
    //   output: indices=[0], data=[3]
    
    // Cell 1 (intron_csc.indices[2:3] = [1], data[2:3] = [3]):
    //   intron 1 → gene 1, count 3 → acc[1] = 3
    //   active = [1]
    //   output: indices=[0,1], data=[3,3]
    
    // Cell 2 (intron_csc.indices[3:5] = [3, 4], data[3:5] = [4, 5]):
    //   intron 3 → gene 2, count 4 → acc[2] = 4
    //   intron 4 → gene 3, count 5 → acc[3] = 5
    //   active = [2, 3]
    //   output: indices=[0,1,2,3], data=[3,3,4,5]
    
    assert(out.indices[0] == 0);
    assert(out.data[0] == 3);    // cell 0, gene 0
    
    assert(out.indices[1] == 1);
    assert(out.data[1] == 3);    // cell 1, gene 1
    
    assert(out.indices[2] == 2);
    assert(out.data[2] == 4);    // cell 2, gene 2
    
    assert(out.indices[3] == 3);
    assert(out.data[3] == 5);    // cell 2, gene 3
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: collapse_intron_to_gene - out-of-range gene mapping
// ============================================================================
void test_collapse_intron_to_gene_out_of_range_gene() {
    std::cout << "test_collapse_intron_to_gene_out_of_range_gene... ";
    
    MockCSC intron_csc;
    intron_csc.nrows = 5;
    intron_csc.ncols = 1;
    intron_csc.indptr = {0, 3};
    intron_csc.indices = {0, 1, 2};
    intron_csc.data = {2, 3, 4};
    
    MockGeneModel gm;
    gm.n_genes_ = 3;
    // intron 0 → gene 0 (valid)
    // intron 1 → gene 5 (out of range, should be skipped)
    // intron 2 → gene 1 (valid)
    gm.mapping_ = {0, 5, 1};
    
    auto out = collapse_intron_to_gene(intron_csc, gm);
    
    assert(out.nrows == 3);
    assert(out.ncols == 1);
    assert(out.indices.size() == 2);  // only 2 valid genes
    assert(out.indices[0] == 0);
    assert(out.indices[1] == 1);
    assert(out.data[0] == 2);  // gene 0, count from intron 0
    assert(out.data[1] == 4);  // gene 1, count from intron 2 (intron 1 was skipped)
    assert(out.indptr[0] == 0);
    assert(out.indptr[1] == 2);
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: collapse_intron_to_gene - output type conversion (uint16 → uint32)
// ============================================================================
void test_collapse_intron_to_gene_type_conversion() {
    std::cout << "test_collapse_intron_to_gene_type_conversion... ";
    
    MockCSC intron_csc;
    intron_csc.nrows = 3;
    intron_csc.ncols = 1;
    intron_csc.indptr = {0, 2};
    intron_csc.indices = {0, 1};
    intron_csc.data = {100, 200};  // uint16_t
    
    MockGeneModel gm;
    gm.n_genes_ = 2;
    gm.mapping_ = {0, 1};
    
    auto out = collapse_intron_to_gene(intron_csc, gm);
    
    // Output should be uint32_t (since input is integral)
    // For uint16_t input, output type is uint32_t per collapse_intron_to_gene logic
    assert(out.data[0] == 100);
    assert(out.data[1] == 200);
    
    std::cout << "PASS" << std::endl;
}

// ============================================================================
// Test: collapse_intron_to_gene - no introns in cell
// ============================================================================
void test_collapse_intron_to_gene_cell_with_no_introns() {
    std::cout << "test_collapse_intron_to_gene_cell_with_no_introns... ";
    
    MockCSC intron_csc;
    intron_csc.nrows = 5;
    intron_csc.ncols = 2;
    intron_csc.indptr = {0, 2, 2};  // cell 0 has 2 introns, cell 1 has 0
    intron_csc.indices = {0, 1};
    intron_csc.data = {5, 7};
    
    MockGeneModel gm;
    gm.n_genes_ = 3;
    gm.mapping_ = {0, 1, 2, 0, 1};
    
    auto out = collapse_intron_to_gene(intron_csc, gm);
    
    assert(out.nrows == 3);
    assert(out.ncols == 2);
    assert(out.indptr[0] == 0);
    assert(out.indptr[1] == 2);  // cell 0 has 2 entries
    assert(out.indptr[2] == 2);  // cell 1 has 0 entries
    
    assert(out.indices[0] == 0);
    assert(out.indices[1] == 1);
    assert(out.data[0] == 5);
    assert(out.data[1] == 7);
    
    std::cout << "PASS" << std::endl;
}

}  // namespace singlet

int main() {
    using namespace singlet;
    
    std::cout << "Running velocity.h unit tests...\n" << std::endl;
    
    // make_empty_gene_csc tests
    test_make_empty_gene_csc_dimensions();
    test_make_empty_gene_csc_zero_indptr();
    test_make_empty_gene_csc_empty_data();
    
    // collapse_intron_to_gene tests
    test_collapse_intron_to_gene_empty_matrix();
    test_collapse_intron_to_gene_single_cell_single_intron();
    test_collapse_intron_to_gene_multiple_introns_same_gene();
    test_collapse_intron_to_gene_multiple_genes_same_cell();
    test_collapse_intron_to_gene_multiple_cells();
    test_collapse_intron_to_gene_out_of_range_gene();
    test_collapse_intron_to_gene_type_conversion();
    test_collapse_intron_to_gene_cell_with_no_introns();
    
    std::cout << "\nAll tests passed!" << std::endl;
    return 0;
}
