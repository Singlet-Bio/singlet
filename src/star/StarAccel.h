#ifndef H_StarAccel
#define H_StarAccel
// Switches for exact (output-identical) speedups of the mapping loop.

// Window assignment (ReadAlign_stitchPieces.cpp) scans every SA locus of every stored seed, up to seedMultimapNmax
// (10000) per seed, and calls assignAlignToWindow for each; for a locus in a genome bin without a window that call
// returns at once with no side effect. With g_starFastAssign the strand conversion is precomputed per seed and the
// empty-bin test is made in the loop, so assignAlignToWindow is called for exactly the same loci, in the same order.
// SINGLET_STAR_FASTASSIGN=0 restores the original loop.
extern bool g_starFastAssign;

#endif
