# Brainstorm: what would actually move the frontier (Fable 5.1, 2026-10-21)

This note is the source of the "idea_*" experiment wave. Each idea states the mechanism, why it should win bits or
speed, how it stays bit-exact and deterministic, and what would falsify it. Agents implementing an idea read this
section plus PLAN.md sections 1-3.

## 0. Position on "recursive compression"

Lossless coding cannot beat the entropy of the process that generated the data; that is the source coding theorem and
it is not a limitation of "statistical" methods that an "algorithmic" method escapes. Kolmogorov complexity is the
right concept for *constants*: the exon-interval table, the gene order, the 10x barcode whitelist are all outputs of
short programs run on the reference, so they cost zero bits if encoder and decoder share the reference. That is the
entire legitimate content of the "algorithmic complexity" angle here, and it is worth 3-5 MB per file (PLAN.md
section 3). The counts themselves are a sample from a stochastic process (UMI capture from a cell's transcriptome,
ambient RNA in empty droplets); their entropy under the true process is irreducible, and recursively compressing an
output that is already at the model's cross-entropy returns a ratio of 1.0 (Track D measures this).

What *is* real and what we pursue:

- **Conditional entropy with shared side information.** The decoder may know a lot: a reference vocabulary, a model
  of gene expression trained on other studies, and everything it has already decoded from this file. Every bit of
  that knowledge moves the bound from H(X) to H(X | side info). The whole game is making the side information
  predictive and the decoder fast at using it.
- **Chain-rule factorisation as "recursion".** Coding gene totals first, then intervals given genes, then counts
  given presence, then empties given the decoded cells, is a hierarchy where each layer conditions on the previous
  one. It is the chain rule, not magic, but it is how you organise a model-driven decoder so that each stage is
  cheap and the expensive model is evaluated once per cell, not once per nonzero.
- **The file as its own side information.** Cells decoded earlier are the best predictor for cells decoded later
  (same sample, same cell types, same ambient profile). Ordering cells so that similar cells are adjacent and
  predicting each from its decoded neighbours gives cell-type-specific probabilities with *no shipped model* and no
  cross-validation caveats.

What we do not pursue: lossy "semantic" codes, latent drift correction, anything that does not reproduce the matrix.

## 1. Where the bits are (what any model must explain)

Per nonzero we pay for three things: *which feature* (row id, ~log2(m/k) bits for a cell with k nonzeros out of m
features under a uniform model; a good model lowers this toward -log2 p(feature | cell)), *how many* (75-99% ones,
<= 0.5 bit under a per-gene model), and *which cell* (barcode; negligible after delta coding). Today's VOCSC+zstd
spends 6.8-14 bits/nnz on the matrix payload. The uniform set-coding bound for pbmc_v2_typical exon
(k ~ 410 of m = 310,797) is log2(310797/410) + 1.44 ~ 11 bits per entry, so zstd is already exploiting the gene
frequency skew. A per-feature frequency model (order-0 over features) gives H ~ Σ -log2 p_f, typically 8-9 bits;
conditioning on depth and on cell type is where the next 2-4 bits are.

## 2. Ideas

### Idea 1 - Survival-function gap coding (model-driven set coding in O(nnz log m), not O(m))

A per-(feature, depth-bucket) presence model p_f(d) is the natural first model, but arithmetic-coding one binary
decision per feature per cell is O(m x n) = 3e10 decisions for a 96k-column file: too slow to ever decode fast.
Instead code the *gap* to the next present feature directly: with features in a fixed order, the probability that the
next present feature after position i is j is  P(j | i) = Π_{i<t<j}(1 - p_t) · p_j. Precompute the prefix sum of
-log2(1 - p_t) (fixed-point) per depth bucket; then the CDF of j is a monotone table lookup and decoding is a binary
search over the table slice: O(log m) per nonzero, and the per-cell cost is independent of m. With integer tables the
decoder is deterministic. Expected gain: reach the order-0-per-feature × depth cross-entropy (Track D's estimate)
at > 20 Mnnz/s. Falsified if the quantisation needed to fit 310k features × 16 buckets in 8 MiB costs > 0.3 bits/nnz;
mitigation: model at gene level (38k) with a within-gene interval table (Idea 3).

### Idea 2 - Neighbour-conditioned cells: the file as its own model

Order the called cells so neighbours are similar (cheap: sort by a 32-dim hash/projection of log-counts of the top
genes, or greedy nearest-neighbour chaining in that space). Code cell j's presence set with probabilities
p_f = mix(α · neighbour-estimate, (1-α) · global prior) where the neighbour estimate is a running integer count table
over the last K decoded cells (or the decoded cells of the same cluster), updated identically on both sides; α and K
fixed small integers. Then code the *empty droplets* last, against the ambient profile estimated from the decoded
cells (depth-weighted sum of counts, integer), which is exactly the multinomial model Track D identifies as the floor
for empties, with zero side information. Expected gain: the cell-type-specific bits that a static model cannot have,
and the empties at their floor. Cost: the cell order (n log2 n bits, ~1.7 bytes/cell for 10k cells, or free if the
order is stored implicitly by sorting barcodes last). Falsified if adaptive mixing beats the static per-gene×depth
model by < 0.3 bits/nnz on called cells.

### Idea 3 - Gene / interval factorisation and exon|intron joint coding

Rows are exon or intron intervals of genes. Code each cell as gene totals (where biology lives), then the split of
each gene's c UMIs over its intervals as a multinomial with gene-specific interval probabilities (3' bias makes this
peaked; shipped as a model, ~310k x 1 byte quantised). Measure H(interval | gene, count) from data: if it is
< 1 bit/UMI the factorisation is nearly free and the gene-level model does all the work. Second, the intron matrix of
the same sample shares columns and cell structure with the exon matrix: code intron given exon (per gene,
P(intron present | exon count of that gene, depth)). Measure the mutual information; expected to be large for
nuclei (brain_sn_*), moderate for PBMC. Falsified if MI < 0.2 bits per intron nonzero.

### Idea 4 - Low-rank manifold prior (the generative prior, done honestly)

Train out of sample: gene loadings W (G x k, k ~ 16-32, int8) as the shipped model (38k x 32 ~ 1.2 MB). Per cell,
the encoder fits a k-vector h (quantised to 4 bits each, ~16 bytes of side info) and both sides compute
π = softmax(W h) with integer LUT exponentials; the cell's counts are then coded as Multinomial(depth, π) (presence and
counts together via the survival trick of Idea 1 on the per-cell CDF). Gain = reduction of cross-entropy from
per-gene×depth to per-cell π, minus 16 bytes/cell. Decode cost: one G x k integer matvec per called cell (1.2M MACs,
~0.1 ms scalar, 10x less with AVX-512 int8), none for empties. Falsified if the out-of-sample gain is < the side
info + the cost of not knowing the cell's depth-adjusted dropout curve; in that case Idea 2 (neighbours) is the
cheaper way to get cell-type specificity.

### Idea 5 - Decode-speed architecture for a model-driven coder

Interleaved rANS (4-8 states) with 12-16-bit fixed-point frequencies looked up from the model tables; cells in
blocks of 256 so a block's tables stay in L2; no per-symbol divisions in the decoder (reciprocal tables); scalar
reference + AVX-512 path. Target: >= 50 Mnnz/s for the Idea 1 coder, >= 200 Mnnz/s for a static-table coder.

### Idea 6 - Reference-generated constants cost nothing

Feature names (vocabulary id), gene order, 10x barcode whitelist (code colnames as a sorted subset of the whitelist:
~log2(U/n)+1.44 bits per barcode), column pointers derived from decoded counts. This is Track A's territory and the
"Kolmogorov" sliver that is real.

## 3. Hypotheses to test this wave

H1: Idea 1 reaches within 0.2 bits/nnz of the per-gene×depth cross-entropy at >= 20 Mnnz/s decode.
H2: Idea 2 beats the static model by >= 0.5 bits/nnz on called cells and puts empties within 2% of the multinomial
    floor, with no shipped model.
H3: H(interval | gene, count) < 1 bit per UMI, and I(exon; intron) per intron nonzero > 0.3 bits for snRNA.
H4: Idea 4 beats per-gene×depth by >= 1 bit/nnz on called cells out of sample, net of side information.
Each agent reports its numbers against these; a falsified hypothesis is a result, not a failure.
