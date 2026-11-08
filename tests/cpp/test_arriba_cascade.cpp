// SPDX-License-Identifier: MIT
// tests/cpp/test_arriba_cascade.cpp
//
// The port of Arriba's cascade (include/singlet/pileup/arriba_cascade.h) on a synthetic GTF and STAR junction file: the annotation index lookup,
// splice-site tests, contig patterns, k-mers, the single-end split-read record, then the whole cascade: a spliced inter-chromosomal fusion with
// three molecules passes (its UMI duplicate is removed by 'duplicates'), a read-through to a gene 2 kb downstream is removed by 'read_through',
// a single-molecule spliced fusion is discarded (min_support, then in_vitro), a read from an unlisted barcode never enters, and a poly-A read trips 'low_entropy'.

#include <unistd.h>

#include <cstdlib>
#include <fstream>
#include <string>

#include "singlet/pileup/arriba_cascade.h"
#include "test_harness.h"

using namespace singlet::arriba;

namespace {
void put(const std::string& path, const std::string& text) { std::ofstream(path) << text; }

std::string gtf_exon(const std::string& chr, int s, int e, const std::string& strand, const std::string& gene, const std::string& tx, bool cds) {
    std::string attr = "gene_id \"ENSG_" + gene + ".1\"; transcript_id \"ENST_" + tx + ".1\"; gene_name \"" + gene + "\";";
    std::string l = chr + "\ttest\texon\t" + std::to_string(s) + "\t" + std::to_string(e) + "\t.\t" + strand + "\t.\t" + attr + "\n";
    if (cds) l += chr + "\ttest\tCDS\t" + std::to_string(s) + "\t" + std::to_string(e) + "\t.\t" + strand + "\t0\t" + attr + "\n";
    return l;
}

// donor chr, brk, strand, acceptor chr, brk, strand, type, repL, repR, read, startA, cigarA, startB, cigarB, nAln, maxScore, nonChim, score, best,
// PEmerged, cb, umi
std::string jl(const std::string& c1, int p1, const std::string& c2, int p2, const std::string& read, int startA, const std::string& cigA, int startB,
               const std::string& cigB, const std::string& cb, const std::string& umi) {
    return c1 + "\t" + std::to_string(p1) + "\t+\t" + c2 + "\t" + std::to_string(p2) + "\t+\t1\t0\t0\t" + read + "\t" + std::to_string(startA) + "\t" + cigA +
           "\t" + std::to_string(startB) + "\t" + cigB + "\t1\t70\t0\t70\t70\t0\t" + cb + "\t" + umi + "\n";
}
}  // namespace

int main() {
    char tmpl[] = "/tmp/arriba_cascade_XXXXXX";
    const std::string dir = mkdtemp(tmpl);
    const std::string gtf = dir + "/genes.gtf", junc = dir + "/Chimeric.out.junction", lab = dir + "/labels.tsv";
    put(gtf, gtf_exon("chr1", 1001, 1100, "+", "GA", "TA", true) + gtf_exon("chr1", 2001, 2100, "+", "GA", "TA", false) +
                 gtf_exon("chr1", 4001, 4100, "+", "GC", "TC", true) + gtf_exon("chr1", 4501, 4600, "+", "GC", "TC", true) +
                 gtf_exon("chr2", 50001, 50100, "+", "GB", "TB", true) + gtf_exon("chr2", 60001, 60100, "+", "GB", "TB", true) +
                 gtf_exon("chr3", 70001, 70100, "+", "GD", "TD", true) + gtf_exon("chr3", 80001, 80100, "+", "GD", "TD", true));

    // ---- annotation units
    {
        annotation_t a;
        assembly_t none;
        CHECK(read_annotation_gtf(gtf, none, a) && a.genes.size() == 4 && a.exons.size() == 8, "GTF: four genes, eight exons");
        make_annotation_index(a.exons, a.exon_index, a.contigs.size());
        make_annotation_index(a.genes, a.gene_index, a.contigs.size());
        gene_set_t g;
        get_annotation_by_coordinate(a.contigs.find("chr1"), 1500, 1500, g, a.gene_index);
        CHECK(g.size() == 1 && g[0]->name == "GA", "intronic position annotates to its gene");
        get_annotation_by_coordinate(a.contigs.find("chr1"), 3000, 3000, g, a.gene_index);
        CHECK(g.empty(), "intergenic position has no gene");
        gene_t ga = a.gene_names.at("GA");
        CHECK(ga->is_protein_coding && ga->start == 1000 && ga->end == 2099, "gene bounds are zero-based; CDS marks protein coding");
        CHECK(is_breakpoint_spliced(ga, DOWNSTREAM, 1099, a.exon_index), "end of exon 1 is a donor splice site");
        CHECK(is_breakpoint_spliced(ga, DOWNSTREAM, 1101, a.exon_index), "two bases off still counts (MAX_SPLICE_SITE_DISTANCE)");
        CHECK(!is_breakpoint_spliced(ga, DOWNSTREAM, 1103, a.exon_index), "four bases off does not");
        CHECK(!is_breakpoint_spliced(ga, DOWNSTREAM, 2099, a.exon_index), "end of the last coding exon is not a donor site");
        CHECK(is_breakpoint_spliced(ga, UPSTREAM, 2000, a.exon_index), "start of exon 2 is an acceptor splice site");
        CHECK(get_spliced_distance(a.contigs.find("chr1"), 1050, 2050, ga, a.exon_index) < 1000, "spliced distance skips the intron");
    }
    CHECK(is_interesting_contig("chr1", "1 2 X Y AC_* NC_*") && is_interesting_contig("NC_001802.1", "AC_* NC_*"), "contig patterns match");
    CHECK(!is_interesting_contig("chrUn_KI270302v1", "1 2 X Y AC_* NC_*") && !is_interesting_contig("chr12", "1 2"), "contig patterns reject");
    CHECK(kmer_to_int("TGCA", 0, 4) == 27, "k-mer code T=0 G=1 C=2 A=3");
    CHECK(strip_ensembl_version_number("ENSG00000186092.7") == "ENSG00000186092" && strip_ensembl_version_number("X.1") == "X.1", "version stripping");

    // ---- the split-read record
    {
        alignment_t a, b;
        a.contig = 0; a.strand = FORWARD; parse_cigar("30M40S", a.cigar); a.start = 1070; a.end = 1099;
        b.contig = 1; b.strand = FORWARD; parse_cigar("30S40M", b.cigar); b.start = 60000; b.end = 60039;
        mates_t m;
        CHECK(make_split_read(a, b, nullptr, m) && m.size() == 3, "junction line becomes mate 1, split read and supplementary");
        CHECK(m[SPLIT_READ].start == 60000 && m[SUPPLEMENTARY].start == 1070, "the longer segment is the split read");
        CHECK(m[SPLIT_READ].strand == FORWARD && m[MATE1].strand == REVERSE && !m[MATE1].first_in_pair, "strands follow Arriba's single-end flip");
        alignment_t c = b, d = a;
        parse_cigar("30S60M", c.cigar); c.end = 60059;
        parse_cigar("40M50S", d.cigar); d.end = 1109;
        mates_t m2;
        CHECK(make_split_read(d, c, nullptr, m2) && m2[SUPPLEMENTARY].end == 1099, "overlapping segments are disjoined at the supplementary");
    }

    // ---- the cascade
    put(lab, "barcode\tlabel\nAAAACCCCGGGGTTTT\tK562\nCCCCAAAAGGGGTTTT\tK562\n");
    std::string j;
    // GA exon 1 -> GB exon 2: three molecules plus a UMI duplicate
    j += jl("chr1", 1101, "chr2", 60001, "r1", 1071, "30M40S", 60001, "30S40M", "AAAACCCCGGGGTTTT", "AAAAAAAAAAAA");
    j += jl("chr1", 1101, "chr2", 60001, "r2", 1071, "30M40S", 60001, "30S40M", "AAAACCCCGGGGTTTT", "AAAAAAAAAAAA");
    j += jl("chr1", 1101, "chr2", 60001, "r3", 1071, "30M40S", 60001, "30S40M", "AAAACCCCGGGGTTTA", "CCCCCCCCCCCC");  // one mismatch to cell 1
    j += jl("chr1", 1101, "chr2", 60001, "r4", 1071, "30M40S", 60001, "30S40M", "CCCCAAAAGGGGTTTT", "GGGGGGGGGGGG");
    // GA exon 1 -> GC exon 1, 2 kb downstream: read-through
    j += jl("chr1", 1101, "chr1", 4001, "r5", 1071, "30M40S", 4001, "30S40M", "AAAACCCCGGGGTTTT", "TTTTTTTTTTTT");
    // GA -> GD: a single molecule
    j += jl("chr1", 1101, "chr3", 80001, "r6", 1071, "30M40S", 80001, "30S40M", "AAAACCCCGGGGTTTT", "ACACACACACAC");
    // a barcode outside the labelled cells
    j += jl("chr1", 1101, "chr2", 60001, "r7", 1071, "30M40S", 60001, "30S40M", "GGGGGGGGGGGGGGGG", "AAAAAAAAAAAA");
    put(junc, j);
    config_t cfg;
    cfg.input.junction_path = junc;
    detail::for_each_line(lab, [&](const std::string& l) { std::string bc = l.substr(0, l.find('\t')); if (bc != "barcode") cfg.input.barcodes.push_back(bc); });
    cfg.gtf_path = gtf;
    cfg.mapped_reads = 1000000;
    result_t r;
    run(cfg, r);
    CHECK(r.ok, "cascade runs");
    CHECK(r.load.lines == 7 && r.load.unmatched == 1 && r.load.loaded == 6, "six reads from labelled cells enter, one unlisted barcode does not");
    auto filter_of = [&](const std::string& n) { return r.alignments.at(n + ",1").filter; };
    CHECK(filter_of("r2") == FILTER_duplicates && filter_of("r1") == FILTER_none, "the second read of a molecule is a duplicate");
    CHECK(filter_of("r5") == FILTER_read_through, "a junction 2 kb downstream into the next gene is a read-through");
    const fusion_t* ab = nullptr;
    const fusion_t* ad = nullptr;
    for (auto& kv : r.fusions) {
        if (kv.second.gene1->name == "GA" && kv.second.gene2->name == "GB") ab = &kv.second;
        if (kv.second.gene1->name == "GA" && kv.second.gene2->name == "GD") ad = &kv.second;
    }
    CHECK(ab && ab->filter == FILTER_none && ab->supporting_reads() == 3, "GA-GB passes with three molecules");
    CHECK(ab && ab->spliced1 && ab->spliced2 && ab->breakpoint1 == 1099 && ab->breakpoint2 == 60000, "both breakpoints at splice sites");
    CHECK(ab && ab->evalue < 0.3 && ab->confidence >= CONFIDENCE_MEDIUM, "GA-GB has a passing e-value and confidence");
    // min_support discards it first; because it is spliced, in_vitro then re-examines it and, with one read between the most-read genes, claims it
    CHECK(ad && ad->filter == FILTER_in_vitro, "a single-molecule spliced fusion is discarded (min_support, then in_vitro)");
    const std::string out = dir + "/calls.tsv.gz";
    CHECK(write_fusions(r, out, false), "calls written");
    std::vector<std::string> rows;
    detail::for_each_line(out, [&](const std::string& l) { rows.push_back(l); });
    CHECK(rows.size() == 2 && rows[1].compare(0, 6, "GA\tGB\t") == 0, "one call, GA-GB");

    // ---- low entropy on a poly-A read
    {
        chimeric_alignments_t ca;
        alignment_t a, b;
        a.contig = 0; a.strand = FORWARD; parse_cigar("30M40S", a.cigar); a.start = 1070; a.end = 1099;
        b.contig = 1; b.strand = FORWARD; parse_cigar("30S40M", b.cigar); b.start = 60000; b.end = 60039;
        const std::string poly(70, 'A');
        mates_t m;
        make_split_read(a, b, &poly, m);
        CHECK(m.has_sequence, "read sequence attached");
        ca["r1,1"] = m;
        CHECK(filter_low_entropy(ca, 3, 0.6, 100) == 0 && ca["r1,1"].filter == FILTER_low_entropy, "a poly-A read is low entropy");
    }

    std::system(("rm -rf " + dir).c_str());
    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail == 0 ? 0 : 1;
}
