// E15 (docs/gpu-engine/fable/E15-posterior.md): a plain STAR built from this branch's vendored STAR (src/star, CPU only), for the
// binned-quality truth run of jobs/binned.sbatch: the pipeline never gives STAR the barcode read's own qualities (its .1fq path writes
// 'F', its legacy FASTQ path runs CB_samTagOut), so STAR runs here on the original FASTQs. Built by jobs/starbuild.sbatch.
int star_main_impl(int argc, char* argv[]);
int main(int argc, char* argv[]) { return star_main_impl(argc, argv); }
