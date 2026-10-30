# SNP false-alt study (D6)

Evidence and the rule R12 are in `docs/research-beta/snp-false-alt.md`. Scripts run on Clipper beside the private donor VCFs and write counts only (plus a pickle of per-site rows with no site ids, kept on Clipper, never copied):

- `strat1.py RUNS TRUTH_DIR VARIANT_DIR OUT.txt POOL` - join Singlet donor VCFs to the benchmark's truth, stratify false alts.
- `sweep.py OUT.txt pkl...` - alternative caller rules on the same counts.
- `resid.py OUT.txt pkl...` - what remains after R12.
- `strat2.sbatch` - runs the five Kang pools (jobs 792789 on the 0.0597 run, 792790 on the campaign runs).
