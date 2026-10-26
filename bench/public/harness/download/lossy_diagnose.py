"""Where does Singlet's default download path change the cDNA read?

    python lossy_diagnose.py REF.fastq DEFAULT.fastq LOSSLESS.fastq [MAX_READS]

Three decodes of the same run, in file order: NCBI's fasterq-dump (REF), `singlet download` defaults then `singlet decode` (DEFAULT),
and `singlet download --no-trim --quality full` (LOSSLESS). Reads are compared in lockstep (order is checked first). Reports, for
DEFAULT vs REF: the share of reads whose sequence differs; whether the change is a 3' trim, a 5' trim, a substitution or an N change;
the length distribution; the quality alphabet of each file and how many reads have a different quality string; and the share of
bases that are N. Standard library only.
"""
import collections
import sys


def reads(path, limit):
    with open(path) as fh:
        for i, h in enumerate(fh):
            if i % 4 == 0:
                name = h
            elif i % 4 == 1:
                seq = h.rstrip("\n")
            elif i % 4 == 3:
                yield seq, h.rstrip("\n")
                if limit and (i + 1) // 4 >= limit:
                    return


def main():
    ref, dfl, lsl = sys.argv[1:4]
    limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    c = collections.Counter()
    lens = collections.Counter()
    qa = {"ref": set(), "default": set(), "lossless": set()}
    changed_examples = []
    pos_sub = collections.Counter()
    trim3_len = collections.Counter()
    for (rs, rq), (ds, dq), (ls, lq) in zip(reads(ref, limit), reads(dfl, limit), reads(lsl, limit)):
        c["reads"] += 1
        qa["ref"].update(rq); qa["default"].update(dq); qa["lossless"].update(lq)
        c["lossless_seq_differs"] += ls != rs
        c["lossless_qual_differs"] += lq != rq
        if ds != rs:
            c["default_seq_differs"] += 1
            if len(ds) < len(rs):
                if rs.startswith(ds):
                    c["default_3prime_trim"] += 1
                    trim3_len[len(rs) - len(ds)] += 1
                elif rs.endswith(ds):
                    c["default_5prime_trim"] += 1
                else:
                    c["default_shorter_other"] += 1
            elif len(ds) == len(rs):
                d = [i for i in range(len(rs)) if rs[i] != ds[i]]
                c["default_same_length_substitution"] += 1
                c["default_sub_to_N"] += all(ds[i] == "N" for i in d)
                c["default_sub_from_N"] += all(rs[i] == "N" for i in d)
                for i in d:
                    pos_sub[i] += 1
            else:
                c["default_longer"] += 1
            if len(changed_examples) < 5:
                changed_examples.append((rs, ds))
        lens[len(ds)] += 1
        if dq != rq:
            c["default_qual_differs"] += 1
        c["ref_N_bases"] += rs.count("N")
        c["default_N_bases"] += ds.count("N")
        c["bases"] += len(rs)
    n = max(c["reads"], 1)
    print(f"reads compared: {c['reads']:,}")
    for k in ("lossless_seq_differs", "lossless_qual_differs", "default_seq_differs", "default_3prime_trim", "default_5prime_trim", "default_shorter_other",
              "default_same_length_substitution", "default_sub_to_N", "default_sub_from_N", "default_longer", "default_qual_differs"):
        print(f"  {k}: {c[k]:,} ({c[k] / n:.3%})")
    print("  3' trim length (bases removed) top:", trim3_len.most_common(8))
    print("  substitution positions (0-based) top:", pos_sub.most_common(8))
    print("  default read length distribution top:", lens.most_common(6))
    print(f"  N bases: ref {c['ref_N_bases']:,}, default {c['default_N_bases']:,} of {c['bases']:,}")
    for k, v in qa.items():
        s = "".join(sorted(v))
        print(f"  quality alphabet {k}: {len(v)} symbols: {s[:60]}")
    for rs, ds in changed_examples:
        print("  example ref    :", rs)
        print("  example default:", ds)


if __name__ == "__main__":
    main()
