#!/usr/bin/env python
"""Assemble notes/w4_corpus.md from results/w4_corpus_head.md + results/w4_corpus_verdicts.md + generated tables."""
import os
HERE = os.path.dirname(os.path.abspath(__file__)); BC = os.path.dirname(HERE); R = os.path.join(BC, "results")
parts = [open(os.path.join(R, f)).read() for f in ("w4_corpus_head.md", "w4_corpus_verdicts.md")]
parts.append("\n## 5. Generated tables (tools/w4_corpus_report.py)\n\n" + open(os.path.join(R, "w4_corpus_tables.md")).read().replace("\n### ", "\n#### "))
parts.append("\n## 6. Generated tables (tools/w4_corpus_report2.py)\n\n" + open(os.path.join(R, "w4_corpus_tables2.md")).read().replace("\n### ", "\n#### "))
open(os.path.join(BC, "notes", "w4_corpus.md"), "w").write("\n".join(parts))
