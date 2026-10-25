#!/usr/bin/env python3
"""Print the SRA runs of a cohort `gsm` field: 'GSM1;GSM2', ranges 'GSM3836772-GSM3837155', notes in brackets ignored.

Uses the ENA portal API (a GEO sample's ENA sample_alias is its GSM accession). Standard library only.
"""
import json
import re
import sys
import urllib.parse
import urllib.request

API = "https://www.ebi.ac.uk/ena/portal/api/search"


def gsms(field: str):
    for part in re.sub(r"\(.*?\)", "", field).split(";"):
        part = part.strip()
        m = re.fullmatch(r"GSM(\d+)-GSM(\d+)", part)
        if m:
            yield from (f"GSM{i}" for i in range(int(m.group(1)), int(m.group(2)) + 1))
        elif part.startswith("GSM"):
            yield part


def runs(gsm_list):
    out = []
    ids = list(gsm_list)
    for i in range(0, len(ids), 100):
        q = " OR ".join(f'sample_alias="{g}"' for g in ids[i:i + 100])
        url = API + "?" + urllib.parse.urlencode({"result": "read_run", "query": q, "fields": "run_accession,sample_alias",
                                                  "format": "json", "limit": 0})
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "singlet-bench/1.0"}), timeout=120) as r:
            out += [d["run_accession"] for d in json.load(r)]
    return sorted(set(out))


if __name__ == "__main__":
    print("\n".join(runs(gsms(sys.argv[1]))))
