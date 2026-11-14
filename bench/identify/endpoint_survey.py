"""Metadata-only endpoint survey for runs whose deposited copy has no barcode read: SDL original-format objects, ENA submitted and BAM, GEO supplementary files, sibling runs.
curl, public APIs, default headers, no read data.   usage: endpoint_survey.py PANEL.tsv IDS.txt OUT.tsv"""
import csv, json, re, subprocess, sys, time

PANEL, IDFILE, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
IDS = [l.strip() for l in open(IDFILE) if l.strip()]
panel = {r['id']: r for r in csv.DictReader(open(PANEL), delimiter='	')}


def curl(url, tries=3):
    for k in range(tries):
        p = subprocess.run(['curl', '-s', '-L', '--max-time', '60', url], capture_output=True)
        if p.returncode == 0 and p.stdout:
            return p.stdout.decode('utf-8', 'replace')
        time.sleep(2)
    return ''


rows = []


def add(i, run, ep, status, ev, layout=''):
    rows.append([i, run, ep, status, ev, layout])


for i in IDS:
    run = panel[i]['source']
    # 1. SDL: every non-sra object of the run
    sdl = "https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=%s&accept-alternate-locations=yes" % run
    t = curl(sdl)
    try:
        files = json.loads(t)['result'][0].get('files', [])
    except Exception:
        files = None
    if files is None:
        add(i, run, 'sdl_original', 'unknown', sdl, 'no parsable answer')
    else:
        orig = [f for f in files if f.get('type') != 'sra']
        if not orig:
            add(i, run, 'sdl_original', 'absent', sdl, 'objects: ' + ','.join(sorted({f.get('type', '?') for f in files})))
        for f in orig:
            locs = f.get('locations', [])
            free = [l for l in locs if l.get('link') and not l.get('payRequired')]
            pay = [l for l in locs if l.get('payRequired') or l.get('rehydrationRequired') or not l.get('link')]
            st = 'present' if free else ('restricted' if pay else 'unknown')
            nm = f.get('name', '')
            kind = 'BAM' if nm.endswith('.bam') or '.bam.' in nm else ('FASTQ' if 'fastq' in nm.lower() or 'fq' in nm.lower() else f.get('type', ''))
            add(i, run, 'sdl_original', st, (free[0]['link'] if free else sdl), '%s %s type=%s size=%s' % (kind, nm, f.get('type'), f.get('size')))
    # 2. ENA filereport
    fields = 'run_accession,experiment_accession,sample_accession,secondary_sample_accession,sample_alias,experiment_alias,submitted_ftp,submitted_format,bam_ftp,fastq_ftp,read_count,base_count'
    url = 'https://www.ebi.ac.uk/ena/portal/api/filereport?accession=%s&result=read_run&fields=%s&format=tsv' % (run, fields)
    t = curl(url)
    rd = list(csv.DictReader(t.splitlines(), delimiter='\t'))
    if not rd:
        add(i, run, 'ena_submitted', 'unknown', url, 'no filereport row')
        add(i, run, 'ena_bam', 'unknown', url, 'no filereport row')
        exp = sample = gsm = ''
    else:
        r0 = rd[0]
        exp, sample = r0['experiment_accession'], r0['sample_accession']
        gsm = next((a for a in (r0.get('sample_alias', ''), r0.get('experiment_alias', '')) if a.startswith('GSM')), '')
        if i.startswith('GSM'):
            gsm = i
        sub = r0.get('submitted_ftp', '')
        add(i, run, 'ena_submitted', 'present' if sub else 'absent', url, (r0.get('submitted_format', '') + ' ' + sub).strip())
        bam = r0.get('bam_ftp', '')
        add(i, run, 'ena_bam', 'present' if bam else 'absent', url, ('BAM ' + bam) if bam else '')
    # 3. GEO supplementary files, GSM and GSE
    if gsm:
        soft = 'https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=%s&targ=self&form=text&view=brief' % gsm
        t = curl(soft)
        sup = re.findall(r'!Sample_supplementary_file_\d+ = (\S+)', t)
        gse = re.findall(r'!Sample_series_id = (GSE\d+)', t)
        keep = [s for s in sup if s.upper() != 'NONE']
        hit = [s for s in keep if re.search(r'\.(bam|fastq|fq)(\.gz)?(\.\d+)?$', s, re.I) or 'bam' in s.lower() or 'fastq' in s.lower()]
        add(i, run, 'geo_gsm_suppl', 'present' if hit else 'absent', soft, ('; '.join(hit) if hit else ('files: ' + '; '.join(keep)[:200] if keep else 'none listed')))
        if gse:
            g = gse[0]
            gd = 'https://ftp.ncbi.nlm.nih.gov/geo/series/%snnn/%s/suppl/' % (g[:-3], g)
            t = curl(gd)
            names = re.findall(r'href="([^"?/][^"]*)"', t)
            names = [n for n in names if n not in ('../',)]
            hit = [n for n in names if re.search(r'bam|fastq|\.fq', n, re.I)]
            add(i, run, 'geo_gse_suppl', 'present' if hit else ('absent' if names else 'unknown'), gd, ('; '.join(hit) if hit else ('files: ' + '; '.join(names)[:200] if names else 'empty or unreadable listing')))
        else:
            add(i, run, 'geo_gse_suppl', 'unknown', soft, 'no series id in the GSM record')
    else:
        add(i, run, 'geo_gsm_suppl', 'unknown', url, 'no GSM alias in ENA for this run')
        add(i, run, 'geo_gse_suppl', 'unknown', url, 'no GSM alias in ENA for this run')
    # 4. sibling runs of the same experiment and sample
    sib = {}
    for key in (exp, sample):
        if not key:
            continue
        u = 'https://www.ebi.ac.uk/ena/portal/api/filereport?accession=%s&result=read_run&fields=run_accession,experiment_accession,read_count,base_count,fastq_ftp&format=tsv' % key
        for r in csv.DictReader(curl(u).splitlines(), delimiter='\t'):
            sib[r['run_accession']] = r
    others = {k: v for k, v in sib.items() if k != run}
    found = []
    for k, v in sorted(others.items()):
        try:
            rc, bc = int(v['read_count']), int(v['base_count'])
        except Exception:
            continue
        per = bc / rc if rc else 0
        nf = len([x for x in v.get('fastq_ftp', '').split(';') if x])
        # a run holding only the barcode read: one short read per spot (26 to 28 or 8 to 10 bases); a per-spot length over a file count is not exact
        if rc and (24 <= per <= 30 or 8 <= per <= 10 or (nf == 1 and per <= 30)):
            found.append('%s %.0f bases per spot' % (k, per))
    if not others:
        add(i, run, 'sibling_runs', 'absent', 'ENA filereport by experiment %s and sample %s' % (exp, sample), 'no other run')
    else:
        add(i, run, 'sibling_runs', 'present' if found else 'absent', 'ENA filereport by experiment %s and sample %s' % (exp, sample),
            ('%d other runs; barcode-length: ' % len(others)) + ('; '.join(found[:6]) if found else 'none'))
    print(i, 'done', file=sys.stderr)

with open(OUT, 'w', newline='') as f:
    w = csv.writer(f, delimiter='\t', lineterminator='\n')
    w.writerow(['id', 'run', 'endpoint', 'status', 'evidence', 'layout_or_note'])
    w.writerows(rows)
print(len(rows), 'lines')
