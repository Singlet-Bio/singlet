#!/usr/bin/env python3
"""Aggregate the ingest study's raw JSONL (bench/gpu/ingest/results_raw/) into bench/gpu/results/ingest.json and print the tables.

  summarize.py [--out ../results/ingest.json]

Inputs (copied back from Clipper job dirs, small result files only):
  results_raw/<sample>_<jobid>.jsonl        run_sample.sbatch (transform variants, reader, GPU decode, identity, CPU compat, m8 baseline)
  results_raw/cpu_<sample>_<jobid>.jsonl    cpu_extra.sbatch  (zstd level ladder, CPU-engine compatibility)
  results_raw/dl*.txt                       download.sbatch / lite_compare.sbatch logs (parsed by hand into DOWNLOAD below)
"""
import json, glob, os, sys, re

here = os.path.dirname(os.path.abspath(__file__))
raw = os.path.join(here, 'results_raw')
out = os.path.join(here, '..', 'results', 'ingest.json')
if '--out' in sys.argv:
    out = sys.argv[sys.argv.index('--out') + 1]


def load(path):
    rows = []
    for l in open(path):
        l = l.strip()
        if l.startswith('{'):
            try:
                rows.append(json.loads(l))
            except Exception:
                pass
    return rows


def pick(rows, run):
    for r in rows:
        if r.get('run') == run:
            return r
    return None


def enc(r):
    if not r:
        return None
    return {k: r[k] for k in ('codec', 'raw_sections', 'level', 'threads', 'crc', 'chunk_pairs', 'pairs_in', 'pairs_out', 'dropped', 'unknown_bc', 'n_exc_N', 'in_bytes', 'out_bytes',
                              'out_over_in', 'out_bytes_per_pair', 'in_bytes_per_pair', 'wall_s', 'cpu_s_total', 'cpu_s_per_Mpairs', 'Mpairs_per_s', 'in_MBps', 'out_MBps',
                              'cpu_split_per_Mpairs', 'section_bytes') if k in r}


def rd(r):
    if not r:
        return None
    keys = ('mode', 'codec', 'threads', 'batch_chunks', 'crc', 'pairs', 'file_bytes_read', 'bytes_per_pair', 'wall_s', 'Mpairs_per_s', 'file_MBps', 'cpu_s_total', 'cpu_s_per_Mpairs',
            'cpu_reader_threads_s_per_Mpairs', 'cpu_gpu_thread_s_per_Mpairs', 'crc_mismatch', 'status_ok')
    return {k: r[k] for k in keys if k in r}


def gd(r):
    if not r:
        return None
    keys = ('codec', 'batch_chunks', 'pairs_per_iter', 'comp_bytes_per_iter', 'raw_bytes_per_iter', 'reps', 'ms_median', 'ms_min', 'decoded_GBps_median', 'decoded_GBps_best',
            'comp_GBps_median', 'Mpairs_per_s_median', 'Mpairs_per_s_best', 'status_ok', 'nvcomp_temp_bytes')
    return {k: r[k] for k in keys if k in r}


def build_sample(label, run_file, cpu_file):
    rows = load(run_file); cpu = load(cpu_file) if cpu_file and os.path.exists(cpu_file) else []
    S = {'label': label, 'files': {'run': os.path.basename(run_file), 'cpu': os.path.basename(cpu_file) if cpu_file else None}}
    info = next((r for r in rows if r.get('info') == 'node'), {})
    S['job'] = info.get('job'); S['host'] = info.get('host'); S['source_1fq_bytes'] = info.get('q_bytes')
    S['selftest'] = pick(rows, 'selftest')
    S['transform'] = {r['run']: enc(r) for r in rows if r.get('run', '').startswith('enc_')}
    S['transform_levels'] = {r['run']: enc(r) for r in cpu if r.get('run', '').startswith('enc_level')}
    S['reader_e2e'] = {r['run']: rd(r) for r in rows if r.get('run', '').startswith('read_')}
    S['gpu_decode'] = {r['run']: gd(r) for r in rows if r.get('run', '').startswith('gpu_decode')}
    S['identity_vs_1fq_decode'] = {r['run']: {k: r[k] for k in r if k.startswith('verify') or k in ('pairs', 'status_ok')} for r in rows if r.get('run', '').startswith('verify')}
    S['m8_1fq_baseline'] = {r['run']: {k: r.get(k) for k in ('threads', 'pairs', 'wall_s', 'pairs_per_s_M_total', 'host_core_s_per_Mpair', 'per_thread_ns_per_pair', 'file_B_per_pair', 'payload_B_per_pair')}
                            for r in rows if r.get('run', '').startswith('m8_')}
    S['cpu_engine_compat'] = {}
    for src in (rows, cpu):
        for r in src:
            if r.get('side'):
                S['cpu_engine_compat'].setdefault(r['run'], {})[r['side']] = {k: r.get(k) for k in ('threads', 'pairs', 'dropped', 'hash', 'wall_s', 'cpu_s', 'cpu_s_per_Mpairs', 'Mpairs_per_s', 'hashing', 'hash_equal_to_1fg')}
    return S


samples = {}
for label, runpat in (('pbmc1k_v3', 'pbmc1k_v3_*.jsonl'), ('GSM6470208', 'GSM6470208_*.jsonl')):
    files = sorted(f for f in glob.glob(os.path.join(raw, runpat)) if 'cancelled' not in f and '/cpu_' not in f.replace('\\', '/'))
    cpus = sorted(glob.glob(os.path.join(raw, 'cpu_%s_*.jsonl' % label)))
    if files:
        samples[label] = build_sample(label, files[-1], cpus[-1] if cpus else None)

res = {'study': 'W11 ingest format (bench/gpu/ingest)', 'format': '.1fg v1', 'samples': samples}
dl = os.path.join(here, 'download_results.json')
if os.path.exists(dl):
    res['download'] = json.load(open(dl))
for f in sorted(glob.glob(os.path.join(raw, 'shuffle_*.jsonl'))):
    lab = os.path.basename(f).split('_')[1] if False else os.path.basename(f)[len('shuffle_'):].rsplit('_', 1)[0]
    res.setdefault('sort_matters', {})[lab] = {r['run']: {'out_bytes_per_pair': r['out_bytes_per_pair'], 'section_bytes_per_pair': {k: v / r['pairs_out'] for k, v in r['section_bytes'].items()},
                                                          'cpu_s_per_Mpairs': r['cpu_s_per_Mpairs']} for r in load(f)}
cp = glob.glob(os.path.join(raw, 'cpupipe_*_enc.json'))
if cp:
    res['cpu_node_pipeline_enc'] = enc(json.load(open(cp[-1])))
res['recommendation'] = {
    'fetch': 'NCBI SRA Lite from the NCBI host (anonymous ranged GETs, 8 streams) to job /tmp, else full .sra from public S3, else ENA fastq.gz; decode the local file, never stream through VDB',
    'format': '.1fg v1: 32,768-pair chunks, CBUMI u64 + R2 2-bit slot + exception list, zstd frames (nvCOMP), CRC32C per section, no qualities',
    'codec': 'zstd level 3 (level 1 if CPU-node time dominates, raw if bytes are free); keep the per-block barcode sort (+17-37% bytes without it)',
    'gpu_batch': '128-256 chunks per nvCOMP batch (320-600 M pairs/s decode)',
    'status': 'prototype, measured on 2 samples (+1 fetched run for the CPU-node stage); identity 0 mismatches on 66.6 M and 121.8 M pairs',
}
os.makedirs(os.path.dirname(out), exist_ok=True)
json.dump(res, open(out, 'w'), indent=1)
print('wrote', out)

# --- tables
for lab, S in samples.items():
    print('\n### %s (job %s)' % (lab, S['job']))
    print('transform (1 thread unless named):')
    for k, v in S['transform'].items():
        if v:
            print('  %-34s out/in %.3f  %.2f B/pair  cpu %.4f core-s/Mpair  %.2f Mpair/s  split %s' % (k, v['out_over_in'], v['out_bytes_per_pair'], v['cpu_s_per_Mpairs'], v['Mpairs_per_s'], v['cpu_split_per_Mpairs']))
    print('gpu decode (resident):')
    for k, v in S['gpu_decode'].items():
        if v:
            print('  %-34s %.1f Mpair/s  %.2f GB/s decoded  ok=%s' % (k, v['Mpairs_per_s_median'], v['decoded_GBps_median'], v['status_ok']))
    print('reader e2e:')
    for k, v in S['reader_e2e'].items():
        if v:
            print('  %-34s %.1f Mpair/s  %.0f MB/s  host %.4f core-s/Mpair (readers %.4f, gpu thread %.4f)' % (k, v['Mpairs_per_s'], v['file_MBps'], v['cpu_s_per_Mpairs'], v['cpu_reader_threads_s_per_Mpairs'], v['cpu_gpu_thread_s_per_Mpairs']))
    print('identity:', json.dumps(S['identity_vs_1fq_decode']))
    print('cpu compat:', json.dumps(S['cpu_engine_compat']))
    print('m8:', json.dumps(S['m8_1fq_baseline']))
