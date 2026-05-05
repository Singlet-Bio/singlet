"""Deep exploration of TAR files - quick version with just file listings."""
import pandas as pd
import os, re, tarfile, io, gzip, urllib.request
from collections import Counter

CAT_PATH = "/mnt/projects/debruinz_project/cellarium/catalog/processing_catalog.parquet"
DATASET_DIR = "/mnt/projects/debruinz_project/cellarium/dataset"
TMPDIR = "/tmp/2c_explore"
os.makedirs(TMPDIR, exist_ok=True)

cat = pd.read_parquet(CAT_PATH)

# Get unique GSEs with processed data AND tar files 
processed_gsms = set()
gsm_to_gse = {}
for gse_d in os.listdir(DATASET_DIR):
    gse_path = os.path.join(DATASET_DIR, gse_d)
    if not os.path.isdir(gse_path): continue
    for gsm_d in os.listdir(gse_path):
        if gsm_d.startswith('GSM') and os.path.exists(os.path.join(gse_path, gsm_d, 'cells.parquet')):
            processed_gsms.add(gsm_d)
            gsm_to_gse[gsm_d] = gse_d

proc = cat[cat['gsm_id'].isin(processed_gsms)]

# Find unique TAR URLs (deduplicate since same tar shows up for every GSM in a GSE)
tar_urls = {}  # gse -> set of tar urls
for _, row in proc.iterrows():
    urls_str = row.get('supplementary_files', '')
    if pd.isna(urls_str): continue
    for url in str(urls_str).split(';'):
        url = url.strip()
        fname = url.split('/')[-1].lower()
        if fname.endswith('.tar') or fname.endswith('.tar.gz'):
            tar_urls.setdefault(row['gse_id'], set()).add(url)

unique_tars = set()
for gse, urls in tar_urls.items():
    unique_tars.update(urls)
print(f"Unique GSEs with TARs: {len(tar_urls)}")
print(f"Unique TAR files: {len(unique_tars)}")

# Sample 5 GSEs, download tars, list contents
sample_gses = list(tar_urls.keys())[:5]
all_inner_exts = Counter()
per_gsm_inner_files = Counter()  # How many GSM-specific files vs shared
meta_re = re.compile(r'meta|annot|obs|cell[_.]?type|cluster|label|barcode|pheno|coldata|ident', re.I)

for gse in sample_gses:
    for tar_url in list(tar_urls[gse])[:1]:
        fname = tar_url.split('/')[-1]
        url = tar_url.replace('ftp://ftp.ncbi.nlm.nih.gov/', 'https://ftp.ncbi.nlm.nih.gov/')
        print(f"\n--- {gse}: {fname} ---")
        
        outfile = os.path.join(TMPDIR, f"{gse}_{fname}")
        try:
            if not os.path.exists(outfile):
                # Check size first
                req = urllib.request.Request(url, method='HEAD')
                resp = urllib.request.urlopen(req, timeout=15)
                size = int(resp.headers.get('Content-Length', 0))
                if size > 50_000_000:  # >50MB, just get listing via partial download
                    print(f"  Large file ({size/1e6:.0f}MB), partial download for listing")
                    req = urllib.request.Request(url)
                    req.add_header('Range', 'bytes=0-2097152')  # 2MB
                    resp = urllib.request.urlopen(req, timeout=30)
                    with open(outfile, 'wb') as f:
                        f.write(resp.read())
                else:
                    print(f"  Downloading ({size/1e6:.1f}MB)...")
                    urllib.request.urlretrieve(url, outfile)
            
            # List contents
            mode = 'r:gz' if fname.endswith('.gz') else 'r'
            try:
                tf = tarfile.open(outfile, mode)
            except tarfile.ReadError:
                tf = tarfile.open(outfile, 'r|' if not fname.endswith('.gz') else 'r|gz')
            
            members = []
            gsm_pattern = re.compile(r'GSM\d+')
            for i, m in enumerate(tf):
                if i > 100: break
                members.append((m.name, m.size))
                # Count per-extension
                bn = m.name.split('/')[-1].lower()
                for ce in ['.gz', '.bz2']:
                    if bn.endswith(ce):
                        bn = bn[:-len(ce)]
                        break
                ext = '.' + bn.rsplit('.', 1)[-1] if '.' in bn else 'noext'
                all_inner_exts[ext] += 1
                # Is this GSM-specific?
                if gsm_pattern.search(m.name):
                    per_gsm_inner_files['gsm_specific'] += 1
                else:
                    per_gsm_inner_files['shared'] += 1
            tf.close()
            
            print(f"  {len(members)} entries (first 100):")
            for name, size in members[:15]:
                marker = " [META]" if meta_re.search(name) else ""
                has_gsm = " [GSM-SPECIFIC]" if gsm_pattern.search(name) else ""
                print(f"    {name[:60]:60s} {size:>10d}{marker}{has_gsm}")
            if len(members) > 15:
                print(f"    ... ({len(members)-15} more)")
                
        except Exception as e:
            print(f"  Error: {e}")

print("\n=== INNER FILE EXTENSIONS ACROSS ALL SAMPLED TARS ===")
for ext, count in all_inner_exts.most_common(15):
    print(f"  {ext:15s} {count:>5d}")

print(f"\n=== GSM-SPECIFIC vs SHARED FILES ===")
print(f"  GSM-specific: {per_gsm_inner_files.get('gsm_specific', 0)}")
print(f"  Shared:       {per_gsm_inner_files.get('shared', 0)}")

# --- Now explore xlsx ---
print("\n\n" + "=" * 80)
print("XLSX EXPLORATION")
print("=" * 80)
xlsx_items = []
for _, row in proc.iterrows():
    urls_str = row.get('supplementary_files', '')
    if pd.isna(urls_str): continue
    for url in str(urls_str).split(';'):
        if '.xlsx' in url.lower():
            xlsx_items.append((row['gsm_id'], row['gse_id'], url))

print(f"\nXLSX files for processed GSMs: {len(xlsx_items)}")
# Show unique filenames
xlsx_fnames = Counter(url.split('/')[-1] for _, _, url in xlsx_items)
print("Unique xlsx filenames:")
for name, count in xlsx_fnames.most_common(15):
    print(f"  {name[:70]:70s} {count:>5d}")

# Download and peek at a few different xlsx files
import subprocess
seen_gses = set()
for gsm, gse, url in xlsx_items:
    if gse in seen_gses: continue
    seen_gses.add(gse)
    if len(seen_gses) > 3: break
    fname = url.split('/')[-1]
    https_url = url.replace('ftp://ftp.ncbi.nlm.nih.gov/', 'https://ftp.ncbi.nlm.nih.gov/')
    outfile = os.path.join(TMPDIR, fname)
    print(f"\n  {gse}/{fname}:")
    try:
        if not os.path.exists(outfile):
            urllib.request.urlretrieve(https_url, outfile)
        # Try reading with pandas
        try:
            df = pd.read_excel(outfile, engine='openpyxl', nrows=5)
        except ImportError:
            df = pd.read_excel(outfile, nrows=5)
        print(f"    Shape: {df.shape}")
        print(f"    Columns: {list(df.columns[:15])}")
        # Check if any column looks like barcodes
        for col in df.columns:
            vals = df[col].dropna().astype(str)
            if any(re.match(r'^[ACGTN]{12,18}(-\d+)?$', v) for v in vals[:5]):
                print(f"    BARCODE COLUMN: {col}")
        print(f"    First row: {dict(list(df.iloc[0].items())[:8])}")
    except Exception as e:
        print(f"    Error: {e}")

# --- Now explore h5seurat more ---
print("\n\n" + "=" * 80)
print("H5SEURAT DEEP EXPLORATION")
print("=" * 80)
import h5py
h5s_items = []
for _, row in proc.iterrows():
    urls_str = row.get('supplementary_files', '')
    if pd.isna(urls_str): continue
    for url in str(urls_str).split(';'):
        if '.h5seurat' in url.lower() or '.h5Seurat' in url:
            h5s_items.append((row['gsm_id'], row['gse_id'], url))

print(f"H5Seurat files: {len(h5s_items)}")
seen = set()
for gsm, gse, url in h5s_items:
    if gse in seen: continue
    seen.add(gse)
    if len(seen) > 2: break
    fname = url.split('/')[-1]
    https_url = url.replace('ftp://ftp.ncbi.nlm.nih.gov/', 'https://ftp.ncbi.nlm.nih.gov/')
    outfile = os.path.join(TMPDIR, fname)
    print(f"\n  {gse}/{fname}:")
    try:
        if not os.path.exists(outfile):
            urllib.request.urlretrieve(https_url, outfile)
        with h5py.File(outfile, 'r') as f:
            if 'meta.data' in f:
                meta = f['meta.data']
                print(f"    meta.data keys: {list(meta.keys())}")
                # Get barcodes/index
                if '_index' in meta:
                    idx = meta['_index'][:5]
                    print(f"    _index (barcodes): {[v.decode() if isinstance(v, bytes) else v for v in idx]}")
                n_cells = len(meta['_index']) if '_index' in meta else 'unknown'
                print(f"    n_cells: {n_cells}")
                # Show sample values for each metadata column
                for k in list(meta.keys())[:10]:
                    d = meta[k]
                    if isinstance(d, h5py.Group):
                        # Factor variable
                        levels = [v.decode() if isinstance(v, bytes) else v for v in d['levels'][:]]
                        print(f"    {k}: Factor with levels {levels[:5]}{'...' if len(levels)>5 else ''}")
                    elif isinstance(d, h5py.Dataset):
                        vals = d[:3]
                        print(f"    {k}: {[v.decode() if isinstance(v, bytes) else v for v in vals]}")
            if 'cell.names' in f:
                cn = f['cell.names'][:3]
                print(f"    cell.names: {[v.decode() if isinstance(v, bytes) else v for v in cn]}")
    except Exception as e:
        print(f"    Error: {e}")

# --- rdata/rda exploration ---
print("\n\n" + "=" * 80)
print("RDATA/RDA EXPLORATION")
print("=" * 80)
rdata_items = []
for _, row in proc.iterrows():
    urls_str = row.get('supplementary_files', '')
    if pd.isna(urls_str): continue
    for url in str(urls_str).split(';'):
        fname = url.split('/')[-1].lower()
        if fname.endswith('.rdata.gz') or fname.endswith('.rda.gz') or fname.endswith('.robj.gz'):
            rdata_items.append((row['gsm_id'], row['gse_id'], url))
print(f"RData/RDA/RObj files: {len(rdata_items)}")
rdata_fnames = Counter(url.split('/')[-1] for _, _, url in rdata_items)
print("Unique filenames:")
for name, count in rdata_fnames.most_common(10):
    print(f"  {name[:70]:70s} {count:>5d}")

print("\nDone.")
