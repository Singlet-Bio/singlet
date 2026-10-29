"""Check the proposed PSI fix: patched C++ (donor priority) == python donor-first recipe bitwise; basis-point member roundtrip."""
import sys, os, subprocess, numpy as np
sys.path.insert(0, 'tools'); sys.path.insert(0, '../../python')
from w4_bugs_psi_recipe import *
for gsm in sys.argv[1:]:
    D = root + gsm + '/'; pre = f'corpus_w4_bugs/psi_fixed_{gsm}'
    subprocess.run([os.path.abspath('bin/w4_bugs_psi_fix.exe'), D + 'sj_counts.1pz', pre], check=True, capture_output=True)
    sj = read_1pz(D + 'sj_counts.1pz'); ev = build_events(sj['rownames']); je2 = assign_donor_first(ev, sj['m'])
    f2, _, _ = psi_f32(sj, je2)
    cpp = np.fromfile(pre + '.fix_f32.bin', dtype=np.float32)
    bp = read_1pz(pre + '.fix_bp.1pz'); v = np.asarray(bp['data']).astype(np.float64) / int(bp['user_kv']['value_scale'])
    ship = np.asarray(read_1pz(D + 'splice_psi.1pz')['data'])
    print(gsm, 'patched C++ == python donor-first (bitwise):', bool((cpp.view(np.uint32) == f2.view(np.uint32)).all()),
          '| vt_code', bp['vt_code'], 'kv', bp['user_kv'], '| max |stored/10000 - PSI| = %.2e' % np.abs(v - cpp).max(),
          '| distinct stored', len(np.unique(bp['data'])), '| file bytes', os.path.getsize(pre + '.fix_bp.1pz'), 'vs shipped', os.path.getsize(D + 'splice_psi.1pz'),
          '| explicit zeros', int((np.asarray(bp['data']) == 0).sum()))
