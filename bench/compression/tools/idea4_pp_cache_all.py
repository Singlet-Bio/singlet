import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import idea4_lib as L, idea4_pp_export as EX
t0 = time.time()
for org in ('human', 'mouse'):
    EX.get_ref(org)
for sid in L.ALL:
    EX.get_piece(sid, 2500, 20000)
    print(sid, 'cached', int(time.time() - t0), 's', flush=True)
