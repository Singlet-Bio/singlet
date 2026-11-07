import os, re, subprocess

REPO = r'C:\Users\zacha\Projects\singlet-bio\.claude\worktrees\release-gate-e1e139'
OUT = r'C:\Users\zacha\AppData\Local\Temp\claude\C--Users-zacha-Projects-singlet-bio--claude-worktrees-release-gate-e1e139\614522b8-15be-415e-92ff-c32adc0f710c\scratchpad\g3'
os.makedirs(OUT, exist_ok=True)
FILES = ['hto_rg.sbatch', 'nh_prep.sbatch', 'score_all.sbatch', 'atacsp_rg.sh', 'run_sample_rg.sbatch', 'pyjob_rg.sbatch', 'ambient_prep.sbatch']


def show(f):
    b = subprocess.check_output(['git', '-C', REPO, 'show', 'origin/prod/campaign:bench/release_gate/' + f])
    return b.decode('utf-8').replace('\r\n', '\n')


for f in FILES:
    s = show(f)
    s = s.replace('d9834b8a', '3eaa041c').replace('22b5782f', '3eaa041c')
    # separate the perf harness tree (src is shared with gate 1 through a symlink)
    s = s.replace('$RG/src/perf', '$RG/src_perf').replace('release-gate/src/perf', 'release-gate2/src_perf')
    s = s.replace('$WORK/release-gate/', '$WORK/release-gate2/')
    s = re.sub(r'singlet/release-gate/', 'singlet/release-gate2/', s)
    s = s.replace('/release-gate/', '/release-gate2/')
    s = s.replace('\\$WORK/release-gate/', '\\$WORK/release-gate2/')
    open(os.path.join(OUT, f), 'w', encoding='utf-8', newline='\n').write(s)
    left = [l for l in s.split('\n') if re.search(r'release-gate(?!2)', l) and not l.lstrip().startswith('#')]
    print(f, 'remaining release-gate (not 2):', len(left))
    for l in left[:3]:
        print('   ', l[:150])
