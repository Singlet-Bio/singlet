set -u
D=/tmp/cptoy_dz; cd $D
B=${BB:-/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-runtime/bin/t2-chimprof-2/singlet-gpu}
PY=/mnt/projects/debruinz_project/debruinz/singlet/envs/pbench/bin/python3
if [ ! -s $D/g2/SA ]; then
$PY - <<'PYEOF'
import random
random.seed(7)
L={}
cur=None
for l in open('/tmp/cptoy_dz/g.fa'):
    if l[0]=='>': cur=l[1:].strip(); L[cur]=[]
    else: L[cur].append(l.strip())
g={k:''.join(v) for k,v in L.items()}
with open('/tmp/cptoy_dz/a.gtf','w') as f:
    n=0
    for c,s in g.items():
        for p in range(1000,len(s)-5000,20000):
            n+=1
            a='gene_id "G%d"; transcript_id "T%d"; gene_name "g%d"; gene_biotype "protein_coding";'%(n,n,n)
            f.write('%s\ttoy\tgene\t%d\t%d\t.\t+\t.\t%s\n'%(c,p,p+3000,a))
            f.write('%s\ttoy\ttranscript\t%d\t%d\t.\t+\t.\t%s\n'%(c,p,p+3000,a))
            f.write('%s\ttoy\texon\t%d\t%d\t.\t+\t.\t%s exon_number "1";\n'%(c,p,p+800,a))
            f.write('%s\ttoy\texon\t%d\t%d\t.\t+\t.\t%s exon_number "2";\n'%(c,p+1500,p+3000,a))
wl=[''.join(random.choice('ACGT') for _ in range(16)) for _ in range(3000)]
open('/tmp/cptoy_dz/wl.txt','w').write('\n'.join(wl)+'\n')
b=open('/tmp/cptoy_dz/b.fq','w')
for i in range(300000):
    s=random.choice(wl)+''.join(random.choice('ACGT') for _ in range(10))
    if random.random()<0.1: s=s[:3]+'N'+s[4:]
    b.write('@%d\n%s\n+\n%s\n'%(i,s,'I'*26))
PYEOF
mkdir -p $D/g2
timeout 50 $B star --runMode genomeGenerate --genomeDir $D/g2 --genomeFastaFiles $D/g.fa --sjdbGTFfile $D/a.gtf --sjdbOverhang 99 --genomeSAindexNbases 8 --runThreadN 2 --outFileNamePrefix $D/gen2_ > $D/gen2.out 2>&1; echo "gen2 rc=$?"
fi
mkdir -p $D/o2
SINGLET_CHIM_SAMPLE=$D/o2/q SINGLET_CHIM_SAMPLE_MODE=${MODE:-bt} SINGLET_CHIM_SAMPLE_HZ=${HZV:-100} timeout 80 $B star --runMode alignReads --genomeDir $D/g2 --soloType CB_UMI_Simple --soloCBwhitelist $D/wl.txt --soloCBlen 16 --soloCBstart 1 --soloUMIstart 17 --soloBarcodeReadLength 0 --soloCellReadStats Standard --soloUMIlen 10 --clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTACATGGG --clip3pAdapterMMp 0.1 --outFilterScoreMin 10 --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --runThreadN ${TH:-8} --genomeLoad NoSharedMemory --readFilesIn $D/r.fq $D/b.fq --soloFeatures Gene --outSAMtype None --outFileNamePrefix $D/o2/c_ --chimSegmentMin 10 --chimJunctionOverhangMin 10 --chimScoreMin 1 --chimScoreDropMax 30 --chimScoreJunctionNonGTAG 0 --chimScoreSeparation 1 --alignSJstitchMismatchNmax 5 -1 5 5 --chimSegmentReadGapMax 3 --chimMultimapNmax 50 --chimOutType Junctions --chimOutJunctionFormat 1 > $D/o2/out.txt 2> $D/o2/err.txt
echo "run rc=$?"; tail -4 $D/o2/err.txt | cut -c1-200; tail -4 $D/o2/c_Log.out | cut -c1-200; ls $D/o2 | tr '\n' ' '
