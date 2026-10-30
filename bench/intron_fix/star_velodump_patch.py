"""Patch STAR 2.7.11b SoloFeature_countVelocyto.cpp: dump every UMI's Velocyto call when STAR_VELO_DUMP is set.

Line format: CB  UMI  gene_id|-  label   (S/U/A counted; M multi-gene UMI, E empty transcript intersection: not counted)
The counting itself is unchanged (the run's Velocyto matrices must equal an unpatched run's).
"""
import sys

p = sys.argv[1]
s = open(p).read()


def rep(old, new):
    global s
    assert s.count(old) == 1, old
    s = s.replace(old, new)


rep('#include "systemFunctions.h"', '#include "systemFunctions.h"\n#include <cstdio>\n#include <cstdlib>')
rep('''    for (uint32 iCB=0; iCB<nCB; iCB++) {//main collapse cycle
        map<uint32,array<uint32,3>> geneC;
        for (auto &umi : cuTrTypes[iCB]) {//cycle over UMIs
            if (umi.second.empty()) //no transcripts in the intesect
                continue;''', '''    const char *velDumpPath = getenv("STAR_VELO_DUMP");
    FILE *velDump = velDumpPath ? fopen(velDumpPath, "w") : NULL;
    for (uint32 iCB=0; iCB<nCB; iCB++) {//main collapse cycle
        map<uint32,array<uint32,3>> geneC;
        for (auto &umi : cuTrTypes[iCB]) {//cycle over UMIs
            if (umi.second.empty()) {//no transcripts in the intesect
                if (velDump) fprintf(velDump, "%s\\t%s\\t-\\tE\\n", pSolo.cbWLstr[indCB[iCB]].c_str(), convertNuclInt64toString(umi.first, pSolo.umiL).c_str());
                continue;
            };''')
rep('''            if (geneI+1==0) //multigene
                continue;''', '''            if (geneI+1==0) {//multigene
                if (velDump) fprintf(velDump, "%s\\t%s\\t-\\tM\\n", pSolo.cbWLstr[indCB[iCB]].c_str(), convertNuclInt64toString(umi.first, pSolo.umiL).c_str());
                continue;
            };
            char velLab;''')
rep('''                geneC[geneI][0]++; //spliced ''', '''                geneC[geneI][0]++; velLab='S'; //spliced ''')
rep('''                geneC[geneI][1]++;//unspliced''', '''                geneC[geneI][1]++; velLab='U';//unspliced''')
rep('''                geneC[geneI][2]++;//ambiguous''', '''                geneC[geneI][2]++; velLab='A';//ambiguous''')
rep('''            nUMIperCB[iCB]++;
        };''', '''            if (velDump) fprintf(velDump, "%s\\t%s\\t%s\\t%c\\n", pSolo.cbWLstr[indCB[iCB]].c_str(), convertNuclInt64toString(umi.first, pSolo.umiL).c_str(), Trans.geID[geneI].c_str(), velLab);
            nUMIperCB[iCB]++;
        };''')
rep('''    //for now only unique mappers are considered for velocyto''', '''    if (velDump) fclose(velDump);
    //for now only unique mappers are considered for velocyto''')
open(p, 'w').write(s)
print('patched', p)
