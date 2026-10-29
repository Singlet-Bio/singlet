"""SAM fields of a STAR transcript, as STAR writes them for a pair: FLAG, POS, CIGAR, PNEXT, per mate.
A transcription of ReadAlign_outputTranscriptSAM.cpp:57-190 (the CIGAR logic is the same in ReadAlign_calcCIGAR.cpp:10-60 and
ReadAlign_alignBAM.cpp:224-275), with the clipping of readLoad.cpp. Used by the tests and by compare_star_sam.py.

exons: [(R, G, L, mate)], canon: canonSJ between consecutive exons (length nExons-1), Str: the transcript strand, chrStart: start of
the chromosome in the concatenated genome. readLength: clipped lengths, readLengthOriginal: lengths in the FASTQ, clip5/clip3: bases
clipped at the 5' and 3' end of each mate (clipMates[mate][0].clippedN and [1]).
"""


def sam_records(exons, canon, Str, chrStart, readLength, readLengthOriginal, clip5, clip3, Lread, primary=True,
                protrude=0, concordant_pair=False, sj_annot=None):
    n = len(exons)
    iExMate = next((i for i in range(n - 1) if canon[i] == -3), n - 1)
    nMates = 2 if iExMate < n - 1 else 1
    flagCommon = 0x1                                    # paired reads (:73)
    if nMates == 1:
        flagCommon |= 0x8                               # :76, mateChr is never set here: not mapped as pair
    else:                                               # :79-83
        e = exons
        if (concordant_pair and protrude > 0) or (
                e[0][1] <= e[iExMate + 1][1] + e[0][0] and e[iExMate][1] + e[iExMate][2] <= e[n - 1][1] + Lread - e[n - 1][0]):
            flagCommon |= 0x2
    leftMate = Str                                      # :94-96
    recs = []
    for imate in range(nMates):
        iEx1 = 0 if imate == 0 else iExMate + 1
        iEx2 = iExMate if imate == 0 else n - 1
        Mate = exons[iEx1][3]
        flag = flagCommon
        if Mate == 0:                                   # :112-114
            flag |= Str * 0x10
            if nMates == 2:
                flag |= (1 - Str) * 0x20
        else:
            flag |= (1 - Str) * 0x10
            if nMates == 2:
                flag |= Str * 0x20
        flag |= 0x40 if Mate == 0 else 0x80
        if not primary:
            flag |= 0x100
        if Str == 0 and Mate == 0:                      # :136-142
            trimL = clip5[Mate]
        elif Str == 0 and Mate == 1:
            trimL = clip3[Mate]
        elif Str == 1 and Mate == 0:
            trimL = clip3[Mate]
        else:
            trimL = clip5[Mate]
        R1 = exons[iEx1][0]
        trimL1 = trimL + R1 - (0 if R1 < readLength[leftMate] else readLength[leftMate] + 1)   # :145
        cig = f"{trimL1}S" if trimL1 > 0 else ""
        for ii in range(iEx1, iEx2 + 1):                # :146-168
            if ii > iEx1:
                gapG = exons[ii][1] - (exons[ii - 1][1] + exons[ii - 1][2])
                gapR = exons[ii][0] - exons[ii - 1][0] - exons[ii - 1][2]
                if gapR > 0:
                    cig += f"{gapR}I"
                if canon[ii - 1] >= 0 or (sj_annot is not None and sj_annot[ii - 1] == 1):
                    cig += f"{gapG}N"
                elif gapG > 0:
                    cig += f"{gapG}D"
            cig += f"{exons[ii][2]}M"
        first = exons[iEx1][0]
        trimR1 = (readLengthOriginal[leftMate] if first < readLength[leftMate]
                  else readLength[leftMate] + 1 + readLengthOriginal[Mate]) - exons[iEx2][0] - exons[iEx2][2] - trimL   # :184-186
        if trimR1 > 0:
            cig += f"{trimR1}S"
        recs.append(dict(mate=Mate, flag=flag, pos=exons[iEx1][1] - chrStart + 1, cigar=cig))
    if nMates == 2:                                     # PNEXT: the other mate's POS (RNEXT "=")
        recs[0]["pnext"], recs[1]["pnext"] = recs[1]["pos"], recs[0]["pos"]
    return recs
