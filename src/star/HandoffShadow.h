#pragma once
// SINGLET_STAR_SHADOW=1: shadow mode for the STAR bypass of hand-off reads (docs/research-beta/star-handoff-bypass.md). It changes no output.
// The live per-read pass (ReadAlign::outputHandoffSummary) keeps working; next to it each thread adds up what that pass did to the run statistics
// (the difference of statsRA around the stats section), and after mapping the same counters are computed in bulk from the per-record summaries
// and the per-read record / clip arrays alone. Both sets are compared and logged to Log.out ("SHADOW ...").
#include "IncludeDefine.h"
#include "Stats.h"
#include "Parameters.h"

extern bool g_shadowOn;
//per hand-off record the length of its (clipped) sequence; set by SeqCacheGpuPrefill.cpp when the summaries are installed
extern const uint32 *g_handoffRecLen;

void shadowAddLive(const Stats &before, const Stats &after);
void shadowCompare(Parameters &P);
