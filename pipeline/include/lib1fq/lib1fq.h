// lib1fq/lib1fq.h — Single include for the .1fq library
//
// Header-only C++17 library for reading and writing .1fq files.
// Depends on: zstd (required), lz4 (optional, via HAS_LZ4 define)

#pragma once

#include "types.h"
#include "packing.h"
#include "compress.h"
#include "writer.h"
#include "reader.h"
#include "protocol.h"
#include "manifest.h"
#include "dedup.h"
#include "folding.h"
#include "packed_align.h"
#include "benchmark.h"
