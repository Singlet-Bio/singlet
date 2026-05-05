#pragma once
// singlet.h — Master include for the singlet C++ library
//
// Usage:
//   #include <singlet/singlet.h>          // everything (excluding STAR)
//   #include <singlet/pz/reader.h>        // just the .1pz codec
//   #include <singlet/pz/writer.h>        // write .1pz files
//   #include <singlet/fq/reader.h>        // just the .1fq codec
//   #include <singlet/pileup/engine.h>    // just the pileup engine
//   #include <singlet/star/star_api.h>    // STAR aligner API
//
// Namespace: singlet::pz, singlet::fq, singlet::pileup
// STAR uses its own global namespace (legacy C++ codebase)

#include "singlet/pz/writer.h"
#include "singlet/pz/reader.h"
#include "singlet/fq/reader.h"
#include "singlet/fq/writer.h"
#include "singlet/pileup/pileup_engine.h"
