#pragma once
// singlet.h — Master include for the singlet C++ library
//
// Usage:
//   #include <singlet/singlet.h>          // everything
//   #include <singlet/pz/reader.h>        // just the .1pz codec
//   #include <singlet/fq/reader.h>        // just the .1fq codec
//   #include <singlet/pileup/engine.h>    // just the pileup engine
//
// Namespace: singlet::pz, singlet::fq, singlet::pileup, singlet::gpu

#include "singlet/pz/writer.h"
#include "singlet/pz/reader.h"
#include "singlet/fq/reader.h"
#include "singlet/fq/writer.h"
#include "singlet/pileup/pileup_engine.h"
