// star_api.h — Declaration for the bundled STAR aligner entry point.
//
// STAR source is vendored in src/star/ (MIT license, see src/star/LICENSE.STAR)
// and compiled as an OBJECT library into the singlify binary.  singlify forks
// a child process and calls star_main_impl() directly rather than exec-ing an
// external binary.  Because STAR runs in a forked child, its internal exit()
// calls terminate only the child, leaving the parent pileup process unaffected.
//
// Usage in singlify (child side of fork):
//   int rc = star_main_impl(args.size(), const_cast<char**>(args.data()));
//   _exit(rc);
#pragma once
extern "C++" int star_main_impl(int argc, char *argv[]);
