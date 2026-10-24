// quick consistency test of G()/log2fix()
#include "idea1_codec.h"
using namespace cs; using namespace cs::i1;
int main(){
  const Tabs& T = tabs();
  // E from slot vs G
  double maxerr=0;
  for (uint64_t slot = 0; slot < FCUT; slot += 9973) {
    uint64_t rem = MSC - slot;
    uint64_t E = ((uint64_t)31<<FR) - log2fix(T,(uint32_t)rem);
    uint64_t c = MSC - G(T,E);
    double d = (double)(int64_t)c - (double)slot;
    maxerr = std::max(maxerr, std::fabs(d));
  }
  std::printf("max |F(E(slot)) - slot| = %.1f units of 2^-31\n", maxerr);
  for (uint64_t x : {0ull, 1ull<<20, 1ull<<24, 3ull<<23, 5ull<<24, 15ull<<24}) std::printf("G(%llu)=%u ref=%.1f\n",(unsigned long long)x,G(T,x), std::ldexp(std::exp2(-(double)x/(1<<FR)),31));
}
