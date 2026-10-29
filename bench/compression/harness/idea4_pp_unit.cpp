#include "idea4_pp_core.h"
#include <cstdio>
#include <random>
using namespace cs::ipp;
int main(){
  std::mt19937_64 rng(5); long bad=0, tot=0; u64 firstD=0;
  for(long t=0;t<200000000;++t){
    u64 D = rng() % (24ull<<32);
    u64 d2 = D + 1 + (rng() % (1ull<<(rng()%24)));
    u32 a=exp_neg_q30(D), b=exp_neg_q30(d2);
    ++tot; if(b>a){ if(!bad) firstD=D; ++bad; if(bad<5) std::printf("non-monotone: D=%llu e=%u  D2=%llu e=%u\n",(unsigned long long)D,a,(unsigned long long)d2,b);} 
  }
  std::printf("pairs %ld nonmonotone %ld\n",tot,bad);
}
