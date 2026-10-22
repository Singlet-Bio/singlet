#include "idea4_core.h"
#include <cstdio>
#include <cmath>
#include <random>
#include <array>
using namespace cs::i4;
int main(){
  // exp accuracy
  double maxerr=0; std::mt19937_64 rng(1);
  for(int t=0;t<2000000;++t){
    double d = std::uniform_real_distribution<double>(0,24)(rng);
    u64 D=(u64)(d*4294967296.0);
    double ref=std::exp(-(double)D/4294967296.0);
    double got=exp_neg_q30(D)/1073741824.0;
    maxerr=std::max(maxerr,std::fabs(ref-got));
  }
  std::printf("exp max abs err %.3e (Q30 step 9.3e-10)\n",maxerr);
  // monotone
  u32 prev=1u<<30; bool mono=true; for(u64 D=0; D<(25ull<<32); D+= 12345677){ u32 e=exp_neg_q30(D); if(e>prev) mono=false; prev=e; }
  std::printf("monotone %d  e(0)=%u\n",mono,exp_neg_q30(0));
  // rans roundtrip
  RansEnc enc; std::vector<std::array<u32,3>> v;
  for(int i=0;i<1000000;++i){ int bits=1+rng()%28; u32 freq=1+rng()%((1u<<bits)>>1 ? (1u<<bits)>>1 : 1); if(freq>(1u<<bits)) freq=1<<bits; u32 start=rng()%((1u<<bits)-freq+1); enc.put(start,freq,bits); v.push_back({start,freq,(u32)bits}); }
  auto out=enc.finish(); std::printf("rans bytes %zu\n",out.size());
  RansDec dec; dec.init(out.data(),out.size()); bool ok=true;
  for(auto&s:v){ u32 sl=dec.slot(s[2]); if(sl<s[0]||sl>=s[0]+s[1]){ok=false;break;} dec.advance(s[0],s[1],s[2]); }
  std::printf("rans roundtrip %d\n",ok);
  u64 r=root2_q62(8); std::printf("2^(1/8)=%.9f\n",(double)r/4611686018427387904.0);
}
