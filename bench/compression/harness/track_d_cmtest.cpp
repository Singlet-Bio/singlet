// Standalone round-trip + ratio + speed test of the Track D context-mixing byte coder.
//   bin/track_d_cmtest.exe file [file...]
#include "common.h"
#include "track_d.h"
#include <cstdio>
using namespace cs;
int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        auto raw = read_file(argv[i]);
        Timer t;
        auto z = dcm::compress(raw.data(), raw.size());
        double te = t.ms();
        Timer t2;
        auto back = dcm::decompress(z.data(), z.size());
        double td = t2.ms();
        std::printf("%s  raw %zu  cm %zu  ratio %.4f  enc %.0f ms (%.2f MB/s)  dec %.0f ms  %s\n", argv[i], raw.size(), z.size(),
                    (double)z.size() / raw.size(), te, raw.size() / 1e3 / te, td, back == raw ? "OK" : "MISMATCH");
    }
    return 0;
}
