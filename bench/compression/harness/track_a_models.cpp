// Track A: write the shared model files (results/track_a_vocab_<build>.bin, results/track_a_whitelist_<name>.bin).
//
//   track_a_models vocab <build> <kind>=<file.1pz> [<kind>=<file.1pz> ...]
//   track_a_models whitelist <name> <whitelist.txt>
#include "common.h"
#include "track_a.h"
#include <iostream>

using namespace cs;

int main(int argc, char** argv) {
    if (argc < 4) { std::cerr << "usage: see source\n"; return 2; }
    std::string cmd = argv[1];
    if (cmd == "vocab") {
        std::string build = argv[2];
        std::vector<std::pair<std::string, std::vector<std::string>>> lists;
        for (int i = 3; i < argc; ++i) {
            std::string a = argv[i];
            auto eq = a.find('=');
            auto M = load_1pz(a.substr(eq + 1));
            lists.push_back({a.substr(0, eq), M.rownames});
            std::printf("%s: %zu names, raw %zu bytes\n", a.substr(0, eq).c_str(), M.rownames.size(), ta::join_nul(M.rownames).size());
        }
        auto bytes = ta::build_vocab_file(lists);
        std::string out = "results/track_a_vocab_" + build + ".bin";
        write_file(out, bytes);
        std::printf("wrote %s: %zu bytes\n", out.c_str(), bytes.size());
    } else if (cmd == "whitelist") {
        std::string name = argv[2];
        std::ifstream f(argv[3]);
        std::string line;
        std::vector<uint32_t> codes;
        while (std::getline(f, line)) {
            if (line.size() < 16) continue;
            uint32_t v;
            if (ta::bc_code(line.substr(0, 16), v)) codes.push_back(v);
        }
        auto bytes = ta::build_whitelist_file(name, codes);
        std::string out = "results/track_a_whitelist_" + name + ".bin";
        write_file(out, bytes);
        std::printf("wrote %s: %zu barcodes -> %zu bytes (%.2f bits/barcode)\n", out.c_str(), codes.size(), bytes.size(),
                    bytes.size() * 8.0 / codes.size());
    }
    return 0;
}
