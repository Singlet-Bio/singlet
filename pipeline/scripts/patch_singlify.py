#!/usr/bin/env python3
"""Patch singlify.cpp to add 3-read ATAC support."""
import sys

path = '/mnt/home/debruinz/Singlet-AI/singlify/src/singlify.cpp'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

changes = 0

# 1. Add reads_r3 variable
old1 = 'static int cmd_encode(int argc, char* argv[]) {\n    std::string reads_r1, reads_r2;\n    std::string output_path;'
new1 = 'static int cmd_encode(int argc, char* argv[]) {\n    std::string reads_r1, reads_r2, reads_r3;  // r3 optional (ATAC 3-read)\n    std::string output_path;'
if old1 in text:
    text = text.replace(old1, new1, 1); changes += 1
else:
    print("ERROR: change 1 not found"); sys.exit(1)

# 2. Peek for R3 in --reads parsing
old2 = '            reads_r1 = argv[++i]; reads_r2 = argv[++i];\n        }\n        else if ((arg == "-o" || arg == "--output") && i+1 < argc)'
new2 = '            reads_r1 = argv[++i]; reads_r2 = argv[++i];\n            if (i+1 < argc && argv[i+1][0] != \'-\') reads_r3 = argv[++i];\n        }\n        else if ((arg == "-o" || arg == "--output") && i+1 < argc)'
if old2 in text:
    text = text.replace(old2, new2, 1); changes += 1
else:
    print("ERROR: change 2 not found"); sys.exit(1)

# 3. encode_atac dispatch in cmd_encode
old3 = '    lib1fq::FastqEncoder encoder;\n    auto stats = encoder.encode(reads_r1, reads_r2, ecfg);\n'
new3 = '''    lib1fq::FastqEncoder encoder;

    if (!reads_r3.empty()) {
        // 3-read ATAC mode: R1 genomic + R2 barcode + R3 genomic
        auto stats = encoder.encode_atac(reads_r1, reads_r2, reads_r3, ecfg);
        std::cerr << "\\n[singlify-encode] Done: " << stats.total_reads << " reads"
                  << ", " << stats.blocks_written << " blocks (ATAC 3-read)\\n";
        std::cerr << "[singlify-encode] Output: " << output_path << "\\n";
        return 0;
    }

    auto stats = encoder.encode(reads_r1, reads_r2, ecfg);
'''
if old3 in text:
    text = text.replace(old3, new3, 1); changes += 1
else:
    print("ERROR: change 3 not found"); sys.exit(1)

# 4. Add has_i2_stream detection in ATAC BC dict section
old4 = (
    '            // ATAC: pre-load BC dict for QNAME injection (stream[0]=R1 genomic,\n'
    '            // stream[1]=R3 genomic; barcode is in the per-read BC dict slot).\n'
    '            std::vector<uint8_t> atac_bc_dict_bytes;\n'
    '            uint16_t atac_bc_len = 0;\n'
    '            if (is_atac_mode && has_bc_dict && bc_dict_n > 0) {\n'
    '                atac_bc_dict_bytes = reader.bc_dict_flat();\n'
    '                atac_bc_len        = reader.bc_length();\n'
    '                std::cerr << "[1fq-decode] ATAC: BC dict " << bc_dict_n\n'
    '                          << " entries \xc3\x97 " << atac_bc_len << "bp\\n";\n'
    '            }'
)
new4 = (
    '            // ATAC: barcode via I2 stream (3-read encode) or BC dict.\n'
    '            std::vector<uint8_t> atac_bc_dict_bytes;\n'
    '            uint16_t atac_bc_len = 0;\n'
    '            bool has_i2_stream = is_atac_mode &&\n'
    '                                 (reader.header().reserved[0] & lib1fq::ExtraStreams::HAS_I2) != 0;\n'
    '            if (has_i2_stream) {\n'
    '                atac_bc_len = reader.header().stream_lengths[2];\n'
    '                std::cerr << "[1fq-decode] ATAC: I2 barcode stream " << atac_bc_len << "bp\\n";\n'
    '            } else if (is_atac_mode && has_bc_dict && bc_dict_n > 0) {\n'
    '                atac_bc_dict_bytes = reader.bc_dict_flat();\n'
    '                atac_bc_len        = reader.bc_length();\n'
    '                std::cerr << "[1fq-decode] ATAC: BC dict " << bc_dict_n\n'
    '                          << " entries \xc3\x97 " << atac_bc_len << "bp\\n";\n'
    '            }'
)
if old4 in text:
    text = text.replace(old4, new4, 1); changes += 1
else:
    print("ERROR: change 4 not found (ATAC BC dict section)")
    # Try to find it
    idx = text.find('ATAC: pre-load BC dict')
    print("  'ATAC: pre-load BC dict' at:", idx)
    sys.exit(1)

# 5. Add I2 stream to ATAC QNAME injection
# Find the QNAME block by its unique start
start_marker = (
    '                    for (uint32_t rep = 0; rep < repeat; ++rep) {\n'
    '                        // Build ATAC QNAME prefix "BC:BARCODE_" when in ATAC mode.\n'
    '                        // Barcode comes from the per-read BC dict slot (fb.bc_indices[i]).\n'
    '                        char atac_qname_prefix[32] = {}; // "BC:XXXXXXXXXXXXXXXX_" max 23 chars\n'
    '                        if (is_atac_mode) {'
)
end_marker = '\n                        }\n\n                        // R1 record'

idx_s = text.find(start_marker)
idx_e = text.find(end_marker, idx_s)
if idx_s < 0 or idx_e < 0:
    print("ERROR: could not find QNAME block markers (start={}, end={})".format(idx_s, idx_e))
    sys.exit(1)

old5 = text[idx_s:idx_e]
new5 = (
    '                    for (uint32_t rep = 0; rep < repeat; ++rep) {\n'
    '                        // Build ATAC QNAME prefix "BC:BARCODE_" when in ATAC mode.\n'
    '                        // Priority: I2 stream > BC dict > N-pad.\n'
    '                        char atac_qname_prefix[32] = {}; // "BC:XXXXXXXXXXXXXXXX_" max 23 chars\n'
    '                        if (is_atac_mode) {\n'
    '                            bool bc_set = false;\n'
    '                            if (has_i2_stream && !blk.i2_data.empty() &&\n'
    '                                i < blk.i2_lengths.size() && blk.i2_lengths[i] > 0) {\n'
    '                                const uint8_t* bcp = blk.i2_seq(i);\n'
    '                                uint16_t blen = blk.i2_len(i);\n'
    '                                int np = 0;\n'
    "                                atac_qname_prefix[np++] = 'B';\n"
    "                                atac_qname_prefix[np++] = 'C';\n"
    "                                atac_qname_prefix[np++] = ':';\n"
    '                                for (uint16_t j = 0; j < blen; ++j)\n'
    '                                    atac_qname_prefix[np++] = NUM_TO_ASCII[bcp[j]];\n'
    "                                atac_qname_prefix[np++] = '_';\n"
    "                                atac_qname_prefix[np]   = '\\0';\n"
    '                                bc_set = true;\n'
    '                            }\n'
    '                            if (!bc_set) {\n'
    '                                uint32_t bc_idx = (!fb.bc_indices.empty() && i < fb.bc_indices.size())\n'
    '                                                  ? fb.bc_indices[i] : UINT32_MAX;\n'
    '                                if (!atac_bc_dict_bytes.empty() && atac_bc_len > 0 &&\n'
    '                                    bc_idx < bc_dict_n) {\n'
    '                                    const uint8_t* bcp = atac_bc_dict_bytes.data() +\n'
    '                                                         static_cast<size_t>(bc_idx) * atac_bc_len;\n'
    '                                    int np = 0;\n'
    "                                    atac_qname_prefix[np++] = 'B';\n"
    "                                    atac_qname_prefix[np++] = 'C';\n"
    "                                    atac_qname_prefix[np++] = ':';\n"
    '                                    for (uint16_t j = 0; j < atac_bc_len; ++j)\n'
    '                                        atac_qname_prefix[np++] = NUM_TO_ASCII[bcp[j]];\n'
    "                                    atac_qname_prefix[np++] = '_';\n"
    "                                    atac_qname_prefix[np]   = '\\0';\n"
    '                                    bc_set = true;\n'
    '                                }\n'
    '                            }\n'
    '                            if (!bc_set) {\n'
    '                                int np = sprintf(atac_qname_prefix, "BC:NNNNNNNNNNNNNNNN_");\n'
    '                                (void)np;\n'
    '                            }\n'
    '                        }'
)
text = text[:idx_s] + new5 + text[idx_e:]
changes += 1

print(f"Applied {changes} changes successfully")
with open(path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Written.")
