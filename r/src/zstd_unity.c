/* singlet R package -- the bundled Zstandard library (src/zstd/, zstd 1.5.6, BSD licence: src/zstd/LICENSE), compiled as one
 * translation unit the way zstd's own single-file build does (build/single_file_libs/zstd-in.c), without the multithreaded
 * compressor, the dictionary builder, legacy formats or assembly. Bundled so the package needs no system libzstd on any platform. */
#define DEBUGLEVEL 0
#define MEM_MODULE
#undef XXH_NAMESPACE
#define XXH_NAMESPACE ZSTD_
#undef XXH_PRIVATE_API
#define XXH_PRIVATE_API
#undef XXH_INLINE_ALL
#define XXH_INLINE_ALL
#define ZSTD_LEGACY_SUPPORT 0
#define ZSTD_TRACE 0
#define ZSTD_DISABLE_ASM 1

#define ZSTD_DEPS_NEED_MALLOC
#define ZSTD_DEPS_NEED_MATH64
#include "zstd/common/zstd_deps.h"

#include "zstd/common/debug.c"
#include "zstd/common/entropy_common.c"
#include "zstd/common/error_private.c"
#include "zstd/common/fse_decompress.c"
#include "zstd/common/threading.c"
#include "zstd/common/pool.c"
#include "zstd/common/zstd_common.c"

#include "zstd/compress/fse_compress.c"
#include "zstd/compress/hist.c"
#include "zstd/compress/huf_compress.c"
#include "zstd/compress/zstd_compress_literals.c"
#include "zstd/compress/zstd_compress_sequences.c"
#include "zstd/compress/zstd_compress_superblock.c"
#include "zstd/compress/zstd_compress.c"
#include "zstd/compress/zstd_double_fast.c"
#include "zstd/compress/zstd_fast.c"
#include "zstd/compress/zstd_lazy.c"
#include "zstd/compress/zstd_ldm.c"
#include "zstd/compress/zstd_opt.c"

#include "zstd/decompress/huf_decompress.c"
#include "zstd/decompress/zstd_ddict.c"
#include "zstd/decompress/zstd_decompress.c"
#include "zstd/decompress/zstd_decompress_block.c"
