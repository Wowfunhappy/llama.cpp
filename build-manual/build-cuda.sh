#!/bin/bash
set -e

cd "$(dirname "$0")/.."

BUILDDIR=build-manual

# CUDA compiler
NVCC=/usr/local/cuda/bin/nvcc
NVCC_FLAGS="-std=c++11 -arch=sm_35 -expt-extended-lambda --use_fast_math"
NVCC_DEFINES="-DNDEBUG -DGGML_USE_CUDA -DGGML_CUDA_NO_VMM"
NVCC_INCLUDES="-I./build-manual -I./ggml/include -I./ggml/src -I./include -I./src"

# Host compiler flags (clang/clang++ on macOS 10.9)
CXX=clang++
CC=clang
COMPAT="-include ./build-manual/compat-macos109.h"
COMMON_FLAGS="-O3 -DNDEBUG -DGGML_USE_CPU -DGGML_USE_BLAS -DGGML_BLAS_USE_ACCELERATE -DGGML_USE_CUDA -DJSON_HAS_CPP_11 -DSTBI_NO_THREAD_LOCALS"
COMMON_FLAGS="$COMMON_FLAGS -mavx2 -mfma -mf16c -mbmi -mbmi2 -msse4.2 -mpopcnt"
COMMON_FLAGS="$COMMON_FLAGS $COMPAT"
COMMON_FLAGS="$COMMON_FLAGS -DGGML_VERSION='\"0.9.7-manual\"' -DGGML_COMMIT='\"manual\"'"
CFLAGS="-std=c11 $COMMON_FLAGS"
CXXFLAGS="-std=c++1y $COMMON_FLAGS"
INCLUDES="-I./build-manual -I./ggml/include -I./ggml/src -I./include -I./src"
INCLUDES="$INCLUDES -I./ggml/src/ggml-cpu -I./ggml/src/ggml-cuda -I./common -I./vendor -I./tools/mtmd"

compile_cu() {
    local cu_file="$1"
    local basename=$(echo "$cu_file" | sed 's|/|_|g; s|\.cu$||')
    local obj_file="${BUILDDIR}/cuda_${basename}.o"

    if [ "$obj_file" -nt "$cu_file" ] 2>/dev/null; then
        echo "  SKIP: $cu_file"
        return
    fi

    echo "  NVCC: $cu_file"
    $NVCC $NVCC_FLAGS $NVCC_DEFINES $NVCC_INCLUDES -c "$cu_file" -o "$obj_file"
}

compile_c() {
    local src_file="$1"
    local extra_flags="${2:-}"
    local basename=$(echo "$src_file" | sed 's|/|_|g; s|\.c$|_c|')
    local obj_file="${BUILDDIR}/host_${basename}.o"

    if [ "$obj_file" -nt "$src_file" ] 2>/dev/null; then
        echo "  SKIP: $src_file"
        return
    fi

    echo "  CC:  $src_file"
    $CC $CFLAGS $INCLUDES $extra_flags -c "$src_file" -o "$obj_file"
}

compile_cpp() {
    local src_file="$1"
    local extra_flags="${2:-}"
    local basename=$(echo "$src_file" | sed 's|/|_|g; s|\.cpp$|_cpp|')
    local obj_file="${BUILDDIR}/host_${basename}.o"

    if [ "$obj_file" -nt "$src_file" ] 2>/dev/null; then
        echo "  SKIP: $src_file"
        return
    fi

    echo "  CXX: $src_file"
    $CXX $CXXFLAGS $INCLUDES $extra_flags -c "$src_file" -o "$obj_file"
}

echo "=== Step 1: Compiling CUDA files ==="

# All CUDA files + template instances (FA enabled, all kernels)
CU_FILES=$(ls ggml/src/ggml-cuda/*.cu \
              ggml/src/ggml-cuda/template-instances/mmq-instance-*.cu \
              ggml/src/ggml-cuda/template-instances/mmf-instance-*.cu \
              ggml/src/ggml-cuda/template-instances/fattn-mma-*.cu \
              ggml/src/ggml-cuda/template-instances/fattn-vec-instance-f16-f16.cu \
              ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q4_0-q4_0.cu \
              ggml/src/ggml-cuda/template-instances/fattn-vec-instance-q8_0-q8_0.cu \
              ggml/src/ggml-cuda/template-instances/fattn-tile-*.cu \
           | grep -v 'fattn-wmma')

for cu_file in $CU_FILES; do
    compile_cu "$cu_file"
done

# Compile fattn-tile stubs (tile kernel needs constexpr config unavailable in nvcc 7.5)
echo "  NVCC: build-manual/fattn-tile-stubs.cu"
$NVCC $NVCC_FLAGS $NVCC_DEFINES $NVCC_INCLUDES -I./ggml/src/ggml-cuda \
    -c build-manual/fattn-tile-stubs.cu -o ${BUILDDIR}/cuda_fattn-tile-stubs.o

echo ""
echo "=== Step 2: Compiling ggml core (C files) ==="

compile_c ggml/src/ggml.c
compile_c ggml/src/ggml-alloc.c
compile_c ggml/src/ggml-quants.c

echo ""
echo "=== Step 3: Compiling ggml core (C++ files) ==="

compile_cpp ggml/src/ggml.cpp
compile_cpp ggml/src/ggml-backend.cpp
compile_cpp ggml/src/ggml-opt.cpp
compile_cpp ggml/src/ggml-threading.cpp
compile_cpp ggml/src/gguf.cpp
compile_cpp ggml/src/ggml-backend-reg.cpp
compile_cpp ggml/src/ggml-backend-dl.cpp

echo ""
echo "=== Step 4: Compiling CPU backend ==="

compile_c ggml/src/ggml-cpu/ggml-cpu.c
compile_c ggml/src/ggml-cpu/quants.c
compile_c ggml/src/ggml-cpu/arch/x86/quants.c
compile_cpp ggml/src/ggml-cpu/ggml-cpu.cpp
compile_cpp ggml/src/ggml-cpu/binary-ops.cpp
compile_cpp ggml/src/ggml-cpu/unary-ops.cpp
compile_cpp ggml/src/ggml-cpu/ops.cpp
compile_cpp ggml/src/ggml-cpu/vec.cpp
compile_cpp ggml/src/ggml-cpu/traits.cpp
# Skip repack.cpp (references AVX-512/AMX symbols from arch/x86/repack.cpp)
# compile_cpp ggml/src/ggml-cpu/repack.cpp
compile_cpp ggml/src/ggml-cpu/hbm.cpp
# Skip arch/x86/repack.cpp (AVX-512/AMX optimizations, not needed on Haswell)
# Skip amx/ (Intel AMX, 12th gen+)
compile_cpp ggml/src/ggml-cpu/llamafile/sgemm.cpp

echo ""
echo "=== Step 5: Compiling BLAS backend ==="

compile_cpp ggml/src/ggml-blas/ggml-blas.cpp

echo ""
echo "=== Step 6: Compiling llama library ==="

for src in \
    llama.cpp llama-adapter.cpp llama-arch.cpp llama-batch.cpp \
    llama-chat.cpp llama-context.cpp llama-cparams.cpp llama-grammar.cpp \
    llama-graph.cpp llama-hparams.cpp llama-impl.cpp llama-io.cpp \
    llama-kv-cache.cpp llama-kv-cache-iswa.cpp \
    llama-memory.cpp llama-memory-hybrid.cpp llama-memory-hybrid-iswa.cpp \
    llama-memory-recurrent.cpp llama-mmap.cpp \
    llama-model-loader.cpp llama-model-saver.cpp llama-model.cpp \
    llama-quant.cpp llama-sampler.cpp llama-vocab.cpp \
    unicode-data.cpp unicode.cpp; do
    compile_cpp "src/${src}"
done

# Model architectures
for src in $(ls src/models/*.cpp); do
    compile_cpp "$src"
done

echo ""
echo "=== Step 7: Compiling common library ==="

COMMON_INCLUDES="-I./common -I./vendor -I./tools/server"

compile_cpp build-manual/build-info.cpp "$COMMON_INCLUDES"

for src in \
    arg.cpp chat-auto-parser-generator.cpp chat-auto-parser-helpers.cpp \
    chat-diff-analyzer.cpp chat-peg-parser.cpp chat.cpp common.cpp \
    console.cpp debug.cpp download.cpp json-partial.cpp \
    json-schema-to-grammar.cpp llguidance.cpp log.cpp \
    ngram-cache.cpp ngram-map.cpp ngram-mod.cpp \
    peg-parser.cpp preset.cpp reasoning-budget.cpp regex-partial.cpp \
    sampling.cpp speculative.cpp unicode.cpp; do
    compile_cpp "common/${src}" "$COMMON_INCLUDES"
done

# Jinja template engine
for src in lexer.cpp parser.cpp runtime.cpp value.cpp string.cpp caps.cpp; do
    compile_cpp "common/jinja/${src}" "$COMMON_INCLUDES"
done

echo ""
echo "=== Step 8a: Compiling mtmd library ==="

MTMD_INCLUDES="-I./tools/mtmd -I./tools/mtmd/models -I./common -I./vendor"
for src in clip.cpp mtmd.cpp mtmd-audio.cpp mtmd-helper.cpp; do
    compile_cpp "tools/mtmd/${src}" "$MTMD_INCLUDES"
done
for src in $(ls tools/mtmd/models/*.cpp); do
    compile_cpp "$src" "$MTMD_INCLUDES"
done

echo ""
echo "=== Step 8: Compiling server-context library ==="

SERVER_INCLUDES="-I./tools/server -I./common -I./vendor"

for src in \
    server-common.cpp server-context.cpp server-http.cpp \
    server-models.cpp server-queue.cpp server-task.cpp; do
    compile_cpp "tools/server/${src}" "$SERVER_INCLUDES"
done
# server.cpp has its own main() - compiled separately, not linked into llama-cli

echo ""
echo "=== Step 9: Compiling llama-cli ==="

compile_cpp tools/cli/cli.cpp "-I./tools/server -I./common -I./vendor"
compile_cpp examples/simple/simple.cpp

echo ""
echo "=== Step 10: Compiling CUDA runtime stub ==="

# The stub intercepts __cudaRegisterFatBinary etc. and forwards to the real
# CUDA runtime via dlsym if available, or no-ops if CUDA isn't installed.
# This lets one binary work with or without CUDA.
$CC -O3 -c build-manual/cuda-stub.c -o ${BUILDDIR}/host_build-manual_cuda-stub_c.o

echo ""
echo "=== Step 10b: Linking ==="

ALL_HOST=$(ls ${BUILDDIR}/host_*.o 2>/dev/null)
CUDA_OBJS=$(ls ${BUILDDIR}/cuda_*.o 2>/dev/null)

# Weak-link CUDA libs: binary loads them if present, ignores if absent.
# The cuda-stub.o provides fallback symbols for static init code.
CUDA_LINK="-L/usr/local/cuda/lib -Wl,-rpath,/usr/local/cuda/lib"
CUDA_LINK="$CUDA_LINK -Wl,-weak-lcudart -Wl,-weak-lcublas"

# llama-cli
CLI_OBJS=$(echo "$ALL_HOST" | grep -v 'examples_simple')
$CXX -o ${BUILDDIR}/llama-cli \
    $CLI_OBJS $CUDA_OBJS \
    -framework Accelerate \
    $CUDA_LINK \
    /usr/local/lib/libMacportsLegacySupport.a \
    -lpthread

# llama-simple
SIMPLE_OBJS=$(echo "$ALL_HOST" | grep -v 'tools_\|common_')
$CXX -o ${BUILDDIR}/llama-simple \
    $SIMPLE_OBJS $CUDA_OBJS \
    -framework Accelerate \
    $CUDA_LINK \
    /usr/local/lib/libMacportsLegacySupport.a \
    -lpthread

echo ""
echo "=== Build complete ==="
echo "Binaries (single file, works with or without CUDA):"
ls -la ${BUILDDIR}/llama-cli ${BUILDDIR}/llama-simple
