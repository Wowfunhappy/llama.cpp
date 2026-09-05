#include "common.cuh"

// The tile kernel needs constexpr launch configs that nvcc 7.5 cannot evaluate. On Kepler
// ggml_cuda_get_best_fattn_kernel never selects it, so the whole family is stubbed out here
// and fattn-tile.cu is left out of the build.

template<int DKQ, int DV>
void ggml_cuda_flash_attn_ext_tile_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("fattn-tile kernel not available on this build");
}

template void ggml_cuda_flash_attn_ext_tile_case< 40,  40>(ggml_backend_cuda_context &, ggml_tensor *);
template void ggml_cuda_flash_attn_ext_tile_case< 64,  64>(ggml_backend_cuda_context &, ggml_tensor *);
template void ggml_cuda_flash_attn_ext_tile_case< 72,  72>(ggml_backend_cuda_context &, ggml_tensor *);
template void ggml_cuda_flash_attn_ext_tile_case< 80,  80>(ggml_backend_cuda_context &, ggml_tensor *);
template void ggml_cuda_flash_attn_ext_tile_case< 96,  96>(ggml_backend_cuda_context &, ggml_tensor *);
template void ggml_cuda_flash_attn_ext_tile_case<112, 112>(ggml_backend_cuda_context &, ggml_tensor *);
template void ggml_cuda_flash_attn_ext_tile_case<128, 128>(ggml_backend_cuda_context &, ggml_tensor *);
template void ggml_cuda_flash_attn_ext_tile_case<256, 256>(ggml_backend_cuda_context &, ggml_tensor *);
template void ggml_cuda_flash_attn_ext_tile_case<576, 512>(ggml_backend_cuda_context &, ggml_tensor *);

void ggml_cuda_flash_attn_ext_tile(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("fattn-tile kernel not available on this build");
}

// WMMA flash attention stub (requires Volta+ CC >= 700)
void ggml_cuda_flash_attn_ext_wmma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx); GGML_UNUSED(dst);
    GGML_ABORT("fattn-wmma kernel not available on Kepler");
}
