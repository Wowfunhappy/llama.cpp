// Only the Kepler table is reached from the constexpr device path; the rest are
// selected at runtime by the host dispatcher.

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_blackwell_ggml_type_mxfp4(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_MXFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    ggml_cuda_mmq_get_config_packed_ampere(type, J, fallback);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_blackwell_ggml_type_nvfp4(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, true)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,   8, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  24, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  40, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    CASE(GGML_TYPE_NVFP4, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_FP4, MMQ_ITER_K_FP4, true, false)
    ggml_cuda_mmq_get_config_packed_ampere(type, J, fallback);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_blackwell(ggml_type type, int J, bool fallback) {
    return
        type == GGML_TYPE_MXFP4 ? ggml_cuda_mmq_get_config_packed_blackwell_ggml_type_mxfp4(type, J, fallback) :
        type == GGML_TYPE_NVFP4 ? ggml_cuda_mmq_get_config_packed_blackwell_ggml_type_nvfp4(type, J, fallback) :
        ggml_cuda_mmq_get_config_packed_ampere(type, J, fallback);
}
