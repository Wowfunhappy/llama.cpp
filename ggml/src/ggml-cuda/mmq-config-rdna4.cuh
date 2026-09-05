// Only the Kepler table is reached from the constexpr device path; the rest are
// selected at runtime by the host dispatcher.

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q1_0(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q1_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q1_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q1_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q1_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q1_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q1_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q1_0, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q1_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q1_0, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q1_0, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q1_0, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q1_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q2_0(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q2_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q2_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q2_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q2_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q2_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_0, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_0, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_0, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_0, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q4_0(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_0, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_0, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_0, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q4_1(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_1, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_1, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_1, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_1, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_1, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_1, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q5_0(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_0, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_0, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_0, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q5_1(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q5_1, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_1, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_1, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_1, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_1, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_1, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_1, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_1, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_1, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_1, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_1, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_1, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q8_0(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q8_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q8_0, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q8_0, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q8_0, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q8_0, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q8_0, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q2_k(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q2_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q2_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q2_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q2_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_K, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q2_K, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q3_k(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q3_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q3_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q3_K, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q3_K, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q3_K, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q3_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q4_k(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_K, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_K, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_K, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q4_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q5_k(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q5_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q5_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_K, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_K, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_K, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_K, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q5_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q6_k(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q6_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q6_K, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q6_K, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q6_K, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q6_K, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_Q6_K, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq1_s(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_IQ1_S, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ1_S, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ1_S, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ1_S, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ1_S, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ1_S, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ1_S, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ1_S, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ1_S, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ1_S, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ1_S, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ1_S, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq2_xxs(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_IQ2_XXS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_XXS, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_XXS, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_XXS, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_XXS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XXS, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XXS, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XXS, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XXS, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XXS, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XXS, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XXS, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq2_xs(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_IQ2_XS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_XS, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_XS, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_XS, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_XS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XS, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XS, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XS, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XS, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XS, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XS, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_XS, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq2_s(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_IQ2_S, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_S, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_S, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_S, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ2_S, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_S, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_S, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_S, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_S, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_S, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_S, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ2_S, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq3_xxs(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_IQ3_XXS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ3_XXS, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ3_XXS, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ3_XXS, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ3_XXS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_XXS, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_XXS, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_XXS, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_XXS, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_XXS, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_XXS, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_XXS, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq3_s(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_IQ3_S, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ3_S, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ3_S, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ3_S, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ3_S, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_S, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_S, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_S, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_S, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_S, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_S, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ3_S, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq4_xs(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ4_XS, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_XS, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_XS, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_XS, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_XS, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_XS, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq4_nl(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_IQ4_NL, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ4_NL, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ4_NL, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ4_NL, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_IQ4_NL, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_NL, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_NL, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_NL, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_NL, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_NL, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_NL, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_IQ4_NL, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_mxfp4(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_MXFP4, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_MXFP4, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_MXFP4, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_MXFP4, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_MXFP4, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_MXFP4, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_MXFP4, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_MXFP4, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_MXFP4, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_MXFP4, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_MXFP4, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_MXFP4, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_nvfp4(ggml_type type, int J, bool fallback) {
    return
    CASE(GGML_TYPE_NVFP4, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_NVFP4, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_NVFP4, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_NVFP4, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, true)
    CASE(GGML_TYPE_NVFP4, 128, 2,  64,  16, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_NVFP4, 128, 2,  64,  32, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_NVFP4, 128, 2,  64,  48, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_NVFP4, 128, 2,  64,  64, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  80, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_NVFP4, 256, 2, 128,  96, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_NVFP4, 256, 2, 128, 112, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false)
    CASE(GGML_TYPE_NVFP4, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4, MMQ_ITER_K, false, false)
    MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}

static __host__ __device__ uint64_t ggml_cuda_mmq_get_config_packed_rdna4(ggml_type type, int J, bool fallback) {
    return
        type == GGML_TYPE_Q1_0 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q1_0(type, J, fallback) :
        type == GGML_TYPE_Q2_0 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q2_0(type, J, fallback) :
        type == GGML_TYPE_Q4_0 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q4_0(type, J, fallback) :
        type == GGML_TYPE_Q4_1 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q4_1(type, J, fallback) :
        type == GGML_TYPE_Q5_0 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q5_0(type, J, fallback) :
        type == GGML_TYPE_Q5_1 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q5_1(type, J, fallback) :
        type == GGML_TYPE_Q8_0 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q8_0(type, J, fallback) :
        type == GGML_TYPE_Q2_K ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q2_k(type, J, fallback) :
        type == GGML_TYPE_Q3_K ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q3_k(type, J, fallback) :
        type == GGML_TYPE_Q4_K ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q4_k(type, J, fallback) :
        type == GGML_TYPE_Q5_K ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q5_k(type, J, fallback) :
        type == GGML_TYPE_Q6_K ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_q6_k(type, J, fallback) :
        type == GGML_TYPE_IQ1_S ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq1_s(type, J, fallback) :
        type == GGML_TYPE_IQ2_XXS ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq2_xxs(type, J, fallback) :
        type == GGML_TYPE_IQ2_XS ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq2_xs(type, J, fallback) :
        type == GGML_TYPE_IQ2_S ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq2_s(type, J, fallback) :
        type == GGML_TYPE_IQ3_XXS ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq3_xxs(type, J, fallback) :
        type == GGML_TYPE_IQ3_S ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq3_s(type, J, fallback) :
        type == GGML_TYPE_IQ4_XS ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq4_xs(type, J, fallback) :
        type == GGML_TYPE_IQ4_NL ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_iq4_nl(type, J, fallback) :
        type == GGML_TYPE_MXFP4 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_mxfp4(type, J, fallback) :
        type == GGML_TYPE_NVFP4 ? ggml_cuda_mmq_get_config_packed_rdna4_ggml_type_nvfp4(type, J, fallback) :
        MMQ_CFG_PACK(GGML_TYPE_COUNT, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, 256, false, true);
}
