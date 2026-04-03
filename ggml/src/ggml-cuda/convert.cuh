#pragma once
#include "common.cuh"

#define CUDA_DEQUANTIZE_BLOCK_SIZE 256

template<typename T>
using to_t_cuda_t = void (*)(const void * x, T * y, int64_t k, cudaStream_t stream);

typedef to_t_cuda_t<float> to_fp32_cuda_t;
typedef to_t_cuda_t<half> to_fp16_cuda_t;
typedef to_t_cuda_t<nv_bfloat16> to_bf16_cuda_t;

to_fp16_cuda_t ggml_get_to_fp16_cuda(ggml_type type);

to_bf16_cuda_t ggml_get_to_bf16_cuda(ggml_type type);

to_fp32_cuda_t ggml_get_to_fp32_cuda(ggml_type type);

// TODO more general support for non-contiguous inputs

template<typename T>
using to_t_nc_cuda_t = void (*)(const void * x, T * y,
    int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
    int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream);

typedef to_t_nc_cuda_t<float> to_fp32_nc_cuda_t;
typedef to_t_nc_cuda_t<half> to_fp16_nc_cuda_t;
typedef to_t_nc_cuda_t<nv_bfloat16> to_bf16_nc_cuda_t;

to_fp32_nc_cuda_t ggml_get_to_fp32_nc_cuda(ggml_type type);
to_fp16_nc_cuda_t ggml_get_to_fp16_nc_cuda(ggml_type type);
to_bf16_nc_cuda_t ggml_get_to_bf16_nc_cuda(ggml_type type);

// C++11/CUDA7.5 compat: use overloaded dispatch instead of if constexpr + implicit conversions
// Default: go through float
template<typename dst_t, typename src_t>
__device__ inline dst_t ggml_cuda_cast(src_t x) {
    return (dst_t)ggml_cuda_to_float(x);
}

// Same type: identity
template<> __device__ inline float       ggml_cuda_cast<float, float>(float x)             { return x; }
template<> __device__ inline __half      ggml_cuda_cast<__half, __half>(__half x)           { return x; }
template<> __device__ inline nv_bfloat16 ggml_cuda_cast<nv_bfloat16, nv_bfloat16>(nv_bfloat16 x) { return x; }
template<> __device__ inline int32_t     ggml_cuda_cast<int32_t, int32_t>(int32_t x)       { return x; }

// half <-> float
template<> __device__ inline float  ggml_cuda_cast<float, __half>(__half x)      { return __half2float(x); }
template<> __device__ inline __half ggml_cuda_cast<__half, float>(float x)       { return __float2half(x); }

// bf16 <-> float
template<> __device__ inline float       ggml_cuda_cast<float, nv_bfloat16>(nv_bfloat16 x) { return __bfloat162float(x); }
template<> __device__ inline nv_bfloat16 ggml_cuda_cast<nv_bfloat16, float>(float x)       { return __float2bfloat16(x); }

// half <-> bf16 (go through float)
template<> __device__ inline nv_bfloat16 ggml_cuda_cast<nv_bfloat16, __half>(__half x)     { return __float2bfloat16(__half2float(x)); }
template<> __device__ inline __half      ggml_cuda_cast<__half, nv_bfloat16>(nv_bfloat16 x){ return __float2half(__bfloat162float(x)); }

// int32 from float/half/bf16
template<> __device__ inline int32_t ggml_cuda_cast<int32_t, float>(float x)             { return (int32_t)x; }
template<> __device__ inline int32_t ggml_cuda_cast<int32_t, __half>(__half x)            { return (int32_t)__half2float(x); }
template<> __device__ inline int32_t ggml_cuda_cast<int32_t, nv_bfloat16>(nv_bfloat16 x) { return (int32_t)__bfloat162float(x); }

// float2 identity
template<> __device__ inline float2 ggml_cuda_cast<float2, float2>(float2 x)              { return x; }

// float2 -> half2/bf162
template<> __device__ inline half2       ggml_cuda_cast<half2, float2>(float2 x)          { return __float22half2_rn(x); }
template<> __device__ inline nv_bfloat162 ggml_cuda_cast<nv_bfloat162, float2>(float2 x)  { return __floats2bfloat162_rn(x.x, x.y); }

// bf162 -> float2
template<> __device__ inline float2 ggml_cuda_cast<float2, nv_bfloat162>(nv_bfloat162 x)  {
    return make_float2(__bfloat162float(x.x), __bfloat162float(x.y));
}

// half2 identity
template<> __device__ inline half2 ggml_cuda_cast<half2, half2>(half2 x)                  { return x; }

// nv_bfloat162 identity
template<> __device__ inline nv_bfloat162 ggml_cuda_cast<nv_bfloat162, nv_bfloat162>(nv_bfloat162 x) { return x; }
