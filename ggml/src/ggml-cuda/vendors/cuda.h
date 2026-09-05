#pragma once

#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <cstring>
#include <vector>

#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#else
// CUDA 7.5-10.x: no hardware BF16, use stub
#include "cuda_bf16.h"
#endif

#ifdef GGML_USE_NCCL
#include <nccl.h>
#endif // GGML_USE_NCCL

#if CUDART_VERSION >= 11080
#include <cuda_fp8.h>
#define FP8_AVAILABLE
#endif // CUDART_VERSION >= 11080

#if CUDART_VERSION >= 12080
#include <cuda_fp4.h>
#endif // CUDART_VERSION >= 12080

#if CUDART_VERSION < 9000
/* CUDA 7.5/8.0: __shfl_*_sync don't exist, use non-sync versions (implicit warp sync on Kepler/Maxwell) */
#define __shfl_sync(mask, val, srcLane, width)       __shfl((val), (srcLane), (width))
#define __shfl_up_sync(mask, val, delta, width)      __shfl_up((val), (delta), (width))
#define __shfl_down_sync(mask, val, delta, width)    __shfl_down((val), (delta), (width))
#define __shfl_xor_sync(mask, val, laneMask, width)  __shfl_xor((val), (laneMask), (width))
#define __all_sync(mask, predicate) __all((predicate))
#define __any_sync(mask, predicate) __any((predicate))
#define __ballot_sync(mask, predicate) __ballot((predicate))
static __device__ __forceinline__ void __syncwarp(unsigned mask = 0xFFFFFFFF) { (void)mask; /* implicit on Kepler */ }

/* cudaStreamWaitEvent gained a default for its flags argument in CUDA 11 */
static inline cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event) {
    return cudaStreamWaitEvent(stream, event, 0);
}

/* cublasSgemmStridedBatched arrived in CUDA 8.0; run the batch as individual GEMMs */
static inline cublasStatus_t cublasSgemmStridedBatched(
        cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
        int m, int n, int k, const float * alpha,
        const float * A, int lda, long long int strideA,
        const float * B, int ldb, long long int strideB,
        const float * beta,
        float * C, int ldc, long long int strideC, int batchCount) {
    for (int i = 0; i < batchCount; ++i) {
        const cublasStatus_t status = cublasSgemm(handle, transa, transb, m, n, k, alpha,
            A + i*strideA, lda, B + i*strideB, ldb, beta, C + i*strideC, ldc);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

/* CUBLAS_TENSOR_OP_MATH and cublasSetMathMode don't exist before CUDA 9.0 */
#define CUBLAS_TENSOR_OP_MATH 0
#define CUBLAS_TF32_TENSOR_OP_MATH 0
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
static inline cublasStatus_t cublasSetMathMode(cublasHandle_t, int) { return CUBLAS_STATUS_SUCCESS; }

/* cudaFuncSetAttribute doesn't exist before CUDA 9.0 */
#define cudaFuncAttributeMaxDynamicSharedMemorySize 0
template<typename T>
static inline cudaError_t cudaFuncSetAttribute(T, int, size_t) { return cudaSuccess; }

/* CUBLAS_DEFAULT_MATH doesn't exist before CUDA 9.0 */
#define CUBLAS_DEFAULT_MATH 0

/* cudaDevAttrCooperativeLaunch doesn't exist before CUDA 9.0 */
#define cudaDevAttrCooperativeLaunch ((cudaDeviceAttr)95)

/* cublasGemmAlgo_t doesn't exist before CUDA 8.0 */
#define CUBLAS_GEMM_DEFAULT 0

/* CUBLAS_GEMM_DEFAULT_TENSOR_OP doesn't exist before CUDA 9.0 */
#define CUBLAS_GEMM_DEFAULT_TENSOR_OP 99

/* CUDA_R_16BF doesn't exist before CUDA 11.0 */
#ifndef CUDA_R_16BF
#define CUDA_R_16BF ((cudaDataType_t)14)
#endif

/* CUDA 7.5 half-precision compatibility:
   In CUDA 7.5, __half is a struct { unsigned short x; } with no implicit float conversions.
   CUDA 9.0+ adds operator float() and __half(float) constructor.
   Provide overloaded helpers so template code compiles for all types. */

static __device__ __forceinline__ half2 make_half2(float a, float b) {
    return __floats2half2_rn(a, b);
}

/* Overloaded to_float: convert any numeric type to float */
static __device__ __forceinline__ float ggml_cuda_to_float(float x) { return x; }
static __device__ __forceinline__ float ggml_cuda_to_float(__half x) { return __half2float(x); }
static __device__ __forceinline__ float ggml_cuda_to_float(nv_bfloat16 x) { return __bfloat162float(x); }
static __device__ __forceinline__ float ggml_cuda_to_float(int x) { return (float)x; }

/* The *Ex GEMM entry points arrive in CUDA 8.0/9.0; run them on the 7.5 primitives.
   cublasSgemmEx already accepts half inputs with an FP32 accumulator, which is what
   the callers ask for, and alpha/beta are host scalars in the pointer mode ggml uses. */

static inline float cublas_compat_scalar(const void * p, cudaDataType_t compute_type) {
    if (compute_type == CUDA_R_16F) {
        unsigned short h;
        memcpy(&h, p, sizeof(h));
        const unsigned sign = (unsigned) (h >> 15) << 31;
        const unsigned exp  = (h >> 10) & 0x1f;
        const unsigned mant = h & 0x3ff;
        unsigned bits;
        if (exp == 0) {
            if (mant == 0) {
                bits = sign;
            } else {
                // subnormal half: renormalize into a float
                unsigned e = 0;
                unsigned m = mant;
                while ((m & 0x400) == 0) {
                    m <<= 1;
                    e++;
                }
                m &= 0x3ff;
                bits = sign | ((127 - 15 - e + 1) << 23) | (m << 13);
            }
        } else if (exp == 0x1f) {
            bits = sign | 0x7f800000u | (mant << 13);
        } else {
            bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    }
    return *(const float *) p;
}

static inline size_t cublas_compat_type_size(cudaDataType_t type) {
    return type == CUDA_R_16F ? sizeof(unsigned short) : sizeof(float);
}

static inline cublasStatus_t cublasGemmEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void * alpha, const void * A, cudaDataType_t Atype, int lda,
    const void * B, cudaDataType_t Btype, int ldb, const void * beta, void * C, cudaDataType_t Ctype, int ldc,
    cudaDataType_t computeType, int) {
    const float alpha_f = cublas_compat_scalar(alpha, computeType);
    const float beta_f  = cublas_compat_scalar(beta,  computeType);
    if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
        return cublasSgemm(handle, transa, transb, m, n, k, &alpha_f,
            (const float *) A, lda, (const float *) B, ldb, &beta_f, (float *) C, ldc);
    }
    return cublasSgemmEx(handle, transa, transb, m, n, k, &alpha_f,
        A, (cublasDataType_t) Atype, lda, B, (cublasDataType_t) Btype, ldb,
        &beta_f, C, (cublasDataType_t) Ctype, ldc);
}

static inline cublasStatus_t cublasGemmBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void * alpha,
    const void * const * Aarray, cudaDataType_t Atype, int lda,
    const void * const * Barray, cudaDataType_t Btype, int ldb,
    const void * beta, void * const * Carray, cudaDataType_t Ctype, int ldc,
    int batchCount, cudaDataType_t computeType, int algo) {
    const float alpha_f = cublas_compat_scalar(alpha, computeType);
    const float beta_f  = cublas_compat_scalar(beta,  computeType);
    if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
        return cublasSgemmBatched(handle, transa, transb, m, n, k, &alpha_f,
            (const float **) Aarray, lda, (const float **) Barray, ldb, &beta_f,
            (float **) Carray, ldc, batchCount);
    }

    // no batched half GEMM here: read the pointer arrays back and run the batch one GEMM at a time
    cudaStream_t stream = nullptr;
    cublasStatus_t status = cublasGetStream(handle, &stream);
    if (status != CUBLAS_STATUS_SUCCESS) {
        return status;
    }
    std::vector<const void *> a_host(batchCount);
    std::vector<const void *> b_host(batchCount);
    std::vector<void *>       c_host(batchCount);
    if (cudaMemcpyAsync(a_host.data(), Aarray, batchCount*sizeof(void *), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(b_host.data(), Barray, batchCount*sizeof(void *), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaMemcpyAsync(c_host.data(), Carray, batchCount*sizeof(void *), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    for (int i = 0; i < batchCount; ++i) {
        status = cublasGemmEx(handle, transa, transb, m, n, k, alpha,
            a_host[i], Atype, lda, b_host[i], Btype, ldb, beta, c_host[i], Ctype, ldc, computeType, algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

static inline cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void * alpha,
    const void * A, cudaDataType_t Atype, int lda, long long strideA,
    const void * B, cudaDataType_t Btype, int ldb, long long strideB,
    const void * beta, void * C, cudaDataType_t Ctype, int ldc, long long strideC,
    int batchCount, cudaDataType_t computeType, int algo) {
    for (int i = 0; i < batchCount; ++i) {
        const cublasStatus_t status = cublasGemmEx(handle, transa, transb, m, n, k, alpha,
            (const char *) A + i*strideA*cublas_compat_type_size(Atype), Atype, lda,
            (const char *) B + i*strideB*cublas_compat_type_size(Btype), Btype, ldb,
            beta,
            (      char *) C + i*strideC*cublas_compat_type_size(Ctype), Ctype, ldc,
            computeType, algo);
        if (status != CUBLAS_STATUS_SUCCESS) {
            return status;
        }
    }
    return CUBLAS_STATUS_SUCCESS;
}

#elif CUDART_VERSION < 11020
#define CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED
#define CUBLAS_TF32_TENSOR_OP_MATH CUBLAS_TENSOR_OP_MATH
#define CUBLAS_COMPUTE_16F CUDA_R_16F
#define CUBLAS_COMPUTE_32F CUDA_R_32F
#define cublasComputeType_t cudaDataType_t
#endif // CUDART_VERSION
