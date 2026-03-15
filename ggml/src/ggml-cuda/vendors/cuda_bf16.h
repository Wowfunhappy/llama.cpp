// Minimal cuda_bf16.h stub for CUDA 7.5 (no hardware BF16 support on Kepler)
#pragma once

#include <cuda_fp16.h>

#ifndef __CUDA_BF16_H__
#define __CUDA_BF16_H__

struct __nv_bfloat16_raw {
    unsigned short x;
};

struct __nv_bfloat162_raw {
    unsigned short x;
    unsigned short y;
};

struct __align__(2) __nv_bfloat16 {
    unsigned short __x;

    __host__ __device__ __nv_bfloat16() : __x(0) {}

    __host__ __device__ explicit __nv_bfloat16(float f) {
        unsigned int bits;
        memcpy(&bits, &f, sizeof(bits));
        unsigned int rounding_bias = (bits >> 16) & 1;
        rounding_bias += 0x7FFF;
        bits += rounding_bias;
        __x = (unsigned short)(bits >> 16);
    }

    __host__ __device__ explicit operator float() const {
        unsigned int bits = ((unsigned int)__x) << 16;
        float f;
        memcpy(&f, &bits, sizeof(f));
        return f;
    }
};

struct __align__(4) __nv_bfloat162 {
    __nv_bfloat16 x;
    __nv_bfloat16 y;

    __host__ __device__ __nv_bfloat162() : x(), y() {}
    __host__ __device__ __nv_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) : x(a), y(b) {}
};

typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

__host__ __device__ inline __nv_bfloat16 __float2bfloat16(float f) {
    return __nv_bfloat16(f);
}

__host__ __device__ inline float __bfloat162float(__nv_bfloat16 h) {
    return (float)h;
}

__host__ __device__ inline __nv_bfloat162 make_bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) {
    return __nv_bfloat162(a, b);
}

__host__ __device__ inline __nv_bfloat162 __floats2bfloat162_rn(float a, float b) {
    return make_bfloat162(__float2bfloat16(a), __float2bfloat16(b));
}

__host__ __device__ inline float __low2float(__nv_bfloat162 h) {
    return __bfloat162float(h.x);
}

__host__ __device__ inline float __high2float(__nv_bfloat162 h) {
    return __bfloat162float(h.y);
}

__host__ __device__ inline __nv_bfloat162 __halves2bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) {
    return make_bfloat162(a, b);
}

__host__ __device__ inline __nv_bfloat16 __low2bfloat16(__nv_bfloat162 h) {
    return h.x;
}

__host__ __device__ inline __nv_bfloat16 __high2bfloat16(__nv_bfloat162 h) {
    return h.y;
}

__host__ __device__ inline __nv_bfloat162 __bfloat162bfloat162(__nv_bfloat16 a) {
    return make_bfloat162(a, a);
}

__host__ __device__ inline __nv_bfloat16 __hneg(__nv_bfloat16 a) {
    return __float2bfloat16(-__bfloat162float(a));
}

__host__ __device__ inline __nv_bfloat162 __hneg2(__nv_bfloat162 a) {
    return make_bfloat162(__hneg(a.x), __hneg(a.y));
}

__host__ __device__ inline __nv_bfloat162 __hadd2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __floats2bfloat162_rn(
        __bfloat162float(a.x) + __bfloat162float(b.x),
        __bfloat162float(a.y) + __bfloat162float(b.y));
}

__host__ __device__ inline __nv_bfloat162 __hsub2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __floats2bfloat162_rn(
        __bfloat162float(a.x) - __bfloat162float(b.x),
        __bfloat162float(a.y) - __bfloat162float(b.y));
}

__host__ __device__ inline __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) {
    return __floats2bfloat162_rn(
        __bfloat162float(a.x) * __bfloat162float(b.x),
        __bfloat162float(a.y) * __bfloat162float(b.y));
}

__host__ __device__ inline __nv_bfloat162 __hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
    return __floats2bfloat162_rn(
        __bfloat162float(a.x) * __bfloat162float(b.x) + __bfloat162float(c.x),
        __bfloat162float(a.y) * __bfloat162float(b.y) + __bfloat162float(c.y));
}

__host__ __device__ inline __nv_bfloat162 __hmax2(__nv_bfloat162 a, __nv_bfloat162 b) {
    float ax = __bfloat162float(a.x), ay = __bfloat162float(a.y);
    float bx = __bfloat162float(b.x), by = __bfloat162float(b.y);
    return __floats2bfloat162_rn(ax > bx ? ax : bx, ay > by ? ay : by);
}

__host__ __device__ inline __nv_bfloat16 __hmax(__nv_bfloat16 a, __nv_bfloat16 b) {
    float fa = __bfloat162float(a), fb = __bfloat162float(b);
    return __float2bfloat16(fa > fb ? fa : fb);
}

#endif // __CUDA_BF16_H__
