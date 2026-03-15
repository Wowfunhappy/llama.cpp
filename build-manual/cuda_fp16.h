/*
 * Replacement cuda_fp16.h for CUDA 7.5 compatibility.
 * Provides modern __half2 layout (__half x, __half y) instead of CUDA 7.5's (unsigned int x).
 * All arithmetic emulated through float for CC 3.5 (Kepler).
 */
#ifndef CUDA_FP16_H_COMPAT
#define CUDA_FP16_H_COMPAT

/* Prevent the original CUDA 7.5 header from being included */
#define CUDA_FP16_H_JNESTUG4

#include <cuda_runtime.h>
#include <string.h> /* for memcpy */

/* ---------- Host-side software conversion helpers (defined early for use in types) ---------- */
static inline float __ggml_half2float_sw(unsigned short h) {
    unsigned int sign = ((unsigned int)(h & 0x8000)) << 16;
    unsigned int exponent = (h >> 10) & 0x1F;
    unsigned int mantissa = h & 0x03FF;
    unsigned int bits;
    if (exponent == 0) {
        if (mantissa == 0) { bits = sign; }
        else {
            exponent = 1;
            while (!(mantissa & 0x0400)) { mantissa <<= 1; exponent--; }
            mantissa &= 0x03FF;
            bits = sign | ((unsigned int)(exponent + 127 - 1) << 23) | (mantissa << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((unsigned int)(exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float f; memcpy(&f, &bits, sizeof(f)); return f;
}
static inline unsigned short __ggml_float2half_sw(float f) {
    unsigned int bits; memcpy(&bits, &f, sizeof(bits));
    unsigned short sign = (unsigned short)((bits >> 16) & 0x8000);
    int exp_val = ((bits >> 23) & 0xFF) - 127;
    unsigned int mant = bits & 0x7FFFFF;
    if (exp_val > 15) return (unsigned short)(sign | 0x7C00);
    if (exp_val > -15) return (unsigned short)(sign | ((exp_val + 15) << 10) | (mant >> 13));
    return sign;
}

/* ---------- Type definitions ---------- */

struct __align__(2) __half {
    unsigned short x;
#if defined(__CUDACC__)
    __host__ __device__ __forceinline__ __half() : x(0) {}
    __host__ __device__ __forceinline__ __half(unsigned short v) : x(v) {}
    __host__ __device__ __forceinline__ __half(float f) {
#ifdef __CUDA_ARCH__
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(x) : "f"(f));
#else
        x = __ggml_float2half_sw(f);
#endif
    }
    __host__ __device__ __forceinline__ operator float() const {
#ifdef __CUDA_ARCH__
        float f; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(x)); return f;
#else
        return __ggml_half2float_sw(x);
#endif
    }
#endif
};

struct __align__(4) __half2 {
    __half x;
    __half y;
#if defined(__CUDACC__)
    __host__ __device__ __forceinline__ __half2() : x(), y() {}
    __host__ __device__ __forceinline__ __half2(__half a, __half b) : x(a), y(b) {}
#endif
};

#ifndef CUDA_NO_HALF
typedef __half half;
typedef __half2 half2;
#endif

/* ---------- Host/device conversion helpers ---------- */

#if defined(__CUDACC__)

/* ---- Core conversions: use struct constructors/operators, also provide standalone functions ---- */

static __host__ __device__ __forceinline__ __half __float2half(const float f) {
    return __half(f);
}

static __host__ __device__ __forceinline__ float __half2float(const __half h) {
    return (float)h;
}

/* NOTE: __float2half_rn(float) is a built-in returning unsigned short.
 * Do NOT redefine it here. Use __float2half() which returns __half. */

/* ---- half2 construction/access ---- */

static __device__ __forceinline__ __half2 __halves2half2(const __half a, const __half b) {
    __half2 r;
    r.x = a;
    r.y = b;
    return r;
}

static __device__ __forceinline__ __half2 make_half2(const __half a, const __half b) {
    return __halves2half2(a, b);
}

static __device__ __forceinline__ __half __low2half(const __half2 h) {
    return h.x;
}

static __device__ __forceinline__ __half __high2half(const __half2 h) {
    return h.y;
}

static __device__ __forceinline__ float __low2float(const __half2 h) {
    return __half2float(h.x);
}

static __device__ __forceinline__ float __high2float(const __half2 h) {
    return __half2float(h.y);
}

static __device__ __forceinline__ __half2 __half2half2(const __half a) {
    return __halves2half2(a, a);
}

static __device__ __forceinline__ __half2 __float2half2_rn(const float f) {
    __half h = __float2half(f);
    return __half2half2(h);
}

static __device__ __forceinline__ __half2 __floats2half2_rn(const float a, const float b) {
    return __halves2half2(__float2half(a), __float2half(b));
}

static __device__ __forceinline__ __half2 __float22half2_rn(const float2 a) {
    return __halves2half2(__float2half(a.x), __float2half(a.y));
}

static __device__ __forceinline__ float2 __half22float2(const __half2 h) {
    float2 r;
    r.x = __half2float(h.x);
    r.y = __half2float(h.y);
    return r;
}

static __device__ __forceinline__ __half2 __lowhigh2highlow(const __half2 h) {
    return __halves2half2(h.y, h.x);
}

static __device__ __forceinline__ __half2 __lows2half2(const __half2 a, const __half2 b) {
    return __halves2half2(a.x, b.x);
}

static __device__ __forceinline__ __half2 __highs2half2(const __half2 a, const __half2 b) {
    return __halves2half2(a.y, b.y);
}

static __device__ __forceinline__ __half2 __low2half2(const __half2 h) {
    return __halves2half2(h.x, h.x);
}

static __device__ __forceinline__ __half2 __high2half2(const __half2 h) {
    return __halves2half2(h.y, h.y);
}

static __device__ __forceinline__ int __hisinf(const __half a) {
    unsigned short bits = a.x;
    /* FP16 inf: exponent all 1s (0x7C00), mantissa all 0s */
    if ((bits & 0x7FFF) == 0x7C00) {
        return (bits & 0x8000) ? -1 : 1;
    }
    return 0;
}

static __device__ __forceinline__ bool __hisnan(const __half a) {
    unsigned short bits = a.x;
    return ((bits & 0x7C00) == 0x7C00) && ((bits & 0x03FF) != 0);
}

/* ---- half2 ldg ---- */
static __device__ __forceinline__ __half2 __ldg(const __half2 *ptr) {
    /* On Kepler, __ldg is a texture cache load */
    unsigned int packed;
    asm("ld.global.nc.u32 %0, [%1];" : "=r"(packed) : "l"(ptr));
    __half2 r;
    r.x.x = (unsigned short)(packed & 0xFFFF);
    r.y.x = (unsigned short)(packed >> 16);
    return r;
}

static __device__ __forceinline__ __half __ldg(const __half *ptr) {
    unsigned short bits;
    asm("ld.global.nc.u16 %0, [%1];" : "=h"(bits) : "l"(ptr));
    __half r;
    r.x = bits;
    return r;
}

/* ---- half2 shuffle ---- */
static __device__ __forceinline__ __half2 __shfl(__half2 var, int delta, int width) {
    unsigned int packed = ((unsigned int)var.y.x << 16) | (unsigned int)var.x.x;
    packed = __shfl(packed, delta, width);
    __half2 r;
    r.x.x = (unsigned short)(packed & 0xFFFF);
    r.y.x = (unsigned short)(packed >> 16);
    return r;
}

static __device__ __forceinline__ __half2 __shfl_up(__half2 var, unsigned int delta, int width) {
    unsigned int packed = ((unsigned int)var.y.x << 16) | (unsigned int)var.x.x;
    packed = __shfl_up(packed, delta, width);
    __half2 r;
    r.x.x = (unsigned short)(packed & 0xFFFF);
    r.y.x = (unsigned short)(packed >> 16);
    return r;
}

static __device__ __forceinline__ __half2 __shfl_down(__half2 var, unsigned int delta, int width) {
    unsigned int packed = ((unsigned int)var.y.x << 16) | (unsigned int)var.x.x;
    packed = __shfl_down(packed, delta, width);
    __half2 r;
    r.x.x = (unsigned short)(packed & 0xFFFF);
    r.y.x = (unsigned short)(packed >> 16);
    return r;
}

static __device__ __forceinline__ __half2 __shfl_xor(__half2 var, int delta, int width) {
    unsigned int packed = ((unsigned int)var.y.x << 16) | (unsigned int)var.x.x;
    packed = __shfl_xor(packed, delta, width);
    __half2 r;
    r.x.x = (unsigned short)(packed & 0xFFFF);
    r.y.x = (unsigned short)(packed >> 16);
    return r;
}

/* ---- half arithmetic (all via float for CC 3.5) ---- */
static __device__ __forceinline__ __half __hadd(const __half a, const __half b) {
    return __float2half(__half2float(a) + __half2float(b));
}
static __device__ __forceinline__ __half __hsub(const __half a, const __half b) {
    return __float2half(__half2float(a) - __half2float(b));
}
static __device__ __forceinline__ __half __hmul(const __half a, const __half b) {
    return __float2half(__half2float(a) * __half2float(b));
}
static __device__ __forceinline__ __half __hfma(const __half a, const __half b, const __half c) {
    return __float2half(__half2float(a) * __half2float(b) + __half2float(c));
}
static __device__ __forceinline__ __half __hneg(const __half a) {
    __half r; r.x = a.x ^ 0x8000u; return r;
}

/* Saturated variants */
static __device__ __forceinline__ float __ggml_saturate(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}
static __device__ __forceinline__ __half __hadd_sat(const __half a, const __half b) {
    return __float2half(__ggml_saturate(__half2float(a) + __half2float(b)));
}
static __device__ __forceinline__ __half __hsub_sat(const __half a, const __half b) {
    return __float2half(__ggml_saturate(__half2float(a) - __half2float(b)));
}
static __device__ __forceinline__ __half __hmul_sat(const __half a, const __half b) {
    return __float2half(__ggml_saturate(__half2float(a) * __half2float(b)));
}
static __device__ __forceinline__ __half __hfma_sat(const __half a, const __half b, const __half c) {
    return __float2half(__ggml_saturate(__half2float(a) * __half2float(b) + __half2float(c)));
}

/* ---- half2 arithmetic ---- */
static __device__ __forceinline__ __half2 __hadd2(const __half2 a, const __half2 b) {
    return __halves2half2(__hadd(a.x, b.x), __hadd(a.y, b.y));
}
static __device__ __forceinline__ __half2 __hsub2(const __half2 a, const __half2 b) {
    return __halves2half2(__hsub(a.x, b.x), __hsub(a.y, b.y));
}
static __device__ __forceinline__ __half2 __hmul2(const __half2 a, const __half2 b) {
    return __halves2half2(__hmul(a.x, b.x), __hmul(a.y, b.y));
}
static __device__ __forceinline__ __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c) {
    return __halves2half2(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y));
}
static __device__ __forceinline__ __half2 __hneg2(const __half2 a) {
    return __halves2half2(__hneg(a.x), __hneg(a.y));
}

/* Saturated half2 */
static __device__ __forceinline__ __half2 __hadd2_sat(const __half2 a, const __half2 b) {
    return __halves2half2(__hadd_sat(a.x, b.x), __hadd_sat(a.y, b.y));
}
static __device__ __forceinline__ __half2 __hsub2_sat(const __half2 a, const __half2 b) {
    return __halves2half2(__hsub_sat(a.x, b.x), __hsub_sat(a.y, b.y));
}
static __device__ __forceinline__ __half2 __hmul2_sat(const __half2 a, const __half2 b) {
    return __halves2half2(__hmul_sat(a.x, b.x), __hmul_sat(a.y, b.y));
}
static __device__ __forceinline__ __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c) {
    return __halves2half2(__hfma_sat(a.x, b.x, c.x), __hfma_sat(a.y, b.y, c.y));
}

/* ---- half comparisons ---- */
static __device__ __forceinline__ bool __heq(const __half a, const __half b)  { return __half2float(a) == __half2float(b); }
static __device__ __forceinline__ bool __hne(const __half a, const __half b)  { return __half2float(a) != __half2float(b); }
static __device__ __forceinline__ bool __hle(const __half a, const __half b)  { return __half2float(a) <= __half2float(b); }
static __device__ __forceinline__ bool __hge(const __half a, const __half b)  { return __half2float(a) >= __half2float(b); }
static __device__ __forceinline__ bool __hlt(const __half a, const __half b)  { return __half2float(a) <  __half2float(b); }
static __device__ __forceinline__ bool __hgt(const __half a, const __half b)  { return __half2float(a) >  __half2float(b); }
static __device__ __forceinline__ bool __hequ(const __half a, const __half b) { return __half2float(a) == __half2float(b); }
static __device__ __forceinline__ bool __hneu(const __half a, const __half b) { return __half2float(a) != __half2float(b); }
static __device__ __forceinline__ bool __hleu(const __half a, const __half b) { return __half2float(a) <= __half2float(b); }
static __device__ __forceinline__ bool __hgeu(const __half a, const __half b) { return __half2float(a) >= __half2float(b); }
static __device__ __forceinline__ bool __hltu(const __half a, const __half b) { return __half2float(a) <  __half2float(b); }
static __device__ __forceinline__ bool __hgtu(const __half a, const __half b) { return __half2float(a) >  __half2float(b); }

/* ---- half2 comparisons (element-wise, return half2 with 0x0000 or 0xFFFF) ---- */
static __device__ __forceinline__ __half __bool_to_half(bool b) {
    __half r; r.x = b ? 0xFFFF : 0x0000; return r;
}
static __device__ __forceinline__ __half2 __heq2(const __half2 a, const __half2 b) {
    return __halves2half2(__bool_to_half(__heq(a.x, b.x)), __bool_to_half(__heq(a.y, b.y)));
}
static __device__ __forceinline__ __half2 __hne2(const __half2 a, const __half2 b) {
    return __halves2half2(__bool_to_half(__hne(a.x, b.x)), __bool_to_half(__hne(a.y, b.y)));
}
static __device__ __forceinline__ __half2 __hle2(const __half2 a, const __half2 b) {
    return __halves2half2(__bool_to_half(__hle(a.x, b.x)), __bool_to_half(__hle(a.y, b.y)));
}
static __device__ __forceinline__ __half2 __hge2(const __half2 a, const __half2 b) {
    return __halves2half2(__bool_to_half(__hge(a.x, b.x)), __bool_to_half(__hge(a.y, b.y)));
}
static __device__ __forceinline__ __half2 __hlt2(const __half2 a, const __half2 b) {
    return __halves2half2(__bool_to_half(__hlt(a.x, b.x)), __bool_to_half(__hlt(a.y, b.y)));
}
static __device__ __forceinline__ __half2 __hgt2(const __half2 a, const __half2 b) {
    return __halves2half2(__bool_to_half(__hgt(a.x, b.x)), __bool_to_half(__hgt(a.y, b.y)));
}
static __device__ __forceinline__ __half2 __hequ2(const __half2 a, const __half2 b) { return __heq2(a, b); }
static __device__ __forceinline__ __half2 __hneu2(const __half2 a, const __half2 b) { return __hne2(a, b); }
static __device__ __forceinline__ __half2 __hleu2(const __half2 a, const __half2 b) { return __hle2(a, b); }
static __device__ __forceinline__ __half2 __hgeu2(const __half2 a, const __half2 b) { return __hge2(a, b); }
static __device__ __forceinline__ __half2 __hltu2(const __half2 a, const __half2 b) { return __hlt2(a, b); }
static __device__ __forceinline__ __half2 __hgtu2(const __half2 a, const __half2 b) { return __hgt2(a, b); }

/* ---- half2 boolean comparisons ---- */
static __device__ __forceinline__ bool __hbeq2(const __half2 a, const __half2 b) { return __heq(a.x, b.x) && __heq(a.y, b.y); }
static __device__ __forceinline__ bool __hbne2(const __half2 a, const __half2 b) { return __hne(a.x, b.x) || __hne(a.y, b.y); }
static __device__ __forceinline__ bool __hble2(const __half2 a, const __half2 b) { return __hle(a.x, b.x) && __hle(a.y, b.y); }
static __device__ __forceinline__ bool __hbge2(const __half2 a, const __half2 b) { return __hge(a.x, b.x) && __hge(a.y, b.y); }
static __device__ __forceinline__ bool __hblt2(const __half2 a, const __half2 b) { return __hlt(a.x, b.x) && __hlt(a.y, b.y); }
static __device__ __forceinline__ bool __hbgt2(const __half2 a, const __half2 b) { return __hgt(a.x, b.x) && __hgt(a.y, b.y); }
static __device__ __forceinline__ bool __hbequ2(const __half2 a, const __half2 b) { return __hbeq2(a, b); }
static __device__ __forceinline__ bool __hbneu2(const __half2 a, const __half2 b) { return __hbne2(a, b); }
static __device__ __forceinline__ bool __hbleu2(const __half2 a, const __half2 b) { return __hble2(a, b); }
static __device__ __forceinline__ bool __hbgeu2(const __half2 a, const __half2 b) { return __hbge2(a, b); }
static __device__ __forceinline__ bool __hbltu2(const __half2 a, const __half2 b) { return __hblt2(a, b); }
static __device__ __forceinline__ bool __hbgtu2(const __half2 a, const __half2 b) { return __hbgt2(a, b); }

/* ---- half2 nan check ---- */
static __device__ __forceinline__ __half2 __hisnan2(const __half2 a) {
    return __halves2half2(__bool_to_half(__hisnan(a.x)), __bool_to_half(__hisnan(a.y)));
}

/* ---- Max/min (not in CUDA 7.5 originally, added later) ---- */
static __device__ __forceinline__ __half __hmax(const __half a, const __half b) {
    float fa = __half2float(a), fb = __half2float(b);
    return __float2half(fa > fb ? fa : fb);
}
static __device__ __forceinline__ __half __hmin(const __half a, const __half b) {
    float fa = __half2float(a), fb = __half2float(b);
    return __float2half(fa < fb ? fa : fb);
}
static __device__ __forceinline__ __half2 __hmax2(const __half2 a, const __half2 b) {
    return __halves2half2(__hmax(a.x, b.x), __hmax(a.y, b.y));
}
static __device__ __forceinline__ __half2 __hmin2(const __half2 a, const __half2 b) {
    return __halves2half2(__hmin(a.x, b.x), __hmin(a.y, b.y));
}

/* ---- Absolute value ---- */
static __device__ __forceinline__ __half __habs(const __half a) {
    __half r; r.x = a.x & 0x7FFF; return r;
}
static __device__ __forceinline__ __half2 __habs2(const __half2 a) {
    return __halves2half2(__habs(a.x), __habs(a.y));
}

/* ---- Math functions ---- */
static __device__ __forceinline__ __half hexp(const __half a) {
    return __float2half(expf(__half2float(a)));
}
static __device__ __forceinline__ __half2 h2exp(const __half2 a) {
    return __halves2half2(hexp(a.x), hexp(a.y));
}
static __device__ __forceinline__ __half hlog(const __half a) {
    return __float2half(logf(__half2float(a)));
}
static __device__ __forceinline__ __half hsqrt(const __half a) {
    return __float2half(sqrtf(__half2float(a)));
}
static __device__ __forceinline__ __half hrsqrt(const __half a) {
    return __float2half(rsqrtf(__half2float(a)));
}
static __device__ __forceinline__ __half hsin(const __half a) {
    return __float2half(sinf(__half2float(a)));
}
static __device__ __forceinline__ __half hcos(const __half a) {
    return __float2half(cosf(__half2float(a)));
}

#endif /* __CUDACC__ */

/* ---------- Host-only fallback implementations ---------- */
#ifndef __CUDACC__

static inline __half __float2half_host(float f) {
    __half h;
    unsigned int bits;
    memcpy(&bits, &f, sizeof(bits));
    unsigned int sign = (bits >> 16) & 0x8000;
    int exp_val = ((bits >> 23) & 0xFF) - 127;
    unsigned int mantissa = bits & 0x7FFFFF;

    if (exp_val > 15) {
        h.x = (unsigned short)(sign | 0x7C00); /* inf */
    } else if (exp_val > -15) {
        unsigned short exp_part = (unsigned short)((exp_val + 15) << 10);
        unsigned short mant_part = (unsigned short)(mantissa >> 13);
        h.x = (unsigned short)(sign | exp_part | mant_part);
    } else {
        h.x = (unsigned short)sign; /* zero/denorm */
    }
    return h;
}

static inline float __half2float_host(__half h) {
    unsigned int sign = ((unsigned int)(h.x & 0x8000)) << 16;
    unsigned int exp_val = (h.x >> 10) & 0x1F;
    unsigned int mantissa = h.x & 0x03FF;
    unsigned int bits;

    if (exp_val == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exp_val = 1;
            while (!(mantissa & 0x0400)) { mantissa <<= 1; exp_val--; }
            mantissa &= 0x03FF;
            bits = sign | ((unsigned int)(exp_val + 127 - 1) << 23) | (mantissa << 13);
        }
    } else if (exp_val == 31) {
        bits = sign | 0x7F800000 | (mantissa << 13);
    } else {
        bits = sign | ((unsigned int)(exp_val + 127 - 15) << 23) | (mantissa << 13);
    }

    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static inline __half __float2half(float f) { return __float2half_host(f); }
static inline float __half2float(__half h) { return __half2float_host(h); }
static inline __half __float2half_rn(float f) { return __float2half_host(f); }

static inline __half2 __halves2half2(__half a, __half b) { __half2 r; r.x = a; r.y = b; return r; }
static inline __half2 make_half2(__half a, __half b) { return __halves2half2(a, b); }
static inline __half __low2half(__half2 h) { return h.x; }
static inline __half __high2half(__half2 h) { return h.y; }
static inline float __low2float(__half2 h) { return __half2float(h.x); }
static inline float __high2float(__half2 h) { return __half2float(h.y); }
static inline __half2 __half2half2(__half a) { return __halves2half2(a, a); }
static inline __half2 __float2half2_rn(float f) { __half h = __float2half(f); return __half2half2(h); }
static inline __half2 __floats2half2_rn(float a, float b) { return __halves2half2(__float2half(a), __float2half(b)); }
static inline __half2 __float22half2_rn(float2 a) { return __halves2half2(__float2half(a.x), __float2half(a.y)); }
static inline float2 __half22float2(__half2 h) { float2 r; r.x = __half2float(h.x); r.y = __half2float(h.y); return r; }
static inline __half2 __lowhigh2highlow(__half2 h) { return __halves2half2(h.y, h.x); }
static inline __half2 __lows2half2(__half2 a, __half2 b) { return __halves2half2(a.x, b.x); }
static inline __half2 __highs2half2(__half2 a, __half2 b) { return __halves2half2(a.y, b.y); }
static inline __half2 __low2half2(__half2 h) { return __halves2half2(h.x, h.x); }
static inline __half2 __high2half2(__half2 h) { return __halves2half2(h.y, h.y); }
static inline __half __hneg(__half a) { __half r; r.x = a.x ^ 0x8000u; return r; }
static inline __half2 __hneg2(__half2 a) { return __halves2half2(__hneg(a.x), __hneg(a.y)); }

#endif /* !__CUDACC__ */

/* ---------- CUDA_FP16_CONSTEXPR: needed by some ggml headers ---------- */
#ifndef __CUDA_FP16_CONSTEXPR__
#define __CUDA_FP16_CONSTEXPR__
#endif

#endif /* CUDA_FP16_H_COMPAT */
