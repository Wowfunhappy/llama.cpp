// Compatibility header for macOS 10.9 (Darwin 13.x)
// Provides declarations for functions available via libMacportsLegacySupport.a
#pragma once

#include <time.h>


// clock_gettime was added in macOS 10.12 but libMacportsLegacySupport provides it
#ifndef CLOCK_MONOTONIC
#define CLOCK_REALTIME  0
#define CLOCK_MONOTONIC 6
#define CLOCK_MONOTONIC_RAW 4
typedef int clockid_t;
#ifdef __cplusplus
extern "C" {
#endif
int clock_gettime(clockid_t clk_id, struct timespec *tp);
#ifdef __cplusplus
}
#endif
#endif

// Apple Clang 6.0's immintrin.h has _mm256_set_m128i but not the float form
#if defined(__clang__) && defined(__APPLE__) && (__clang_major__ < 7)
#define _mm256_set_m128(hi, lo) _mm256_insertf128_ps(_mm256_castps128_ps256(lo), (hi), 1)
#define _mm256_set_m128d(hi, lo) _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), (hi), 1)
#endif
