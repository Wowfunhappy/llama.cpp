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
