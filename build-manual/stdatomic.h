// Minimal stdatomic.h stub for Apple Clang 6.0 on macOS 10.9
// Clang supports _Atomic and __c11_atomic_* builtins even without the header
#pragma once

#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    memory_order_relaxed = __ATOMIC_RELAXED,
    memory_order_consume = __ATOMIC_CONSUME,
    memory_order_acquire = __ATOMIC_ACQUIRE,
    memory_order_release = __ATOMIC_RELEASE,
    memory_order_acq_rel = __ATOMIC_ACQ_REL,
    memory_order_seq_cst = __ATOMIC_SEQ_CST
} memory_order;

typedef _Atomic(int)  atomic_int;
typedef _Atomic(bool) atomic_bool;
typedef _Atomic(int)  atomic_flag;

#define atomic_store(ptr, val)                     __c11_atomic_store(ptr, val, __ATOMIC_SEQ_CST)
#define atomic_store_explicit(ptr, val, order)     __c11_atomic_store(ptr, val, order)
#define atomic_load(ptr)                           __c11_atomic_load(ptr, __ATOMIC_SEQ_CST)
#define atomic_load_explicit(ptr, order)           __c11_atomic_load(ptr, order)
#define atomic_fetch_add(ptr, val)                 __c11_atomic_fetch_add(ptr, val, __ATOMIC_SEQ_CST)
#define atomic_fetch_add_explicit(ptr, val, order) __c11_atomic_fetch_add(ptr, val, order)
#define atomic_fetch_sub(ptr, val)                 __c11_atomic_fetch_sub(ptr, val, __ATOMIC_SEQ_CST)
#define atomic_fetch_sub_explicit(ptr, val, order) __c11_atomic_fetch_sub(ptr, val, order)
#define atomic_compare_exchange_strong(ptr, exp, des) \
    __c11_atomic_compare_exchange_strong(ptr, exp, des, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define atomic_compare_exchange_weak(ptr, exp, des) \
    __c11_atomic_compare_exchange_weak(ptr, exp, des, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)
#define atomic_flag_test_and_set(ptr)              __c11_atomic_exchange(ptr, 1, __ATOMIC_SEQ_CST)
#define atomic_flag_clear(ptr)                     __c11_atomic_store(ptr, 0, __ATOMIC_SEQ_CST)
#define atomic_thread_fence(order)                 __c11_atomic_thread_fence(order)
#define ATOMIC_FLAG_INIT 0
#define ATOMIC_VAR_INIT(value) (value)

#ifdef __cplusplus
}
#endif
