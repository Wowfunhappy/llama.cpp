#pragma once

// std::shared_mutex is C++17 and its out-of-line implementation is missing from
// the libc++ shipped with macOS 10.9; this is the same interface on pthread_rwlock.

#include <pthread.h>
#include <system_error>

class compat_shared_mutex {
public:
    compat_shared_mutex() {
        pthread_rwlock_init(&rwlock, nullptr);
    }
    ~compat_shared_mutex() {
        pthread_rwlock_destroy(&rwlock);
    }

    compat_shared_mutex(const compat_shared_mutex &) = delete;
    compat_shared_mutex & operator=(const compat_shared_mutex &) = delete;

    void lock()          { check(pthread_rwlock_wrlock(&rwlock)); }
    void unlock()        { check(pthread_rwlock_unlock(&rwlock)); }
    void lock_shared()   { check(pthread_rwlock_rdlock(&rwlock)); }
    void unlock_shared() { check(pthread_rwlock_unlock(&rwlock)); }

private:
    static void check(int rc) {
        if (rc != 0) {
            throw std::system_error(rc, std::generic_category(), "compat_shared_mutex");
        }
    }

    pthread_rwlock_t rwlock;
};

class compat_shared_lock {
public:
    explicit compat_shared_lock(compat_shared_mutex & m) : mu(m) {
        mu.lock_shared();
    }
    ~compat_shared_lock() {
        mu.unlock_shared();
    }

    compat_shared_lock(const compat_shared_lock &) = delete;
    compat_shared_lock & operator=(const compat_shared_lock &) = delete;

private:
    compat_shared_mutex & mu;
};
