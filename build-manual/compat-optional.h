#pragma once

// std::optional is C++17; this is the subset the codebase uses. The value is held in
// raw storage so T need not be default-constructible.

#include <cassert>
#include <new>
#include <type_traits>
#include <utility>

struct common_nullopt_t { };
static const common_nullopt_t common_nullopt = common_nullopt_t();

struct common_in_place_t { };
static const common_in_place_t common_in_place = common_in_place_t();

template <typename T>
class common_optional {
    bool has_;
    typename std::aligned_storage<sizeof(T), __alignof(T)>::type buf_;

    T *       ptr()       { return reinterpret_cast<T *>(&buf_); }
    const T * ptr() const { return reinterpret_cast<const T *>(&buf_); }

  public:
    common_optional() : has_(false) { }
    common_optional(common_nullopt_t) : has_(false) { }
    common_optional(const T & v) : has_(true) { new (&buf_) T(v); }
    common_optional(T && v) : has_(true) { new (&buf_) T(std::move(v)); }

    template <typename... Args>
    explicit common_optional(common_in_place_t, Args &&... args) : has_(true) { new (&buf_) T(std::forward<Args>(args)...); }

    common_optional(const common_optional & o) : has_(o.has_) {
        if (has_) { new (&buf_) T(*o.ptr()); }
    }
    common_optional(common_optional && o) : has_(o.has_) {
        if (has_) { new (&buf_) T(std::move(*o.ptr())); }
    }

    ~common_optional() { reset(); }

    common_optional & operator=(common_nullopt_t) { reset(); return *this; }

    common_optional & operator=(const common_optional & o) {
        if (this != &o) {
            reset();
            if (o.has_) { new (&buf_) T(*o.ptr()); has_ = true; }
        }
        return *this;
    }

    common_optional & operator=(common_optional && o) {
        if (this != &o) {
            reset();
            if (o.has_) { new (&buf_) T(std::move(*o.ptr())); has_ = true; }
        }
        return *this;
    }

    common_optional & operator=(const T & v) { reset(); new (&buf_) T(v);            has_ = true; return *this; }
    common_optional & operator=(T && v)      { reset(); new (&buf_) T(std::move(v)); has_ = true; return *this; }

    bool has_value() const { return has_; }
    explicit operator bool() const { return has_; }

    const T & value() const { assert(has_); return *ptr(); }
    T &       value()       { assert(has_); return *ptr(); }

    const T & operator*() const { return *ptr(); }
    T &       operator*()       { return *ptr(); }

    const T * operator->() const { return ptr(); }
    T *       operator->()       { return ptr(); }

    template <typename U>
    T value_or(const U & fallback) const { return has_ ? *ptr() : T(fallback); }

    void reset() {
        if (has_) { ptr()->~T(); has_ = false; }
    }

    template <typename... Args>
    void emplace(Args &&... args) { reset(); new (&buf_) T(std::forward<Args>(args)...); has_ = true; }
};

template <typename T> bool operator==(const common_optional<T> & o, common_nullopt_t) { return !o.has_value(); }
template <typename T> bool operator==(common_nullopt_t, const common_optional<T> & o) { return !o.has_value(); }
template <typename T> bool operator!=(const common_optional<T> & o, common_nullopt_t) { return  o.has_value(); }
template <typename T> bool operator!=(common_nullopt_t, const common_optional<T> & o) { return  o.has_value(); }
