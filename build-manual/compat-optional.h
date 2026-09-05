#pragma once

// std::optional is C++17; this is the subset the codebase uses.

#include <cassert>
#include <utility>

struct common_nullopt_t { };
static const common_nullopt_t common_nullopt = common_nullopt_t();

struct common_in_place_t { };
static const common_in_place_t common_in_place = common_in_place_t();

template <typename T>
class common_optional {
    bool has_ = false;
    T    val_ = T();

  public:
    common_optional() { }
    common_optional(common_nullopt_t) { }
    common_optional(const T & v) : has_(true), val_(v) { }
    common_optional(T && v) : has_(true), val_(std::move(v)) { }

    template <typename... Args>
    explicit common_optional(common_in_place_t, Args &&... args) : has_(true), val_(T(std::forward<Args>(args)...)) { }

    common_optional & operator=(common_nullopt_t) { has_ = false; val_ = T(); return *this; }
    common_optional & operator=(const T & v)      { has_ = true;  val_ = v;   return *this; }

    bool has_value() const { return has_; }
    explicit operator bool() const { return has_; }

    const T & value() const { assert(has_); return val_; }
    T &       value()       { assert(has_); return val_; }

    const T & operator*() const { return val_; }
    T &       operator*()       { return val_; }

    const T * operator->() const { return &val_; }
    T *       operator->()       { return &val_; }

    template <typename U>
    T value_or(const U & fallback) const { return has_ ? val_ : T(fallback); }

    void reset() { has_ = false; val_ = T(); }

    template <typename... Args>
    void emplace(Args &&... args) { val_ = T(std::forward<Args>(args)...); has_ = true; }
};

template <typename T> bool operator==(const common_optional<T> & o, common_nullopt_t) { return !o.has_value(); }
template <typename T> bool operator==(common_nullopt_t, const common_optional<T> & o) { return !o.has_value(); }
template <typename T> bool operator!=(const common_optional<T> & o, common_nullopt_t) { return  o.has_value(); }
template <typename T> bool operator!=(common_nullopt_t, const common_optional<T> & o) { return  o.has_value(); }
