#pragma once

#ifdef _WIN32
#   define WIN32_LEAN_AND_MEAN
#   ifndef NOMINMAX
#       define NOMINMAX
#   endif
#   include <windows.h>
#   include <winevt.h>
#else
#    include <dlfcn.h>
#    include <unistd.h>
#endif
#include <string>
#include <memory>
#include <vector>
#include <system_error>
#include <sys/stat.h>
#include <dirent.h>

// C++14 compat: minimal std::filesystem replacement using POSIX
namespace fs {

class path {
    std::string p;
public:
    path() {}
    path(const std::string & s) : p(s) {}
    path(const char * s) : p(s) {}

    const std::string & native() const { return p; }
    const char * c_str() const { return p.c_str(); }
    std::string string() const { return p; }

    path filename() const {
        auto pos = p.find_last_of('/');
        return pos == std::string::npos ? *this : path(p.substr(pos + 1));
    }

    path extension() const {
        std::string fn = filename().p;
        auto pos = fn.find_last_of('.');
        return pos == std::string::npos ? path("") : path(fn.substr(pos));
    }

    path stem() const {
        std::string fn = filename().p;
        auto pos = fn.find_last_of('.');
        return pos == std::string::npos ? path(fn) : path(fn.substr(0, pos));
    }

    path parent_path() const {
        auto pos = p.find_last_of('/');
        return pos == std::string::npos ? path("") : path(p.substr(0, pos));
    }

    path operator/(const path & other) const {
        if (p.empty()) return other;
        if (p.back() == '/') return path(p + other.p);
        return path(p + "/" + other.p);
    }

    bool empty() const { return p.empty(); }

    friend bool operator==(const path & a, const path & b) { return a.p == b.p; }
    friend bool operator!=(const path & a, const path & b) { return a.p != b.p; }
    friend bool operator<(const path & a, const path & b) { return a.p < b.p; }
};

inline path u8path(const std::string & s) { return path(s); }
inline path u8path(const char * s) { return path(s); }

inline bool exists(const path & p, std::error_code & /*ec*/) {
    struct stat st;
    return stat(p.c_str(), &st) == 0;
}

inline path current_path() {
    char buf[4096];
    if (getcwd(buf, sizeof(buf))) return path(buf);
    return path(".");
}

enum class directory_options { none = 0, skip_permission_denied = 1 };

struct directory_entry {
    fs::path p_;

    const fs::path & path() const { return p_; }

    bool is_regular_file(std::error_code & /*ec*/) const {
        struct stat st;
        if (stat(p_.c_str(), &st) != 0) return false;
        return S_ISREG(st.st_mode);
    }

    // Allow implicit conversion to fs::path for dl_load_library
    operator const fs::path &() const { return p_; }
};

class directory_iterator {
    DIR * dir_;
    std::string base_;
    directory_entry current_;
    bool done_;

    void advance() {
        if (!dir_) { done_ = true; return; }
        struct dirent * ent;
        while ((ent = readdir(dir_)) != nullptr) {
            if (ent->d_name[0] == '.' && (ent->d_name[1] == '\0' ||
                (ent->d_name[1] == '.' && ent->d_name[2] == '\0'))) continue;
            current_.p_ = fs::path(base_ + "/" + ent->d_name);
            return;
        }
        done_ = true;
    }
public:
    directory_iterator() : dir_(nullptr), done_(true) {}
    directory_iterator(const fs::path & p, directory_options = directory_options::none)
        : dir_(opendir(p.c_str())), base_(p.native()), done_(false) {
        advance();
    }
    ~directory_iterator() { if (dir_) closedir(dir_); }

    // Copy: only valid for sentinel (end) iterators
    directory_iterator(const directory_iterator & other)
        : dir_(nullptr), done_(other.done_) {}

    // Move
    directory_iterator(directory_iterator && other)
        : dir_(other.dir_), base_(std::move(other.base_)),
          current_(std::move(other.current_)), done_(other.done_) {
        other.dir_ = nullptr;
        other.done_ = true;
    }
    directory_iterator & operator=(const directory_iterator &) = delete;

    const directory_entry & operator*() const { return current_; }
    directory_iterator & operator++() { advance(); return *this; }
    bool operator!=(const directory_iterator & other) const { return done_ != other.done_; }

    // For range-based for
    directory_iterator & begin() { return *this; }
    directory_iterator end() const { return directory_iterator(); }
};

} // namespace fs

#ifdef _WIN32

using dl_handle = std::remove_pointer_t<HMODULE>;

struct dl_handle_deleter {
    void operator()(HMODULE handle) {
        FreeLibrary(handle);
    }
};

#else

using dl_handle = void;

struct dl_handle_deleter {
    void operator()(void * handle) {
        dlclose(handle);
    }
};

#endif

using dl_handle_ptr = std::unique_ptr<dl_handle, dl_handle_deleter>;

dl_handle * dl_load_library(const fs::path & path);
void * dl_get_sym(dl_handle * handle, const char * name);
const char * dl_error();
