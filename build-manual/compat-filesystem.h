#pragma once

// std::filesystem is C++17; this is the subset the codebase uses, on POSIX calls.

#include <string>
#include <system_error>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>

namespace compat_fs {

inline bool exists(const std::string & p, std::error_code & ec) {
    struct stat st;
    if (stat(p.c_str(), &st) == 0) {
        ec.clear();
        return true;
    }
    ec = (errno == ENOENT) ? std::error_code() : std::error_code(errno, std::generic_category());
    return false;
}

inline bool exists(const std::string & p) {
    std::error_code ec;
    return exists(p, ec);
}

// creates every missing component of the path
inline bool create_directories(const std::string & p, std::error_code & ec) {
    ec.clear();
    if (p.empty()) {
        return false;
    }
    std::string acc;
    size_t i = 0;
    if (p[0] == '/') {
        acc = "/";
        i = 1;
    }
    bool created = false;
    while (i <= p.size()) {
        const size_t sep = p.find('/', i);
        const std::string part = p.substr(i, sep == std::string::npos ? std::string::npos : sep - i);
        if (!part.empty()) {
            if (acc.size() > 1 || (acc.size() == 1 && acc[0] != '/')) {
                acc += "/";
            }
            acc += part;
            if (mkdir(acc.c_str(), 0755) == 0) {
                created = true;
            } else if (errno != EEXIST) {
                ec = std::error_code(errno, std::generic_category());
                return false;
            }
        }
        if (sep == std::string::npos) {
            break;
        }
        i = sep + 1;
    }
    return created;
}

inline size_t file_size(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 ? (size_t) st.st_size : 0;
}

inline size_t file_size(const std::string & p, std::error_code & ec) {
    struct stat st;
    if (stat(p.c_str(), &st) == 0) {
        ec.clear();
        return (size_t) st.st_size;
    }
    ec = std::error_code(errno, std::generic_category());
    return 0;
}

inline std::string parent_path(const std::string & p) {
    const size_t pos = p.find_last_of('/');
    return pos == std::string::npos ? std::string() : p.substr(0, pos);
}

inline bool is_symlink(const std::string & p, std::error_code & ec) {
    struct stat st;
    if (lstat(p.c_str(), &st) != 0) {
        ec = std::error_code(errno, std::generic_category());
        return false;
    }
    ec.clear();
    return S_ISLNK(st.st_mode);
}

inline std::string read_symlink(const std::string & p, std::error_code & ec) {
    char buf[4096];
    const ssize_t n = readlink(p.c_str(), buf, sizeof(buf) - 1);
    if (n < 0) {
        ec = std::error_code(errno, std::generic_category());
        return std::string();
    }
    ec.clear();
    buf[n] = '\0';
    return std::string(buf);
}

inline bool remove(const std::string & p, std::error_code & ec) {
    if (::remove(p.c_str()) == 0) {
        ec.clear();
        return true;
    }
    ec = std::error_code(errno, std::generic_category());
    return false;
}

inline bool is_directory(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

} // namespace compat_fs
