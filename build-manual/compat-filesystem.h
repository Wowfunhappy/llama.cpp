#pragma once

// std::filesystem is C++17; this is the subset the codebase uses, on POSIX calls.

#include <string>
#include <system_error>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>

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

} // namespace compat_fs
