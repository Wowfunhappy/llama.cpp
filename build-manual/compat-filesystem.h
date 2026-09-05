#pragma once

// std::filesystem is C++17; this is the subset the codebase uses, on POSIX calls.

#include <string>
#include <system_error>
#include <cerrno>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cstdio>
#include <dirent.h>
#include <vector>
#include <cstdlib>
#include <climits>

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

inline std::string filename(const std::string & p) {
    const size_t pos = p.find_last_of('/');
    return pos == std::string::npos ? p : p.substr(pos + 1);
}

// appends a path component, inserting a separator when needed
inline std::string join(const std::string & base, const std::string & child) {
    if (base.empty() || (!child.empty() && child[0] == '/')) {
        return child;
    }
    if (base[base.size() - 1] == '/') {
        return base + child;
    }
    return base + "/" + child;
}

// absolute path with symlinks resolved; throws std::system_error on failure
inline std::string canonical(const std::string & p) {
    char buf[PATH_MAX];
    if (realpath(p.c_str(), buf) == nullptr) {
        throw std::system_error(errno, std::generic_category(), "canonical");
    }
    return std::string(buf);
}

inline std::string current_path() {
    char buf[PATH_MAX];
    return getcwd(buf, sizeof(buf)) ? std::string(buf) : std::string();
}

// full paths of the entries in a directory, excluding "." and ".."
inline std::vector<std::string> list_directory(const std::string & p, std::error_code & ec) {
    std::vector<std::string> out;
    DIR * dir = opendir(p.c_str());
    if (!dir) {
        ec = std::error_code(errno, std::generic_category());
        return out;
    }
    ec.clear();
    while (struct dirent * ent = readdir(dir)) {
        const std::string name = ent->d_name;
        if (name == "." || name == "..") {
            continue;
        }
        out.push_back(join(p, name));
    }
    closedir(dir);
    return out;
}

inline bool is_regular_file(const std::string & p) {
    struct stat st;
    return stat(p.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

// ".txt" for "dir/file.txt", empty when the name carries no extension
inline std::string extension(const std::string & p) {
    const std::string name = filename(p);
    const size_t pos = name.find_last_of('.');
    if (pos == std::string::npos || pos == 0) {
        return std::string();
    }
    return name.substr(pos);
}

// full paths of every regular file below a directory; unreadable directories are skipped
inline std::vector<std::string> list_files_recursive(const std::string & root, std::error_code & ec) {
    std::vector<std::string> out;
    std::vector<std::string> pending(1, root);
    ec.clear();
    while (!pending.empty()) {
        const std::string dir = pending.back();
        pending.pop_back();
        std::error_code dir_ec;
        const std::vector<std::string> entries = list_directory(dir, dir_ec);
        if (dir_ec) {
            if (dir == root) {
                ec = dir_ec;
                return out;
            }
            continue;
        }
        for (size_t i = 0; i < entries.size(); i++) {
            struct stat st;
            if (lstat(entries[i].c_str(), &st) != 0) {
                continue;
            }
            if (S_ISDIR(st.st_mode)) {
                pending.push_back(entries[i]);
            } else if (S_ISREG(st.st_mode)) {
                out.push_back(entries[i]);
            }
        }
    }
    return out;
}

} // namespace compat_fs
