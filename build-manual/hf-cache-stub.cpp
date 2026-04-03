// Stub hf-cache implementation for C++14 / macOS 10.9 compat (no std::filesystem)
#include "hf-cache.h"
#include <stdexcept>

namespace hf_cache {

hf_files get_repo_files(const std::string & /*repo_id*/, const std::string & /*token*/) {
    return {};
}

hf_files get_cached_files(const std::string & /*repo_id*/) {
    return {};
}

std::string finalize_file(const hf_file & file) {
    return file.local_path;
}

void migrate_old_cache_to_hf_cache(const std::string & /*token*/, bool /*offline*/) {
    // no-op
}

} // namespace hf_cache
