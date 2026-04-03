// Stub server-tools for C++14 compat (real impl uses std::filesystem)
#include "server-tools.h"

json server_tool::to_json() { return json::object(); }

void server_tools::setup(const std::vector<std::string> &) {}

json server_tools::invoke(const std::string &, const json &) {
    return {{"error", "tools not supported in this build"}};
}
