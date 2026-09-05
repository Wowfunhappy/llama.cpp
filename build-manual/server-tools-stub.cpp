// Stub server-tools: the real implementation needs std::filesystem (C++17).
#include "server-tools.h"

#include <stdexcept>

// the real runtime lives in server-tools.cpp; an empty definition is enough for
// unique_ptr<server_tools_runtime> to be destroyable here
struct server_tools_runtime {};

void server_tool::stream::push(const std::string &) {}

json server_tool::to_json() const { return json::object(); }

server_tools::server_tools() {}

server_tools::~server_tools() {}

void server_tools::setup(const std::vector<std::string> &, server_mcp &, const std::string &) {
    throw std::runtime_error("server tools are not available in this build");
}
