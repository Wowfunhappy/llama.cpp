// Stub server-tools: the real implementation needs std::filesystem (C++17).
#include "server-tools.h"

void server_tool::stream::push(const std::string &) {}

json server_tool::to_json() const { return json::object(); }

server_tools::server_tools() {}

server_tools::~server_tools() {}

void server_tools::setup(const std::vector<std::string> &, server_mcp &, const std::string &) {}
