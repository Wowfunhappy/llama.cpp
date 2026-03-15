#pragma once

#include <nlohmann/json_fwd.hpp>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <functional>
#include <vector>
#include <type_traits>
#include <cassert>

struct common_grammar_builder;

class common_peg_parser_builder;

using common_peg_parser_id = size_t;
constexpr common_peg_parser_id COMMON_PEG_INVALID_PARSER_ID = static_cast<common_peg_parser_id>(-1);

using common_peg_ast_id = size_t;
constexpr common_peg_ast_id COMMON_PEG_INVALID_AST_ID = static_cast<common_peg_ast_id>(-1);

// Lightweight wrapper around common_peg_parser_id for convenience
class common_peg_parser {
    common_peg_parser_id id_;
    common_peg_parser_builder & builder_;

  public:
    common_peg_parser(const common_peg_parser & other) : id_(other.id_), builder_(other.builder_) {}
    common_peg_parser(common_peg_parser_id id, common_peg_parser_builder & builder) : id_(id), builder_(builder) {}

    common_peg_parser & operator=(const common_peg_parser & other);
    common_peg_parser & operator+=(const common_peg_parser & other);
    common_peg_parser & operator|=(const common_peg_parser & other);

    operator common_peg_parser_id() const { return id_; }
    common_peg_parser_id id() const { return id_; }

    common_peg_parser_builder & builder() const { return builder_; }

    // Creates a sequence
    common_peg_parser operator+(const common_peg_parser & other) const;

    // Creates a sequence separated by spaces.
    common_peg_parser operator<<(const common_peg_parser & other) const;

    // Creates a choice
    common_peg_parser operator|(const common_peg_parser & other) const;

    common_peg_parser operator+(const char * str) const;
    common_peg_parser operator+(const std::string & str) const;
    common_peg_parser operator<<(const char * str) const;
    common_peg_parser operator<<(const std::string & str) const;
    common_peg_parser operator|(const char * str) const;
    common_peg_parser operator|(const std::string & str) const;
};

common_peg_parser operator+(const char * str, const common_peg_parser & p);
common_peg_parser operator+(const std::string & str, const common_peg_parser & p);
common_peg_parser operator<<(const char * str, const common_peg_parser & p);
common_peg_parser operator<<(const std::string & str, const common_peg_parser & p);
common_peg_parser operator|(const char * str, const common_peg_parser & p);
common_peg_parser operator|(const std::string & str, const common_peg_parser & p);

enum common_peg_parse_result_type {
    COMMON_PEG_PARSE_RESULT_FAIL            = 0,
    COMMON_PEG_PARSE_RESULT_SUCCESS         = 1,
    COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT = 2,
};

const char * common_peg_parse_result_type_name(common_peg_parse_result_type type);

struct common_peg_ast_node {
    common_peg_ast_id id;
    std::string rule;
    std::string tag;
    size_t start;
    size_t end;
    std::string text;
    std::vector<common_peg_ast_id> children;

    bool is_partial = false;
};

struct common_peg_parse_result;

using common_peg_ast_visitor = std::function<void(const common_peg_ast_node & node)>;

class common_peg_ast_arena {
    std::vector<common_peg_ast_node> nodes_;
  public:
    common_peg_ast_id add_node(
        const std::string & rule,
        const std::string & tag,
        size_t start,
        size_t end,
        const std::string & text,
        std::vector<common_peg_ast_id> children,
        bool is_partial = false
    ) {
        common_peg_ast_id id = nodes_.size();
        nodes_.push_back({id, rule, tag, start, end, text, std::move(children), is_partial});
        return id;
    }

    const common_peg_ast_node & get(common_peg_ast_id id) const { return nodes_.at(id); }

    size_t size() const { return nodes_.size(); }

    void clear() { nodes_.clear(); }

    void visit(common_peg_ast_id id, const common_peg_ast_visitor & visitor) const;
    void visit(const common_peg_parse_result & result, const common_peg_ast_visitor & visitor) const;

    std::string dump();
};

struct common_peg_parse_result {
    common_peg_parse_result_type type = COMMON_PEG_PARSE_RESULT_FAIL;
    size_t start = 0;
    size_t end = 0;

    std::vector<common_peg_ast_id> nodes;

    common_peg_parse_result() = default;

    common_peg_parse_result(common_peg_parse_result_type type, size_t start)
        : type(type), start(start), end(start) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end)
        : type(type), start(start), end(end) {}

    common_peg_parse_result(common_peg_parse_result_type type, size_t start, size_t end, std::vector<common_peg_ast_id> nodes)
        : type(type), start(start), end(end), nodes(std::move(nodes)) {}

    bool fail() const { return type == COMMON_PEG_PARSE_RESULT_FAIL; }
    bool need_more_input() const { return type == COMMON_PEG_PARSE_RESULT_NEED_MORE_INPUT; }
    bool success() const { return type == COMMON_PEG_PARSE_RESULT_SUCCESS; }
};

enum common_peg_parse_flags {
    COMMON_PEG_PARSE_FLAG_NONE    = 0,
    COMMON_PEG_PARSE_FLAG_LENIENT = 1 << 0,
    COMMON_PEG_PARSE_FLAG_DEBUG   = 1 << 1,
};

inline common_peg_parse_flags operator|(common_peg_parse_flags a, common_peg_parse_flags b) {
    return static_cast<common_peg_parse_flags>(int(a) | int(b));
}

inline common_peg_parse_flags & operator|=(common_peg_parse_flags & a, common_peg_parse_flags b) {
    return a = a | b;
}

inline common_peg_parse_flags operator&(common_peg_parse_flags a, common_peg_parse_flags b) {
    return static_cast<common_peg_parse_flags>(int(a) & int(b));
}

inline common_peg_parse_flags operator~(common_peg_parse_flags a) {
    return static_cast<common_peg_parse_flags>(~int(a));
}

struct common_peg_parse_context {
    std::string input;
    common_peg_parse_flags flags;
    common_peg_ast_arena ast;

    int parse_depth;

    common_peg_parse_context(common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE)
        : flags(flags), parse_depth(0) {}

    common_peg_parse_context(const std::string & input, common_peg_parse_flags flags = COMMON_PEG_PARSE_FLAG_NONE)
        : input(input), flags(flags), parse_depth(0) {}

    bool is_lenient() const { return flags & COMMON_PEG_PARSE_FLAG_LENIENT; }
    bool is_debug() const { return flags & COMMON_PEG_PARSE_FLAG_DEBUG; }
};

class common_peg_arena;

// Parser variants
struct common_peg_epsilon_parser {};

struct common_peg_start_parser {};

struct common_peg_end_parser {};

struct common_peg_literal_parser {
    std::string literal;
};

struct common_peg_sequence_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_choice_parser {
    std::vector<common_peg_parser_id> children;
};

struct common_peg_repetition_parser {
    common_peg_parser_id child;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_and_parser {
    common_peg_parser_id child;
};

struct common_peg_not_parser {
    common_peg_parser_id child;
};

struct common_peg_any_parser {};

struct common_peg_space_parser {};

struct common_peg_chars_parser {
    struct char_range {
        uint32_t start;
        uint32_t end;
        bool contains(uint32_t codepoint) const { return codepoint >= start && codepoint <= end; }
    };

    std::string pattern;
    std::vector<char_range> ranges;
    bool negated;
    int min_count;
    int max_count;  // -1 for unbounded
};

struct common_peg_string_parser {
    char delimiter;
};

struct common_peg_until_parser {
    std::vector<std::string> delimiters;
};

struct common_peg_schema_parser {
    common_peg_parser_id child;
    std::string name;
    std::shared_ptr<nlohmann::ordered_json> schema;

    // Indicates if the GBNF should accept a raw string that matches the schema.
    bool raw;
};

struct common_peg_rule_parser {
    std::string name;
    common_peg_parser_id child;
    bool trigger;
};

struct common_peg_ref_parser {
    std::string name;
};

struct common_peg_atomic_parser {
    common_peg_parser_id child;
};

struct common_peg_tag_parser {
    common_peg_parser_id child;
    std::string tag;
};

// Tag enum for parser types (C++14-compatible variant replacement)
enum common_peg_parser_tag {
    PEG_TAG_EPSILON,
    PEG_TAG_START,
    PEG_TAG_END,
    PEG_TAG_LITERAL,
    PEG_TAG_SEQUENCE,
    PEG_TAG_CHOICE,
    PEG_TAG_REPETITION,
    PEG_TAG_AND,
    PEG_TAG_NOT,
    PEG_TAG_ANY,
    PEG_TAG_SPACE,
    PEG_TAG_CHARS,
    PEG_TAG_STRING,
    PEG_TAG_UNTIL,
    PEG_TAG_SCHEMA,
    PEG_TAG_RULE,
    PEG_TAG_REF,
    PEG_TAG_ATOMIC,
    PEG_TAG_TAG,
};

// Type trait to map parser struct -> tag
template <typename T> struct peg_parser_tag;
template <> struct peg_parser_tag<common_peg_epsilon_parser>    { static const common_peg_parser_tag value = PEG_TAG_EPSILON; };
template <> struct peg_parser_tag<common_peg_start_parser>      { static const common_peg_parser_tag value = PEG_TAG_START; };
template <> struct peg_parser_tag<common_peg_end_parser>        { static const common_peg_parser_tag value = PEG_TAG_END; };
template <> struct peg_parser_tag<common_peg_literal_parser>    { static const common_peg_parser_tag value = PEG_TAG_LITERAL; };
template <> struct peg_parser_tag<common_peg_sequence_parser>   { static const common_peg_parser_tag value = PEG_TAG_SEQUENCE; };
template <> struct peg_parser_tag<common_peg_choice_parser>     { static const common_peg_parser_tag value = PEG_TAG_CHOICE; };
template <> struct peg_parser_tag<common_peg_repetition_parser> { static const common_peg_parser_tag value = PEG_TAG_REPETITION; };
template <> struct peg_parser_tag<common_peg_and_parser>        { static const common_peg_parser_tag value = PEG_TAG_AND; };
template <> struct peg_parser_tag<common_peg_not_parser>        { static const common_peg_parser_tag value = PEG_TAG_NOT; };
template <> struct peg_parser_tag<common_peg_any_parser>        { static const common_peg_parser_tag value = PEG_TAG_ANY; };
template <> struct peg_parser_tag<common_peg_space_parser>      { static const common_peg_parser_tag value = PEG_TAG_SPACE; };
template <> struct peg_parser_tag<common_peg_chars_parser>      { static const common_peg_parser_tag value = PEG_TAG_CHARS; };
template <> struct peg_parser_tag<common_peg_string_parser>     { static const common_peg_parser_tag value = PEG_TAG_STRING; };
template <> struct peg_parser_tag<common_peg_until_parser>      { static const common_peg_parser_tag value = PEG_TAG_UNTIL; };
template <> struct peg_parser_tag<common_peg_schema_parser>     { static const common_peg_parser_tag value = PEG_TAG_SCHEMA; };
template <> struct peg_parser_tag<common_peg_rule_parser>       { static const common_peg_parser_tag value = PEG_TAG_RULE; };
template <> struct peg_parser_tag<common_peg_ref_parser>        { static const common_peg_parser_tag value = PEG_TAG_REF; };
template <> struct peg_parser_tag<common_peg_atomic_parser>     { static const common_peg_parser_tag value = PEG_TAG_ATOMIC; };
template <> struct peg_parser_tag<common_peg_tag_parser>        { static const common_peg_parser_tag value = PEG_TAG_TAG; };

// C++14-compatible variant for parser types using type-erased storage
class common_peg_parser_variant {
    struct holder_base {
        virtual ~holder_base() {}
        virtual holder_base * clone() const = 0;
    };

    template <typename T>
    struct holder : holder_base {
        T value;
        holder(const T & v) : value(v) {}
        holder(T && v) : value(std::move(v)) {}
        holder_base * clone() const override { return new holder<T>(value); }
    };

    common_peg_parser_tag tag_;
    holder_base * data_;

public:
    common_peg_parser_variant() : tag_(PEG_TAG_EPSILON), data_(new holder<common_peg_epsilon_parser>(common_peg_epsilon_parser{})) {}

    template <typename T,
        typename std::enable_if<!std::is_same<typename std::decay<T>::type, common_peg_parser_variant>::value, int>::type = 0>
    common_peg_parser_variant(const T & v)
        : tag_(peg_parser_tag<typename std::decay<T>::type>::value)
        , data_(new holder<typename std::decay<T>::type>(v)) {}

    template <typename T>
    common_peg_parser_variant(T && v,
        typename std::enable_if<!std::is_same<typename std::decay<T>::type, common_peg_parser_variant>::value>::type * = nullptr)
        : tag_(peg_parser_tag<typename std::decay<T>::type>::value)
        , data_(new holder<typename std::decay<T>::type>(std::forward<T>(v))) {}

    common_peg_parser_variant(const common_peg_parser_variant & other)
        : tag_(other.tag_), data_(other.data_ ? other.data_->clone() : nullptr) {}

    common_peg_parser_variant(common_peg_parser_variant && other) noexcept
        : tag_(other.tag_), data_(other.data_) { other.data_ = nullptr; }

    common_peg_parser_variant & operator=(const common_peg_parser_variant & other) {
        if (this != &other) {
            delete data_;
            tag_ = other.tag_;
            data_ = other.data_ ? other.data_->clone() : nullptr;
        }
        return *this;
    }

    common_peg_parser_variant & operator=(common_peg_parser_variant && other) noexcept {
        if (this != &other) {
            delete data_;
            tag_ = other.tag_;
            data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }

    ~common_peg_parser_variant() { delete data_; }

    common_peg_parser_tag tag() const { return tag_; }

    template <typename T>
    T * get_if() {
        if (tag_ == peg_parser_tag<T>::value) {
            return &static_cast<holder<T>*>(data_)->value;
        }
        return nullptr;
    }

    template <typename T>
    const T * get_if() const {
        if (tag_ == peg_parser_tag<T>::value) {
            return &static_cast<const holder<T>*>(data_)->value;
        }
        return nullptr;
    }

    template <typename T>
    bool holds() const {
        return tag_ == peg_parser_tag<T>::value;
    }
};

// Free-function helpers matching std:: API patterns
template <typename T>
T * peg_get_if(common_peg_parser_variant * v) {
    return v ? v->get_if<T>() : nullptr;
}

template <typename T>
const T * peg_get_if(const common_peg_parser_variant * v) {
    return v ? v->get_if<T>() : nullptr;
}

template <typename T>
bool peg_holds_alternative(const common_peg_parser_variant & v) {
    return v.holds<T>();
}

// Visitor dispatch for common_peg_parser_variant (C++14-compatible)
template <typename Visitor>
auto peg_visit(Visitor && vis, common_peg_parser_variant & v)
    -> decltype(vis(*(common_peg_epsilon_parser*)nullptr))
{
    switch (v.tag()) {
        case PEG_TAG_EPSILON:    return vis(*v.get_if<common_peg_epsilon_parser>());
        case PEG_TAG_START:      return vis(*v.get_if<common_peg_start_parser>());
        case PEG_TAG_END:        return vis(*v.get_if<common_peg_end_parser>());
        case PEG_TAG_LITERAL:    return vis(*v.get_if<common_peg_literal_parser>());
        case PEG_TAG_SEQUENCE:   return vis(*v.get_if<common_peg_sequence_parser>());
        case PEG_TAG_CHOICE:     return vis(*v.get_if<common_peg_choice_parser>());
        case PEG_TAG_REPETITION: return vis(*v.get_if<common_peg_repetition_parser>());
        case PEG_TAG_AND:        return vis(*v.get_if<common_peg_and_parser>());
        case PEG_TAG_NOT:        return vis(*v.get_if<common_peg_not_parser>());
        case PEG_TAG_ANY:        return vis(*v.get_if<common_peg_any_parser>());
        case PEG_TAG_SPACE:      return vis(*v.get_if<common_peg_space_parser>());
        case PEG_TAG_CHARS:      return vis(*v.get_if<common_peg_chars_parser>());
        case PEG_TAG_STRING:     return vis(*v.get_if<common_peg_string_parser>());
        case PEG_TAG_UNTIL:      return vis(*v.get_if<common_peg_until_parser>());
        case PEG_TAG_SCHEMA:     return vis(*v.get_if<common_peg_schema_parser>());
        case PEG_TAG_RULE:       return vis(*v.get_if<common_peg_rule_parser>());
        case PEG_TAG_REF:        return vis(*v.get_if<common_peg_ref_parser>());
        case PEG_TAG_ATOMIC:     return vis(*v.get_if<common_peg_atomic_parser>());
        case PEG_TAG_TAG:        return vis(*v.get_if<common_peg_tag_parser>());
        default: assert(false && "Unknown parser tag"); return vis(*v.get_if<common_peg_epsilon_parser>());
    }
}

template <typename Visitor>
auto peg_visit(Visitor && vis, const common_peg_parser_variant & v)
    -> decltype(vis(*(const common_peg_epsilon_parser*)nullptr))
{
    switch (v.tag()) {
        case PEG_TAG_EPSILON:    return vis(*v.get_if<common_peg_epsilon_parser>());
        case PEG_TAG_START:      return vis(*v.get_if<common_peg_start_parser>());
        case PEG_TAG_END:        return vis(*v.get_if<common_peg_end_parser>());
        case PEG_TAG_LITERAL:    return vis(*v.get_if<common_peg_literal_parser>());
        case PEG_TAG_SEQUENCE:   return vis(*v.get_if<common_peg_sequence_parser>());
        case PEG_TAG_CHOICE:     return vis(*v.get_if<common_peg_choice_parser>());
        case PEG_TAG_REPETITION: return vis(*v.get_if<common_peg_repetition_parser>());
        case PEG_TAG_AND:        return vis(*v.get_if<common_peg_and_parser>());
        case PEG_TAG_NOT:        return vis(*v.get_if<common_peg_not_parser>());
        case PEG_TAG_ANY:        return vis(*v.get_if<common_peg_any_parser>());
        case PEG_TAG_SPACE:      return vis(*v.get_if<common_peg_space_parser>());
        case PEG_TAG_CHARS:      return vis(*v.get_if<common_peg_chars_parser>());
        case PEG_TAG_STRING:     return vis(*v.get_if<common_peg_string_parser>());
        case PEG_TAG_UNTIL:      return vis(*v.get_if<common_peg_until_parser>());
        case PEG_TAG_SCHEMA:     return vis(*v.get_if<common_peg_schema_parser>());
        case PEG_TAG_RULE:       return vis(*v.get_if<common_peg_rule_parser>());
        case PEG_TAG_REF:        return vis(*v.get_if<common_peg_ref_parser>());
        case PEG_TAG_ATOMIC:     return vis(*v.get_if<common_peg_atomic_parser>());
        case PEG_TAG_TAG:        return vis(*v.get_if<common_peg_tag_parser>());
        default: assert(false && "Unknown parser tag"); return vis(*v.get_if<common_peg_epsilon_parser>());
    }
}

class common_peg_arena {
    std::vector<common_peg_parser_variant> parsers_;
    std::unordered_map<std::string, common_peg_parser_id> rules_;
    common_peg_parser_id root_ = COMMON_PEG_INVALID_PARSER_ID;

  public:
    const common_peg_parser_variant & get(common_peg_parser_id id) const { return parsers_.at(id); }
    common_peg_parser_variant & get(common_peg_parser_id id) { return parsers_.at(id); }

    size_t size() const { return parsers_.size(); }
    bool empty() const { return parsers_.empty(); }

    common_peg_parser_id get_rule(const std::string & name) const;
    bool has_rule(const std::string & name) const { return rules_.find(name) != rules_.end(); }

    common_peg_parser_id root() const { return root_; }
    void set_root(common_peg_parser_id id) { root_ = id; }

    common_peg_parse_result parse(common_peg_parse_context & ctx, size_t start = 0) const;
    common_peg_parse_result parse(common_peg_parser_id id, common_peg_parse_context & ctx, size_t start) const;

    void resolve_refs();

    void build_grammar(const common_grammar_builder & builder, bool lazy = false) const;

    std::string dump(common_peg_parser_id id) const;

    nlohmann::json to_json() const;
    static common_peg_arena from_json(const nlohmann::json & j);

    std::string save() const;
    void load(const std::string & data);

    friend class common_peg_parser_builder;

  private:
    std::string dump_impl(common_peg_parser_id id, std::unordered_set<common_peg_parser_id> & visited) const;

    common_peg_parser_id add_parser(common_peg_parser_variant parser);
    void add_rule(const std::string & name, common_peg_parser_id id);

    common_peg_parser_id resolve_ref(common_peg_parser_id id);
};

class common_peg_parser_builder {
    common_peg_arena arena_;

    common_peg_parser wrap(common_peg_parser_id id) { return common_peg_parser(id, *this); }
    common_peg_parser add(const common_peg_parser_variant & p) { return wrap(arena_.add_parser(p)); }

  public:
    common_peg_parser_builder();

    // Match nothing, always succeed.
    //   S -> ε
    common_peg_parser eps() { return add(common_peg_epsilon_parser{}); }

    // Matches the start of the input.
    //   S -> ^
    common_peg_parser start() { return add(common_peg_start_parser{}); }

    // Matches the end of the input.
    //   S -> $
    common_peg_parser end() { return add(common_peg_end_parser{}); }

    // Matches an exact literal string.
    //   S -> "hello"
    common_peg_parser literal(const std::string & literal) { return add(common_peg_literal_parser{literal}); }

    // Matches a sequence of parsers in order, all must succeed.
    //   S -> A B C
    common_peg_parser sequence() { return add(common_peg_sequence_parser{}); }
    common_peg_parser sequence(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser sequence(const std::vector<common_peg_parser> & parsers);
    common_peg_parser sequence(std::initializer_list<common_peg_parser> parsers);

    // Matches the first parser that succeeds from a list of alternatives.
    //   S -> A | B | C
    common_peg_parser choice() { return add(common_peg_choice_parser{}); }
    common_peg_parser choice(const std::vector<common_peg_parser_id> & parsers);
    common_peg_parser choice(const std::vector<common_peg_parser> & parsers);
    common_peg_parser choice(std::initializer_list<common_peg_parser> parsers);

    // Matches one or more repetitions of a parser.
    //   S -> A+
    common_peg_parser one_or_more(const common_peg_parser & p) { return repeat(p, 1, -1); }

    // Matches zero or more repetitions of a parser, always succeeds.
    //   S -> A*
    common_peg_parser zero_or_more(const common_peg_parser & p) { return repeat(p, 0, -1); }

    // Matches zero or one occurrence of a parser, always succeeds.
    //   S -> A?
    common_peg_parser optional(const common_peg_parser & p) { return repeat(p, 0, 1); }

    // Positive lookahead: succeeds if child parser succeeds, consumes no input.
    //   S -> &A
    common_peg_parser peek(const common_peg_parser & p) { return add(common_peg_and_parser{p}); }

    // Negative lookahead: succeeds if child parser fails, consumes no input.
    //   S -> !A
    common_peg_parser negate(const common_peg_parser & p) { return add(common_peg_not_parser{p}); }

    // Matches any single character.
    //   S -> .
    common_peg_parser any() { return add(common_peg_any_parser{}); }

    // Matches between min and max repetitions of characters from a character class.
    //   S -> [a-z]{m,n}
    //
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser chars(const std::string & classes, int min = 1, int max = -1);

    // Creates a lightweight reference to a named rule (resolved during build()).
    // Use this for forward references in recursive grammars.
    //   expr_ref -> expr
    common_peg_parser ref(const std::string & name) { return add(common_peg_ref_parser{name}); }

    // Matches zero or more whitespace characters (space, tab, newline).
    //   S -> [ \t\n]*
    common_peg_parser space() { return add(common_peg_space_parser{}); }

    // Matches all characters until a delimiter is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until(const std::string & delimiter) { return add(common_peg_until_parser{{delimiter}}); }

    // Matches all characters until one of the delimiters in the list is found (delimiter not consumed).
    //   S -> (!delim .)*
    common_peg_parser until_one_of(const std::vector<std::string> & delimiters) { return add(common_peg_until_parser{delimiters}); }

    // Matches everything
    //   S -> .*
    common_peg_parser rest() { return until_one_of({}); }

    // Matches between min and max repetitions of a parser (inclusive).
    //   S -> A{m,n}
    // Use -1 for max to represent unbounded repetition (equivalent to {m,})
    common_peg_parser repeat(const common_peg_parser & p, int min, int max) { return add(common_peg_repetition_parser{p, min,max}); }

    // Matches exactly n repetitions of a parser.
    //   S -> A{n}
    common_peg_parser repeat(const common_peg_parser & p, int n) { return repeat(p, n, n); }

    // Matches a double-quoted string: '"' content '"' space
    common_peg_parser double_quoted_string();

    // Matches a single-quoted string: "'" content "'" space
    common_peg_parser single_quoted_string();

    // Matches a string that accepts both double-quoted and single-quoted styles.
    common_peg_parser quoted_string();

    // Matches string content without the surrounding delimiter.
    common_peg_parser string_content(char delimiter);

    // Creates a complete JSON parser supporting objects, arrays, strings, numbers, booleans, and null.
    //   value -> object | array | string | number | true | false | null
    common_peg_parser json();
    common_peg_parser json_object();
    common_peg_parser json_string();
    common_peg_parser json_array();
    common_peg_parser json_number();
    common_peg_parser json_bool();
    common_peg_parser json_null();

    // Matches a JSON object member with a key and associated parser as the
    // value.
    common_peg_parser json_member(const std::string & key, const common_peg_parser & p);

    // Creates a complete Python format parser supporting dicts, arrays, strings, numbers, booleans, and None.
    // Differs from JSON: uses True/False/None, accepts both single and double-quoted strings.
    //   value -> dict | array | string | number | True | False | None
    common_peg_parser python_value();
    common_peg_parser python_dict();
    common_peg_parser python_string();
    common_peg_parser python_array();
    common_peg_parser python_number();
    common_peg_parser python_bool();
    common_peg_parser python_null();

    // A marker, i.e. text delimited by a pair of <> or []
    common_peg_parser marker();

    // Wraps a parser with JSON schema metadata for grammar generation.
    // Used internally to convert JSON schemas to GBNF grammar rules.
    common_peg_parser schema(const common_peg_parser & p, const std::string & name, const nlohmann::ordered_json & schema, bool raw = false);

    // Creates a named rule, stores it in the grammar, and returns a ref.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", json_obj | json_arr | ...)
    common_peg_parser rule(const std::string & name, const common_peg_parser & p, bool trigger = false);

    // Creates a named rule using a builder function, and returns a ref.
    // If trigger=true, marks this rule as an entry point for lazy grammar generation.
    //   auto json = p.rule("json", [&]() { return json_object() | json_array() | ... })
    common_peg_parser rule(const std::string & name, const std::function<common_peg_parser()> & builder, bool trigger = false);

    // Creates a trigger rule. When generating a lazy grammar from the parser,
    // only trigger rules and descendents are emitted.
    common_peg_parser trigger_rule(const std::string & name, const common_peg_parser & p) { return rule(name, p, true); }
    common_peg_parser trigger_rule(const std::string & name, const std::function<common_peg_parser()> & builder) { return rule(name, builder, true); }

    // Creates an atomic parser. Atomic parsers do not create an AST node if
    // the child results in a partial parse, i.e. NEEDS_MORE_INPUT. This is
    // intended for situations where partial output is undesirable.
    common_peg_parser atomic(const common_peg_parser & p) { return add(common_peg_atomic_parser{p}); }

    // Tags create nodes in the generated AST for semantic purposes.
    // Unlike rules, you can tag multiple nodes with the same tag.
    common_peg_parser tag(const std::string & tag, const common_peg_parser & p) { return add(common_peg_tag_parser{p.id(), tag}); }

    void set_root(const common_peg_parser & p);

    common_peg_arena build();
};

// Helper function for building parsers
common_peg_arena build_peg_parser(const std::function<common_peg_parser(common_peg_parser_builder & builder)> & fn);
