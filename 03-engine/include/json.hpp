// Small strict JSON value/parser for the model header and local HTTP protocol.
// No third-party runtime dependencies. Strings use UTF-8; numbers must be finite.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace hybrid {
inline void append_utf8(std::string& s, uint32_t c) {
    if (c <= 0x7f) s += char(c);
    else if (c <= 0x7ff) { s += char(0xc0 | (c >> 6)); s += char(0x80 | (c & 63)); }
    else if (c <= 0xffff) { s += char(0xe0 | (c >> 12)); s += char(0x80 | ((c >> 6) & 63)); s += char(0x80 | (c & 63)); }
    else { s += char(0xf0 | (c >> 18)); s += char(0x80 | ((c >> 12) & 63)); s += char(0x80 | ((c >> 6) & 63)); s += char(0x80 | (c & 63)); }
}

class Json {
public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;
    Json() : value(nullptr) {}
    Json(std::nullptr_t) : value(nullptr) {}
    Json(bool x) : value(x) {}
    Json(int x) : value(double(x)) {}
    Json(size_t x) : value(double(x)) {}
    Json(double x) : value(x) { if (!std::isfinite(x)) throw std::runtime_error("Nonfinite JSON number"); }
    Json(const char* x) : value(std::string(x)) {}
    Json(std::string x) : value(std::move(x)) {}
    Json(Array x) : value(std::move(x)) {}
    Json(Object x) : value(std::move(x)) {}
    bool null() const { return std::holds_alternative<std::nullptr_t>(value); }
    bool is_string() const { return std::holds_alternative<std::string>(value); }
    const std::string& str() const { return std::get<std::string>(value); }
    double number() const { return std::get<double>(value); }
    int integer() const {
        double d = number();
        if (d < std::numeric_limits<int>::min() || d > std::numeric_limits<int>::max() || std::floor(d) != d)
            throw std::runtime_error("Expected a bounded integer");
        return int(d);
    }
    bool boolean() const { return std::get<bool>(value); }
    const Array& array() const { return std::get<Array>(value); }
    const Object& object() const { return std::get<Object>(value); }
    bool has(const std::string& key) const { return object().count(key) != 0; }
    const Json& at(const std::string& key) const { return object().at(key); }
    std::string dump() const {
        if (null()) return "null";
        if (auto b = std::get_if<bool>(&value)) return *b ? "true" : "false";
        if (auto n = std::get_if<double>(&value)) { std::ostringstream s; s << std::setprecision(17) << *n; return s.str(); }
        if (auto s = std::get_if<std::string>(&value)) {
            std::string out = "\"";
            const char* hex = "0123456789abcdef";
            for (unsigned char c : *s) {
                if (c == '"' || c == '\\') { out += '\\'; out += char(c); }
                else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
                else out += char(c);
            }
            return out + '"';
        }
        std::string out;
        if (auto a = std::get_if<Array>(&value)) {
            out = "["; for (const auto& x : *a) { if (out.size() > 1) out += ','; out += x.dump(); } return out + ']';
        }
        out = "{";
        for (const auto& x : object()) { if (out.size() > 1) out += ','; out += Json(x.first).dump() + ':' + x.second.dump(); }
        return out + '}';
    }
    static Json parse(const std::string& text);
};

inline Json Json::parse(const std::string& text) {
    struct Parser {
        const std::string& s; size_t p = 0;
        [[noreturn]] void fail() { throw std::runtime_error("Invalid JSON at byte " + std::to_string(p)); }
        void ws() { while (p < s.size() && (s[p]==' ' || s[p]=='\r' || s[p]=='\n' || s[p]=='\t')) ++p; }
        bool eat(char c) { ws(); if (p < s.size() && s[p] == c) { ++p; return true; } return false; }
        uint32_t hex4() {
            uint32_t n = 0;
            for (int i = 0; i < 4; ++i) {
                if (p == s.size()) fail();
                char c = s[p++]; int v = c >= '0' && c <= '9' ? c-'0' : c >= 'a' && c <= 'f' ? c-'a'+10 : c >= 'A' && c <= 'F' ? c-'A'+10 : -1;
                if (v < 0) fail(); n = (n << 4) | uint32_t(v);
            }
            return n;
        }
        std::string string() {
            if (!eat('"')) fail();
            std::string out;
            while (p < s.size()) {
                unsigned char c = s[p++];
                if (c == '"') return out;
                if (c < 32) fail();
                if (c != '\\') { out += char(c); continue; }
                if (p == s.size()) fail(); c = s[p++];
                switch(c) {
                    case '"': case '\\': case '/': out += char(c); break;
                    case 'b': out += '\b'; break; case 'f': out += '\f'; break;
                    case 'n': out += '\n'; break; case 'r': out += '\r'; break; case 't': out += '\t'; break;
                    case 'u': {
                        uint32_t cp = hex4();
                        if (cp >= 0xd800 && cp <= 0xdbff) {
                            if (s.substr(p, 2) != "\\u") fail(); p += 2; uint32_t low = hex4();
                            if (low < 0xdc00 || low > 0xdfff) fail(); cp = 0x10000 + ((cp-0xd800)<<10) + low-0xdc00;
                        } else if (cp >= 0xdc00 && cp <= 0xdfff) fail();
                        append_utf8(out, cp); break;
                    }
                    default: fail();
                }
            }
            fail();
        }
        Json read(int depth) {
            if (depth > 64) fail(); ws(); if (p == s.size()) fail();
            char c = s[p];
            if (c == '"') return string();
            if (eat('{')) {
                Object o; if (eat('}')) return o;
                do { auto k = string(); if (!eat(':')) fail(); auto v = read(depth+1); if (!o.emplace(k, std::move(v)).second) fail(); } while (eat(','));
                if (!eat('}')) fail(); return o;
            }
            if (eat('[')) {
                Array a; if (eat(']')) return a;
                do { a.push_back(read(depth+1)); } while (eat(',')); if (!eat(']')) fail(); return a;
            }
            for (auto word : {"true", "false", "null"}) {
                std::string w(word); if (s.compare(p, w.size(), w) == 0) { p += w.size(); return w == "null" ? Json() : Json(w == "true"); }
            }
            size_t begin = p;
            if (s[p] == '-') ++p;
            if (p == s.size()) fail();
            if (s[p] == '0') ++p;
            else { if (s[p] < '1' || s[p] > '9') fail(); while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p; }
            if (p < s.size() && s[p] == '.') { ++p; size_t b = p; while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p; if (b == p) fail(); }
            if (p < s.size() && (s[p]=='e' || s[p]=='E')) {
                ++p; if (p < s.size() && (s[p]=='+' || s[p]=='-')) ++p;
                size_t b = p; while (p < s.size() && s[p] >= '0' && s[p] <= '9') ++p; if (b == p) fail();
            }
            return Json(std::stod(s.substr(begin, p-begin)));
        }
    } parser{text};
    auto result = parser.read(0); parser.ws(); if (parser.p != text.size()) parser.fail(); return result;
}
} // namespace hybrid
