#pragma once
#include "json.hpp"
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace hybrid {
using Vec = std::vector<float>;
struct Message { std::string role, content; };
struct GenerationOptions {
    int max_tokens = 256, max_think_tokens = 128, top_k = 40;
    double temperature = 0.8, top_p = 0.95, repeat_penalty = 1.0;
    int repeat_last_n = 64;
    uint32_t seed = 42;
    bool thinking = false;
    std::vector<std::string> stop;
    void validate() const;
};
struct Result {
    std::string answer, reasoning, finish_reason = "length";
    std::vector<int> ids;
    int prompt_tokens = 0;
    bool think_truncated = false;
    double prefill_ms = 0, decode_ms = 0;
    Json json() const;
};
// Callback returns false to cancel generation (e.g. a disconnected HTTP client).
using Emit = std::function<bool(const std::string& channel, const std::string& text)>;

class Runtime {
public:
    Runtime(const std::string& path, int threads = 0, int context = 0);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    const Json& metadata() const;
    int context_size() const;
    int thread_count() const;
    int seen_tokens() const;
    size_t cache_bytes() const;
    void reset();
    // CPU decode mutates a single session's preallocated cache; not thread-safe.
    Vec forward(int token, bool logits = true);
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;
    std::vector<int> chat_prompt(const std::vector<Message>& messages, bool thinking) const;
    Result chat(const std::vector<Message>& messages, const GenerationOptions& options, Emit emit = {});
    Result complete(const std::string& prompt, const GenerationOptions& options, Emit emit = {});
    Json benchmark(int prompt_tokens, int generated_tokens, int repeats);
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    Result generate(const std::vector<int>& prompt, const GenerationOptions& options, bool chat, Emit emit);
};

std::vector<Message> parse_messages(const Json& value);
GenerationOptions parse_options(const Json& value);
void serve(Runtime& runtime, const std::string& host, int port);
} // namespace hybrid
