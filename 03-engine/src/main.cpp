#include "runtime.hpp"
#include <fstream>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>
#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

using namespace hybrid;
namespace {
const char* help = R"HELP(hybrid-cpu: standalone C++17 Hybrid100M CPU inference

Usage: hybrid-cpu COMMAND --model model.h100 [options]
Commands:
  chat       Stream a chat response; interactive when --prompt is absent
  complete   Continue raw text (useful before SFT)
  serve      Local HTTP: /health, /v1/models, /v1/chat/completions
  bench      Fixed-length warm CPU prefill/decode benchmark
  info       Model metadata and allocated cache size
  tokenize   Encode --prompt and print IDs + decoded text
  logits     Validation: forward comma-separated --ids, return logits at each step

Common: --threads N (0=auto, at most 8) --ctx N (0=model limit)
Generation: --prompt TEXT | --prompt-file FILE, --system TEXT, --json
  --max-tokens 256 --temperature 0.8 --top-k 40 --top-p 0.95 --seed 42
  --repeat-penalty 1.0 --repeat-last-n 64 --stop TEXT
  --thinking --max-think-tokens 128 --hide-thinking
Server: --host 127.0.0.1 --port 8080
Bench: --prompt-tokens 128 --decode-tokens 32 --repeats 3
Interactive: /reset, /exit; Ctrl+C exits. All text uses UTF-8.
)HELP";

struct Args {
    std::map<std::string,std::string> values;
    bool has(const std::string& key) const { return values.count(key)!=0; }
    std::string get(const std::string& key,const std::string& fallback="") const {auto it=values.find(key); return it==values.end() ? fallback : it->second;}
    int integer(const std::string& key,int fallback) const {if(!has(key)) return fallback; size_t n=0; int v=std::stoi(get(key),&n); if(n!=get(key).size()) throw std::runtime_error("Invalid integer: "+key); return v;}
    double number(const std::string& key,double fallback) const {if(!has(key)) return fallback; size_t n=0; double v=std::stod(get(key),&n); if(n!=get(key).size()) throw std::runtime_error("Invalid number: "+key); return v;}
};
Args parse(int argc,char** argv) {
    std::set<std::string> flags{"thinking","hide-thinking","json"};
    std::set<std::string> values{"model","threads","ctx","prompt","prompt-file","system","max-tokens","max-think-tokens","temperature","top-k","top-p","seed","repeat-penalty","repeat-last-n","stop","host","port","prompt-tokens","decode-tokens","repeats","ids"};
    Args out;
    for(int i=2;i<argc;++i) {
        std::string key=argv[i]; if(key.substr(0,2)!="--") throw std::runtime_error("Expected --option: "+key); key=key.substr(2);
        if(out.has(key)) throw std::runtime_error("Duplicate option: "+key);
        if(flags.count(key)) out.values[key]="true";
        else if(values.count(key) && i+1<argc) out.values[key]=argv[++i];
        else throw std::runtime_error("Unknown option or missing value: --"+key);
    }
    if(!out.has("model")) throw std::runtime_error("--model is required");
    if(out.has("prompt") && out.has("prompt-file")) throw std::runtime_error("Choose --prompt or --prompt-file");
    return out;
}
std::string prompt(const Args& args) {
    if(!args.has("prompt-file")) return args.get("prompt");
    if(args.get("prompt-file")=="-") return std::string(std::istreambuf_iterator<char>(std::cin),{});
    std::ifstream f(std::filesystem::u8path(args.get("prompt-file")),std::ios::binary); if(!f) throw std::runtime_error("Cannot open prompt file");
    return std::string(std::istreambuf_iterator<char>(f),{});
}
GenerationOptions options(const Args& a) {
    GenerationOptions o;
    o.max_tokens=a.integer("max-tokens",256); o.max_think_tokens=a.integer("max-think-tokens",128); o.thinking=a.has("thinking");
    o.temperature=a.number("temperature",0.8); o.top_k=a.integer("top-k",40); o.top_p=a.number("top-p",0.95);
    o.repeat_penalty=a.number("repeat-penalty",1); o.repeat_last_n=a.integer("repeat-last-n",64);
    int seed=a.integer("seed",42); if(seed<0) throw std::runtime_error("seed must be nonnegative"); o.seed=uint32_t(seed);
    if(a.has("stop")) o.stop.push_back(a.get("stop")); o.validate(); return o;
}
Result display(Runtime& runtime,const Args& args,const GenerationOptions& opt,const std::vector<Message>& messages,const std::string* raw=nullptr) {
    std::string channel;
    Emit emit;
    if(!args.has("json")) emit=[&](const std::string& next,const std::string& text) {
        if(next=="think" && args.has("hide-thinking")) return true;
        if(next!=channel) {
            if(channel=="think") std::cout << "</think>\n";
            if(next=="think") std::cout << "<think>\n";
            channel=next;
        }
        std::cout << text << std::flush; return true;
    };
    auto result=raw ? runtime.complete(*raw,opt,emit) : runtime.chat(messages,opt,emit);
    if(args.has("json")) std::cout << result.json().dump() << '\n';
    else {if(channel=="think") std::cout << "</think>"; std::cout << '\n';}
    std::cerr << "[prompt=" << result.prompt_tokens << ", generated=" << result.ids.size() << ", prefill=" << result.prefill_ms << " ms, decode+output=" << result.decode_ms << " ms, finish=" << result.finish_reason << "]\n";
    return result;
}
}
int main(int argc,char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8); SetConsoleCP(CP_UTF8);
    // The CRT's narrow argv uses the legacy system code page. Read the real
    // UTF-16 command line so Chinese prompts/paths and emoji survive intact.
    int wide_count=0;
    wchar_t** wide=CommandLineToArgvW(GetCommandLineW(),&wide_count);
    if(!wide) {std::cerr<<"Cannot read Unicode command line\n"; return 1;}
    std::vector<std::string> utf8_args;
    for(int i=0;i<wide_count;++i) {
        int n=WideCharToMultiByte(CP_UTF8,0,wide[i],-1,nullptr,0,nullptr,nullptr);
        std::string text(size_t(n),'\0'); WideCharToMultiByte(CP_UTF8,0,wide[i],-1,text.data(),n,nullptr,nullptr);
        text.pop_back(); utf8_args.push_back(std::move(text));
    }
    LocalFree(wide);
    std::vector<char*> pointers; for(auto& arg:utf8_args) pointers.push_back(arg.data());
    argc=wide_count; argv=pointers.data();
#endif
    try {
        if(argc<2 || std::string(argv[1])=="--help" || std::string(argv[1])=="help") {std::cout<<help; return 0;}
        std::string command=argv[1]; auto args=parse(argc,argv);
        Runtime runtime(args.get("model"),args.integer("threads",0),args.integer("ctx",0));
        if(command=="info") {std::cout << Json(Json::Object{{"metadata",runtime.metadata()},{"threads",runtime.thread_count()},{"context",runtime.context_size()},{"cache_bytes",runtime.cache_bytes()}}).dump() << '\n'; return 0;}
        if(command=="tokenize") {auto ids=runtime.encode(prompt(args)); Json::Array a; for(int id:ids) a.emplace_back(id); std::cout<<Json(Json::Object{{"ids",a},{"text",runtime.decode(ids)}}).dump()<<'\n'; return 0;}
        if(command=="logits") {
            if(!args.has("ids")) throw std::runtime_error("--ids is required");
            std::istringstream stream(args.get("ids")); std::string item; Json::Array rows;
            while(std::getline(stream,item,',')) {size_t n=0; int id=std::stoi(item,&n); if(n!=item.size()) throw std::runtime_error("Invalid token ID"); Json::Array row; for(float f:runtime.forward(id)) row.emplace_back(double(f)); rows.emplace_back(std::move(row));}
            if(rows.empty()) throw std::runtime_error("Empty IDs"); std::cout<<Json(rows).dump()<<'\n'; return 0;
        }
        if(command=="bench") {std::cout<<runtime.benchmark(args.integer("prompt-tokens",128),args.integer("decode-tokens",32),args.integer("repeats",3)).dump()<<'\n'; return 0;}
        if(command=="serve") {serve(runtime,args.get("host","127.0.0.1"),args.integer("port",8080)); return 0;}
        if(command!="chat" && command!="complete") throw std::runtime_error("Unknown command: "+command);
        auto opt=options(args); std::vector<Message> messages;
        if(args.has("system")) messages.push_back({"system",args.get("system")});
        if(command=="complete" || args.has("prompt") || args.has("prompt-file")) {
            auto text=prompt(args);
            if(command=="complete") display(runtime,args,opt,messages,&text);
            else {messages.push_back({"user",text}); display(runtime,args,opt,messages);}
            return 0;
        }
        if(args.has("json")) throw std::runtime_error("Interactive chat does not support --json; provide --prompt");
        std::cerr<<"Interactive chat: /reset clears history; /exit quits.\n";
        for(std::string text;;) {
            std::cout<<"\nyou> "<<std::flush; if(!std::getline(std::cin,text) || text=="/exit") break;
            if(text=="/reset") {messages.clear(); if(args.has("system")) messages.push_back({"system",args.get("system")}); runtime.reset(); continue;}
            if(text.empty()) continue;
            messages.push_back({"user",text}); std::cout<<"assistant> "<<std::flush;
            try {auto result=display(runtime,args,opt,messages); messages.push_back({"assistant",result.answer});}
            catch(const std::exception& e) {messages.pop_back(); std::cerr<<"Error: "<<e.what()<<" (use /reset to clear history)\n";}
        }
        return 0;
    } catch(const std::exception& e) {std::cerr<<"Error: "<<e.what()<<'\n'; return 1;}
}
