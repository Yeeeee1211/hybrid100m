#include "runtime.hpp"
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstring>
#include <iostream>
#include <sstream>
#ifdef _WIN32
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace hybrid {
namespace {
#ifdef _WIN32
using Handle=SOCKET;
constexpr Handle invalid=INVALID_SOCKET;
void close_socket(Handle s) {closesocket(s);}
struct Network {Network(){WSADATA w; if(WSAStartup(MAKEWORD(2,2),&w)) throw std::runtime_error("WSAStartup failed");} ~Network(){WSACleanup();}};
#else
using Handle=int;
constexpr Handle invalid=-1;
void close_socket(Handle s) {close(s);}
struct Network {Network(){std::signal(SIGPIPE,SIG_IGN);}};
#endif
struct Socket {
    Handle fd;
    explicit Socket(Handle h):fd(h) {if(h==invalid) throw std::runtime_error("Socket operation failed");}
    ~Socket(){close_socket(fd);}
    Socket(const Socket&)=delete;
    Socket& operator=(const Socket&)=delete;
};
bool send_all(Handle fd,const std::string& text) {
    size_t off=0;
    while(off<text.size()) {int n=int(send(fd,text.data()+off,int(std::min(size_t(1<<20),text.size()-off)),0)); if(n<=0) return false; off+=size_t(n);}
    return true;
}
void response(Handle fd,int status,const Json& body) {
    auto data=body.dump(); std::string label=status==200 ? "OK" : status==404 ? "Not Found" : "Bad Request";
    send_all(fd,"HTTP/1.1 "+std::to_string(status)+" "+label+"\r\nContent-Type: application/json; charset=utf-8\r\nConnection: close\r\nContent-Length: "+std::to_string(data.size())+"\r\n\r\n"+data);
}
Json error(const std::string& message) {return Json::Object{{"error",Json::Object{{"message",message},{"type","invalid_request_error"}}}};}
struct Request {std::string method,path,body;};
Request read_request(Handle fd) {
    std::string data; char block[4096]; size_t separator;
    while((separator=data.find("\r\n\r\n"))==std::string::npos) {
        if(data.size()>16384) throw std::runtime_error("HTTP headers too large");
        int n=int(recv(fd,block,sizeof(block),0)); if(n<=0) throw std::runtime_error("Incomplete request"); data.append(block,size_t(n));
    }
    if(separator>16384) throw std::runtime_error("HTTP headers too large");
    std::istringstream header(data.substr(0,separator)); Request request; std::string line,version;
    std::getline(header,line); std::istringstream first(line); first>>request.method>>request.path>>version;
    if(version!="HTTP/1.1" && version!="HTTP/1.0") throw std::runtime_error("Invalid HTTP version");
    size_t length=0; bool has_length=false;
    while(std::getline(header,line)) {
        if(!line.empty() && line.back()=='\r') line.pop_back(); auto colon=line.find(':'); if(colon==std::string::npos) throw std::runtime_error("Invalid header");
        std::string key=line.substr(0,colon),value=line.substr(colon+1); for(char& c:key) c=char(std::tolower(static_cast<unsigned char>(c)));
        auto begin=value.find_first_not_of(" \t"); value=begin==std::string::npos ? "" : value.substr(begin); auto end=value.find_last_not_of(" \t"); if(end!=std::string::npos) value.resize(end+1);
        if(key=="transfer-encoding") throw std::runtime_error("Chunked request bodies are not supported; send Content-Length");
        if(key=="content-length") {
            if(has_length || value.empty() || value.find_first_not_of("0123456789")!=std::string::npos) throw std::runtime_error("Invalid Content-Length");
            length=size_t(std::stoull(value)); if(length>1024*1024) throw std::runtime_error("Request body exceeds 1 MiB"); has_length=true;
        }
    }
    request.body=data.substr(separator+4);
    while(request.body.size()<length) {int n=int(recv(fd,block,int(std::min(sizeof(block),length-request.body.size())),0)); if(n<=0) throw std::runtime_error("Incomplete body"); request.body.append(block,size_t(n));}
    request.body.resize(length); return request;
}
Json chunk(const std::string& id,size_t created,const Json& delta,const Json& finish=Json()) {
    return Json::Object{{"id",id},{"object","chat.completion.chunk"},{"created",created},{"model","hybrid100m"},{"choices",Json::Array{Json::Object{{"index",0},{"delta",delta},{"finish_reason",finish}}}}};
}
}
void serve(Runtime& runtime,const std::string& host,int port) {
    if(port<1 || port>65535) throw std::runtime_error("port must be in [1,65535]");
    Network network; Socket listener(socket(AF_INET,SOCK_STREAM,IPPROTO_TCP));
    sockaddr_in address{}; address.sin_family=AF_INET; address.sin_port=htons(uint16_t(port));
    if(inet_pton(AF_INET,host.c_str(),&address.sin_addr)!=1) throw std::runtime_error("--host must be an IPv4 address");
#ifndef _WIN32
    int yes=1; setsockopt(listener.fd,SOL_SOCKET,SO_REUSEADDR,&yes,sizeof(yes));
#endif
    if(bind(listener.fd,reinterpret_cast<sockaddr*>(&address),sizeof(address))!=0 || listen(listener.fd,8)!=0) throw std::runtime_error("Cannot bind/listen on requested address");
    std::cerr<<"hybrid100m HTTP listening on http://"<<host<<':'<<port<<" (one request at a time)\n";
    size_t sequence=0;
    for(;;) {
        Socket client(accept(listener.fd,nullptr,nullptr));
#ifdef _WIN32
        DWORD timeout=30000; setsockopt(client.fd,SOL_SOCKET,SO_RCVTIMEO,reinterpret_cast<const char*>(&timeout),sizeof(timeout)); setsockopt(client.fd,SOL_SOCKET,SO_SNDTIMEO,reinterpret_cast<const char*>(&timeout),sizeof(timeout));
#else
        timeval timeout{30,0}; setsockopt(client.fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout)); setsockopt(client.fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
#endif
        bool streaming=false;
        try {
            auto request=read_request(client.fd);
            if(request.method=="GET" && request.path=="/health") {response(client.fd,200,Json::Object{{"status","ok"},{"model","hybrid100m"}}); continue;}
            if(request.method=="GET" && request.path=="/v1/models") {response(client.fd,200,Json::Object{{"object","list"},{"data",Json::Array{Json::Object{{"id","hybrid100m"},{"object","model"},{"owned_by","local"}}}}}); continue;}
            if(request.method!="POST" || request.path!="/v1/chat/completions") {response(client.fd,404,error("Unknown endpoint")); continue;}
            auto body=Json::parse(request.body);
            // A deliberately small, explicit compatibility surface; don't silently
            // pretend to honor tool calls, multimodal inputs or unknown parameters.
            const std::vector<std::string> supported={"model","messages","stream","max_tokens","max_think_tokens","temperature","top_k","top_p","seed","repeat_penalty","repeat_last_n","stop","enable_thinking","n","stream_options"};
            for(const auto& field:body.object()) if(std::find(supported.begin(),supported.end(),field.first)==supported.end()) throw std::runtime_error("Unsupported request field: "+field.first);
            if(body.has("model") && body.at("model").str()!="hybrid100m") throw std::runtime_error("model must be hybrid100m");
            if(body.has("n") && body.at("n").integer()!=1) throw std::runtime_error("Only n=1 is supported");
            bool include_usage=false;
            if(body.has("stream_options")) {const auto& so=body.at("stream_options"); if(so.object().size()!=1 || !so.has("include_usage")) throw std::runtime_error("Only stream_options.include_usage is supported"); include_usage=so.at("include_usage").boolean();}
            auto options=parse_options(body); auto messages=parse_messages(body.at("messages"));
            auto prompt=runtime.chat_prompt(messages,options.thinking);
            int budget=std::min(options.max_tokens,runtime.context_size()-int(prompt.size()));
            if(budget<(options.thinking ? 3 : 1)) throw std::runtime_error("Prompt leaves insufficient context");
            size_t created=size_t(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
            std::string id="chatcmpl-h100-"+std::to_string(created)+"-"+std::to_string(++sequence);
            streaming=body.has("stream") && body.at("stream").boolean();
            Emit emit;
            if(streaming) {
                if(!send_all(client.fd,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n")) continue;
                if(!send_all(client.fd,"data: "+chunk(id,created,Json::Object{{"role","assistant"}}).dump()+"\n\n")) continue;
                emit=[&](const std::string& channel,const std::string& text) {return send_all(client.fd,"data: "+chunk(id,created,Json::Object{{channel=="think" ? "reasoning_content" : "content",text}}).dump()+"\n\n");};
            }
            auto result=runtime.chat(messages,options,emit);
            Json usage=Json::Object{{"prompt_tokens",result.prompt_tokens},{"completion_tokens",result.ids.size()},{"total_tokens",size_t(result.prompt_tokens)+result.ids.size()}};
            if(streaming) {
                if(result.finish_reason=="cancelled") continue;
                if(!send_all(client.fd,"data: "+chunk(id,created,Json::Object{},result.finish_reason).dump()+"\n\n")) continue;
                if(include_usage) send_all(client.fd,"data: "+Json(Json::Object{{"id",id},{"object","chat.completion.chunk"},{"created",created},{"model","hybrid100m"},{"choices",Json::Array{}},{"usage",usage}}).dump()+"\n\n");
                send_all(client.fd,"data: [DONE]\n\n");
            } else response(client.fd,200,Json::Object{{"id",id},{"object","chat.completion"},{"created",created},{"model","hybrid100m"},{"choices",Json::Array{Json::Object{{"index",0},{"message",Json::Object{{"role","assistant"},{"content",result.answer},{"reasoning_content",result.reasoning}}},{"finish_reason",result.finish_reason}}}},{"usage",usage}});
        } catch(const std::exception& e) {
            if(streaming) send_all(client.fd,"data: "+error(e.what()).dump()+"\n\ndata: [DONE]\n\n");
            else response(client.fd,400,error(e.what()));
        }
    }
}
} // namespace hybrid
