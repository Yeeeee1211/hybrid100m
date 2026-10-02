# Hybrid100M C++ CPU 推理框架

这是为当前模型实现的独立 **C++17** 推理程序。运行时只需要可执行文件和一个 `.h100` 模型文件，**不依赖 Python、PyTorch、CUDA、FLA 或外部分词服务**。Python 只用于离线导出权重和数值对照测试。

提供流式命令行聊天、多轮历史、原始文本续写、可见 think、采样、FP32/Q8 导出、CPU 线程设置、测速，以及本地 HTTP/SSE 接口。`.h100` 是本项目的版本化文件格式，不是 GGUF，不能直接交给 llama.cpp 加载。

推理主干现支持 **KDA 逐通道遗忘门、低秩门投影、sigmoid 输出门，以及 YaRN RoPE**；旧 Gated DeltaNet / 原始 RoPE 权重仍可加载。训练用 MTP 辅助层不会导出到推理程序，因此普通自回归推理不增加 MTP 计算。分词行为、训练时的 tokenizer 文件和 token IDs 均保持一致。

| 配置节点 | C++ 推理计算 |
|---|---|
| `linear_attention="kda"` | `a_up(a_down(x))` 产生逐 key 维遗忘门，递归状态逐行衰减后进行 delta 更新 |
| KDA 输出门 | `sigmoid(z_up(z_down(x)))` 乘以按头 RMSNorm 后的递归输出 |
| `linear_attention="gated_delta"` | 保留旧的逐头标量衰减与 SiLU 输出门 |
| `rope_type="yarn"` | 根据原始窗口、factor、beta_fast/beta_slow 混合频率，并对 cos/sin 乘 YaRN attention factor |
| `rope_type="default"` | 保留原始 RoPE 频率及幅度 |
| `mtp_num_layers` | 仅保留训练配置元数据，推理使用主干，不运行 MTP 层 |

## 1. 编译或使用已有程序

本次在 Windows x64 上编译并实测的通用程序位于 `cpu/build/hybrid-cpu.exe`，另有针对本机指令集优化的 `cpu/build/hybrid-cpu-native.exe`。后者速度更快，但只适用于本机或兼容指令集的 CPU；跨机器部署优先使用通用程序或在目标机重新编译。`cpu/build/hybrid-cpu-linux-x64` 是交叉编译的 Linux x64 程序，尚未在 Linux 上运行验证。构建目录不纳入源码版本管理；自行复制部署时，带上适合目标系统的可执行文件和导出的权重即可。

自己编译需要 CMake >= 3.16 和 C++17 编译器。从项目根目录执行：

```sh
cmake -S cpu -B cpu/build-release -DCMAKE_BUILD_TYPE=Release
cmake --build cpu/build-release --config Release -j
```

Linux 的程序路径为 `cpu/build-release/hybrid-cpu`；Windows Visual Studio 生成器通常输出 `cpu/build-release/Release/hybrid-cpu.exe`。Windows 需安装“使用 C++ 的桌面开发”构建工具。源码没有需要在线下载的 C/C++ 库。

如果只在编译机器或兼容 CPU 上使用，可添加 `-DHYBRID_NATIVE=ON`：GCC/Clang 使用 `-march=native`，MSVC 使用 `/arch:AVX2`。AVX2 构建对矩阵向量乘法提供显式 SIMD 路径；通用构建保留标量/编译器自动向量化路径。不要把针对新指令集的二进制直接部署到旧 CPU。

## 2. 导出训练权重

在训练环境或装有 CPU PyTorch 的机器上执行。使用 `train.py` 保存的 **`model.pt` + `config.json`**，以及训练时原封不动的 `tokenizer.json`。不需要训练器状态 `ckpt.pt`。

```sh
python -m pip install tokenizers numpy
python cpu/export.py --run-dir runs/pre_2k --tokenizer tokenizer.json --output models/hybrid100m-q8.h100 --quant q8
```

默认推荐先导出 Q8；需要保留 FP32 对照时：

```sh
python cpu/export.py --run-dir runs/pre_2k --tokenizer tokenizer.json --output models/hybrid100m-f32.h100 --quant f32
```

导出包含权重、配置、分词表、BPE 合并顺序、特殊 token IDs 和 Unicode 分类表。Unicode 表由实际使用的 Hugging Face tokenizer 生成，因此不会把 Python 与 Rust 不同版本的 Unicode 分类混用。导出器只接受项目 `train_tokenizer.py` 对应的 ByteLevel BPE 配置，遇到不支持的预处理或不匹配的权重会报错。

导出器先校验完整训练 checkpoint 的张量名称、形状和有限数值，再移除 `mtp.*` 训练辅助参数。元数据保存 `inference_branch="main"` 和 `omitted_training_tensors`，并保留训练配置供追溯。推理程序只检查、加载主干张量。新结构需重新导出，并使用本次重新编译的程序；旧二进制不支持新的 KDA 张量名称。

若旧 tokenizer 缺少某些基础字节，导出时会警告；C++ 在遇到无法表示的输入字节时明确报错。**不要为了通过导出而修改已训练模型的词表或 token IDs。** 新训练的 tokenizer 宜包含完整 256 字节字母表；旧模型仍需使用原 tokenizer。

## 3. 对话与流式输出

以下 Windows 示例从项目根目录运行；Linux 将程序路径换成自己的 `hybrid-cpu` 路径。

```powershell
# 单次回答：输出会随解码逐步显示
.\cpu\build\hybrid-cpu.exe chat --model models/hybrid100m-q8.h100 --threads 4 --ctx 2048 --prompt "你好，请介绍一下自己。" --max-tokens 128

# 多轮交互：/reset 清空历史，/exit 退出
.\cpu\build\hybrid-cpu.exe chat --model models/hybrid100m-q8.h100 --threads 4 --ctx 4096

# 输出模型生成的 think，然后输出答案
.\cpu\build\hybrid-cpu.exe chat --model models/hybrid100m-q8.h100 --prompt "小明有2个苹果，又买了3个，现在有几个？" --thinking --max-think-tokens 64 --max-tokens 192

# 仅预训练模型先测原始续写；不套聊天/SFT模板
.\cpu\build\hybrid-cpu.exe complete --model models/hybrid100m-q8.h100 --prompt "春天来了，" --max-tokens 128
```

`--prompt-file prompt.txt` 读取 UTF-8 文件，`--prompt-file -` 从标准输入读取。默认流式文本写 stdout，耗时写 stderr。`--json` 改为一次输出完整结果，含 answer、reasoning、生成 token IDs、结束原因和耗时。`--hide-thinking` 只隐藏终端中的 think 文本，仍然执行这部分生成；`--json` 仍含 reasoning 字段。

| 选项 | 默认值 | 作用 |
|---|---:|---|
| `--threads` | 0 | 自动选至多 8 个线程；可用 1/2/4/8 实测比较 |
| `--ctx` | 0 | 使用模型最大上下文；可设 2048/4096/10240，不能超过模型配置 |
| `--max-tokens` | 256 | 新生成 token 总预算，包含 think 和控制标记 |
| `--temperature` | 0.8 | 设 0 使用贪心解码 |
| `--top-k` / `--top-p` | 40 / 0.95 | 采样候选过滤；top-k=0 关闭 top-k |
| `--seed` | 42 | 同一程序/平台上重复随机采样；不保证跨标准库逐 token 相同 |
| `--repeat-penalty` | 1.0 | 大于 1 时惩罚最近重复 token |
| `--repeat-last-n` | 64 | 重复惩罚回看窗口，包括 prompt 和已生成内容 |
| `--stop` | 无 | 遇到指定答案文本停止；停止文本不会输出 |
| `--thinking` | 关闭 | 启用已定义的 think 协议 |
| `--max-think-tokens` | 128 | 到预算后强制进入 answer 阶段 |

UTF-8 字节跨 token 时会缓存到字符完整再显示，停止词也可跨多个 token。聊天遇到 `</answer>` 停止；原始续写遇到 EOS 停止。think 内容是模型生成的说明，能力和正确性仍取决于训练与 SFT。

多轮历史使用与 Python `ChatTemplate` 一致的编码：历史 assistant 只保留答案。每轮重新预填充历史，轮内复用递归状态和 KV 缓存。上下文不足会报错，可用 `/reset` 清空；不会静默丢掉旧消息，也不会扩展已训练的 RoPE 上限。

## 4. 本地 HTTP / SSE

```powershell
.\cpu\build\hybrid-cpu.exe serve --model models/hybrid100m-q8.h100 --threads 4 --ctx 4096 --host 127.0.0.1 --port 8080
```

服务提供 `GET /health`、`GET /v1/models` 和 `POST /v1/chat/completions`。这是常用聊天接口的一个子集：文本 messages、`n=1`、流式/非流式响应。模型名为 `hybrid100m`。

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"hybrid100m","messages":[{"role":"user","content":"你好"}],"max_tokens":128,"temperature":0.8,"stream":true,"stream_options":{"include_usage":true}}'
```

上例是 Bash 的引号写法。PowerShell 可使用：

```powershell
$body = @{model="hybrid100m"; messages=@(@{role="user"; content="你好"}); max_tokens=128; stream=$false} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

额外支持 `enable_thinking`、`max_think_tokens`、`top_k`、`repeat_penalty`、`repeat_last_n`、`seed`；`stop` 可以是字符串或最多 16 个字符串。think 通过 `reasoning_content` 返回，答案使用 `content`。SSE 以 `[DONE]` 结束，可通过 `stream_options.include_usage` 获取包含控制标记在内的 token 用量。

当前服务逐个处理请求，每个请求独立重置状态；客户端需传完整历史。没有并发批处理、工具调用、多模态、鉴权或 TLS，默认绑定本机地址，适合作为本地应用后端。未实现的请求字段会报错。客户端断开且输出失败时会停止该次生成；长 prompt 的预填充期间不会主动探测断开。

## 5. CPU 性能与内存

```powershell
.\cpu\build\hybrid-cpu.exe info --model models/hybrid100m-q8.h100 --ctx 10240
.\cpu\build\hybrid-cpu.exe bench --model models/hybrid100m-q8.h100 --threads 4 --ctx 2048 --prompt-tokens 128 --decode-tokens 32 --repeats 3
```

`bench` 先预热，再分别测 prompt 处理与逐 token 解码，输出 JSON。它忽略 EOS，按固定 token 数计算，所以不会因随机模型提前停止而得到误导结果。聊天返回的 `decode_ms` 还包含采样和输出回调耗时，不能直接等同于纯推理吞吐。

Q8 使用**矩阵逐行对称 INT8 权重 + FP32 scale**，计算时逐行读取并进行浮点乘加，不把整份权重解量化回 FP32。embedding 与输出头共享同一份量化权重；卷积、归一化、RoPE、激活、KV 缓存和 KDA/Delta 状态均为 FP32。KDA 的低秩门矩阵同样支持逐行 Q8。它不是 GGUF Q8_0，也不是 INT8 激活/INT8 GEMM；没有 Q4 支持。量化质量需用训练后的评测集单独检查。

新默认 20 层结构的预分配缓存为 **54.10 MiB（ctx=10240）**，其中 5 层标准注意力 K/V 约 50 MiB，其余是 KDA 状态与卷积历史。这是缓存字节数，不是进程总内存。低秩门没有跨 token 的额外缓存。旧 16 层结构的相应缓存为 43.28 MiB；`info` 可读取导出模型的实际配置和缓存字节数。文件大小以导出器输出为准，MTP 参数不计入推理文件。

本版本使用逐 token 预填充。它避免了全长 T×T 注意力矩阵，但长 prompt 仍需逐 token 执行模型，标准注意力层的累计预填充工作量仍随长度平方增长。尚无批量矩阵预填充、跨请求前缀复用、mmap、连续批处理或专门的 ARM NEON 内核。**不能据短上下文测速推断 10K 的首字延迟。**

## 6. 验证与实测范围

```sh
python -m unittest discover -s cpu -p test_runtime.py -v
python cpu/smoke_full.py --binary cpu/build/hybrid-cpu.exe
```

用 `HYBRID_CPU_BINARY` 环境变量可选择另一份已编译程序；未编译时原生测试会跳过。集成测试需要 Python、CPU PyTorch、tokenizers 和本机回环网络权限；测试不会正式训练或修改你的训练权重。

2026-09-30，新默认结构的 Windows x64 / PyTorch 2.4.1+cpu 验证：

- 通用构建的 **13 项集成测试通过**，覆盖 KDA + YaRN 的 FP32/Q8 逐 token 缓存对照、超过原始 RoPE 窗口的位置、训练用 MTP 参数省略，以及原有 GDN/旧 v1、10K、聊天、分词、HTTP/SSE 等行为。
- 完整默认随机模型包含 **104,555,708 个训练参数**；去掉 MTP 辅助层的推理主干为 **100,162,108 参数**。本机优化构建的 5-token 对照最大绝对误差为 FP32 `4.95e-6`、Q8 `4.77e-6`，Q8 对照使用相同量化后解量化的 PyTorch 权重。
- 相应推理文件约 **382.11 MiB（FP32）/ 96.58 MiB（Q8）**，预分配 ctx=10240 缓存实测 **54.10 MiB**，14 个 MTP 训练张量省略。临时 tokenizer 会影响少量文件字节；实际训练 tokenizer 保持原样。
- 完整报告为 `cpu/build/full-smoke-p1.json`，包含本机优化构建的短上下文测速。它验证实现与导出，不证明模型质量、量化无损或 10K 的首字延迟。Linux x64 程序已重新交叉编译，尚未在 Linux 实机上运行。

以下 2026-09-25 实测来自旧 16 层 Gated DeltaNet 结构，Windows x64 / Intel Core 5 220H / PyTorch 2.4.1+cpu；吞吐数据不代表新默认结构：

- 11 项原生集成测试全部通过：FP32 对照、Q8 解量化对照、中文/Unicode/空白/特殊标记分词、贪心聊天协议、think 强制切换、跨 token 停止词、HTTP/SSE 与请求隔离、错误处理等。
- 小模型实际执行到 10,240-token 缓存边界，并检查了部分 RoPE、非共享输出头和卷积 kernel=1 的变体。
- 完整 **99,232,352 参数**随机模型的 5-token logits 对照：两种构建的 FP32/Q8 最大绝对误差均低于 `8e-6`。Q8 的对照使用同样量化后解量化的 PyTorch 权重；这验证了计算实现，不证明训练后量化无损。
- 同机 4 线程、8-token prompt + 8 次解码、预热后测量两轮：本机优化构建的 FP32 解码约 **86–89 token/s**，Q8 约 **194–196 token/s**；通用构建分别约 **61–64** 和 **64–66 token/s**。这是随机权重、很短上下文下的本机样本，其他 CPU、线程数与上下文需要重新测量。
- 本机优化与通用构建的报告分别在 `cpu/build/full-smoke.json`、`cpu/build/full-smoke-portable.json`；随机模型临时文件会在测试后删除。重新运行脚本时可用 `--report` 指定报告路径。未验证完整 99M 模型的 10K 预填充耗时、训练后答案质量、Linux/ARM 实机表现。

## 文件格式与嵌入接口

`.h100` v1：所有整数/浮点采用 little-endian。文件头为 8 字节 `HYB100M\0`、u32 版本、长度前缀 JSON 元数据；接着保存原始 token 字节、11 个特殊 IDs、256 个基础字节 IDs、Unicode 区间和 BPE 合并三元组。最后保存命名张量（名称、类型、维数、形状、u64 负载字节数及负载）。字符串/数量使用 u32 长度。FP32 类型为 0；逐行 Q8 类型为 1，负载为行 scales（FP32）后接 row-major INT8。共享输出头不重复写入。加载会检查架构、名称、形状、长度、token 范围及有限数值。文件布局没有变化；通过 JSON 中的 `linear_attention` 和 `rope_type` 选择数学路径。缺少这两个字段的旧 v1 文件按 Gated DeltaNet / 原始 RoPE 处理。

`include/runtime.hpp` 暴露 `Runtime`：`encode`、`decode`、`forward`、`reset`、`chat`、`complete` 与 `benchmark`。将 `hybrid_runtime` CMake 目标链接进自己的 C++ 程序即可。一个 Runtime 持有一个可变会话缓存，不可并发调用；回调返回 false 可取消输出。

分词行为参考固定版本的 [Hugging Face ByteLevel 实现](https://github.com/huggingface/tokenizers/blob/v0.21.4/tokenizers/src/pre_tokenizers/byte_level.rs)，模型数学以本项目 `hybrid100m/model.py` 和 `ops.py` 为对照。所有 C++ 推理代码都在本目录，未导入 llama.cpp 或第三方推理运行库。
