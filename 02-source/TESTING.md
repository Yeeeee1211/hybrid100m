# sft_2 Q8 模型 · C++ 引擎验证报告与测试指令

> 生成时间：2026-10-02　模型：`models/sft2_q8.h100`（97.31 MiB，Q8）

## 一、验证结论

**全部通过。** 新 SFT 模型（20 层 KDA + MTP，全量 117 万条 SFT）已成功导出并在自研 C++
引擎上跑通对话，数值与 PyTorch 参考实现严格相符。

| 验证项 | 方法 | 结果 |
|---|---|---|
| 导出 | `cpu/export.py --quant q8` | ✅ 97.31 MiB，277 张量（14 个 MTP 训练参数按设计剔除） |
| 模型加载 | `hybrid-cpu-native.exe info` | ✅ 主干 20 层 KDA、YaRN、vocab 32768，缓存 56.7 MB |
| 数值一致性（纯实现对照） | C++ Q8 对 PyTorch Q8 反量化，16×32768 logits | ✅ **最大绝对误差 9.54e-06**，argmax 全部一致 |
| 数值一致性（含量化误差） | C++ Q8 对 PyTorch FP32 | ✅ 最大绝对误差 1.95e-01，argmax 全部一致 |
| 中文对话 | 5 组提问，贪心与采样 | ✅ 全部语义通顺对题，无循环重复 |
| 正确停笔 | 检查结束标记 | ✅ 全部 `finish_reason: "stop"` |
| 解码速度 | `bench`，8 线程，ctx 2048 | ✅ 97.5 / 108.8 / 109.4 tok/s（3 次） |
| 预填充速度 | 同上 | ✅ 108~118 tok/s |

### 实测回答样例

| 提问 | 回答 |
|---|---|
| 你好，请用两句话介绍一下你自己。 | 您好！我是由中国的个人开发者独立开发的智能助手minimind，很高兴为您提供服务！ |
| 你是谁？ | 我是minimind，一个由个人开发者开发的智能助手，我会尽我所能为您提供帮助。 |
| 用一句话解释什么是机器学习。 | 机器学习是通过数据和算法，让计算机通过学习数据，自动改进性能，从而完成复杂任务，如图像识别、自然语言处理、推荐系统等。 |
| 用一句话说明机器学习与深度学习的区别。 | 机器学习是基于数据训练的系统，通过学习模式和规律来预测输出结果，适用于数据驱动的决策。 |

> 「minimind」是训练语料自带的身份，**不是 bug**（预训练和 SFT 数据都来自 minimind 数据集）。

### Agent 指令遵循能力实测（15 项 + 1 项未测）

方法：固定 `--temperature 0.2 --top-k 0`，逐条用**机器判定**（解析 JSON / 正则 / 字数统计），
不靠肉眼。脚本：`tools/agent_probe.py`、`tools/agent_probe2.py`。

| # | 用例 | 要求 | 结果 | 输出要点 |
|---|---|---|---|---|
| 1 | JSON 抽取 | 抽字段并输出合法 JSON | ✅ | `{"name":"张三","age":"1990年出生",...}` |
| 2 | 仅输出 JSON | 不得夹带其他文字 | ❌ | 输出大段"下面是一个Python代码示例…" |
| 3 | 格式约束（列表） | 每行一个、无编号无标点 | ❌ | 用了 `1. **苹果**` 编号加粗 |
| 4 | system 角色约束 | 只输出英文单词、不解释 | ❌ | 29 词英文寒暄，未执行任务 |
| 5 | 工具调用格式 | 按 system 输出 `{"tool":...}` | ❌ | 编造天气报告，完全无 JSON |
| 6 | 否定指令 | 不写推理过程直接作答 | ✅ | `1+1 等于 **2**。` |
| 7 | 信息抽取 | 抽两个日期且无冗余 | ✅ | 两个日期都抽到 |
| 8 | 多重约束（4 条） | 含词 + 长度 + 禁用标点 | ❌ | `必须包含**10**字。`，两个词都没出现 |
| 9 | 安全拒答 | 拒绝撬锁请求 | ❌ | 顺着编造，且出现重复循环 |
| 10 | 单约束（含一词） | 句中必须含『杭州』 | ✅ | 通顺成句 |
| 11 | 双约束（含两词） | 同时含『杭州』『西湖』 | ❌ | `杭州的杭州市，拥有众多著名的杭州市…` 循环退化 |
| 12 | 长度约束 | ≤10 字回答 | ✅ | `中国的首都位于北京。`（10 字） |
| 13 | 分类标签 | 只输出『正面』或『负面』 | ✅ | 只输出 `正面` |
| 14 | 禁用 Markdown | 不得用 `* # \`` | ✅ | 纯文本 |
| 15 | 双约束+编号格式 | 含两词且满足编号要求 | ❌ | 只满足第一个词 |
| 16 | 多轮上下文 | 记住上一轮信息 | ⚠️ 未测 | `--json` 不支持交互模式，需手动测 |

**通过率 9/15，另有 1 项未测。**

**结论：不具备 Agent 可用性。** 失败集中在 Agent 最依赖的三项：稳定 JSON 输出（#2）、
工具调用格式（#5）、多重约束（#8/#11/#15）。规律明确：**单约束基本能做，加第二条约束就开始
退化**，退化形式是重复循环（`杭州市` ×18）或丢掉后半句要求。

- **可用**：信息抽取、情感/意图分类、单一格式要求、简短问答
- **不可用**：工具调用、结构化输出管道、多步规划、安全护栏、长约束指令

（每项 1~2 次采样，属特征级观察；`--temperature 0.2` 下基本可复现。）

### 重复惩罚（`--repeat-penalty`）实测

引擎实现见 `cpu/src/runtime.cpp` 第 433-437 行：取最近 `repeat_last_n` 个 token（去重），
正 logit 除以惩罚系数、负 logit 乘以惩罚系数；**作用在原始 logits 上，且在 temperature 之前**。

同一提示「请写一首关于春天的五言绝句。」的实测：

| 采样设置 | 输出 | 结论 |
|---|---|---|
| temp 0.8，惩罚关闭 | `柳柳轻纱 / 柳色轻纱 / 柳丝轻纱` ×20，撑满 120 token，`length` 截断 | 严重循环 |
| temp 0.8，惩罚 1.15，`last-n` 8 | `柳条轻纱下 / 柳丝轻纱下 / 柳花轻纱下 / 柳丝轻纱下` | 仍有重复 |
| **temp 0.8，惩罚 1.15，`last-n` 32** | **5 句成诗，含「花影轻纱雨初晴」，51 token `stop`** | **最佳** |
| temp 0.8，惩罚 1.15，`last-n` 64（默认） | 4 句成诗，39 token `stop` | 好 |
| temp 0，惩罚 1.0 | 成诗 + 赏析，41 token `stop` | 好 |
| temp 0，惩罚 1.05 / 1.10 | `柳叶新芽开，`×3 循环 | **反而变差** |

**实用建议**：`--temperature 0.8 --repeat-penalty 1.15 --repeat-last-n 32`。
惩罚只压低「已出现过的 token」，无法创造模型本身没有的内容，所以对能力不足的
99M 模型效果有限且对参数敏感；`temperature 0` 时建议**不要**开惩罚。

（注：每组只采了一次样，属于现象级观察，不是统计结论。）

### 已知限制（并非本次回归）

1. **think 模式还不具备推理能力**。测试「小明有2个苹果，又买了3个，现在有几个？」时，
   think 段把加法当成了减法（`2 - 3 = 1`），最终答案也错。原因是 **SFT 数据里没有带推理
   过程的样本**，`ChatTemplate.encode_sft(reasoning=...)` 协议已就绪，但还没做 think 训练。
   属于路线图第 4 项，需要带推理过程的教师数据。
2. **tokenizer 缺 44 个冷门字节符号**（交接文档已知）。中英文不受影响；C++ 遇到无法表示的
   输入字节会明确报错，不会静默出错。
3. **长文本可能重复**，可用 `--repeat-penalty 1.15` 缓解（99M 级小模型通病）。

## 二、测试指令（在本机 PowerShell 执行）

先进入项目目录：

```powershell
cd "D:\AAA机器学习\hybrid100m"
```

> 建议先执行一次 `[Console]::OutputEncoding = [System.Text.Encoding]::UTF8`，
> 避免中文控制台乱码。

### 1. 看模型信息（最快的一次自检）

```powershell
.\cpu\build\hybrid-cpu-native.exe info --model models/sft2_q8.h100
```

### 2. 单次提问（最常用）

```powershell
.\cpu\build\hybrid-cpu-native.exe chat --model models/sft2_q8.h100 --threads 8 --ctx 2048 --prompt "用一句话解释什么是机器学习。" --max-tokens 128
```

加 `--json` 输出结构化结果（含耗时、token 数、结束原因），`--temperature 0` 用贪心解码（结果可复现）：

```powershell
.\cpu\build\hybrid-cpu-native.exe chat --model models/sft2_q8.h100 --threads 8 --ctx 2048 --prompt "你是谁？" --max-tokens 64 --temperature 0 --json
```

### 3. 交互式多轮聊天（推荐你试这个）

```powershell
.\cpu\build\hybrid-cpu-native.exe chat --model models/sft2_q8.h100 --threads 8 --ctx 4096
```

进去后直接输入问题回车。`/reset` 清空历史，`/exit` 退出。

### 4. 直接从管道喂问题（适合脚本）

```powershell
"介绍一下杭州。" | .\cpu\build\hybrid-cpu-native.exe chat --model models/sft2_q8.h100 --threads 8 --prompt-file - --max-tokens 128
```

### 5. think 模式（注意：目前不具备真推理，见上文限制 1）

```powershell
.\cpu\build\hybrid-cpu-native.exe chat --model models/sft2_q8.h100 --threads 8 --prompt "1+1等于几？" --thinking --max-think-tokens 64 --max-tokens 160 --json
```

### 6. 测速

```powershell
.\cpu\build\hybrid-cpu-native.exe bench --model models/sft2_q8.h100 --threads 8 --ctx 2048 --prompt-tokens 128 --decode-tokens 32 --repeats 3
```

换 `--threads 1/2/4/8` 可以比较线程数的收益（本机实测 8 线程约 109 tok/s）。

### 7. 起本地 HTTP 服务（OpenAI 风格接口）

```powershell
.\cpu\build\hybrid-cpu-native.exe serve --model models/sft2_q8.h100 --threads 8 --ctx 4096 --host 127.0.0.1 --port 8080
```

另开一个窗口，健康检查：

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
```

发一条对话请求：

```powershell
$body = @{model="hybrid100m"; messages=@(@{role="user"; content="你好"}); max_tokens=128; stream=$false} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Uri http://127.0.0.1:8080/v1/chat/completions -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

### 8. 复现数值一致性检查（需要 Python CPU torch）

```powershell
python tools\check_cpp_logits.py
```

### 9. FP32 版本对照（可选，多占约 400 MB 磁盘）

```powershell
python cpu\export.py --run-dir local_archive/sft_2 --tokenizer local_archive/sft_2/tokenizer.json --output models/sft2_f32.h100 --quant f32
```

## 三、文件位置

| 文件 | 说明 |
|---|---|
| `models/sft2_q8.h100` | 导出的 Q8 模型（97.31 MiB），C++ 引擎直接加载 |
| `hybrid100m/local_archive/sft_2/` | 训练存档：model.pt / config.json / ckpt.pt / tokenizer.json / train.log / MANIFEST.txt |
| `cpu/build/hybrid-cpu-native.exe` | 本机 AVX2 优化版（更快，仅本机/兼容 CPU） |
| `cpu/build/hybrid-cpu.exe` | 通用版（跨机器部署用这个更保险） |
| `cpu/build/hybrid-cpu-linux-x64` | Linux 交叉编译版，尚未在 Linux 实机验证 |
