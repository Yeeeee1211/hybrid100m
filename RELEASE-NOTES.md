# hybrid100m 发行包

> 组装时间：2026-10-02　模型：104,555,708 参数（20 层 KDA + MTP，中文对话模型）
> 本目录是从 `D:\AAA机器学习\hybrid100m\` 整理出的**自包含副本**，可独立运行与再训练。

## 一句话

一个 99M 级中文对话模型，含**完整源码**、**自研纯 C++ 推理引擎**（不依赖 Python/PyTorch/CUDA）
和**从零训练的全部权重**（预训练底座 + SFT 对话模型）。

## 目录结构

```
hybrid100m-release/
├── 01-model/          模型文件（2.6 GiB）
│   ├── sft2_q8.h100           ← 直接用这个跑推理（97 MiB，Q8 量化）
│   ├── sft_2/                 对话模型完整存档
│   │   ├── model.pt           对话模型权重（FP32，399 MiB）
│   │   ├── config.json        模型配置（与权重严格配套）
│   │   ├── ckpt.pt            含优化器的存档，可继续微调（862 MiB）
│   │   ├── tokenizer.json     分词器（训练时原文件）
│   │   ├── train.log          完整 SFT 日志（36,235 步）
│   │   ├── MANIFEST.txt       训练配置、最终指标、全部 sha256
│   │   ├── verify_local.json  本机 CPU 独立验证结果
│   │   └── agent_probe.json   Agent 指令遵循测试原始数据
│   ├── pre_full/              预训练底座（同结构）
│   └── tools/                 复现验证脚本
│       ├── check_cpp_logits.py     C++ vs PyTorch 数值对照
│       ├── verify_sft_local.py     本机 CPU 独立验证
│       └── agent_probe*.py         指令遵循测试
├── 02-source/         Python 源码与训练脚本（0.2 MiB）
│   ├── hybrid100m/            模型核心：config / model / ops / losses / optim / data / chat
│   ├── train.py               预训练脚本
│   ├── train_sft.py           SFT 微调脚本
│   ├── train_tokenizer.py     分词器训练
│   ├── pretokenize.py         语料预编码
│   ├── tests/                 单元测试（67 项通过）
│   ├── README.md              架构规格（权威文档）
│   └── TESTING.md             本次全套测试报告
└── 03-engine/         C++ 推理引擎（11.8 MiB）
    ├── src/ include/          C++17 源码（零外部依赖）
    ├── build/
    │   ├── hybrid-cpu-native.exe  本机 AVX2 优化版（更快，推荐本机用）
    │   ├── hybrid-cpu.exe         通用版（跨机器部署用这个更保险）
    │   └── hybrid-cpu-linux-x64   Linux 交叉编译版（尚未在 Linux 实机验证）
    └── README.md              引擎完整文档（CLI / HTTP / 参数）
```

## 最快的试法（Windows PowerShell）

```powershell
cd "D:\AAA机器学习\hybrid100m-release"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # 防中文乱码

# 单次提问
.\03-engine\build\hybrid-cpu-native.exe chat --model 01-model\sft2_q8.h100 --threads 8 --ctx 2048 --prompt "用一句话解释什么是机器学习。" --max-tokens 128

# 多轮交互聊天（/reset 清历史，/exit 退出）
.\03-engine\build\hybrid-cpu-native.exe chat --model 01-model\sft2_q8.h100 --threads 8 --ctx 4096

# 测速
.\03-engine\build\hybrid-cpu-native.exe bench --model 01-model\sft2_q8.h100 --threads 8 --ctx 2048 --prompt-tokens 128 --decode-tokens 32 --repeats 3
```

本机实测：**解码 97~109 tok/s、预填充 108~118 tok/s**（8 线程，Q8）。

## 模型状态与实测结论

| 项 | 结果 |
|---|---|
| 架构 | 20 层混合主干：15 × KDA 线性注意力 + 5 × Gated GQA，hidden 512，MTP 1 层 |
| 参数 | 104,555,708（embedding 与 LM head 共享），词表 32,768 |
| 预训练 | 全量语料 16.2 亿 tokens，1 个 epoch（12,328 步），**验证 loss 1.4434** |
| SFT | 全量 117 万条对话，1 个 epoch（36,235 步），**验证 loss 2.1796** |
| 数值一致性 | C++ Q8 对 PyTorch Q8 反量化，最大绝对误差 **9.54e-06**，argmax 全一致 |
| 对话能力 | ✅ 通顺、对题、能正确停笔 |
| 重复惩罚 | ✅ 可用；推荐 `--temperature 0.8 --repeat-penalty 1.15 --repeat-last-n 32` |
| **Agent 指令遵循** | ❌ **9/15 通过，不具备 Agent 可用性**（见 TESTING.md） |
| think 推理 | ❌ 无推理能力（SFT 数据未含推理过程样本） |

## 用已知限制

1. **不能做 Agent**：工具调用、稳定 JSON 输出、多重约束全部失败。可用场景限于信息抽取、
   情感/意图分类、单格式要求、简短问答。
2. **think 模式是空壳**：协议保留但未训练，模型会给出错误的"推理"。
3. **长文本会重复**：99M 小模型通病，用重复惩罚缓解。
4. **tokenizer 缺 44 个冷门字节符号**：中英文不受影响，引擎遇到无法表示的字节会明确报错。

## 再训练怎么做

源码在 `02-source/`，训练脚本用法见其中 `README.md`（架构权威规格）与各脚本的 `--help`。
SFT 脚本需要 CUDA + FLA + CCE（安装方式见 `02-source/pyproject.toml` 的 `cuda` extra）。
只用本机 CPU 验证模型数学正确性：

```powershell
cd 02-source
python -m pip install -e ".[training]"
python -m unittest discover -s tests -v
```

## 完整性校验

所有关键文件的 sha256 记录在各 `/MANIFEST.txt` 中。组装后已做逐文件哈希比对，
与原始工作目录**完全一致**，并从本目录实际启动 C++ 引擎验证可用。
