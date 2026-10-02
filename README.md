# hybrid100m

从零训练的中文对话小模型（**104,555,708 参数**），附带**自研纯 C++ 推理引擎**——运行时不需要
Python、PyTorch、CUDA 或任何外部依赖，一个可执行文件加一个模型文件即可跑起来。

本项目完整开源了「**分词器 → 预训练 → SFT → 量化导出 → CPU 推理**」的全链路实现与训练脚本，
包含全部可复现的超参、训练日志和数值验证脚本。

```
架构        20 层混合主干（15 × KDA 线性注意力 + 5 × Gated GQA），hidden 512，FFN 1920
参数量      104,555,708（embedding 与 LM head 共享），词表 32,768
上下文      10,240 tokens（YaRN 扩展）
训练硬件    单张 RTX 4090 24GB
预训练      16.2 亿 tokens，1 个 epoch（12,328 步），验证 loss 1.4434
SFT         117 万条对话，1 个 epoch（36,235 步），验证 loss 2.1796
CPU 推理     解码 97~109 tok/s、预填充 108~118 tok/s（8 线程，Q8）
许可证       MIT（代码）
```

## 模型下载

本仓库**只包含代码**，模型权重托管在 Hugging Face：

```bash
# 方式一：huggingface-cli
pip install -U "huggingface_hub[cli]"
huggingface-cli download <你的用户名>/hybrid100m-sft2 --local-dir 01-model

# 方式二：git lfs
git lfs install
git clone https://huggingface.co/<你的用户名>/hybrid100m-sft2 01-model
```

下载后目录应为：

```
01-model/
├── sft2_q8.h100         ← C++ 引擎直接加载这个（97.3 MiB）
├── model.pt             FP32 权重（可选，做 FP32 推理或继续训练时用）
└── config.json          模型配置
```

然后即可运行：

```powershell
.\03-engine\build\hybrid-cpu-native.exe chat --model 01-model\sft2_q8.h100 --threads 8 --ctx 2048 --prompt "你好"
```

> 若暂时不想下载权重，也可以只克隆代码、自行训练（见 §4.2），或用 `03-engine/export.py`
> 从自己的 checkpoint 导出 `.h100`。

---

## 一、训练数据

### 1.1 数据来源

预训练与 SFT 数据均来自开源数据集 **`gongjy/minimind_dataset`**：

- 数据集主页：<https://www.modelscope.cn/datasets/gongjy/minimind_dataset>
- 上游项目（数据与训练方法参考）：<https://github.com/jingyaogong/minimind>
- Hugging Face 镜像：<https://huggingface.co/datasets/jingyaogong/minimind_dataset>

> ⚠️ **数据许可证提醒：该数据集标注为 CC-BY-NC-4.0（署名—非商业性使用）。**
> 本仓库的**代码**采用 MIT 许可证；但由该数据集训练得到的**模型权重**受数据来源条款约束，
> 不可直接用于商业目的。开源或对外提供权重前，请到数据集主页复核最新许可条款。
> 如需商用，请另行获取数据授权，或更换为可商用数据集重新训练。

### 1.2 本项目实际用到的文件

| 阶段 | 数据文件 | 原始体积 | 说明 |
|---|---|---|---|
| 预训练 | `pretrain_t2t.jsonl` | 8.27 GB | 中文通用文本，逐行 JSON，每行一篇文档（取 `text` 字段） |
| SFT | `sft_t2t.jsonl` | 14.10 GB | 多轮对话，逐行 JSON，**本项目只用了其中 117 万条** |

数据集中的 `pretrain_t2t_mini.jsonl`（1.24 GB）与 `sft_t2t_mini.jsonl`（1.74 GB）为小规模版本，
本项目正式训练未使用（第一轮 2.42 亿 tokens 的可行性试验用的是 mini 版，未计入本文结果）。

### 1.3 数据处理流程

```
原始 jsonl
   │  pretokenize.py     逐行读取 → 自训 tokenizer 编码 → 追加 <|eos|> → uint32 流
   ▼
data_full/train.bin      1,615.9M tokens（6.16 GiB）
data_full/val.bin        5.0M tokens（从流末尾切出 0.5%）
   │
   ▼  train.py           随机窗口采样，窗口长度 2048
```

SFT 数据另经 `sft_prep.py` 处理：

```
sft_t2t.jsonl → ChatTemplate 编码 → assistant 回复以外的位置标记为 -100（不计算 loss）
              → sft_data.jsonl（117 万行，每行 {"input_ids": [...], "labels": [...]}）
```

**实测数据统计**（由 `verify_pretrain_data.py` 的 12 项检查确认）：

| 项 | 数值 |
|---|---|
| 预训练总 tokens | 1,615,900,000（train 1,615.9M + val 5.0M） |
| 预训练文档数 | 8,468,827 篇 |
| 平均文档长度 | **186 tokens** |
| EOS 密度 | 0.537%（文档边界正常） |
| token ID 最大值 | 32,762（词表 32,768，全部合法） |
| 8×2048 窗口内 token 种类 | 5,088（多样性正常） |
| 随机窗口解码可读率 | 100.0% |
| SFT 样本数 | 1,171,226 条 |
| SFT 平均长度 | 162 tokens/条（**其中被监督的仅 35 tokens/条**） |

> **已知限制：语料平均文档仅 186 tokens**，这决定了长上下文能力无法只靠本语料获得。
> 若要做 4K/10K 长上下文训练，需要额外引入真实长文档与检索型样本。

### 1.4 分词器

**未使用任何现成词表**，用 `train_tokenizer.py` 在自建语料上训练 ByteLevel BPE：

| 项 | 值 |
|---|---|
| 词表大小 | 32,768 |
| 特殊 token | 11 个（id 0–10，见下表） |
| 文件 | `01-model/*/tokenizer.json`（2.67 MiB） |

| id | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| token | `<\|pad\|>` | `<\|bos\|>` | `<\|eos\|>` | `<\|im_start\|>` | `<\|im_end\|>` | `<\|think_on\|>` | `<\|think_off\|>` | `<think>` | `</think>` | `<answer>` | `</answer>` |

**已知缺陷：该分词器缺少 44 个冷门字节符号**（ByteLevel 基础字节表不完整）。中英文不受影响，
但 C++ 引擎遇到无法表示的输入字节时会明确报错而非静默出错。后续重训分词器应包含完整 256 字节表。

---

## 二、核心算法

在标准 decoder-only 主干上接入以下算法节点，**全部自行实现，无 fork 依赖**。

### 2.1 混合注意力主干：KDA 线性注意力 + Gated GQA

**论文**：[Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692)

主干布局 `[KDA, KDA, KDA, Gated GQA] × 5`，即 **75% 线性注意力 / 25% 全注意力**：

| 组件 | 配置 |
|---|---|
| KDA（Kimi Delta Attention） | 4 头 × 128 维，**每个 key 通道独立的 log-decay**，Q/K L2 归一化，核大小 4 的因果深度卷积，两组 rank=128 的低秩门投影 |
| Gated GQA | 8 Q / 2 KV 头，head_dim=64，Q/K RMSNorm，**按头拆分的 sigmoid 输出门** |

KDA 的核心是**逐通道记忆衰减**（相对原始 Gated DeltaNet 的关键改进）：

```
g      = -exp(A_log) * softplus(a + dt_bias)     # 每个 key 通道一个衰减值，不再是逐头标量
S_bar  = diag(exp(g_t)) @ S_(t-1)                # 先按 key 行衰减状态
S_t    = (I - beta_t * k_t k_tᵀ) @ S_bar + beta_t * k_t v_tᵀ   # delta-rule 擦写
y_t    = (q_t / sqrt(K)) @ S_t
```

低秩门控（`a_down → a_up` 产生遗忘门，`z_down → z_up → sigmoid` 产生输出门）让门控参数量
远小于全宽投影。推理时递归状态保持 FP32。

**双向实现**：训练走 FLA 的 chunk kernel（`fla.ops.kda.chunk_kda`，CUDA BF16）；
`ops.py` 另有纯 PyTorch 的逐时刻参考实现（`recurrent_kda_reference`）用于数学验证与 CPU 测试，
两者在单元测试中被要求数值一致。

### 2.2 Muon 优化器

**论文**：[Muon is Scalable for LLM Training](https://github.com/MoonshotAI/Moonlight)

| 参数类型 | 优化器 | 说明 |
|---|---|---|
| 隐藏层二维矩阵 | **Muon**（NS5 正交化，momentum 0.95，Nesterov） | 更新尺度 `0.2*sqrt(max(rows, cols))` |
| embedding / LM head、卷积、向量、归一化 | AdamW | 共享 embedding 只注册一次 |

规范化常数与衰减项**不做 weight decay**。非有限梯度会在任何参数更新前**拒绝整步**。
单 GPU 实现，支持标准 `state_dict` 与 PyTorch scheduler 恢复。

### 2.3 串行 Multi-Token Prediction（MTP）

**论文**：[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) §4.3

训练时挂 1 个 MTP 分支：融合主干表示 `h_i` 与共享 embedding 的下一 token，经一个**完整因果 GQA block**
预测再下一个 token：

```
loss = ntp_loss + 0.1 * mtp_loss
```

MTP 分支**只用于训练**，导出时被剔除（`.h100` 中记录 `inference_branch="main"`，
本次导出剔除 14 个张量），因此推理不增加任何额外计算。SFT 阶段按各层有效目标数加权统计，
避免不同监督长度的均值被等权相加。

### 2.4 融合交叉熵与有效位置过滤

**论文**：[Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009)

- **先过滤 `labels = -100` 再投影**，只为被监督的位置计算 LM projection
- CUDA 默认走 CCE（`cut-cross-entropy`，Apple 官方实现，pinned commit `b7a0279`；
  `filter_eps=None` 不做近似梯度跳过，`accum_e_fp32/accum_c_fp32` 保证 FP32 累加）
- CPU 回退到 `loss_chunk_size=128` 的分块 CE + 检查点路径
- 全部位置被忽略时返回**可反向的零**，不产生 NaN

### 2.5 YaRN 长上下文扩展

**论文**：[YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)

固定配置：原始窗口 2048、factor = 5、上限 10240，beta_fast/slow = 32/1，
attention scaling `1 + 0.1*log(5)` 同时作用于 cos/sin。**训练全程频率固定**，
不按当前长度动态重算。RoPE 仅用于全注意力层，完整 64 维旋转。

### 2.6 工程实现要点

| 技术 | 做法 |
|---|---|
| 梯度检查点 | 每层与 MTP 分支独立 checkpoint，显存换时间 |
| 混合精度 | **FP32 参数 + BF16 autocast**；递归状态、衰减指数、归一化统计、RoPE 相位保持 FP32 |
| 残差结构 | Pre-RMSNorm 两次串行相加：`u = x + Mixer(Norm(x))`，`y = u + FFN(Norm(u))` |
| 完整样本 packing | `PackedCollator`（best-fit decreasing）+ `cu_seqlens` 重置 KDA 递归状态；卷积显式屏蔽跨文档 tap；全注意力逐文档调用 causal SDPA，**不分配全局 T×T mask** |
| 上下文课程 | `ContextCurriculum` 按**实际处理 tokens** 选择长度（本语料未启用 4K/10K 阶段） |
| C++ 引擎 | `.h100` 自定义格式（FP32 / 逐行 INT8 Q8），含 BPE 分词、流式 CLI、多轮、HTTP/SSE、bench |

---

## 三、训练结果

### 3.1 两个阶段

| 阶段 | 配置 | 步数 | 最终验证 loss | 耗时 |
|---|---|---|---|---|
| 预训练 | 1.62B tokens，seq 2048，micro-batch 32 × accum 2（131,072 tokens/步），lr 3e-4 余弦到 3e-5，warmup 500 | 12,328（= 1 epoch） | **1.4434** | 约 13.5 小时 |
| SFT | 117 万条对话，micro-batch 32 × accum 1，lr 5e-5 余弦到 5e-6，warmup 200 | 36,235（= 1 epoch） | **2.1796** | 约 4.8 小时 |

优化器均为 Muon + AdamW，梯度裁剪 1.0，权重衰减 0.1（预训练）/ 0（SFT）。

### 3.2 引擎数值一致性

| 验证项 | 结果 |
|---|---|
| C++ Q8 对 PyTorch Q8 反量化（16×32768 logits） | **最大绝对误差 9.54e-06**，argmax 全部一致 |
| C++ Q8 对 PyTorch FP32（含量化误差） | 最大绝对误差 1.95e-01，argmax 全部一致 |
| 解码速度 | 97.5 / 108.8 / 109.4 tok/s（8 线程，ctx 2048） |
| 预填充速度 | 108~118 tok/s |

### 3.3 能力评估（诚实记录）

| 能力 | 结论 |
|---|---|
| 中文对话 | ✅ 通顺、对题、能正确停笔 |
| 信息抽取 / 情感分类 / 单格式要求 | ✅ 可用 |
| **Agent 指令遵循** | ❌ **15 项测试通过 9 项，不具备 Agent 可用性** |
| 工具调用 / 稳定 JSON 输出 / 多重约束 | ❌ 全部失败 |
| think 推理模式 | ❌ 无推理能力（SFT 数据未含推理过程样本） |
| 长文本生成 | ⚠️ 会重复循环，需 `--repeat-penalty` 缓解 |

**失败规律**：单约束基本能满足，**加上第二条约束即开始退化**（重复循环或丢掉后半句要求）。
这是 104M 参数规模 + 通用对话 SFT 数据的**能力天花板**，不是调参能解决的。
完整 16 项明细见 [`02-source/TESTING.md`](02-source/TESTING.md)。

---

## 四、快速开始

### 4.1 直接跑推理（不需要 Python / GPU）

```powershell
# Windows：单次提问
.\03-engine\build\hybrid-cpu-native.exe chat --model 01-model\sft2_q8.h100 --threads 8 --ctx 2048 --prompt "用一句话解释什么是机器学习。" --max-tokens 128

# 多轮交互聊天（/reset 清历史，/exit 退出）
.\03-engine\build\hybrid-cpu-native.exe chat --model 01-model\sft2_q8.h100 --threads 8 --ctx 4096

# 测速
.\03-engine\build\hybrid-cpu-native.exe bench --model 01-model\sft2_q8.h100 --threads 8 --ctx 2048 --prompt-tokens 128 --decode-tokens 32 --repeats 3
```

CLI / HTTP 接口的完整参数见 [`03-engine/README.md`](03-engine/README.md)。

### 4.2 从源码训练

需要 CUDA + FLA + CCE（安装方式见 `02-source/pyproject.toml` 的 `cuda` extra）：

```bash
cd 02-source
pip install -e ".[cuda,training]"
python -m unittest discover -s tests -v          # 67 项单元测试

# 1) 训练分词器（可选，本项目使用已发布的 tokenizer.json）
python train_tokenizer.py --data-dir <jsonl目录> --output tokenizer.json

# 2) 预编码语料
python pretokenize.py --data-dir <minimind_dataset> --tokenizer tokenizer.json \
    --out-dir data_full --file pretrain_t2t.jsonl
python verify_pretrain_data.py --data-dir data_full

# 3) 预训练
python train.py --data-dir data_full --out-dir runs/pre_full \
    --seq-len 2048 --micro-batch 32 --accum 2 --lr 3e-4 --warmup 500 \
    --max-steps 12328 --save-every 500 --val-every 250

# 4) 准备 SFT 数据并微调
python sft_prep.py --data-dir <minimind_dataset> --tokenizer tokenizer.json --out sft_data.jsonl
python train_sft.py --data sft_data.jsonl --init-from runs/pre_full --tokenizer tokenizer.json \
    --out-dir runs/sft_2 --micro-batch 32 --accum 1 --epochs 1 --max-samples 0 --lr 5e-5

# 5) 导出为引擎格式
python ../03-engine/export.py --run-dir runs/sft_2 --tokenizer tokenizer.json \
    --output sft2_q8.h100 --quant q8
```

### 4.3 复现数值验证

```bash
python 01-model/tools/check_cpp_logits.py    # C++ 引擎 vs PyTorch logits 对照
python 01-model/tools/verify_sft_local.py    # 本机 CPU 独立验证
```

---

## 五、目录结构

```
.
├── 01-model/          模型配置与训练记录（权重从 Hugging Face 下载，见「模型下载」）
│   ├── sft_2/                SFT 模型的 config.json + 训练日志 + 验证报告 + MANIFEST
│   ├── pre_full/             预训练底座的 config.json + 训练日志 + 验证报告 + MANIFEST
│   └── tools/                复现验证脚本（数值对照 / 本机验证 / Agent 测试）
├── 02-source/         Python 源码
│   ├── hybrid100m/           核心：config / model / ops / losses / optim / data / chat
│   ├── train*.py             训练脚本
│   └── tests/                单元测试（67 项）
├── 03-engine/         C++ 推理引擎
│   ├── src/ include/         C++17 源码，零外部依赖
│   └── build/                已编译二进制（Windows x64 / Linux x64）
├── LICENSE            MIT（代码）
└── README.md
```

> **仓库内容边界**：本仓库只含**代码、配置、分词器、训练日志与验证报告**（约 4 MiB）。
> 所有模型权重（`*.h100` / `model.pt` / `ckpt.pt`）均不纳入版本管理，统一托管在 Hugging Face，
> 下载方式见开头「模型下载」一节。

---

## 六、已知限制

1. **权重受数据来源的非商业限制**（数据集 CC-BY-NC-4.0），商用需另行获取授权或更换数据集重训；
   代码本身为 MIT 许可证。
2. **不具备 Agent 能力**：工具调用、稳定结构化输出、多约束指令均不可用。
3. **think 模式是空壳**：协议已实现但未训练，会输出错误的"推理"。
4. **长文本重复**：小模型通病，建议 `--repeat-penalty 1.15 --repeat-last-n 32`。
5. **分词器缺 44 个字节符号**：中英文不受影响，冷门字节输入会明确报错。
6. **语料平均文档仅 186 tokens**，长上下文能力需要额外真实长文数据。
7. **Linux 二进制尚未实机验证**（交叉编译产物）。

---

## 七、致谢

- 数据集与训练方法参考：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)
- 线性注意力 kernel：[flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
- 融合交叉熵：[apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy)
- 各算法节点对应论文见「核心算法」各节链接
