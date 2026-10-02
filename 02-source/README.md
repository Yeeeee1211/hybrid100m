# Hybrid100M 核心算法

从随机权重开始训练的中文因果语言模型，目标硬件为单张 **RTX 4090 / 24GB**。本次一次性接入 P0/P1：过滤有效 loss 位置、融合交叉熵、KDA、低秩门控、串行 MTP、Muon + AdamW、20 层主干、固定 YaRN、上下文课程、隔离 packing 和教师文本 SFT 辅助接口。**tokenizer、词表与 token ID 映射保持原样，残差仍为 Pre-RMSNorm 的两次串行相加。**

推理主干 **100,162,108** 个独立参数；默认一个 MTP 训练分支增加 **4,393,600**，训练总计 **104,555,708**。embedding 与 LM head 共享。正式训练、语料处理、教师答案校验、学习率调度和训练预算由你的训练脚本负责；本仓库提供核心算法及验证接口。

CPU 部署见 [独立 C++ 推理框架](cpu/README.md)：支持 FP32/Q8、原生中文 BPE、缓存、流式聊天、think 控制和 HTTP/SSE。导出器严格加载完整训练 checkpoint，然后省略 MTP 分支；运行时无需 Python/PyTorch/CUDA。

## P0/P1 与论文节点

| 优先级 | 论文 / 官方实现 | 接入的关键节点 | 本项目实现 |
|---|---|---|---|
| P0 | [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009) | LM projection / CE 的显存占用 | `losses.py`：先过滤 `labels=-100`，CUDA 默认 CCE；关闭近似梯度过滤，FP32 梯度累加；保留 checkpoint 分块 CE |
| P1 | [Kimi Linear: An Expressive, Efficient Attention Architecture](https://arxiv.org/abs/2510.26692) | KDA 的逐通道记忆衰减、门控与 chunk 算法 | `model.py` / `ops.py`：每个 key 通道独立 log-decay，rank=128 的衰减/输出门投影，sigmoid 输出门，FLA chunk/recurrent |
| P1 | [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) | 串行 Multi-Token Prediction，共享 embedding/head | `MTPBlock`：融合主干表示和下一 token embedding，通过一个完整因果 GQA block 预测再下一 token；默认辅助权重 0.1 |
| P1 | [Muon is Scalable for LLM Training](https://github.com/MoonshotAI/Moonlight) | 隐藏矩阵的正交化更新和 RMS 尺度匹配 | `optim.py`：NS5 Muon；embedding/head、卷积和向量保留 AdamW；规范化/衰减常数零 weight decay |
| P1 | [GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models](https://arxiv.org/abs/2508.06471) | 深宽权衡与 thinking/direct 数据模式 | 主干改为 20×512、FFN=1920；沿用已有两种对话模式。尺寸是本项目选择，不是 GLM 原模型配置 |
| P1 | [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)；DeepSeek-V3 §4.3 | RoPE 频率缩放、长上下文训练课程 | 固定 2048 基准 / factor=5 / 10240 上限；课程由 `ContextCurriculum` 按实际处理 tokens 选择 |
| P1 | [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) | 教师生成的推理文本用于小模型 SFT | `encode_distillation_records` / `mix_sft_examples`：用原 tokenizer 编码已校验的短推理与直接回答；不做跨词表 logits KL |
| P1 | [FLA KDA 接口](https://github.com/fla-org/flash-linear-attention/tree/v0.4.0/fla/ops/kda) | `cu_seqlens` 文档边界 | `PackedCollator` + 模型入口：完整样本 best-fit packing，隔离卷积、线性状态、全注意力和 NTP/MTP 目标 |

这些节点组成一次联合改动；本项目没有做逐项消融，也没有验证训练后的能力增益。QK-Norm 保留，Muon 没有复制 MLA 专用的 QK-Clip。P2 的 MLA、MoE、稀疏注意力、FP8、残差重构和额外调度方案不在本次范围。

## 默认结构

| 配置 | 默认值 |
|---|---|
| 主干布局 | `[KDA, KDA, KDA, Gated GQA] × 5`，75% 线性 / 25% 全注意力 |
| 主干 | 20 层，hidden=512，SwiGLU intermediate=1920 |
| 全注意力 | 8 Q / 2 KV 头，head_dim=64；Q/K RMSNorm；按头拆分的 sigmoid 输出门 |
| KDA | 4 头 × 128 维，Q/K L2 归一化，核大小 4 的因果深度卷积；两组 rank=128 线性门投影 |
| RoPE | 仅全注意力，完整 64 维旋转，theta=10000；固定 YaRN，original=2048，factor=5，beta_fast/slow=32/1 |
| 上下文 | 输入、生成的 think 和答案合计最多 10240 tokens |
| 词表 | 32768，包含原有全部特殊 token；embedding / LM head 共享 |
| MTP | 1 个训练专用完整 GQA block，共享 embedding/head，loss 权重 0.1 |
| 精度 | FP32 参数 + BF16 autocast；递归状态、衰减指数、归一化统计、RoPE 相位和 CCE 梯度累加为 FP32 |
| 显存控制 | 层和 MTP checkpoint；CUDA 默认 CCE；备用 CE checkpoint 分块大小 128 |

主干参数组成：embedding 16,777,216；FFN 58,982,400；15 个 KDA 模块 19,793,340；5 个全注意力模块 4,588,160；主干归一化 20,992。

残差计算保持：`u = x + Mixer(RMSNorm(x))`，`y = u + SwiGLU(RMSNorm(u))`。KDA 的衰减为 `g = -exp(A_log) * softplus(a + dt_bias)`，其中 `a`、`dt_bias` 为每个 key 通道的值；先按 key 行衰减状态，再做 delta-rule 擦写。低秩投影之间没有非线性。CPU oracle 用于数学检查，5B-token 训练必须使用 FLA。

YaRN 从训练起点就固定，所有课程阶段用相同频率与 attention scaling；不根据当前长度动态重算配置。默认 scaling 为 `1 + 0.1 * log(5)`，同时作用于 cos/sin。10K 配置本身不会学出长上下文能力，需要真实长文和关联/检索样本。

## 安装与验证

Python >= 3.10。CPU 数学检查：

```bash
python -m pip install -e '.[training]'
python -m unittest discover -s tests -v
```

默认配置要求 CUDA / FLA / Flash SDPA。CPU 小模型检查显式设置 `linear_backend="reference", sdpa_backend="math"`；CPU loss 使用过滤后的分块 CE，不需要 CCE 包。`training` extra 提供现有脚本及桥接测试使用的 numpy/tokenizers；只使用核心模型可安装 `pip install -e .`，不会改变已有 tokenizer 文件。

4090 云端使用 Linux 和与驱动匹配的 PyTorch CUDA wheel。先安装适合服务器的 PyTorch CUDA 构建，再安装：

```bash
python -m pip install -e '.[cuda,training]'
python -m unittest discover -s tests -v
```

CUDA extra 要求 PyTorch >= 2.7、Triton >= 3.3、FLA >= 0.4,<0.7。CCE 固定为 [Apple 官方实现的 commit b7a02791](https://github.com/apple/ml-cross-entropy/commit/b7a02791b234e187b524fb1dba6a812d521b203a)，安装需要 Git 和访问 GitHub；这个版本提供本项目使用的 FP32 累加接口。不要用旧的 PyPI 版本覆盖它。缺失 FLA、CCE 或 Flash 内核会明确失败；没有 CUDA 的测试显示 skipped。

可在服务器运行一次完整模型的合成显存检查，默认 microbatch=1：

```bash
HYBRID100M_FULL_GPU_TEST=1 HYBRID100M_GPU_SEQ_LEN=2048 python -m unittest discover -s tests -p test_cuda.py -v
HYBRID100M_FULL_GPU_TEST=1 HYBRID100M_GPU_SEQ_LEN=4096 python -m unittest discover -s tests -p test_cuda.py -v
HYBRID100M_FULL_GPU_TEST=1 HYBRID100M_GPU_SEQ_LEN=10240 python -m unittest discover -s tests -p test_cuda.py -v
```

检查包括 MTP、真实 Muon + AdamW 更新，以及保留梯度/优化器状态后的第二次前向反向；打印参数数、峰值显存和耗时。它不读取语料、不保存训练权重，首次编译耗时不能代表稳态吞吐。`HYBRID100M_GPU_BATCH_SIZE` 修改本次 microbatch；`HYBRID100M_GPU_LOSS_BACKEND=chunked` 可检查备用 CE；`HYBRID100M_GPU_LOSS_CHUNK_SIZE` 仅对分块路径有效。这些环境变量不修改保存配置。

从 microbatch=1 开始，通过梯度累积确定有效 token batch；保持 checkpointing 和 `return_logits=False`。CCE 使用 `filter_eps=None`，没有开启近似梯度跳过。`loss_backend="auto"` 仅在 CCE 包未安装时警告并切换分块路径，API/内核错误照常抛出；默认 `"cce"` 则严格要求安装。最终 10K 显存、吞吐与收敛仍以你的 4090 实测为准。

2026-09-30 本地验证环境是 Windows / Python 3.12 / PyTorch 2.4.1+cpu：Python 测试 63 项通过、5 项 CUDA 测试因无 GPU 跳过，尚未执行 CUDA 内核或正式语料训练。验证覆盖 KDA 矩阵更新及梯度、loss 过滤、Muon 更新/恢复、MTP 对齐、packing 隔离、YaRN、缓存和现有训练脚本接线；原生 C++ 13 项集成测试通过。完整新主干导出对照通过：FP32 对原权重的最大误差约 4.95e-6；Q8 对同一量化权重的 Python 反量化参考误差约 4.77e-6，这个数不是 Q8 相对原 FP32 模型的量化误差。报告见 [full-smoke-p1.json](cpu/build/full-smoke-p1.json)。Linux 程序仅交叉编译，尚未实机运行。

## 接入训练代码

以下只展示一次核心更新，不包含 trainer。不要把模型权重直接转成 BF16。

```python
import torch
from hybrid100m import HybridConfig, HybridForCausalLM, build_optimizer

config = HybridConfig.load("config.json")
model = HybridForCausalLM(config).cuda().train()  # FP32 master weights
optimizer = build_optimizer(model, muon_lr=3e-4, adamw_lr=3e-4)

# input_ids: CUDA int64 [B,T]；预训练 labels=input_ids，SFT labels 遮罩 prompt。
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(input_ids, labels=input_ids)
output.loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

`3e-4` 是可配置起点，未做学习率搜索。Muon 只更新隐藏层二维矩阵，NS 迭代 5 次、momentum=0.95、Nesterov 开启，更新尺度为 `0.2*sqrt(max(rows, columns))`。共享 embedding/head 只注册一次并走 AdamW；向量、归一化、`A_log/dt_bias` 零 weight decay。它是单 GPU 优化器，支持标准 `state_dict` 和 PyTorch scheduler；不包含分布式 Muon。非有限梯度会在任何参数更新前拒绝整步。

现有 `train.py / train_sft.py` 已做最小接线：KDA 配置派发 Muon + AdamW，旧 GDN 保留原 AdamW 分组以兼容优化器恢复；SFT 累积按 NTP 与各层 MTP 各自的有效目标数加权。预训练脚本的验证指标仍是 NTP，SFT 验证显式包含配置的 MTP 项，两者不能直接混作同一指标。没有运行正式训练。

损失与输出约定：

- `labels` 与 `input_ids` 同形状、**未 shift**；无监督位置是 `-100`。NTP 用位置 i 预测 i+1。
- MTP 第 k 层融合上一深度的 `h_i` 与共享 embedding 的 `token_(i+k)`，预测 `token_(i+k+1)`。默认 k=1，`loss = ntp_loss + 0.1 * mtp_loss`；多层配置取辅助层均值。
- 过滤 prompt/padding 只省掉这些位置的 LM projection，保留其主干计算与上下文梯度。
- 训练模式有 labels 时自动计算 MTP。`compute_mtp=False` 关闭辅助分支；eval 默认只计算 NTP，需辅助指标时显式传 `compute_mtp=True`。推理只运行主干，不使用 MTP 或投机解码。
- `output.loss_sum / output.num_loss_tokens` 是 NTP 的和/有效目标数；`mtp_loss_sums / mtp_num_tokens` 给出各辅助深度统计。全部忽略时返回可反向的零。统计跨 microbatch 应按各自目标数加权，不能把不同监督长度的均值直接等权相加。
- 传 labels 默认不返回 logits；`return_logits=True` 会产生词表大小的输出。不传 labels 默认返回 logits；推理 `logits_to_keep=1` 只投影最后位置。
- `attention_mask` 支持非空右 padding。单独使用它时一行是连续序列，EOS 不自动重置状态。严格隔离使用下文的 `segment_ids`。
- cache 仅用于 eval + no_grad/inference_mode、无 padding 的等长 batch，不接收 `segment_ids/position_ids`；KV 缓存保持 2 个头，线性状态是 FP32。训练样本之间不传 cache。

## 完整样本 packing 与长上下文

```python
from hybrid100m import PackedCollator

# samples 是已由原 tokenizer 编码的完整样本，带未shift的input_ids/labels。
collator = PackedCollator(2048, template.tokens.pad, config.vocab_size)
batch = {name: tensor.cuda() for name, tensor in collator(samples).items()}
with torch.autocast("cuda", dtype=torch.bfloat16):
    output = model(**batch)
```

collator 对有限样本池做 best-fit decreasing，输出 `input_ids、labels、attention_mask、segment_ids、position_ids`。样本不截断，原 assistant-only labels 保留；过长样本明确报错。padding 的 segment ID 为 -1；每个文档的位置从 0 开始，一个 segment ID 在同一行只能出现一个连续区段。

模型将有效 tokens 压紧为 B=1，KDA 用 `cu_seqlens` 重置状态，卷积显式屏蔽跨文档 tap，全注意力逐文档调用 causal Flash SDPA，不分配全局 T×T block mask。NTP 和 MTP 均删除跨边界目标；返回 logits 时恢复原 batch 形状。大量极短文档会增加 SDPA 调用开销，这个正确性优先的 packing 路径没有额外引入 flash-attn 包。

```python
from hybrid100m import ContextCurriculum, ContextStage

curriculum = ContextCurriculum([
    ContextStage(2048, budget_2k),
    ContextStage(4096, budget_4k),
    ContextStage(10240, budget_10k),
])
seq_length = curriculum.seq_length(processed_tokens)
```

三个 budget 必须由你给出；`processed_tokens` 统计实际经过模型的非 padding tokens，重复采样重复计数。调整 collator/采样器后使用新长度，**不修改 RoPE 参数**。`finished(processed_tokens)` 表示课程预算耗尽，helper 本身不会启动/停止训练。约 5B 预训练 tokens 的最终计数使用现有 tokenizer 统计，不再训练或更换 tokenizer。

现有 `train.bin` 是无文档边界元数据的连续 token 流，不能直接恢复文档隔离；`train.py` 继续按连续窗口读取它。现有 SFT 脚本继续长度分桶与右 padding。使用本节的隔离 packing/自动课程时，需要你的数据管线接入完整样本与课程预算；本次没有改 `pretokenize.py` 或重写数据文件。

长上下文关联应位于同一文档/segment；把相关段落拆成隔离 segment 会失去它们的关联。`build_retrieval_probe` 接受已经 tokenized 的事实、干扰、问题和答案，生成 2K/4K/10K 检索样本，只裁剪干扰填充并只监督答案；准确率由外部生成评测决定。

## 原 tokenizer、SFT 与可见 think

继续使用已有 tokenizer 和 `hybrid100m.chat.ChatTemplate`，保留既有 11 个特殊标记与其 ID。`ChatTemplate` 读取实际特殊 ID，不假定它们是 0～10。本次没有修改 `chat.py` 或 tokenizer 实现。

```python
from hybrid100m.chat import ChatTemplate
from hybrid100m import encode_distillation_records, mix_sft_examples

template = ChatTemplate(tokenizer)
messages = [{"role": "user", "content": "每盒3个，两盒有多少个？"}]
direct = encode_distillation_records(template, [
    {"messages": messages, "answer": "一共有6个。", "answer_verified": True},
])
thinking = encode_distillation_records(template, [
    {"messages": messages, "answer": "一共有6个。",
     "reasoning": "每盒3个，两盒是3×2＝6。", "answer_verified": True},
])
samples = mix_sft_examples(direct, thinking, reasoning_fraction=0.5, total_samples=100)
```

教师输出作为文本用原 tokenizer 编码，`answer_verified=True` 表示你已经外部检查答案；helper 不调用教师、不自动验证答案。可用 `max_sample_tokens` 限制完整样本长度，超长则报错。`mix_sft_examples` 对两个池有放回采样，比例/预算由你决定；示例 0.5 不是经过优化的配方。优先短、可核验推理，同时保留直接回答模式。

```python
from hybrid100m.chat import generate_chat

# model 已加载你训练出的权重；tokenizer 与训练时相同。
result = generate_chat(
    model, template, messages, enable_thinking=True,
    max_think_tokens=128, max_new_tokens=256,
    temperature=0.8, top_k=40, top_p=0.95,
)
print("<think>\n" + result.reasoning + "\n</think>")
print(result.answer)
```

普通聊天 `enable_thinking=False`；`on_token(token_id, channel)` 支持 `think/answer/control` 流式显示。推理预算耗尽时控制器补入 `</think><answer>` 转入答案，这些标记也占生成预算。可见 think 是模型生成的推理说明，其质量来自训练；格式与算法测试不等于已具备推理能力。

## 保存、恢复与兼容性

```python
config.save("config.json")
torch.save(model.state_dict(), "model.pt")
torch.save(optimizer.state_dict(), "optimizer.pt")

restored = HybridForCausalLM(HybridConfig.load("config.json"))
restored.load_state_dict(torch.load("model.pt", map_location="cpu", weights_only=True))
restored_optimizer = build_optimizer(restored, muon_lr=3e-4, adamw_lr=3e-4)
restored_optimizer.load_state_dict(torch.load("optimizer.pt", map_location="cpu", weights_only=True))
```

继续训练还要保存 scheduler、随机数、数据游标和实际处理 token 数。训练配置与 `HybridConfig()` 默认值一致，包含所有新字段。

旧配置缺少新字段时，`HybridConfig.load` 显式恢复 `gated_delta / default RoPE / mtp=0 / chunked loss`，仍可加载原 16 层 checkpoint。**新默认的 20 层 KDA + MTP 权重形状发生变化，不能直接加载旧权重。** 不要用新配置替换旧 checkpoint 的配套配置；固定的 tokenizer/token ID 保持兼容。C++ 运行时同时支持旧 v1 GDN 与新 KDA 格式。

底层参考还包括 [Qwen3.5-0.8B 配置](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/config.json)、[PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) 和 [FLA 安装说明](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)。本项目按约 0.1B 预算独立组织核心代码，无法直接加载上述厂商的模型权重。
