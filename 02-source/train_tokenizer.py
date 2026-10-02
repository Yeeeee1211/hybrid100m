"""Train the hybrid100m tokenizer (ByteLevel BPE, vocab_size=32768 incl. 11 special tokens).

Reads raw text from the jsonl files inside minimind_dataset (pretrain_t2t*.jsonl,
each line {"text": "..."} or {"conversations": [...]}) and writes a single
tokenizer.json usable by both tokenizers.Tokenizer and hybrid100m.chat.ChatTemplate.

Usage:
    python train_tokenizer.py --data-dir /root/autodl-tmp/minimind_dataset \
        --out /root/autodl-tmp/tokenizer.json
"""

import argparse
import json
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors, trainers

from hybrid100m.chat import SPECIAL_TOKENS

# 预训练原文文件优先级；mini 是大文件的小份子集，先读它更快
PRETRAIN_FILES = ("pretrain_t2t_mini.jsonl", "pretrain_t2t.jsonl")


def extract_text(obj):
    """从一条 jsonl 记录里提取纯文本，兼容 minimind 的几种字段布局。"""
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None
    text = obj.get("text")
    if isinstance(text, str):
        return text
    conv = obj.get("conversations")
    if isinstance(conv, list):
        parts = []
        for turn in conv:
            if not isinstance(turn, dict):
                continue
            content = turn.get("content") or turn.get("value") or ""
            role = turn.get("role") or turn.get("from") or ""
            if content:
                parts.append(f"{role}: {content}" if role else content)
        return "\n".join(parts)
    return None


def resolve_files(data_dir: Path):
    files = [data_dir / name for name in PRETRAIN_FILES if (data_dir / name).exists()]
    if files:
        return files[:1]  # 用 mini 就足够，避免重复内容
    found = sorted(data_dir.rglob("*.jsonl"))
    if not found:
        sys.exit(f"错误: 在 {data_dir} 下没有找到任何 .jsonl 文件。"
                 f"用 ls -lh {data_dir} 检查文件后把结果发给协助者。")
    return [p for p in found if p.name.startswith("pretrain")] or found[:1]


def iter_texts(data_dir: Path, sample_chars: int):
    """Yield raw text documents from jsonl files until sample_chars is reached."""
    files = resolve_files(data_dir)
    print(f"使用数据文件: {[f.name for f in files]}")
    total = 0
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = extract_text(obj)
                if text:
                    yield text
                    total += len(text)
                    if total >= sample_chars:
                        print(f"已采样 {total / 1e6:.0f}M 字符，停止读取")
                        return


def main():
    parser = argparse.ArgumentParser(description="训练 hybrid100m 的 32768 词表 tokenizer")
    parser.add_argument("--data-dir", required=True, help="minimind_dataset 所在目录")
    parser.add_argument("--out", default="tokenizer.json", help="输出的 tokenizer.json 路径")
    parser.add_argument("--sample-chars", type=int, default=300_000_000,
                        help="用于训练 tokenizer 的最大字符数（默认 3 亿，足够且更快）")
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

    trainer = trainers.BpeTrainer(
        vocab_size=32768,  # 总词表，包含 11 个特殊 token
        special_tokens=list(SPECIAL_TOKENS),
        min_frequency=2,
        show_progress=True,
    )
    print(f"开始训练 tokenizer（词表 {trainer.vocab_size}，采样上限 {args.sample_chars / 1e6:.0f}M 字符）...")
    tokenizer.train_from_iterator(iter_texts(Path(args.data_dir), args.sample_chars), trainer)

    # 验证：11 个特殊 token 必须各自编码为恰好 1 个 token 且互不冲突
    # （这是 hybrid100m.chat.ChatTokens.from_tokenizer 的硬性要求）
    ids = []
    for token in SPECIAL_TOKENS:
        enc = tokenizer.encode(token, add_special_tokens=False)
        if len(enc.ids) != 1:
            sys.exit(f"验证失败: {token!r} 编码成了 {len(enc.ids)} 个 token")
        ids.append(enc.ids[0])
    if len(set(ids)) != len(ids):
        sys.exit(f"验证失败: 特殊 token ID 冲突: {list(zip(SPECIAL_TOKENS, ids))}")
    print("特殊 token ID 分配:")
    for token, tid in zip(SPECIAL_TOKENS, ids):
        print(f"  {tid:>6}  {token}")

    # 端到端检查：ChatTemplate 能直接读取这个 tokenizer
    try:
        from hybrid100m.chat import ChatTemplate
        template = ChatTemplate(tokenizer)
        probe = template.encode_sft(
            [{"role": "user", "content": "测试"}], answer="好的", reasoning="因为测试"
        )
        assert len(probe["input_ids"]) == len(probe["labels"])
        print("ChatTemplate 端到端检查通过")
    except Exception as e:  # noqa: BLE001
        sys.exit(f"ChatTemplate 检查失败: {e}")

    tokenizer.save(args.out)
    size = tokenizer.get_vocab_size()
    print(f"完成: 词表 {size} 个 token，已保存到 {args.out}")


if __name__ == "__main__":
    main()
