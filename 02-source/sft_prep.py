"""Prepare SFT data: minimind conversations -> ChatTemplate-encoded (input_ids, labels).

Reads sft_t2t*.jsonl from minimind_dataset, converts every assistant turn into one
training row via ChatTemplate.encode_sft (prompt masked to -100), and writes a
jsonl file with {"input_ids": [...], "labels": [...]} per line.

Usage:
    python sft_prep.py --data-dir /root/autodl-tmp/minimind_dataset \
        --tokenizer /root/autodl-tmp/tokenizer.json \
        --out /root/autodl-tmp/data/sft_data.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

from tokenizers import Tokenizer

from hybrid100m.chat import ChatTemplate

SFT_FILES = ("sft_t2t_mini.jsonl", "sft_t2t.jsonl")
ROLE_MAP = {
    "user": "user", "human": "user",
    "assistant": "assistant", "gpt": "assistant", "chatgpt": "assistant", "bot": "assistant",
    "system": "system",
}


def extract_conversations(obj):
    conv = obj.get("conversations") if isinstance(obj, dict) else None
    if conv is None and isinstance(obj, dict):
        conv = obj.get("messages")
    if not isinstance(conv, list):
        return None
    turns = []
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        raw_role = str(turn.get("role") or turn.get("from") or "").lower()
        content = turn.get("content") or turn.get("value") or turn.get("text") or ""
        role = ROLE_MAP.get(raw_role)
        if role and content:
            turns.append({"role": role, "content": str(content)})
    merged = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["content"] += "\n" + turn["content"]
        else:
            merged.append(dict(turn))
    return merged or None


def resolve_files(data_dir: Path):
    files = [data_dir / name for name in SFT_FILES if (data_dir / name).exists()]
    files = files[:1]  # 先用 mini，避免大文件重复内容
    if not files:
        found = sorted(data_dir.rglob("*.jsonl"))
        files = [p for p in found if p.name.startswith("sft")] or []
    if not files:
        sys.exit(f"错误: 在 {data_dir} 下没有找到 sft_*.jsonl 对话数据")
    return files


def main():
    parser = argparse.ArgumentParser(description="把 minimind 对话数据编码成 SFT 训练样本")
    parser.add_argument("--data-dir", required=True, help="minimind_dataset 目录")
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json 路径")
    parser.add_argument("--out", required=True, help="输出 jsonl 路径")
    parser.add_argument("--max-len", type=int, default=1024, help="单条样本最大 token 数，超长丢弃")
    parser.add_argument("--limit", type=int, default=0, help="最多读取多少行原始数据，0 = 全部")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    template = ChatTemplate(tokenizer)
    files = resolve_files(Path(args.data_dir))
    print(f"使用数据文件: {[f.name for f in files]}")

    stats = {"lines": 0, "convs": 0, "rows": 0, "skipped_role": 0,
             "skipped_long": 0, "skipped_bad": 0}
    total_tokens = 0
    train_tokens = 0
    preview_done = False
    started = time.time()

    with open(args.out, "w", encoding="utf-8") as out:
        for path in files:
            with open(path, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    stats["lines"] += 1
                    if args.limit and stats["lines"] > args.limit:
                        break
                    try:
                        conv = extract_conversations(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if not conv:
                        continue
                    stats["convs"] += 1

                    if not preview_done and conv:
                        print("首条对话预览（确认格式是否正确）:")
                        for turn in conv[:4]:
                            print(f"  [{turn['role']}] {turn['content'][:60]}")
                        preview_done = True

                    # 每个 assistant 回合生成一条独立样本：历史作为 prompt（-100），
                    # 新回复参与 loss。这正好匹配 ChatTemplate.encode_sft 的协议。
                    for i, turn in enumerate(conv):
                        if turn["role"] != "assistant":
                            continue
                        history = conv[:i]
                        if not history or history[-1]["role"] != "user":
                            stats["skipped_role"] += 1
                            continue
                        try:
                            item = template.encode_sft(history, answer=turn["content"])
                        except ValueError:
                            stats["skipped_bad"] += 1
                            continue
                        n = len(item["input_ids"])
                        if n > args.max_len:
                            stats["skipped_long"] += 1
                            continue
                        if n < 10:
                            stats["skipped_bad"] += 1
                            continue
                        out.write(json.dumps(item, ensure_ascii=False) + "\n")
                        stats["rows"] += 1
                        total_tokens += n
                        train_tokens += sum(1 for t in item["labels"] if t != -100)
                        if stats["rows"] % 100000 == 0:
                            print(f"  已生成 {stats['rows']} 条样本 "
                                  f"({total_tokens / 1e6:.0f}M tokens, "
                                  f"{(time.time() - started) / 60:.1f} 分钟)", flush=True)

    minutes = (time.time() - started) / 60
    if stats["rows"] == 0:
        sys.exit("错误: 一条样本都没生成！大概率是对话字段格式不匹配，把上面打印的预览发给协助者。")
    print(f"完成，耗时 {minutes:.1f} 分钟:")
    print(f"  原始行 {stats['lines']}, 对话 {stats['convs']}, 生成样本 {stats['rows']}")
    print(f"  跳过: 角色不符 {stats['skipped_role']}, 超长(>{args.max_len}) {stats['skipped_long']}, "
          f"异常 {stats['skipped_bad']}")
    print(f"  总 tokens {total_tokens / 1e6:.1f}M, 其中参与 loss 的回复 tokens {train_tokens / 1e6:.1f}M")
    print(f"  输出: {args.out}")


if __name__ == "__main__":
    main()
