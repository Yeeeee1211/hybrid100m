"""Pre-tokenize raw text (jsonl) into uint32 token streams for training.

Reads pretrain_t2t*.jsonl from the minimind_dataset download, encodes each
document with the trained tokenizer, appends the <|eos|> id, and writes
<out-dir>/train.bin and <out-dir>/val.bin (last val-fraction carved out).

Usage:
    python pretokenize.py --data-dir /root/autodl-tmp/minimind_dataset \
        --tokenizer /root/autodl-tmp/tokenizer.json --out-dir /root/autodl-tmp/data \
        --max-tokens 300000000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from hybrid100m.chat import SPECIAL_TOKENS

WRITE_MB = 64  # write buffer in tokens (~256 MiB as uint32)
PRETRAIN_FILES = ("pretrain_t2t_mini.jsonl", "pretrain_t2t.jsonl")


def extract_text(obj):
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


def iter_docs(data_dir: Path, file_override: str = ""):
    if file_override:
        path = data_dir / file_override
        if not path.exists():
            sys.exit(f"错误: --file 指定的文件不存在: {path}")
        files = [path]
    else:
        files = [data_dir / name for name in PRETRAIN_FILES if (data_dir / name).exists()]
        files = files[:1]
        if not files:
            found = sorted(data_dir.rglob("*.jsonl"))
            files = [p for p in found if p.name.startswith("pretrain")] or found[:1]
    if not files:
        sys.exit(f"错误: 在 {data_dir} 下没有找到任何 .jsonl 文件")
    print(f"使用数据文件: {[f.name for f in files]}")
    for path in files:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    text = extract_text(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if text and text.strip():
                    yield text


def main():
    parser = argparse.ArgumentParser(description="把 jsonl 原文预编码成 uint32 token 流")
    parser.add_argument("--data-dir", required=True, help="jsonl 数据目录")
    parser.add_argument("--tokenizer", required=True, help="train_tokenizer.py 产出的 tokenizer.json")
    parser.add_argument("--out-dir", required=True, help="train.bin / val.bin 输出目录")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="最多编码多少 token，0 = 全部数据（首次建议 3 亿）")
    parser.add_argument("--val-frac", type=float, default=0.005, help="留作验证集的比例（默认 0.5%%）")
    parser.add_argument("--file", default="",
                        help="只处理指定的 jsonl 文件名（如 pretrain_t2t.jsonl）；默认自动优先 mini 小文件")
    args = parser.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|eos|>")
    if eos_id is None:
        sys.exit("错误: tokenizer 里没有 <|eos|>，请先用 train_tokenizer.py 生成")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "train.bin"

    buffer = np.empty(WRITE_MB, dtype=np.uint32)
    filled = 0
    total_tokens = 0
    total_docs = 0
    started = time.time()
    writer = open(train_path, "wb")

    def flush():
        nonlocal filled
        writer.write(buffer[:filled].tobytes())
        filled = 0

    def append(ids):
        nonlocal filled, total_tokens
        n = len(ids)
        total_tokens += n
        pos = 0
        while pos < n:
            take = min(n - pos, WRITE_MB - filled)
            buffer[filled:filled + take] = ids[pos:pos + take]
            filled += take
            pos += take
            if filled == WRITE_MB:
                flush()

    limit = args.max_tokens if args.max_tokens > 0 else float("inf")
    for doc in iter_docs(Path(args.data_dir), args.file):
        ids = tokenizer.encode(doc, add_special_tokens=False).ids
        append(np.asarray(ids + [eos_id], dtype=np.uint32))
        total_docs += 1
        if total_docs % 20000 == 0:
            rate = total_tokens / max(time.time() - started, 1e-6)
            print(f"  已编码 {total_docs} 篇 / {total_tokens / 1e6:.0f}M tokens "
                  f"({rate / 1e3:.0f}K tokens/s)", flush=True)
        if total_tokens >= limit:
            print(f"达到 --max-tokens={args.max_tokens}，停止读取")
            break

    flush()
    writer.close()
    if total_tokens < 1_000_000:
        sys.exit(f"错误: 只编码出 {total_tokens} 个 token，数据量太少，请检查数据文件")

    # 从流末尾切出验证集
    n = os.path.getsize(train_path) // 4
    n_val = min(int(n * args.val_frac), 5_000_000)
    if n_val > 0:
        data = np.memmap(train_path, dtype=np.uint32, mode="r")
        np.asarray(data[n - n_val:], dtype=np.uint32).tofile(out_dir / "val.bin")
        del data
        with open(train_path, "r+b") as f:
            f.truncate((n - n_val) * 4)

    minutes = (time.time() - started) / 60
    print(f"完成: {total_docs} 篇文档, 编码 {total_tokens / 1e6:.0f}M tokens, 耗时 {minutes:.1f} 分钟")
    print(f"  train.bin: {(n - n_val) / 1e6:.1f}M tokens -> {train_path}")
    print(f"  val.bin:   {n_val / 1e6:.1f}M tokens -> {out_dir / 'val.bin'}")
    print(f"词表大小 {tokenizer.get_vocab_size()}，特殊 token: {len(SPECIAL_TOKENS)} 个")


if __name__ == "__main__":
    main()
