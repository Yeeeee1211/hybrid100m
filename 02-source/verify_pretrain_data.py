"""Verify the full-corpus pretraining data before starting the big run.

Run on the server:
    python verify_pretrain_data.py --data-dir /root/autodl-tmp/data_full \
        --tokenizer /root/autodl-tmp/tokenizer.json \
        --log /root/autodl-tmp/pretokenize_full.log

Checks: pretokenize process finished, files exist and sizes are sane, token
values are valid (< vocab, no stray pad), stream decodes back to readable
text, and cross-checks against the pretokenize log. Exit code 0 = pass.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

VOCAB = 32768
WINDOW = 2048
N_WINDOWS = 8
MINI_TOKENS = 242_000_000  # 第一轮 mini 语料规模，用于对照

results = []  # (name, ok: True/False/None(警告), detail)


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'✅' if ok else '❌'} {name}" + (f" — {detail}" if detail else ""))


def warn(name, detail=""):
    results.append((name, None, detail))
    print(f"⚠️  {name}" + (f" — {detail}" if detail else ""))


def human_tokens(n):
    return f"{n / 1e6:.1f}M" if n < 1e9 else f"{n / 1e9:.2f}B"


def main():
    parser = argparse.ArgumentParser(description="验证全量预编码数据")
    parser.add_argument("--data-dir", default="/root/autodl-tmp/data_full")
    parser.add_argument("--tokenizer", default="/root/autodl-tmp/tokenizer.json")
    parser.add_argument("--log", default="/root/autodl-tmp/pretokenize_full.log")
    parser.add_argument("--full-expected", type=float, default=1.4e9,
                        help="全量语料最低 token 预期（按 mini 压缩率估算 ~1.68B）")
    args = parser.parse_args()

    print(f"=== 验证目标: {args.data_dir} ===\n")

    # 0. pretokenize 是否还在运行
    try:
        out = subprocess.run(["pgrep", "-af", "pretokenize.py"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            warn("pretokenize 进程仍在运行（数字不是最终值，稍后重跑本脚本）",
                 out.stdout.strip()[:120])
        else:
            check("pretokenize 进程已结束", True)
    except FileNotFoundError:
        warn("无法检查进程（pgrep 不可用），请自行确认编码已停止")

    # 1. 文件存在与基本大小
    data_dir = Path(args.data_dir)
    train_path, val_path = data_dir / "train.bin", data_dir / "val.bin"
    if not train_path.exists():
        check("train.bin 存在", False, f"{train_path} 不存在 — 预编码尚未完成或失败，查看 {args.log}")
        sys.exit(1)
    train_size = os.path.getsize(train_path)
    check("train.bin 存在", True, f"{train_size / 2**20:.0f} MiB")
    check("文件大小是 uint32 的整数倍", train_size % 4 == 0,
          f"{train_size} 字节" if train_size % 4 else "对齐正确")

    # 2. token 总量与全量预期
    train_tokens = train_size // 4
    val_tokens = 0
    if val_path.exists():
        val_tokens = os.path.getsize(val_path) // 4
        check("val.bin 存在", True, human_tokens(val_tokens) + " tokens")
    else:
        warn("val.bin 不存在（流末尾切分步骤没跑完？）")
    total = train_tokens + val_tokens
    print(f"\n   实际总 tokens: {human_tokens(total)} （对照: mini 全量 {human_tokens(MINI_TOKENS)}）\n")
    if total >= args.full_expected:
        check("token 总量达到全量语料预期", True, f">= {human_tokens(args.full_expected)}")
    elif total >= 1.0e9:
        warn("token 总量低于估算值", f"{human_tokens(total)} < {human_tokens(args.full_expected)}，"
                                "可能是语料压缩率不同，确认数据源后可接受")
    elif total >= 0.4e9:
        check("token 总量达到全量语料预期", False,
              f"只有 {human_tokens(total)} —— 疑似只编码了 mini 文件，检查 --file 参数")
    else:
        check("token 总量达到全量语料预期", False, f"只有 {human_tokens(total)}，数据严重不足")

    # 3. 与 pretokenize 日志交叉核对
    log_path = Path(args.log)
    if log_path.exists():
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        m_done = re.search(r"完成: (\d+) 篇文档, 编码 (\d+)M tokens", text)
        m_train = re.search(r"train\.bin: ([\d.]+)M tokens", text)
        m_val = re.search(r"val\.bin:\s+([\d.]+)M tokens", text)
        if m_done:
            check("日志显示编码正常完成", True,
                  f"{m_done.group(1)} 篇文档, {m_done.group(2)}M tokens")
        else:
            warn("日志里没有找到「完成」行", "编码可能中途失败或仍在进行")
        if m_train and m_val:
            log_total = (float(m_train.group(1)) + float(m_val.group(1))) * 1e6
            if abs(log_total - total) < 1_000_000:
                check("日志数字与实际文件一致（切分步骤完整）", True)
            else:
                check("日志数字与实际文件一致（切分步骤完整）", False,
                      f"日志 {human_tokens(log_total)} vs 文件 {human_tokens(total)}")
        else:
            warn("日志里没有 train.bin/val.bin 明细行")
    else:
        warn("找不到 pretokenize 日志，跳过日志核对")

    # 4. token 内容抽样体检
    tokenizer = Tokenizer.from_file(args.tokenizer)
    eos_id = tokenizer.token_to_id("<|eos|>")
    rng = np.random.default_rng(42)
    data = np.memmap(train_path, dtype=np.uint32, mode="r")
    offsets = rng.integers(0, max(len(data) - WINDOW, 1), N_WINDOWS)
    sample = np.concatenate([data[o:o + WINDOW] for o in offsets])
    max_id = int(sample.max())
    pad_share = float((sample == 0).mean())
    eos_share = float((sample == eos_id).mean()) if eos_id is not None else -1
    unique = len(np.unique(sample))
    check("token ID 全部 < vocab_size(32768)", max_id < VOCAB, f"max={max_id}")
    check("无异常 pad token（流中不应出现 id 0）", pad_share == 0.0,
          f"占比 {pad_share:.4%}" if pad_share else "干净")
    check("EOS 密度合理（文档边界正常）", 0.0005 < eos_share < 0.05,
          f"{eos_share:.3%}（平均文档 ≈ {1 / max(eos_share, 1e-9):.0f} tokens）")
    check("窗口内 token 多样性正常", unique > 500, f"8×2048 窗口内 {unique} 种不同 token")

    # 5. 解码回读：随机窗口应能解出可读文本
    text = tokenizer.decode(sample[:256].tolist(), skip_special_tokens=True)
    if text:
        bad = sum(1 for ch in text if ch == "\ufffd" or (not ch.isprintable() and ch != "\n"))
        ratio = 1 - bad / len(text)
        check("随机窗口能解码为可读文本", ratio > 0.95, f"可读率 {ratio:.1%}")
        print("\n   解码样例: " + text[:150].replace("\n", "⏎") + " ...\n")
    else:
        check("随机窗口能解码为可读文本", False, "解码结果为空")

    # 总结
    failed = [r for r in results if r[1] is False]
    warned = [r for r in results if r[1] is None]
    print("=== 总结 ===")
    if failed:
        print(f"❌ {len(failed)} 项未通过 → 不要开始训练，先解决上面标 ❌ 的问题")
        sys.exit(1)
    print(f"✅ 全部 {len(results) - len(warned)} 项检查通过"
          + (f"（另有 {len(warned)} 条警告）" if warned else ""))
    print("\n数据就绪。注意：2026-09-30 起主干已换成 20 层 KDA + MTP（Muon 优化器），"
          "旧的 16 层 pre_2k 权重形状不兼容，不能 --resume 续训，需要从头开始新主干训练：")
    print("  cd /root/hybrid100m && nohup python train.py --data-dir " + str(data_dir) +
          " --out-dir /root/autodl-tmp/runs/pre_full \\\n"
          "      --seq-len 2048 --micro-batch 8 --accum 4 --lr 3e-4 --max-steps <预算步数> \\\n"
          "      --save-every 500 --val-every 250 > /root/autodl-tmp/runs/pre_full.log 2>&1 &")


if __name__ == "__main__":
    main()
