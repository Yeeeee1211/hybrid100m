"""Pretraining script for hybrid100m on a single RTX 4090.

Loads config.json, trains on the uint32 token stream produced by pretokenize.py
with BF16 autocast, gradient accumulation, cosine LR schedule, periodic validation
and full checkpoint/resume support.

Example:
    python train.py --data-dir /root/autodl-tmp/data \
        --out-dir /root/autodl-tmp/runs/pre_2k \
        --seq-len 2048 --micro-batch 1 --accum 32 --max-steps 500
"""

import argparse
import dataclasses
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from hybrid100m import HybridConfig, HybridForCausalLM
from hybrid100m.optim import build_optimizer as build_core_optimizer


def build_optimizer(model, lr, weight_decay):
    """New KDA models use Muon; legacy GDN retains resumable AdamW groups."""
    if model.config.linear_attention == "kda":
        return build_core_optimizer(model, muon_lr=lr, adamw_lr=lr, weight_decay=weight_decay)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "A_log" in name or "dt_bias" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, foreach=False)


def lr_at(step, total, warmup, lr, min_frac):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))


def get_batch(data, batch_size, seq_len, device):
    ix = torch.randint(len(data) - seq_len, (batch_size,))
    batch = np.stack([data[i:i + seq_len] for i in ix.tolist()]).astype(np.int64)
    return torch.from_numpy(batch).to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, val_data, args, device, windows=16):
    model.eval()
    losses = []
    for _ in range(windows):
        x = get_batch(val_data, args.micro_batch, args.seq_len, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            losses.append(model(x, labels=x).loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(out_dir, model, optimizer, step, args):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "args": vars(args),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        },
    }
    torch.save(ckpt, out_dir / "ckpt.pt")
    torch.save(model.state_dict(), out_dir / "model.pt")
    model.config.save(out_dir / "config.json")


def main():
    parser = argparse.ArgumentParser(description="hybrid100m 预训练")
    parser.add_argument("--data-dir", required=True, help="pretokenize.py 的输出目录（含 train.bin/val.bin）")
    parser.add_argument("--out-dir", required=True, help="checkpoint 输出目录")
    parser.add_argument("--config", default="config.json", help="模型配置文件")
    parser.add_argument("--seq-len", type=int, default=2048, help="序列长度（2K → 4K → 10K 课程）")
    parser.add_argument("--micro-batch", type=int, default=1, help="物理 batch，先从 1 开始测显存")
    parser.add_argument("--accum", type=int, default=32, help="梯度累积步数（有效 batch = micro-batch × accum）")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--min-lr-frac", type=float, default=0.1, help="余弦退火的最低 lr 比例")
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--wd", type=float, default=0.1, help="矩阵参数的 weight decay")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--loss-chunk", type=int, default=0, help="覆盖 config 的 loss_chunk_size，0 = 不覆盖")
    parser.add_argument("--resume", action="store_true", help="从 out-dir/ckpt.pt 恢复训练")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    device = "cuda"
    config = HybridConfig.load(args.config)
    if args.loss_chunk:
        config = dataclasses.replace(config, loss_chunk_size=args.loss_chunk)
    if args.seq_len > config.max_position_embeddings:
        sys.exit(f"错误: seq_len {args.seq_len} 超过 max_position_embeddings {config.max_position_embeddings}")

    model = HybridForCausalLM(config).to(device)
    print(f"GPU: {torch.cuda.get_device_name(0)} | 参数量: {model.num_parameters():,}")
    print(f"结构: {config.layer_types.count('linear')} 线性层 + {config.layer_types.count('full')} 全注意力层, "
          f"seq_len={args.seq_len}, micro_batch={args.micro_batch}, accum={args.accum}")

    train_path, val_path = Path(args.data_dir) / "train.bin", Path(args.data_dir) / "val.bin"
    if not train_path.exists():
        sys.exit(f"错误: 找不到 {train_path}，先运行 pretokenize.py")
    train_data = np.memmap(train_path, dtype=np.uint32, mode="r")
    val_data = np.memmap(val_path, dtype=np.uint32, mode="r") if val_path.exists() else None
    print(f"数据: train {len(train_data) / 1e6:.1f}M tokens"
          + (f", val {len(val_data) / 1e6:.1f}M tokens" if val_data is not None else ", 无验证集"))

    optimizer = build_optimizer(model, args.lr, args.wd)
    out_dir = Path(args.out_dir)
    start_step = 0
    ckpt_path = out_dir / "ckpt.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        torch.set_rng_state(ckpt["rng"]["torch"].cpu())
        torch.cuda.set_rng_state_all([s.cpu() if s.device.type != "cpu" else s for s in ckpt["rng"]["cuda"]])
        np.random.set_state(ckpt["rng"]["numpy"])
        random.setstate(ckpt["rng"]["python"])
        print(f"已从 step {start_step} 恢复训练")
    elif args.resume:
        print("未找到已有 checkpoint，从头开始训练")

    tokens_per_step = args.micro_batch * args.accum * args.seq_len
    model.train()
    print(f"开始训练: {args.max_steps} 步, 每步 {tokens_per_step:,} tokens "
          f"(fp32 参数 + bf16 autocast, loss_chunk={config.loss_chunk_size})")

    step_losses = []
    step_started = time.perf_counter()
    try:
        for step in range(start_step, args.max_steps):
            lr = lr_at(step, args.max_steps, args.warmup, args.lr, args.min_lr_frac)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad(set_to_none=True)
            for _ in range(args.accum):
                x = get_batch(train_data, args.micro_batch, args.seq_len, device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(x, labels=x).loss
                (loss / args.accum).backward()
                step_losses.append(loss.item())

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
            optimizer.step()

            if (step + 1) % args.log_every == 0:
                dt = time.perf_counter() - step_started
                mean_loss = sum(step_losses) / len(step_losses)
                allocated = torch.cuda.max_memory_allocated() / 2**30
                reserved = torch.cuda.max_memory_reserved() / 2**30
                # dt 覆盖 log_every 步，分子必须乘上 log_every，否则吞吐被低估 log_every 倍
                tokens_per_second = tokens_per_step * len(step_losses) / max(dt, 1e-9)
                print(f"step {step + 1:>6}/{args.max_steps} | loss {mean_loss:.4f} | "
                      f"lr {lr:.2e} | grad_norm {grad_norm:.2f} | "
                      f"{tokens_per_second / 1e3:.1f}K tok/s | "
                      f"显存 {allocated:.2f}/{reserved:.2f} GiB", flush=True)
                step_losses = []
                step_started = time.perf_counter()

            if val_data is not None and (step + 1) % args.val_every == 0:
                val_loss = evaluate(model, val_data, args, device)
                print(f"step {step + 1:>6} | 验证 loss {val_loss:.4f}", flush=True)

            if (step + 1) % args.save_every == 0 or step + 1 == args.max_steps:
                save_checkpoint(out_dir, model, optimizer, step + 1, args)
                print(f"step {step + 1:>6} | 已保存 checkpoint 到 {out_dir}", flush=True)
    except KeyboardInterrupt:
        print("\n检测到手动中断，正在保存当前进度...")
        save_checkpoint(out_dir, model, optimizer, step + 1, args)
        print(f"进度已保存（step {step + 1}），用 --resume 可继续训练")
        return

    print("训练完成。用 sample.py 可以对话测试，或继续下一段课程（4K/10K）。")


if __name__ == "__main__":
    main()
