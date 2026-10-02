"""SFT training for hybrid100m: fine-tune a pretrained checkpoint on chat data.

Loads model.pt + config.json from the pretrain run directory, trains on the
jsonl produced by sft_prep.py (variable-length rows, right padding, loss only
on assistant completions), and saves a chat-capable model into --out-dir.

Usage:
    python train_sft.py --data /root/autodl-tmp/data/sft_data.jsonl \
        --init-from /root/autodl-tmp/runs/pre_2k \
        --tokenizer /root/autodl-tmp/tokenizer.json \
        --out-dir /root/autodl-tmp/runs/sft_1 \
        --micro-batch 8 --accum 4 --epochs 2
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from hybrid100m import HybridConfig, HybridForCausalLM
from hybrid100m.chat import ChatTemplate
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
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8, foreach=False,
    )


def lr_at(step, total, warmup, lr, min_frac):
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return lr * (min_frac + (1 - min_frac) * 0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))


def make_microbatches(samples, micro_batch, rng):
    """按长度分桶组 micro-batch，减少 padding 浪费；返回打乱顺序的 batch 列表。"""
    order = sorted(range(len(samples)), key=lambda i: len(samples[i][0]))
    batches = [order[i:i + micro_batch] for i in range(0, len(order), micro_batch)]
    rng.shuffle(batches)
    for batch in batches:
        rng.shuffle(batch)
    return batches


def collate(samples, indices, pad_id, device):
    rows = [samples[i] for i in indices]
    max_len = max(len(ids) for ids, _ in rows)
    b = len(rows)
    input_ids = torch.full((b, max_len), pad_id, dtype=torch.int64)
    labels = torch.full((b, max_len), -100, dtype=torch.int64)
    mask = torch.zeros((b, max_len), dtype=torch.bool)
    for row, (ids, labs) in enumerate(rows):
        n = len(ids)
        input_ids[row, :n] = torch.from_numpy(ids.astype(np.int64))
        labels[row, :n] = torch.from_numpy(labs.astype(np.int64))
        mask[row, :n] = True
    return (input_ids.to(device, non_blocking=True),
            labels.to(device, non_blocking=True),
            mask.to(device, non_blocking=True))


def group_token_counts(samples, group):
    """NTP counts: unshifted label zero is never a prediction target."""
    return [sum(int((samples[i][1][1:] != -100).sum()) for i in indices) for indices in group]


def _prediction_counts(samples, indices, mtp_depths):
    # Samples are complete unpadded rows, so each row's target slices can be
    # counted on the host before backward. No second forward is necessary.
    return tuple(sum(int((samples[i][1][depth + 1:] != -100).sum()) for i in indices)
                 for depth in range(mtp_depths + 1))


def _mean_objective(ntp_sum, ntp_count, mtp_sums, mtp_counts, mtp_weight):
    result = ntp_sum / max(ntp_count, 1)
    if mtp_sums:
        result += mtp_weight * sum(total / max(count, 1) for total, count in zip(mtp_sums, mtp_counts)) / len(mtp_sums)
    return result


def train_epoch(model, samples, batches, optimizer, device, pad_id, args,
                schedule, step_offset, log_prefix, out_dir):
    """一个 epoch 的训练循环，返回 (有效 tokens 上的平均 loss, 完成的优化器步数)。"""
    total_loss, total_tokens, steps_done = 0.0, 0, 0
    mtp_depths = model.config.mtp_num_layers if model.config.mtp_loss_weight > 0 else 0
    mtp_sums, mtp_counts = [0.0] * mtp_depths, [0] * mtp_depths
    group, group_counts = [], []
    started = time.perf_counter()
    grad_norm = 0.0

    def flush():
        # started 必须在 nonlocal 里：flush 内既读（算 dt）又写（重置计时）
        nonlocal group, group_counts, total_loss, total_tokens, steps_done, grad_norm, started
        if not group:
            return
        denominators = [max(sum(counts[depth] for counts in group_counts), 1)
                        for depth in range(mtp_depths + 1)]
        for indices in group:
            input_ids, labels, mask = collate(samples, indices, pad_id, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = model(input_ids, labels=labels, attention_mask=mask, compute_mtp=bool(mtp_depths))
                objective = output.loss_sum / denominators[0]
                if mtp_depths:
                    objective = objective + model.config.mtp_loss_weight * sum(
                        total / denominators[depth + 1]
                        for depth, total in enumerate(output.mtp_loss_sums)
                    ) / mtp_depths
            objective.backward()
            total_loss += output.loss_sum.item()
            total_tokens += int(output.num_loss_tokens.item())
            for depth, (total, count) in enumerate(zip(output.mtp_loss_sums, output.mtp_num_tokens)):
                mtp_sums[depth] += total.item()
                mtp_counts[depth] += int(count.item())
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip).item()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        steps_done += 1
        if steps_done % args.log_every == 0:
            dt = time.perf_counter() - started
            allocated = torch.cuda.max_memory_allocated() / 2**30
            lr = schedule(step_offset + steps_done)
            print(f"{log_prefix} step {step_offset + steps_done} | "
                  f"loss {_mean_objective(total_loss, total_tokens, mtp_sums, mtp_counts, model.config.mtp_loss_weight):.4f} | lr {lr:.2e} | "
                  f"grad_norm {grad_norm:.2f} | {total_tokens / dt / 1e3:.1f}K tok/s | "
                  f"显存 {allocated:.2f} GiB", flush=True)
            started = time.perf_counter()
        if args.save_every > 0 and steps_done % args.save_every == 0:
            # 全量 SFT 一轮要数小时，按步存档避免崩溃/断电丢光进度
            save(out_dir, model, optimizer, step_offset + steps_done, args,
                 model_step=step_offset + steps_done)
            model.train()
        group, group_counts = [], []

    for indices in batches:
        group.append(indices)
        group_counts.append(_prediction_counts(samples, indices, mtp_depths))
        if len(group) == args.accum:
            flush()
    flush()
    return _mean_objective(total_loss, total_tokens, mtp_sums, mtp_counts, model.config.mtp_loss_weight), steps_done


@torch.no_grad()
def evaluate(model, samples, batches, device, pad_id):
    model.eval()
    loss_sum, token_sum = 0.0, 0
    mtp_depths = model.config.mtp_num_layers if model.config.mtp_loss_weight > 0 else 0
    mtp_sums, mtp_counts = [0.0] * mtp_depths, [0] * mtp_depths
    for indices in batches:
        input_ids, labels, mask = collate(samples, indices, pad_id, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(input_ids, labels=labels, attention_mask=mask, compute_mtp=bool(mtp_depths))
        loss_sum += output.loss_sum.item()
        token_sum += int(output.num_loss_tokens.item())
        for depth, (total, count) in enumerate(zip(output.mtp_loss_sums, output.mtp_num_tokens)):
            mtp_sums[depth] += total.item()
            mtp_counts[depth] += int(count.item())
    model.train()
    return _mean_objective(loss_sum, token_sum, mtp_sums, mtp_counts, model.config.mtp_loss_weight)


def save(out_dir, model, optimizer, epoch, args, model_step=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "model_step": model_step, "args": vars(args)}, out_dir / "ckpt.pt")
    torch.save(model.state_dict(), out_dir / "model.pt")
    model.config.save(out_dir / "config.json")


def main():
    parser = argparse.ArgumentParser(description="hybrid100m SFT 微调")
    parser.add_argument("--data", required=True, help="sft_prep.py 的输出 jsonl")
    parser.add_argument("--init-from", required=True, help="预训练运行目录（含 model.pt 和 config.json）")
    parser.add_argument("--tokenizer", required=True, help="tokenizer.json（用于读取 <|pad|> 的 ID）")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--micro-batch", type=int, default=8, help="物理 batch（变长短句，可比预训练大）")
    parser.add_argument("--accum", type=int, default=4, help="梯度累积（有效 batch = 8×4 = 32 条样本）")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=150000, help="最多用多少条样本，0 = 全部")
    parser.add_argument("--max-len", type=int, default=1024, help="丢弃超过此长度的样本")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--min-lr-frac", type=float, default=0.1)
    parser.add_argument("--wd", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.01)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=300,
                        help="每多少优化器步存一次档（0 = 只在 epoch 结束存）；全量跑一轮数小时，"
                             "中途崩溃/断电不至于全丢")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    device = "cuda"

    init_dir = Path(args.init_from)
    config = HybridConfig.load(init_dir / "config.json")
    model = HybridForCausalLM(config)
    model.load_state_dict(torch.load(init_dir / "model.pt", map_location="cpu", weights_only=True))
    model = model.to(device)
    print(f"GPU: {torch.cuda.get_device_name(0)} | 从 {init_dir} 加载预训练权重 "
          f"({model.num_parameters():,} 参数)")

    pad_id = ChatTemplate(Tokenizer.from_file(args.tokenizer)).tokens.pad

    print("加载 SFT 数据...")
    samples = []
    with open(args.data, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            ids = np.asarray(item["input_ids"], dtype=np.int32)
            if len(ids) > args.max_len:
                continue
            samples.append((ids, np.asarray(item["labels"], dtype=np.int32)))
    if not samples:
        sys.exit("错误: 没有读到任何样本，检查 sft_prep.py 的输出")
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    if args.max_samples and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
    n_val = max(1, int(len(samples) * args.val_frac))
    val_samples, train_samples = samples[:n_val], samples[n_val:]
    lens = [len(s[0]) for s in train_samples]
    print(f"样本: 训练 {len(train_samples)}, 验证 {n_val} | 长度中位数 {int(np.median(lens))}, 最长 {max(lens)}")

    optimizer = build_optimizer(model, args.lr, args.wd)
    out_dir = Path(args.out_dir)
    start_epoch = 0
    if args.resume and (out_dir / "ckpt.pt").exists():
        ckpt = torch.load(out_dir / "ckpt.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"]
        print(f"已从 epoch {start_epoch} 恢复"
              + (f"（该 epoch 内已训练到 step {ckpt['model_step']}，本实现会重跑当前 epoch）"
                 if ckpt.get("model_step") is not None else ""))
    elif args.resume:
        print("未找到已有 checkpoint，从头开始 SFT")

    probe_batches = make_microbatches(train_samples, args.micro_batch, random.Random(args.seed))
    steps_per_epoch = (len(probe_batches) + args.accum - 1) // args.accum
    total_steps = steps_per_epoch * (args.epochs - start_epoch)
    schedule = lambda s: lr_at(s, total_steps, args.warmup, args.lr, args.min_lr_frac)  # noqa: E731
    print(f"开始 SFT: {args.epochs} epochs, 每 epoch 约 {steps_per_epoch} 步, 共 {total_steps} 步, "
          f"有效 batch {args.micro_batch}×{args.accum} 条样本")

    model.train()
    step_offset = 0
    current_epoch = start_epoch
    try:
        for epoch in range(start_epoch, args.epochs):
            current_epoch = epoch
            batches = make_microbatches(train_samples, args.micro_batch,
                                        random.Random(args.seed + epoch))
            train_loss, _ = train_epoch(model, train_samples, batches, optimizer, device,
                                        pad_id, args, schedule, step_offset,
                                        log_prefix=f"epoch {epoch + 1}", out_dir=out_dir)
            step_offset += steps_per_epoch
            val_batches = make_microbatches(val_samples, args.micro_batch,
                                            random.Random(args.seed))
            val_loss = evaluate(model, val_samples, val_batches, device, pad_id)
            print(f"epoch {epoch + 1}/{args.epochs} 完成 | 训练 loss {train_loss:.4f} | "
                  f"验证 loss {val_loss:.4f}", flush=True)
            save(out_dir, model, optimizer, epoch + 1, args)
            print(f"  已保存到 {out_dir}", flush=True)
    except KeyboardInterrupt:
        print("\n检测到手动中断，保存进度（--resume 会重跑当前未完成的 epoch）...")
        save(out_dir, model, optimizer, current_epoch, args)
        return

    print("SFT 完成！用 sample.py --run-dir 指向本目录即可对话测试。")


if __name__ == "__main__":
    main()
