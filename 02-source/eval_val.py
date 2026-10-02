"""Quick checkpoint sanity check: evaluate validation loss without training.

Usage:
    python eval_val.py --run-dir /root/autodl-tmp/runs/pre_2k --data-dir /root/autodl-tmp/data
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from hybrid100m import HybridConfig, HybridForCausalLM


def main():
    parser = argparse.ArgumentParser(description="评估一个 checkpoint 的验证集 loss")
    parser.add_argument("--run-dir", required=True, help="训练输出目录（含 ckpt.pt / config.json）")
    parser.add_argument("--data-dir", required=True, help="数据目录（含 val.bin）")
    parser.add_argument("--windows", type=int, default=20, help="评估多少个随机窗口")
    parser.add_argument("--micro-batch", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = HybridConfig.load(run_dir / "config.json")
    model = HybridForCausalLM(config).cuda().eval()

    ckpt = torch.load(run_dir / "ckpt.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"checkpoint 记录的进度: step {ckpt.get('step')}")

    val = np.memmap(Path(args.data_dir) / "val.bin", dtype=np.uint32, mode="r")
    if len(val) <= args.seq_len:
        raise SystemExit("val.bin 太小，先运行 pretokenize.py")

    losses = []
    with torch.no_grad():
        for _ in range(args.windows):
            ix = torch.randint(len(val) - args.seq_len, (args.micro_batch,))
            batch = np.stack([val[i:i + args.seq_len] for i in ix.tolist()]).astype(np.int64)
            x = torch.from_numpy(batch).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                losses.append(model(x, labels=x).loss.item())

    mean_loss = sum(losses) / len(losses)
    print(f"验证 loss: {mean_loss:.4f}（{args.windows} 个窗口，每窗 {args.micro_batch}×{args.seq_len} tokens）")
    print("判读: 随机模型 ≈ 10.4；< 7 说明存档是真训练过的；≈ 9~10.4 说明存档是废的，应重新开训")


if __name__ == "__main__":
    main()
