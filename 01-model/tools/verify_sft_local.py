"""Local CPU verification of the archived SFT model (runs/sft_2).

Key difference from the pretrain check: this one drives the *chat* template, so
it proves instruction-following rather than plain continuation.
"""

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
RUN = HERE / "local_archive" / "sft_2"

import torch  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from hybrid100m import HybridConfig, HybridForCausalLM  # noqa: E402


def log(message):
    print(message, flush=True)


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)

    log("[1/5] 加载 config 与 model.pt（严格模式）")
    cfg = HybridConfig.load(RUN / "config.json")
    cfg = dataclasses.replace(cfg, linear_backend="reference", sdpa_backend="math")
    model = HybridForCausalLM(cfg).eval()
    state = torch.load(RUN / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    log(f"      {cfg.num_hidden_layers} 层 | {cfg.linear_attention} | MTP={cfg.mtp_num_layers} | "
        f"{len(state)} 张量加载通过 | 参数量 {sum(p.numel() for p in model.parameters()):,}")

    tokenizer = Tokenizer.from_file(str(HERE / "local_archive" / "pre_full" / "tokenizer.json"))
    t = {"pad": tokenizer.token_to_id("<|pad|>"), "bos": tokenizer.token_to_id("<|bos|>"),
         "eos": tokenizer.token_to_id("<|eos|>"), "im_start": tokenizer.token_to_id("<|im_start|>"),
         "im_end": tokenizer.token_to_id("<|im_end|>")}
    log(f"[2/5] tokenizer: vocab={tokenizer.get_vocab_size()} 特殊 token id={t}")

    question = "用一句话解释什么是机器学习。"
    prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
    ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    log(f"[3/5] 对话 prompt: {question!r} -> {len(ids)} tokens")

    log("[4/5] 生成回答（贪心，最多 40 token）")
    x = torch.tensor([ids], dtype=torch.long)
    generated, started = [], time.time()
    with torch.no_grad():
        for _ in range(40):
            nxt = int(torch.argmax(model(x).logits[0, -1]))
            if nxt in (t["eos"], t["im_end"]):
                break
            generated.append(nxt)
            x = torch.cat([x, torch.tensor([[nxt]])], dim=1)
    elapsed = time.time() - started
    answer = tokenizer.decode(generated, skip_special_tokens=True)
    log(f"      用时 {elapsed:.1f}s（{len(generated)/max(elapsed,1e-9):.1f} tok/s CPU）")
    log(f"      回答: {answer!r}")

    log("[5/5] 正确停笔检查")
    stopped = nxt in (t["eos"], t["im_end"])

    report = {
        "run": "sft_2",
        "params": sum(p.numel() for p in model.parameters()),
        "state_dict_tensors": len(state),
        "question": question,
        "answer": answer,
        "stopped_correctly": bool(stopped),
        "cpu_tok_per_s": round(len(generated) / max(elapsed, 1e-9), 2),
    }
    # 先落盘再打印：控制台是 GBK，emoji/部分字符会抛 UnicodeEncodeError
    (RUN / "verify_local.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[5/5] 正确停笔: {stopped} | 报告 -> {RUN / 'verify_local.json'}", flush=True)


if __name__ == "__main__":
    main()
