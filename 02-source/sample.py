"""Chat with a trained hybrid100m checkpoint.

Loads model.pt + config.json from a run directory and the tokenizer used in
training, then chats in the terminal. Note: a pretrained-only model has not seen
SFT data yet, so answers will be weak; this is a pipeline sanity check.

Usage:
    python sample.py --run-dir /root/autodl-tmp/runs/pre_2k \
        --tokenizer /root/autodl-tmp/tokenizer.json
"""

import argparse
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer

from hybrid100m import HybridConfig, HybridForCausalLM
from hybrid100m.chat import ChatTemplate, generate_chat


def main():
    parser = argparse.ArgumentParser(description="用训练好的模型对话")
    parser.add_argument("--run-dir", required=True, help="train.py 的输出目录（含 model.pt 和 config.json）")
    parser.add_argument("--tokenizer", required=True, help="训练时用的 tokenizer.json")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = HybridConfig.load(run_dir / "config.json")
    model = HybridForCausalLM(config)
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"已加载 {run_dir / 'model.pt'}（参数量 {model.num_parameters():,}，运行在 {device}）")

    tokenizer = Tokenizer.from_file(args.tokenizer)
    template = ChatTemplate(tokenizer)

    print("输入问题开始对话，输入 quit 退出。仅预训练的模型不会正确聊天，输出乱text属正常，SFT 后才会好。")
    while True:
        try:
            user = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() == "quit":
            break
        messages = [{"role": "user", "content": user}]
        result = generate_chat(
            model, template, messages,
            enable_thinking=False, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=40, top_p=0.95,
        )
        print(f"模型: {result.answer}")


if __name__ == "__main__":
    main()
