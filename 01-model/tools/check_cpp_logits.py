"""Independent numerical check: C++ Q8 engine logits vs PyTorch oracle.

Reimplements the spirit of cpu/test_runtime.py's logits test without its
TemporaryDirectory harness (which the DSH sandbox blocks), using the real
exported SFT model instead of a synthetic one.

Two comparisons are made, and the distinction matters:
  1. C++ Q8 vs Python FP32  -> overall quantization + kernel error
  2. C++ Q8 vs Python Q8 dequantized -> pure kernel/implementation agreement
"""

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from hybrid100m import HybridConfig, HybridForCausalLM  # noqa: E402

RUN = HERE / "local_archive" / "sft_2"
H100 = HERE / "models" / "sft2_q8.h100"
BINARY = HERE / "cpu" / "build" / "hybrid-cpu-native.exe"

IDS = [1, 3, 1732, 528, 183, 343, 10541, 10466, 9689, 5205, 349, 11735, 2141, 2378, 349, 4]


def cpp_logits(ids, threads=4):
    out = subprocess.run(
        [str(BINARY), "logits", "--model", str(H100), "--threads", str(threads),
         "--ids", ",".join(map(str, ids))],
        capture_output=True, text=True, check=True,
    )
    return torch.tensor(json.loads(out.stdout))


def quantize_q8(weight):
    scale = weight.abs().amax(1) / 127
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (weight / scale[:, None]).round().clamp(-127, 127)
    return q * scale[:, None]


def main():
    torch.set_num_threads(8)
    torch.manual_seed(0)

    cfg = HybridConfig.load(RUN / "config.json")
    cfg = dataclasses.replace(cfg, linear_backend="reference", sdpa_backend="math")
    model = HybridForCausalLM(cfg).eval()
    state = torch.load(RUN / "model.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)

    x = torch.tensor([IDS], dtype=torch.long)
    with torch.no_grad():
        fp32 = model(x).logits[0]

    print(f"prompt ids      : {IDS}")
    print(f"PyTorch FP32    : logits {tuple(fp32.shape)}")

    # 用「Python 端 Q8 反量化后的权重」再算一遍，作为纯实现对照
    q8_state = {}
    for name, tensor in state.items():
        q8_state[name] = quantize_q8(tensor) if (tensor.ndim == 2 and not name.startswith("mtp.")) else tensor
    model_q8 = HybridForCausalLM(cfg).eval()
    model_q8.load_state_dict(q8_state)
    with torch.no_grad():
        py_q8 = model_q8(x).logits[0]

    cpp = cpp_logits(IDS)
    print(f"C++ Q8 engine   : logits {tuple(cpp.shape)}")

    def report(label, actual, expected):
        diff = (actual - expected).abs()
        denom = expected.abs().clamp_min(1e-6)
        print(f"\n{label}")
        print(f"  最大绝对误差 : {diff.max().item():.3e}")
        print(f"  平均绝对误差 : {diff.mean().item():.3e}")
        print(f"  最大相对误差 : {(diff / denom).max().item():.3e}")
        print(f"  argmax 一致  : {bool((actual.argmax(-1) == expected.argmax(-1)).all())}")

    report("[1] C++ Q8  vs  PyTorch FP32（含量化误差）", cpp, fp32)
    report("[2] C++ Q8  vs  PyTorch Q8 反量化（纯实现对照）", cpp, py_q8)

    ok = torch.allclose(cpp, py_q8, atol=2e-2, rtol=2e-2)
    print(f"\n结论: {'一致（C++ 实现与 Python 参考实现相符）' if ok else '不一致，需要排查'}")


if __name__ == "__main__":
    main()
