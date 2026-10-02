"""Focused follow-up probes: locate exactly where instruction following breaks."""

import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent.parent
BINARY = HERE / "cpu" / "build" / "hybrid-cpu-native.exe"
MODEL = HERE / "models" / "sft2_q8.h100"


def chat(prompt, *, system=None, max_tokens=64, temperature=0.2):
    cmd = [str(BINARY), "chat", "--model", str(MODEL), "--threads", "8", "--ctx", "2048",
           "--prompt", prompt, "--max-tokens", str(max_tokens),
           "--temperature", str(temperature), "--top-k", "0", "--json"]
    if system:
        cmd += ["--system", system]
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", cwd=HERE)
    try:
        return json.loads(out.stdout)["answer"].strip()
    except Exception:
        return f"<解析失败: {out.stdout[:150]} {out.stderr[:150]}>"


def show(title, prompt, answer, verdict, **kw):
    print(f"[{verdict}] {title}")
    print(f"    prompt: {prompt!r}")
    print(f"    output: {answer!r}\n")


print("=== A. 单约束：只含一个词 ===")
p = "写一句话，必须包含『杭州』这个词。"
a = chat(p, max_tokens=64)
show("单约束-含词", p, a, "PASS" if "杭州" in a else "FAIL")

print("=== B. 双约束：含两个词 ===")
p = "写一句话，必须同时包含『杭州』和『西湖』这两个词。"
a = chat(p, max_tokens=64)
show("双约束-两个词", p, a, "PASS" if ("杭州" in a and "西湖" in a) else "FAIL")

print("=== C. 长度约束 ===")
p = "用不超过 10 个字回答：中国的首都是哪里？"
a = chat(p, max_tokens=64)
show("长度约束-10字", p, a, "PASS" if len(a) <= 14 else "FAIL")

print("=== D. 分类标签（Agent 常用） ===")
p = "判断情感倾向，只输出『正面』或『负面』，不要其他文字：这家餐厅服务很好，但菜太难吃了。"
a = chat(p, max_tokens=32)
ok = a.strip() in ("正面", "负面") or (len(a) <= 6 and ("正面" in a or "负面" in a))
show("分类-只输出标签", p, a, "PASS" if ok else "FAIL")

print("=== E. 禁止 Markdown ===")
p = "输出纯文本，不要使用任何 Markdown 符号（不要星号、井号、反引号）。介绍杭州。"
a = chat(p, max_tokens=100)
bad = any(ch in a for ch in "*#`_")
show("禁用 Markdown", p, a, "PASS" if not bad else "FAIL")

print("=== F. 多轮上下文保持 ===")
cmd = [str(BINARY), "chat", "--model", str(MODEL), "--threads", "8", "--ctx", "2048",
       "--max-tokens", "48", "--temperature", "0", "--top-k", "0", "--json"]
out = subprocess.run(cmd, input="我叫小明。\n我今年几岁？\n",
                     capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=HERE)
print("[多轮] 输入两轮：'我叫小明。' 然后 '我今年几岁？'")
print(f"    引擎输出(截断): {out.stdout[:300]!r}")
print(f"    stderr: {out.stderr[:200]!r}\n")

print("=== G. 重复触发场景（多约束失败时的退化模式） ===")
p = "按以下要求写一句话：(1) 必须包含『杭州』；(2) 必须包含『西湖』。"
a = chat(p, max_tokens=100)
show("双约束+编号格式", p, a, "PASS" if ("杭州" in a and "西湖" in a) else "FAIL")
