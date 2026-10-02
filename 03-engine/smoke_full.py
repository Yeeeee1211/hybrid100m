"""Optional default-model random-weight correctness/memory/short CPU benchmark.

This is not a model-quality evaluation. Temporary models are deleted on exit.
Run after compiling: python cpu/smoke_full.py --binary cpu/build/hybrid-cpu.exe
"""
import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid100m import HybridConfig, HybridForCausalLM
from hybrid100m.chat import SPECIAL_TOKENS
from export import export_model, unicode_ranges


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--report', default='cpu/build/full-smoke.json')
    args = parser.parse_args()
    binary = Path(args.binary).resolve()
    torch.set_num_threads(4)
    torch.manual_seed(17)
    config = HybridConfig()
    model = HybridForCausalLM(replace(config, linear_backend='reference', sdpa_backend='math')).eval()
    report = dict(note='Random default-model weights; short-context correctness/throughput only, not trained capability or 10K speed.',
                  platform=platform.platform(), cpu=platform.processor(), torch=torch.__version__,
                  binary=binary.name, binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),
                  parameters=model.num_parameters(),
                  inference_parameters=sum(parameter.numel() for name, parameter in model.named_parameters()
                                           if not name.startswith('mtp.')), runs={})
    if sys.platform == 'win32':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            report['cpu'] = winreg.QueryValueEx(key, 'ProcessorNameString')[0]
    with tempfile.TemporaryDirectory(prefix='full-smoke-', dir=binary.parent) as directory:
        directory = Path(directory)
        config.save(directory / 'config.json')
        torch.save(model.state_dict(), directory / 'model.pt')
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.train_from_iterator(['中文测试 hello system user assistant'] * 10,
            trainers.BpeTrainer(vocab_size=320, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                special_tokens=list(SPECIAL_TOKENS), show_progress=False))
        tokenizer_path = directory / 'tokenizer.json'
        tokenizer.save(str(tokenizer_path))
        ranges = unicode_ranges()
        ids = torch.tensor([[1, 34, 72, 219, 86]])
        with torch.no_grad(): original = model(ids).logits[0].clone()
        for quant in ('f32', 'q8'):
            path = directory / f'model-{quant}.h100'
            size = export_model(directory, tokenizer_path, path, quant, ranges=ranges)
            if quant == 'q8':
                with torch.no_grad():
                    for parameter in model.parameters():
                        if parameter.ndim == 2:
                            scale = parameter.abs().amax(1) / 127
                            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                            parameter.copy_((parameter / scale[:, None]).round().clamp(-127, 127) * scale[:, None])
            with torch.no_grad(): expected = model(ids).logits[0]
            def run(command, *options):
                output = subprocess.check_output([str(binary), command, '--model', str(path), *map(str, options)], timeout=120)
                return json.loads(output)
            actual = torch.tensor(run('logits', '--ids', '1,34,72,219,86', '--threads', 4))
            torch.testing.assert_close(actual, expected, atol=8e-5, rtol=3e-4)
            entry = dict(file_bytes=size, native_vs_oracle_max_abs=float((actual - expected).abs().max()),
                         quant_vs_original_rms=float((expected - original).square().mean().sqrt()),
                         info=run('info', '--threads', 4), benchmarks={})
            for threads in (1, 4):
                entry['benchmarks'][threads] = run('bench', '--threads', threads, '--prompt-tokens', 8, '--decode-tokens', 8, '--repeats', 2)
            report['runs'][quant] = entry
            print(f'{quant}: {size / 2**20:.2f} MiB, native error {entry["native_vs_oracle_max_abs"]:.3g}', flush=True)
        Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Report: {args.report}')


if __name__ == '__main__':
    main()
