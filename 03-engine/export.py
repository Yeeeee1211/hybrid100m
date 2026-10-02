"""Offline export: training model.pt + config.json + tokenizer.json -> one .h100.

Python/PyTorch/tokenizers are required ONLY for this conversion/validation step.
The C++ executable has no dependency on them. See README.md for the v1 format.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import struct
import sys
import tempfile

import torch
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import ByteLevel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hybrid100m import HybridConfig, HybridForCausalLM
from hybrid100m.chat import ChatTokens, SPECIAL_TOKENS


def byte_map():
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs.copy()
    extra = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + extra)
            extra += 1
    return dict(zip(bs, map(chr, cs)))


def unicode_ranges():
    """Export the installed tokenizer's actual regex categories, not Python's UCD.

    Probe sentinels distinguish L/N/whitespace/other for every Unicode scalar.
    That keeps C++ regex-free and avoids Unicode-version differences with Rust.
    This takes several seconds once per export, never at inference time.
    """
    pre = ByteLevel(add_prefix_space=False)
    ranges, start, previous = [], 0, 0
    for cp in range(0x110000):
        kind = 0
        if cp == 32:
            # Literal ASCII space can attach to the following L/N regex branch.
            kind = 3
        elif not 0xD800 <= cp <= 0xDFFF:
            ch = chr(cp)
            parts = pre.pre_tokenize_str('a' + ch + '1')
            if parts[0][1] == (0, 2):
                kind = 1
            elif parts[-1][1] == (1, 3):
                kind = 2
            elif len(pre.pre_tokenize_str('!' + ch + '!')) > 1:
                kind = 3
        if kind != previous:
            if previous:
                ranges.append((start, cp - 1, previous))
            start, previous = cp, kind
    if previous:
        ranges.append((start, 0x10FFFF, previous))
    return ranges


def tokenizer_data(path, vocab_size):
    tokenizer = Tokenizer.from_file(str(path))
    data = json.loads(tokenizer.to_str())
    model, pre = data['model'], data['pre_tokenizer']
    if (model['type'] != 'BPE' or data['normalizer'] is not None
            or pre is None or pre.get('type') != 'ByteLevel'
            or pre.get('add_prefix_space') or not pre.get('use_regex', True)
            or data['decoder'] is None or data['decoder'].get('type') != 'ByteLevel'
            or data.get('truncation') is not None or data.get('padding') is not None):
        raise ValueError('Export requires the ByteLevel BPE tokenizer from train_tokenizer.py (no normalization/padding/truncation)')
    if (model.get('dropout') or model.get('unk_token') is not None
            or model.get('continuing_subword_prefix') or model.get('end_of_word_suffix')
            or model.get('byte_fallback') or model.get('ignore_merges')):
        raise ValueError('Unsupported BPE option; use the training tokenizer without modifying its IDs')
    for item in data['added_tokens']:
        if (item['content'] not in SPECIAL_TOKENS or not item['special']
                or item['single_word'] or item['lstrip'] or item['rstrip'] or item['normalized']):
            raise ValueError('Only the eleven reserved, exact-match special tokens are supported')
    if {i['content'] for i in data['added_tokens']} != set(SPECIAL_TOKENS):
        raise ValueError('All eleven chat markers must be registered as special tokens')
    vocab = model['vocab']
    size = tokenizer.get_vocab_size()
    if size > vocab_size or set(vocab.values()) != set(range(size)):
        raise ValueError('Tokenizer IDs must be contiguous and fit within the model vocabulary')
    mapping = byte_map()
    inverse = {c: b for b, c in mapping.items()}
    tokens = [b''] * size
    for text, index in vocab.items():
        tokens[index] = text.encode('utf-8') if text in SPECIAL_TOKENS else bytes(inverse[c] for c in text)
    base = [vocab.get(mapping[i], -1) for i in range(256)]
    missing = base.count(-1)
    if missing:
        print(f'Warning: tokenizer lacks {missing} byte symbols. Native encode rejects input requiring a missing byte; do not change a trained vocabulary.', file=sys.stderr)
    merges = []
    for pair in model['merges']:
        a, b = pair.split(' ') if isinstance(pair, str) else pair
        merges.append((vocab[a], vocab[b], vocab[a + b]))
    return tokens, ChatTokens.from_tokenizer(tokenizer).all_ids, base, merges


def export_model(run_dir, tokenizer_path, output, quant='q8', ranges=None):
    run_dir, output = Path(run_dir), Path(output)
    config = HybridConfig.load(run_dir / 'config.json')
    if quant not in ('f32', 'q8'):
        raise ValueError('quant must be f32 or q8')
    # model.pt is the tensor-only file emitted by train.py; never unpickle ckpt.pt.
    state = torch.load(run_dir / 'model.pt', map_location='cpu', weights_only=True)
    with torch.device('meta'):
        expected = HybridForCausalLM(config).state_dict()
    if set(state) != set(expected):
        raise ValueError('Checkpoint tensor names do not match the configured architecture')
    for name, tensor in state.items():
        if tensor.shape != expected[name].shape or not tensor.is_floating_point() or not bool(tensor.isfinite().all()):
            raise ValueError(f'Invalid shape/dtype/nonfinite values: {name}')
    if config.tie_word_embeddings:
        if not torch.equal(state['embedding.weight'], state['lm_head.weight']):
            raise ValueError('Tied embedding and LM head have different weights')
        del state['lm_head.weight']
    # MTP supervises additional future tokens during training. Native autoregressive
    # inference uses only the main branch, with unchanged tokenizer/head weights.
    omitted = [name for name in state if name.startswith('mtp.')]
    for name in omitted:
        del state[name]
    tokens, special, base, merges = tokenizer_data(tokenizer_path, config.vocab_size)
    ranges = unicode_ranges() if ranges is None else ranges
    metadata = dict(architecture='hybrid100m-v1', config=asdict(config), quantization=quant,
                    tokenizer='bytelevel-bpe-v1', tensor_count=len(state),
                    inference_branch='main', omitted_training_tensors=len(omitted))
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + '.', suffix='.tmp', dir=output.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            def u32(x): f.write(struct.pack('<I', x))
            def blob(x): u32(len(x)); f.write(x)
            f.write(b'HYB100M\0')
            u32(1)
            blob(json.dumps(metadata, ensure_ascii=True, separators=(',', ':')).encode())
            u32(len(tokens))
            for text in tokens: blob(text)
            f.write(struct.pack('<11I', *special))
            f.write(struct.pack('<256i', *base))
            u32(len(ranges))
            for entry in ranges: f.write(struct.pack('<IIB', *entry))
            u32(len(merges))
            for entry in merges: f.write(struct.pack('<III', *entry))
            u32(len(state))
            for name, tensor in state.items():
                t = tensor.detach().float().contiguous()
                is_q8 = quant == 'q8' and t.ndim == 2
                blob(name.encode('ascii')); u32(int(is_q8)); u32(t.ndim)
                for dim in t.shape: u32(dim)
                if is_q8:
                    scale = t.abs().amax(1) / 127
                    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                    q = (t / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
                    payload = scale.numpy().astype('<f4').tobytes() + q.numpy().tobytes()
                else:
                    payload = t.numpy().astype('<f4').tobytes()
                f.write(struct.pack('<Q', len(payload))); f.write(payload)
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return output.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--quant', choices=('f32', 'q8'), default='q8')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1: parser.error('--threads must be positive')
    torch.set_num_threads(args.threads)
    size = export_model(args.run_dir, args.tokenizer, args.output, args.quant)
    print(f'Exported {args.output}: {size / 2**20:.2f} MiB ({args.quant})')


if __name__ == '__main__':
    main()
