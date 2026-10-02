"""Native integration tests against the actual PyTorch model and HF tokenizer.

Build hybrid-cpu first, then run: python -m unittest discover -s cpu -p test_runtime.py -v
HYBRID_CPU_BINARY may select another executable. HTTP tests bind loopback only.
"""
from dataclasses import replace
import http.client
import json
import os
from pathlib import Path
import random
import socket
import struct
import subprocess
import sys
import tempfile
import unittest

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cpu'))
from export import export_model, unicode_ranges
from hybrid100m import HybridConfig, HybridForCausalLM
from hybrid100m.chat import ChatTemplate, SPECIAL_TOKENS, generate_chat

BINARY = Path(os.environ.get('HYBRID_CPU_BINARY', ROOT / 'cpu/build' / ('hybrid-cpu.exe' if os.name == 'nt' else 'hybrid-cpu')))


@unittest.skipUnless(BINARY.is_file(), 'Build the native executable first')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(2026)
        cls.temporary = tempfile.TemporaryDirectory(prefix='native-tests-', dir=ROOT / 'cpu/build')
        cls.directory = Path(cls.temporary.name)
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        corpus = ["你好，我是中文模型。测试 2+3=5! 😀🚀", "Hello I'm testing: we're they've you'll don't!",
                  "  spaces\n\n\t world  \n", "aaaaabaaaa 中文中文 éé １２３ ²Ⅲ —", "system\nuser\nassistant\n"]
        tokenizer.train_from_iterator(corpus * 20, trainers.BpeTrainer(vocab_size=400, min_frequency=1,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=list(SPECIAL_TOKENS), show_progress=False))
        cls.tokenizer = tokenizer
        cls.tokenizer_path = cls.directory / 'tokenizer.json'
        tokenizer.save(str(cls.tokenizer_path))
        cls.config = HybridConfig(vocab_size=tokenizer.get_vocab_size(), hidden_size=32, intermediate_size=64,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
            linear_num_heads=2, linear_head_dim=8, linear_backend='reference', sdpa_backend='math',
            max_position_embeddings=256, gradient_checkpointing=False,
            linear_attention='gated_delta', rope_type='default', mtp_num_layers=0)
        cls.model = HybridForCausalLM(cls.config).eval()
        cls.config.save(cls.directory / 'config.json')
        torch.save(cls.model.state_dict(), cls.directory / 'model.pt')
        cls.ranges = unicode_ranges()
        cls.paths = {}
        for quant in ('f32', 'q8'):
            path = cls.directory / f'{quant}.h100'
            export_model(cls.directory, cls.tokenizer_path, path, quant, ranges=cls.ranges)
            cls.paths[quant] = path
        # A deterministic endless model exercises forced think transitions/stops.
        zero = HybridForCausalLM(cls.config).eval()
        with torch.no_grad():
            for parameter in zero.parameters(): parameter.zero_()
            zero.embedding.weight.fill_(1)
            zero.embedding.weight[11].fill_(2)
            zero.final_norm.weight.fill_(1)
        cls.zero = zero
        torch.save(zero.state_dict(), cls.directory / 'model.pt')
        cls.paths['zero'] = cls.directory / 'zero.h100'
        export_model(cls.directory, cls.tokenizer_path, cls.paths['zero'], 'f32', ranges=cls.ranges)
        torch.save(cls.model.state_dict(), cls.directory / 'model.pt')

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def run_native(self, command, *args, model='f32', threads=1, check=True):
        return subprocess.run([str(BINARY), command, '--model', str(self.paths.get(model, model)),
                               '--threads', str(threads), *map(str, args)],
                              capture_output=True, text=True, encoding='utf-8', timeout=60, check=check)

    def test_fp32_logits_match_pytorch_all_positions_and_threads(self):
        ids = torch.tensor([[1, 32, 76, 15, 229, 48, 111, 78, 3, 81, 64]])
        with torch.no_grad(): expected = self.model(ids).logits[0]
        for threads in (1, 3):
            result = self.run_native('logits', '--ids', ','.join(map(str, ids[0].tolist())), threads=threads)
            actual = torch.tensor(json.loads(result.stdout))
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=6e-5)

    def test_q8_matches_dequantized_pytorch_and_reduces_file(self):
        model = HybridForCausalLM(self.config).eval()
        state = self.model.state_dict()
        for key, value in state.items():
            if value.ndim == 2:
                scale = value.abs().amax(1) / 127
                scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                state[key] = (value / scale[:, None]).round().clamp(-127, 127) * scale[:, None]
        model.load_state_dict(state)
        ids = torch.tensor([[1, 66, 87, 117, 43, 75, 16]])
        with torch.no_grad(): expected = model(ids).logits[0]
        actual = torch.tensor(json.loads(self.run_native('logits', '--ids', ','.join(map(str, ids[0].tolist())), model='q8', threads=2).stdout))
        torch.testing.assert_close(actual, expected, atol=6e-6, rtol=8e-5)
        self.assertLess(self.paths['q8'].stat().st_size, self.paths['f32'].stat().st_size * 0.45)

    def test_kda_yarn_fp32_q8_cache_and_training_only_mtp_export(self):
        cfg = replace(self.config, linear_attention='kda', linear_gate_rank=8,
                      rope_type='yarn', rope_factor=8.0,
                      rope_original_max_position_embeddings=32, mtp_num_layers=1)
        model = HybridForCausalLM(cfg).eval()
        # Exercise nonuniform per-channel decay as well as positions beyond the
        # original RoPE window, rather than comparing only tiny initial outputs.
        with torch.no_grad():
            for layer in model.layers:
                if hasattr(layer.mixer, 'dt_bias'):
                    layer.mixer.dt_bias.copy_(torch.linspace(-2.5, 1.0, layer.mixer.dt_bias.numel()))
        state = model.state_dict()
        ids = torch.tensor([[1] + [(i * 19 + 31) % cfg.vocab_size for i in range(82)]])
        try:
            cfg.save(self.directory / 'config.json')
            torch.save(state, self.directory / 'model.pt')
            for quant in ('f32', 'q8'):
                path = self.directory / f'kda-{quant}.h100'
                export_model(self.directory, self.tokenizer_path, path, quant, ranges=self.ranges)
                oracle = HybridForCausalLM(cfg).eval()
                dequantized = dict(state)
                if quant == 'q8':
                    for name, value in state.items():
                        if value.ndim == 2 and not name.startswith('mtp.'):
                            scale = value.abs().amax(1) / 127
                            scale = torch.where(scale == 0, torch.ones_like(scale), scale)
                            dequantized[name] = (value / scale[:, None]).round().clamp(-127, 127) * scale[:, None]
                oracle.load_state_dict(dequantized)
                with torch.no_grad():
                    expected = oracle(ids).logits[0]
                actual = torch.tensor(json.loads(self.run_native('logits', '--ids', ','.join(map(str, ids[0].tolist())), model=path, threads=3).stdout))
                torch.testing.assert_close(actual, expected, atol=8e-6, rtol=1e-4)
                info = json.loads(self.run_native('info', model=path).stdout)
                self.assertEqual(info['metadata']['inference_branch'], 'main')
                self.assertGreater(info['metadata']['omitted_training_tensors'], 0)
        finally:
            self.config.save(self.directory / 'config.json')
            torch.save(self.model.state_dict(), self.directory / 'model.pt')

    def test_legacy_v1_metadata_without_new_fields_stays_gated_delta(self):
        payload = self.paths['f32'].read_bytes()
        metadata_length = struct.unpack_from('<I', payload, 12)[0]
        metadata = json.loads(payload[16:16 + metadata_length])
        for name in ('linear_attention', 'linear_gate_rank', 'rope_type', 'rope_factor',
                     'rope_original_max_position_embeddings', 'rope_beta_fast', 'rope_beta_slow',
                     'mtp_num_layers', 'mtp_loss_weight', 'loss_backend'):
            metadata['config'].pop(name, None)
        metadata.pop('inference_branch', None)
        metadata.pop('omitted_training_tensors', None)
        encoded = json.dumps(metadata, separators=(',', ':')).encode()
        path = self.directory / 'legacy-v1.h100'
        path.write_bytes(payload[:12] + struct.pack('<I', len(encoded)) + encoded + payload[16 + metadata_length:])
        ids = torch.tensor([[1, 32, 76, 15, 229]])
        with torch.no_grad():
            expected = self.model(ids).logits[0]
        actual = torch.tensor(json.loads(self.run_native('logits', '--ids', '1,32,76,15,229', model=path).stdout))
        torch.testing.assert_close(actual, expected, atol=4e-6, rtol=6e-5)

    def test_tokenizer_matches_chinese_unicode_spaces_contractions_and_markers(self):
        probes = ["你好，我是中文模型。 😀🚀", "I'm we're they've you'll don't I'M", "\t  Hello  \n\n world \r\n",
                  "aaaaabaaaa", "éé １２３ ²Ⅲ 中文", "<|im_start|>user\n你好<|im_end|>", "a\u2003\u2028 z", ""]
        rng = random.Random(7)
        for _ in range(12):
            points = [rng.randrange(0x110000) for _ in range(24)]
            probes.append(''.join('a' + chr(c) + '1 !' for c in points if not 0xD800 <= c <= 0xDFFF))
        for text in probes:
            with self.subTest(text=repr(text)):
                result = json.loads(self.run_native('tokenize', '--prompt', text).stdout)
                expected = self.tokenizer.encode(text, add_special_tokens=False).ids
                self.assertEqual(result['ids'], expected)
                self.assertEqual(result['text'], self.tokenizer.decode(expected, skip_special_tokens=False))

    def test_chat_template_and_greedy_protocol_match_python(self):
        template = ChatTemplate(self.tokenizer)
        for thinking in (False, True):
            messages = [{'role': 'system', 'content': '你好'}, {'role': 'user', 'content': '2+3?'}]
            expected = generate_chat(self.model, template, messages, enable_thinking=thinking,
                                     max_new_tokens=9, max_think_tokens=2, temperature=0)
            args = ['--prompt', '2+3?', '--system', '你好', '--json', '--temperature', '0', '--max-tokens', '9', '--max-think-tokens', '2']
            if thinking: args.append('--thinking')
            actual = json.loads(self.run_native('chat', *args).stdout)
            self.assertEqual(actual['generated_ids'], expected.generated_ids)
            self.assertEqual(actual['answer'], expected.answer)
            self.assertEqual(actual['reasoning'], expected.reasoning)
            self.assertEqual(actual['prompt_tokens'], len(template.encode_prompt(messages, enable_thinking=thinking)))

    def test_forced_think_budget_streaming_and_stop(self):
        args = ['--prompt', 'hello', '--temperature', '0', '--max-tokens', '8', '--thinking', '--max-think-tokens', '2']
        result = json.loads(self.run_native('chat', *args, '--json', model='zero').stdout)
        self.assertTrue(result['think_truncated'])
        expected = generate_chat(self.zero, ChatTemplate(self.tokenizer), [{'role': 'user', 'content': 'hello'}],
                                 enable_thinking=True, max_new_tokens=8, max_think_tokens=2, temperature=0)
        self.assertEqual(result['generated_ids'], expected.generated_ids)
        stream = self.run_native('chat', *args, model='zero').stdout
        self.assertEqual(stream, '<think>\n' + result['reasoning'] + '</think>\n' + result['answer'] + '\n')
        token_text = self.tokenizer.decode([result['generated_ids'][0]], skip_special_tokens=False)
        stopped = json.loads(self.run_native('chat', '--prompt', 'hello', '--json', '--temperature', '0', '--max-tokens', '8', '--stop', token_text * 2, model='zero').stdout)
        self.assertEqual(stopped['answer'], '')
        self.assertEqual(stopped['finish_reason'], 'stop')
        self.assertEqual(len(stopped['generated_ids']), 2)

    def test_context_limits_and_bad_options_are_errors(self):
        for extra in (['--ctx', '257'], ['--ctx', '1'], ['--temperature', '-1'], ['--top-p', '0'], ['--threads', '-1']):
            # --threads is already supplied; duplicate-option rejection is expected.
            result = self.run_native('chat', '--prompt', 'hello', *extra, check=False)
            self.assertNotEqual(result.returncode, 0)
        prompt = ChatTemplate(self.tokenizer).encode_prompt([{'role': 'user', 'content': 'hello'}])
        result = json.loads(self.run_native('chat', '--prompt', 'hello', '--json', '--max-tokens', '20', '--ctx', str(len(prompt) + 2)).stdout)
        self.assertEqual(len(result['generated_ids']), 2)

    def test_seed_and_raw_completion(self):
        args = ('--prompt', '你好', '--json', '--max-tokens', '5', '--seed', '123')
        a = json.loads(self.run_native('complete', *args).stdout)
        b = json.loads(self.run_native('complete', *args).stdout)
        self.assertEqual(a['generated_ids'], b['generated_ids'])
        self.assertEqual(a['prompt_tokens'], 1 + len(self.tokenizer.encode('你好').ids))

    def test_corrupt_models_are_rejected(self):
        payload = self.paths['f32'].read_bytes()
        for data in (b'bad magic', payload[:-10], payload + b'x'):
            path = self.directory / 'bad.h100'; path.write_bytes(data)
            result = self.run_native('info', model=path, check=False)
            self.assertNotEqual(result.returncode, 0)

    def test_benchmark_and_cache_allocation(self):
        info = json.loads(self.run_native('info', '--ctx', '64').stdout)
        cfg = self.config
        expected = (3 * (3 * cfg.linear_num_heads * cfg.linear_head_dim * cfg.conv_kernel_size
                        + cfg.linear_num_heads * cfg.linear_head_dim**2)
                    + 2 * 64 * cfg.num_key_value_heads * cfg.head_dim) * 4
        self.assertEqual(info['cache_bytes'], expected)
        bench = json.loads(self.run_native('bench', '--prompt-tokens', '3', '--decode-tokens', '2', '--repeats', '2', threads=2).stdout)
        self.assertEqual(len(bench['runs']), 2)
        self.assertGreater(bench['runs'][0]['decode_tokens_per_second'], 0)

    def test_10k_context_and_alternative_weight_shapes(self):
        # A small model keeps this boundary test affordable while exercising
        # actual 10K RoPE positions and cache writes in the same native kernels.
        long_config = replace(self.config, max_position_embeddings=10240,
                              conv_kernel_size=1, rotary_fraction=0.5, tie_word_embeddings=False)
        model = HybridForCausalLM(long_config).eval()
        path = self.directory / 'long.h100'
        try:
            long_config.save(self.directory / 'config.json')
            torch.save(model.state_dict(), self.directory / 'model.pt')
            export_model(self.directory, self.tokenizer_path, path, 'f32', ranges=self.ranges)
            ids = torch.tensor([[1, 32, 78, 56, 99]])
            with torch.no_grad(): expected = model(ids).logits[0]
            actual = torch.tensor(json.loads(self.run_native('logits', '--ids', '1,32,78,56,99', model=path).stdout))
            torch.testing.assert_close(actual, expected, atol=4e-6, rtol=8e-5)
            bench = json.loads(self.run_native('bench', '--prompt-tokens', '10239', '--decode-tokens', '1', '--repeats', '1', model=path).stdout)
            self.assertEqual(bench['context'], 10240)
            failed = self.run_native('bench', '--prompt-tokens', '10240', '--decode-tokens', '1', model=path, check=False)
            self.assertNotEqual(failed.returncode, 0)
        finally:
            self.config.save(self.directory / 'config.json')
            torch.save(self.model.state_dict(), self.directory / 'model.pt')

    @unittest.skipIf(os.environ.get('HYBRID_SKIP_HTTP') == '1', 'Loopback access disabled explicitly')
    def test_http_json_sse_reset_and_validation(self):
        with socket.socket() as s:
            s.bind(('127.0.0.1', 0)); port = s.getsockname()[1]
        process = subprocess.Popen([str(BINARY), 'serve', '--model', str(self.paths['zero']), '--threads', '2', '--port', str(port)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        try:
            ready = process.stderr.readline()
            self.assertIn('HTTP listening', ready)
            def request(method, path, body=None):
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=10)
                try:
                    connection.request(method, path, body=None if body is None else json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    response = connection.getresponse()
                    return response.status, response.read().decode('utf-8')
                finally: connection.close()
            self.assertEqual(request('GET', '/health')[0], 200)
            self.assertEqual(json.loads(request('GET', '/v1/models')[1])['data'][0]['id'], 'hybrid100m')
            body = dict(model='hybrid100m', messages=[dict(role='user', content='你好')], max_tokens=8,
                        temperature=0, enable_thinking=True, max_think_tokens=2)
            status, data = request('POST', '/v1/chat/completions', body)
            self.assertEqual(status, 200); normal = json.loads(data)
            # The same server instance resets both recurrence and KV state per request.
            self.assertEqual(normal['choices'], json.loads(request('POST', '/v1/chat/completions', body)[1])['choices'])
            body.update(stream=True, stream_options={'include_usage': True})
            status, data = request('POST', '/v1/chat/completions', body)
            self.assertEqual(status, 200)
            self.assertTrue(data.endswith('data: [DONE]\n\n'))
            chunks = [json.loads(line[6:]) for line in data.splitlines() if line.startswith('data: ') and line != 'data: [DONE]']
            answer = ''.join(c['choices'][0]['delta'].get('content', '') for c in chunks if c['choices'])
            reasoning = ''.join(c['choices'][0]['delta'].get('reasoning_content', '') for c in chunks if c['choices'])
            self.assertEqual(answer, normal['choices'][0]['message']['content'])
            self.assertEqual(reasoning, normal['choices'][0]['message']['reasoning_content'])
            self.assertEqual(chunks[-1]['usage'], normal['usage'])
            bad = dict(body, tools=[])
            self.assertEqual(request('POST', '/v1/chat/completions', bad)[0], 400)
            self.assertEqual(request('POST', '/v1/chat/completions', dict(body, top_p=0))[0], 400)
            self.assertEqual(request('GET', '/missing')[0], 404)
        finally:
            process.terminate()
            process.communicate(timeout=10)


if __name__ == '__main__':
    unittest.main()
