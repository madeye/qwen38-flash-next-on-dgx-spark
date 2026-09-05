"""Validate launch arguments without a GPU, real weights, or Docker mutations."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServeConfigTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        cache = Path(self.temp.name)
        snapshot = cache / 'hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/test'
        snapshot.mkdir(parents=True)
        hybrid = snapshot.with_name('test-fp8hybrid')
        hybrid.mkdir()
        (hybrid / '.prepared').touch()
        self.env = {'PATH': os.environ['PATH'], 'HOME': str(cache),
                    'HF_CACHE': str(cache), 'DRY_RUN': '1'}

    def launch(self, script='serve.sh', **overrides):
        result = subprocess.run(['bash', str(ROOT / 'scripts' / script)],
                                env={**self.env, **overrides},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return shlex.split(result.stdout.splitlines()[0])

    def test_base_profile(self):
        args = self.launch()
        for flag, value in {'--max-model-len': '262144', '--max-num-seqs': '4',
                            '--max-num-batched-tokens': '2048',
                            '--kv-cache-dtype': 'fp8_e4m3',
                            '--kv-cache-memory': '9663676416'}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('-cc.cudagraph_capture_sizes=[4,8,12,16]', args)
        self.assertIn('-cc.cudagraph_mode=PIECEWISE', args)
        self.assertTrue(any('vllm::ple_mmap_lookup' in arg for arg in args))
        self.assertEqual(json.loads(args[args.index('--speculative-config') + 1])
                         ['num_speculative_tokens'], 3)
        self.assertNotIn('--hf-overrides', args)

    def test_long_context_profiles(self):
        for script, dtype, pool in [('serve-500k.sh', 'fp8_e4m3', '9663676416'),
                                    ('serve-500k-fp8.sh', 'fp8_e4m3', '9663676416'),
                                    ('serve-500k-bf16.sh', 'auto', '17179869184')]:
            with self.subTest(script=script):
                args = self.launch(script)
                self.assertEqual(args[args.index('--max-model-len') + 1], '524288')
                self.assertEqual(args[args.index('--kv-cache-dtype') + 1], dtype)
                self.assertEqual(args[args.index('--kv-cache-memory') + 1], pool)
                self.assertIn('-cc.cudagraph_capture_sizes=[4]', args)
                spec = json.loads(args[args.index('--speculative-config') + 1])
                self.assertEqual(spec['max_model_len'], 524288)

    def test_overrides_and_no_mtp(self):
        args = self.launch(MTP='0', SEQS='3', MAX_NUM_BATCHED_TOKENS='8192')
        self.assertNotIn('--speculative-config', args)
        self.assertIn('-cc.cudagraph_capture_sizes=[1,2,3]', args)
        self.assertEqual(args[args.index('--max-num-batched-tokens') + 1], '8192')
        args = self.launch(CUDAGRAPH_CAPTURE_SIZES='16,4,12,4')
        self.assertIn('-cc.cudagraph_capture_sizes=[4,12,16]', args)
        for script in ['serve.sh', 'serve-500k.sh', 'serve-500k-fp8.sh', 'serve-500k-bf16.sh']:
            args = self.launch(script, CUDAGRAPH_CAPTURE_SIZES='', KV_BYTES='')
            self.assertFalse(any('cudagraph_capture_sizes=' in arg for arg in args))
            self.assertNotIn('--kv-cache-memory', args)

    def test_invalid_settings(self):
        for overrides in [{'SEQS': '0'}, {'MTP': '-1'}, {'MTP': '03'},
                          {'MAX_NUM_BATCHED_TOKENS': 'bad'},
                          {'CUDAGRAPH_CAPTURE_SIZES': '4,0'},
                          {'CUDAGRAPH_CAPTURE_SIZES': '4,broken'},
                          {'CTX': '524288', 'YARN': '0'}]:
            with self.subTest(overrides=overrides):
                result = subprocess.run(['bash', str(ROOT / 'scripts/serve.sh')],
                                        env={**self.env, **overrides},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('run -d', result.stdout)


if __name__ == '__main__':
    unittest.main()
