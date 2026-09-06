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
        model = cache / 'official-nvidia-checkpoint'
        model.mkdir()
        (model / 'config.json').write_text('{}')
        snapshot = cache / 'hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4/snapshots/test'
        snapshot.mkdir(parents=True)
        hybrid = snapshot.with_name('test-fp8hybrid')
        hybrid.mkdir()
        (hybrid / '.prepared').touch()
        self.env = {'PATH': os.environ['PATH'], 'HOME': str(cache),
                    'HF_CACHE': str(cache), 'MODEL_HOST': str(model), 'DRY_RUN': '1'}

    def launch(self, script='serve.sh', **overrides):
        result = subprocess.run(['bash', str(ROOT / 'scripts' / script)],
                                env={**self.env, **overrides},
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        return shlex.split(result.stdout.splitlines()[0])

    def test_base_profile(self):
        args = self.launch()
        for flag, value in {'--max-model-len': '262144', '--max-num-seqs': '6',
                            '--max-num-batched-tokens': '4096',
                            '--kv-cache-dtype': 'fp8_e4m3',
                            '--gpu-memory-utilization': '0.80'}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn('--network', args)
        self.assertEqual(args[args.index('--network') + 1], 'host')
        self.assertIn('--no-enable-prefix-caching', args)
        graph = json.loads(args[args.index('--compilation-config') + 1])
        self.assertEqual(graph['cudagraph_mode'], 'FULL_DECODE_ONLY')
        self.assertEqual(graph['cudagraph_capture_sizes'], [4, 8, 12, 16, 20, 24])
        self.assertTrue(any('full-recipe-patch/ple_layer.py' in arg for arg in args))
        self.assertEqual(json.loads(args[args.index('--speculative-config') + 1])
                         ['num_speculative_tokens'], 3)
        self.assertEqual(args[args.index('--served-model-name') + 1], 'qwen3.8-flash-next')

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
        args = self.launch(MTP='4', SEQS='3', CHUNK='8192', CAPTURE_SIZES='16,4,12,4', PREFIX_CACHE='1')
        self.assertEqual(args[args.index('--max-num-batched-tokens') + 1], '8192')
        self.assertIn('--enable-prefix-caching', args)
        self.assertEqual(json.loads(args[args.index('--compilation-config') + 1])
                         ['cudagraph_capture_sizes'], [16, 4, 12, 4])
        args = self.launch(CAPTURE_SIZES='')
        self.assertNotIn('cudagraph_capture_sizes', args[args.index('--compilation-config') + 1])

    def test_invalid_settings(self):
        for overrides in [{'SEQS': '0'}, {'MTP': '0'}, {'MTP': '03'},
                          {'CHUNK': 'bad'}, {'CAPTURE_SIZES': '4,0'},
                          {'CAPTURE_SIZES': '4,broken'}, {'PREFIX_CACHE': 'yes'}]:
            with self.subTest(overrides=overrides):
                result = subprocess.run(['bash', str(ROOT / 'scripts/serve.sh')],
                                        env={**self.env, **overrides},
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn('run -d', result.stdout)


if __name__ == '__main__':
    unittest.main()
