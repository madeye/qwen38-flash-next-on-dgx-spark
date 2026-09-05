#!/usr/bin/env python3
"""Exercise an exact token count against the local API; save timings and usage.

Synthetic capacity test, not a long-context quality benchmark. No dependencies.
"""
import argparse
import json
import threading
import time
import urllib.request
import uuid
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--base', default='http://127.0.0.1:18300')
parser.add_argument('--tokens', type=int, default=500000)
parser.add_argument('--output', default='/tmp/flash-context-validation.json')
args = parser.parse_args()
model = 'qwen3.8-flash-next'


def post(path, payload):
    request = urllib.request.Request(args.base + path,
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    return urllib.request.urlopen(request, timeout=3600)


def tokenize(text):
    with post('/tokenize', {'model': model, 'prompt': text,
                           'add_special_tokens': False}) as response:
        return json.load(response)


metadata = json.load(urllib.request.urlopen(args.base + '/v1/models', timeout=10))
served = next(item for item in metadata['data'] if item['id'] == model)
assert served['max_model_len'] >= args.tokens + 64, served
prefix = tokenize(f'Record batch {uuid.uuid4()}. Read these archive entries.\n')['tokens']
filler = tokenize('The archive records routine deliveries of paper, ink, and envelopes.\n')['tokens']
suffix = tokenize('\nEnd of archive. Question: What is 17 times 23? Answer with the number only.\nAnswer:')['tokens']
remaining = args.tokens - len(prefix) - len(suffix)
assert remaining > 0
prompt = prefix + (filler * ((remaining // len(filler)) + 1))[:remaining] + suffix
assert len(prompt) == args.tokens
stop = threading.Event()
memory = []


def monitor():
    while not stop.is_set():
        fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        memory.append(int(fields['MemAvailable'].split()[0]) / 1024**2)
        stop.wait(1)


thread = threading.Thread(target=monitor, daemon=True)
thread.start()
start = time.monotonic()
first = None
text = ''
usage = None
try:
    with post('/v1/completions', {'model': model, 'prompt': prompt,
            'max_tokens': 64, 'temperature': 0, 'stream': True,
            'stream_options': {'include_usage': True}}) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            data = line[6:].strip()
            if data == b'[DONE]':
                break
            event = json.loads(data)
            if 'error' in event:
                raise RuntimeError(event['error'])
            for choice in event.get('choices', []):
                chunk = choice.get('text', '')
                if chunk and first is None:
                    first = time.monotonic()
                text += chunk
            usage = event.get('usage') or usage
finally:
    stop.set()
    thread.join()
elapsed = time.monotonic() - start
result = {'served_max_model_len': served['max_model_len'], 'usage': usage,
          'ttft_seconds': None if first is None else first - start,
          'elapsed_seconds': elapsed, 'text': text,
          'min_mem_available_gib': min(memory),
          'capacity_pass': bool(usage and usage['prompt_tokens'] == args.tokens and text),
          'arithmetic_pass': '391' in text}
Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
assert result['capacity_pass'], result
