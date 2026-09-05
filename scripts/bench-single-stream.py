#!/usr/bin/env python3
"""Three sequential, non-thinking 384-token samples; writes /tmp/flash-single-stream.json."""
import json, time, urllib.request
from pathlib import Path
prompts = ['Explain how database transactions prevent lost updates. Give a concrete example with two clients and discuss isolation levels. Write about 400 words.', 'Write a Python function that merges overlapping closed intervals. Include type hints, explain the algorithm, and show three examples.', 'Explain how TCP handles packet loss and congestion. Describe retransmission, acknowledgements, and congestion control with an example. Write about 400 words.']
results = []
for prompt in prompts:
    payload = {'model': 'qwen3.8-flash-next', 'messages': [{'role': 'user', 'content': prompt}], 'chat_template_kwargs': {'enable_thinking': False}, 'max_tokens': 384, 'temperature': 0, 'stream': True, 'stream_options': {'include_usage': True}}
    request = urllib.request.Request('http://127.0.0.1:18300/v1/chat/completions', data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    start = time.monotonic()
    first = None
    last = None
    text = ''
    usage = None
    with urllib.request.urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b'data: '):
                continue
            raw = line[6:].strip()
            if raw == b'[DONE]':
                break
            event = json.loads(raw)
            if 'error' in event:
                raise RuntimeError(event['error'])
            for choice in event.get('choices', []):
                delta = choice.get('delta', {})
                chunk = delta.get('content') or delta.get('reasoning') or ''
                if chunk:
                    last = time.monotonic()
                    if first is None:
                        first = last
                    text += chunk
            usage = event.get('usage') or usage
    result = {'prompt': prompt, 'usage': usage, 'ttft_seconds': first - start, 'decode_tokens_per_second': (usage['completion_tokens'] - 1) / (last - first), 'elapsed_seconds': time.monotonic() - start, 'text': text}
    results.append(result)
    print(json.dumps({k: v for k, v in result.items() if k != 'text'}), flush=True)
Path('/tmp/flash-single-stream.json').write_text(json.dumps(results, indent=2) + '\n')
