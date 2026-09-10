#!/usr/bin/env python3
"""Functional serving checks after changing the draft vocabulary/SSM precision.

Run after /health succeeds. Uses only the Python standard library.
This is a smoke/retrieval check, not a speed comparison or a quality benchmark.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:18300")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = "qwen3.8-flash-next"
    results = []

    def post(path, body):
        req = Request(args.base + path, json.dumps(body).encode(),
                      {"Content-Type": "application/json"})
        with urlopen(req, timeout=600) as response:
            return json.load(response)

    def check(name, messages, expected, max_tokens=128):
        start = time.monotonic()
        response = post("/v1/chat/completions", {
            "model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        })
        content = response["choices"][0]["message"].get("content") or ""
        row = {
            "name": name, "elapsed_seconds": round(time.monotonic() - start, 3),
            "usage": response["usage"], "content": content,
            "passed": all(value.lower() in content.lower() for value in expected),
        }
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return content

    def message(text):
        return {"role": "user", "content": text}

    with urlopen(args.base + "/health", timeout=5) as response:
        assert response.status == 200
    check("arithmetic", [message("What is 1234 times 17? Reply with digits only.")], ["20978"])
    check("chinese", [message("中国的首都是哪里？只回答城市名。")], ["北京"])

    def concurrent_check(i):
        value = f"concurrency-{i}-verified"
        return check(f"concurrent-{i}", [message(f"Repeat exactly: {value}")], [value], 32)

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(concurrent_check, range(6)))

    lines = [f"Record {i:05d}: copper paper window marble." for i in range(2500)]
    facts = [
        "The archive access phrase is cedar-4821.",
        "The maintenance access phrase is quartz-7395.",
        "The inventory access phrase is willow-1608.",
    ]
    for fraction, fact in zip((0.05, 0.50, 0.95), facts):
        lines[int(len(lines) * fraction)] = fact
    text = "Read these records and remember the three access phrases.\n" + "\n".join(lines)
    text += "\nReturn the archive, maintenance, and inventory access phrases, with their labels."
    messages = [message(text)]
    answer = check("long-context-three-needles", messages,
                   ["cedar-4821", "quartz-7395", "willow-1608"])
    messages += [{"role": "assistant", "content": answer},
                 message("Repeat only the maintenance access phrase from those records.")]
    check("long-context-followup", messages, ["quartz-7395"], 32)

    report = {
        "base": args.base, "passed": all(row["passed"] for row in results),
        "checks": results,
        "limitations": "Functional checks only; no matched speed/quality comparison or 262K-context test.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit("One or more serving checks failed")


if __name__ == "__main__":
    main()
