#!/usr/bin/env python3
"""Compare Apex greedy output token IDs with a standalone pinned llama.cpp caller.

The reference program uses llama.cpp C APIs directly, bypassing Apex Engine.
Both binaries intentionally use the SAME pinned llama.cpp runtime. This is
adapter correctness testing, not an independently implemented transformer
or a claim that Apex is faster than llama.cpp.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone
from uuid import uuid4

from phase2_baseline import digest_file, parse_stats

PREFIX_RE = re.compile(r"^[0-9]+(?:,[0-9]+)*$")


def read_ids(source: str, name: str) -> list[int]:
    matches = [line[len(name):] for line in source.splitlines() if line.startswith(name)]
    if len(matches) != 1:
        raise ValueError(f"Expected one {name} record, found {len(matches)}")
    encoded = matches[0]
    if not encoded:
        return []
    if not PREFIX_RE.fullmatch(encoded):
        raise ValueError(f"Malformed {name} record: {encoded[:80]}")
    return [int(v) for v in encoded.split(",")]


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout, check=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apex", type=Path, required=True, help="Apex real GGUF binary")
    parser.add_argument("--reference", type=Path, required=True,
                        help="apex_llama_reference executable")
    parser.add_argument("--model", type=Path, required=True, help="Pinned GGUF file")
    prompt_group = parser.add_mutually_exclusive_group()
    prompt_group.add_argument("--prompt", default="Once upon a time")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--ctx-size", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    if args.ctx_size < 16 or args.max_tokens <= 0 or args.gpu_layers < 0 or args.timeout <= 0:
        parser.error("ctx-size >= 16, max-tokens > 0, gpu-layers >= 0, timeout > 0 required")

    apex, reference, model = (p.resolve() for p in
                              (args.apex, args.reference, args.model))
    for path in (apex, reference, model):
        if not path.is_file():
            parser.error(f"File not found: {path}")

    prompt = (args.prompt_file.read_text(encoding="utf-8") if args.prompt_file
              else args.prompt)
    if not prompt:
        parser.error("Prompt must not be empty")

    now = datetime.now(timezone.utc)
    directory = args.out.resolve() / (
        now.strftime("%Y%m%dT%H%M%SZ") + "-reference-" + uuid4().hex[:8])
    directory.mkdir(parents=True, exist_ok=False)

    apex_command = [
        str(apex), "--model", str(model), "--prompt", prompt,
        "--ctx-size", str(args.ctx_size), "--max-tokens", str(args.max_tokens),
        "--temperature", "0", "--gpu-layers", str(args.gpu_layers),
        "--print-token-ids",
    ]
    reference_command = [
        str(reference), str(model), prompt, str(args.ctx_size),
        str(args.max_tokens), str(args.gpu_layers),
    ]
    report = {
        "reference_kind": "standalone direct caller of pinned llama.cpp C API; "
                          "same upstream weights and compute as Apex",
        "timestamp_utc": now.isoformat(),
        "apex_binary_sha256": digest_file(apex),
        "reference_binary_sha256": digest_file(reference),
        "model": str(model),
        "model_size_bytes": model.stat().st_size,
        "model_sha256": digest_file(model),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "configuration": {
            "ctx_size": args.ctx_size,
            "max_tokens": args.max_tokens,
            "gpu_layers": args.gpu_layers,
            "temperature": 0,
        },
        "matches": False,
    }
    try:
        for name, command in (("apex", apex_command), ("reference", reference_command)):
            completed = run(command, args.timeout)
            (directory / (name + ".stdout.txt")).write_text(completed.stdout,
                                                             encoding="utf-8")
            (directory / (name + ".stderr.log")).write_text(completed.stderr,
                                                            encoding="utf-8")
            if completed.returncode != 0:
                raise RuntimeError(f"{name} returned {completed.returncode}; see raw log")
            if name == "apex":
                apex_ids = read_ids(completed.stderr, "APEX_TOKEN_IDS=")
                metrics = parse_stats(completed.stderr)
                if metrics["output_tokens"] != len(apex_ids):
                    raise ValueError("Apex stats and token ID count differ")
                report["apex_metrics"] = metrics
                report["apex_token_ids"] = apex_ids
            else:
                prompt_ids = read_ids(completed.stdout, "REF_PROMPT_IDS=")
                reference_ids = read_ids(completed.stdout, "REF_TOKEN_IDS=")
                report["reference_prompt_tokens"] = len(prompt_ids)
                report["reference_token_ids"] = reference_ids

        if report["reference_prompt_tokens"] != report["apex_metrics"]["prompt_tokens"]:
            raise ValueError("Reference and Apex tokenized prompt lengths differ")
        report["matches"] = report["apex_token_ids"] == report["reference_token_ids"]
        if not report["matches"]:
            raise ValueError("Apex and direct llama.cpp reference token IDs differ")
        print(f"PASS: {len(report['apex_token_ids'])} generated token IDs match "
              f"the pinned upstream reference; prompt length "
              f"{report['reference_prompt_tokens']}")
        result = 0
    except (RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as exc:
        report["error"] = str(exc)
        print(f"FAIL: {exc}", file=sys.stderr)
        result = 1
    finally:
        (directory / "comparison.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Evidence: {directory / 'comparison.json'}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
