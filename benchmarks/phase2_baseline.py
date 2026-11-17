#!/usr/bin/env python3
"""Repeatable Apex Phase 2 diagnostic runs. Standard library only.

Each sample starts a NEW Apex process. Warmups are excluded from statistics,
but they do not warm an in-process model or KV cache. Do not call this a
controlled hot-vs-cold benchmark or a reference llama.cpp comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DEFAULT_PROMPT = (
    "<|im_start|>system\n"
    "You are a concise technical assistant.\n"
    "<|im_end|>\n"
    "<|im_start|>user\n"
    "Explain three benefits and one drawback of running AI models locally.\n"
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
)
FLOAT_FIELDS = ("load_s", "ttft_s", "prefill_tok_s", "decode_tok_s")
INT_FIELDS = ("prompt_tokens", "output_tokens")


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_stats(stderr: str) -> dict:
    """Read the last well-formed Apex metrics record, or fail explicitly."""
    lines = [line.partition("[Apex] ")[2] for line in stderr.splitlines()
             if "[Apex] prompt_tokens=" in line]
    if not lines:
        raise ValueError("No [Apex] metrics line; check real backend and stderr log")
    result = {}
    for pair in lines[-1].split():
        if "=" not in pair:
            raise ValueError("Malformed Apex metrics field")
        key, value = pair.split("=", 1)
        if key in result:
            raise ValueError(f"Duplicate Apex metrics field: {key}")
        result[key] = value
    for name in INT_FIELDS:
        result[name] = int(result[name])
        if result[name] < 0:
            raise ValueError(f"Negative Apex token count: {name}")
    for name in FLOAT_FIELDS:
        result[name] = float(result[name])
        if not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f"Invalid Apex timing/speed value: {name}")
    if result.get("stop") not in ("length", "eos"):
        raise ValueError(f"Unsupported Apex stop reason: {result.get('stop')}")
    if result["prompt_tokens"] <= 0:
        raise ValueError("No prompt tokens; likely not real model inference")
    return result


def summarize(samples: list[dict]) -> dict:
    if not samples:
        raise ValueError("At least one measured sample is required")
    keys = (*FLOAT_FIELDS, "prompt_tokens", "output_tokens")
    return {
        key: {
            "median": statistics.median(row["metrics"][key] for row in samples),
            "min": min(row["metrics"][key] for row in samples),
            "max": max(row["metrics"][key] for row in samples),
        }
        for key in keys
    }


def optional_command(args: list[str]) -> str | None:
    try:
        process = subprocess.run(args, capture_output=True, text=True,
                                 timeout=15, check=False)
        if process.returncode == 0:
            return process.stdout.strip() or process.stderr.strip() or None
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apex", type=Path, required=True, help="Compiled real-backend Apex binary")
    parser.add_argument("--model", type=Path, required=True, help="Existing local GGUF file")
    parser.add_argument("--label", required=True, help="For example: qwen7b-cpu or qwen7b-gpu")
    parser.add_argument("--gpu-layers", type=int, default=0)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1, help="Excluded independent processes")
    parser.add_argument("--runs", type=int, default=5, help="Measured independent processes")
    parser.add_argument("--timeout", type=int, default=1800, help="Timeout per process in seconds")
    parser.add_argument("--prompt-file", type=Path, help="UTF-8 prompt; default is fixed Qwen ChatML")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    if args.runs < 1 or args.warmup < 0 or args.max_tokens < 1 or args.ctx_size < 16:
        parser.error("runs >= 1, warmup >= 0, max-tokens >= 1 and ctx-size >= 16 required")
    if args.gpu_layers < 0 or args.timeout < 1:
        parser.error("gpu-layers >= 0 and timeout >= 1 required")
    if not args.label or not all(c.isalnum() or c in "_-" for c in args.label):
        parser.error("label must contain only letters, digits, _ or -")

    apex = args.apex.resolve()
    model = args.model.resolve()
    if not apex.is_file():
        parser.error(f"Apex binary not found: {apex}")
    if not model.is_file():
        parser.error(f"GGUF model not found: {model}")
    if args.prompt_file:
        prompt = args.prompt_file.read_text(encoding="utf-8").replace("\r\n", "\n")
    else:
        prompt = DEFAULT_PROMPT
    if not prompt.strip():
        parser.error("Prompt is empty")

    root = Path(__file__).resolve().parents[1]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out.resolve() / (timestamp + "-" + args.label + "-" + uuid4().hex[:8])
    out.mkdir(parents=True, exist_ok=False)
    command = [
        str(apex), "--model", str(model), "--prompt", prompt,
        "--ctx-size", str(args.ctx_size), "--max-tokens", str(args.max_tokens),
        "--temperature", "0", "--gpu-layers", str(args.gpu_layers),
    ]
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "run_kind": "new-process-per-sample; cache-warming only, not hot in-process inference",
        "apex_executable": str(apex),
        "apex_binary_sha256": digest_file(apex),
        "apex_commit": optional_command(["git", "-C", str(root), "rev-parse", "HEAD"]),
        "model": str(model),
        "model_size_bytes": model.stat().st_size,
        "model_sha256": digest_file(model),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_length_bytes": len(prompt.encode("utf-8")),
        "os": platform.platform(),
        "cpu": platform.processor(),
        "nvidia_smi": optional_command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"]),
        "nvcc": optional_command(["nvcc", "--version"]),
        "configuration": {
            "ctx_size": args.ctx_size,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "gpu_layers": args.gpu_layers,
            "warmup": args.warmup,
            "runs": args.runs,
        },
        "samples": [],
    }

    try:
        for index in range(args.warmup + args.runs):
            warmup = index < args.warmup
            name = f"sample-{index + 1:02d}"
            print(f"[{args.label}] {name}: {'warmup' if warmup else 'measurement'}", flush=True)
            try:
                process = subprocess.run(command, capture_output=True, text=True,
                                         encoding="utf-8", errors="replace", timeout=args.timeout,
                                         check=False)
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(f"{name} exceeded {args.timeout}s timeout") from exc
            (out / f"{name}.stdout.txt").write_text(process.stdout, encoding="utf-8")
            (out / f"{name}.stderr.log").write_text(process.stderr, encoding="utf-8")
            if process.returncode:
                raise RuntimeError(f"{name}: Apex exited {process.returncode}; see stderr log")
            metrics = parse_stats(process.stderr)
            metadata["samples"].append({
                "sample": index + 1,
                "warmup": warmup,
                "metrics": metrics,
                "output_sha256": hashlib.sha256(process.stdout.encode("utf-8")).hexdigest(),
            })
            print(f"  decode={metrics['decode_tok_s']:.2f} tok/s; "
                  f"prefill={metrics['prefill_tok_s']:.2f} tok/s; "
                  f"tokens={metrics['output_tokens']}", flush=True)
    except (RuntimeError, ValueError, KeyError) as exc:
        metadata["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        return_code = 1
    else:
        measured = [x for x in metadata["samples"] if not x["warmup"]]
        metadata["summary"] = summarize(measured)
        print(f"Median decode: {metadata['summary']['decode_tok_s']['median']:.2f} tok/s")
        return_code = 0
    finally:
        (out / "report.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Report: {out / 'report.json'}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
