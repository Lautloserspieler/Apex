#!/usr/bin/env python3
"""Apex Phase 3 benchmark matrix (issue #3, step 1: instrument before optimizing).

Standard library only. For every configuration in the matrix

    prompt x context size x max tokens x GPU layers x threads

the harness starts ``--processes`` independent Apex processes. Each process
loads the model once and generates ``--repeat`` times (``apex --repeat``):
run 1 is the *cold* in-process run (first context allocation, first touch of
memory-mapped weights), later runs are *warm*. Cold and warm samples are
summarized separately and never mixed.

Correctness is checked alongside every performance run:
  * greedy decoding only (temperature 0), and every run of a configuration
    must produce identical token IDs (determinism / KV-reset check);
  * optionally (``--reference``) the token IDs must exactly match the
    standalone pinned llama.cpp caller ``apex_llama_reference``.

Outputs (below ``--out``/<UTC timestamp>-<label>-<id>/):
  report.json   environment, build info, every raw run, summaries
  runs.csv      one row per generation run
  summary.csv   one row per configuration x phase x metric
  summary.md    human-readable report
  raw/          per-process Apex metrics JSON, stdout and stderr

All figures are Apex's own diagnostics. They are not a like-for-like
comparison with llama.cpp tools unless the conditions are matched as
described in docs/benchmark-methodology.md.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_reference import read_ids  # noqa: E402
from phase2_baseline import digest_file, optional_command  # noqa: E402

SCHEMA = "apex.metrics/1"
REPORT_SCHEMA = "apex.phase3-bench/1"
DEFAULT_PROMPT_DIR = Path(__file__).resolve().parent / "phase3_prompts"
CONTEXT_REJECTION = "exceeds --ctx-size"

# Derived per-run metrics, in report order. None means "not measured".
METRICS = (
    "model_load_s", "tokenize_s", "context_setup_s", "ttft_s", "prefill_tok_s",
    "decode_tok_s", "ms_per_output_token", "request_s", "peak_rss_mib",
    "peak_vram_mib", "avg_power_watts", "energy_j_per_token",
    "vram_mib_per_ctx_token",
)


@dataclass(frozen=True)
class Config:
    prompt_name: str
    ctx_size: int
    max_tokens: int
    gpu_layers: int
    threads: int

    @property
    def key(self) -> str:
        return (f"{self.prompt_name}-ctx{self.ctx_size}-out{self.max_tokens}"
                f"-gpu{self.gpu_layers}-t{self.threads}")

    def as_dict(self) -> dict:
        return {
            "prompt": self.prompt_name, "ctx_size": self.ctx_size,
            "max_tokens": self.max_tokens, "gpu_layers": self.gpu_layers,
            "threads": self.threads,
        }


# --------------------------------------------------------------------------
# Parsing and validation
# --------------------------------------------------------------------------

def parse_int_list(text: str, name: str, minimum: int) -> list[int]:
    try:
        values = [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError(f"{name}: expected comma-separated integers, got {text!r}") from exc
    if not values:
        raise ValueError(f"{name}: at least one value required")
    if any(v < minimum for v in values):
        raise ValueError(f"{name}: values must be >= {minimum}")
    if len(set(values)) != len(values):
        raise ValueError(f"{name}: duplicate values")
    return values


def parse_prompt_specs(specs: list[str] | None) -> dict[str, Path]:
    if not specs:
        return {"short": DEFAULT_PROMPT_DIR / "short.txt",
                "long": DEFAULT_PROMPT_DIR / "long.txt"}
    prompts: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise ValueError(f"--prompt expects NAME=PATH, got {spec!r}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
            raise ValueError(f"Prompt name must use letters, digits, _ or -: {name!r}")
        if name in prompts:
            raise ValueError(f"Duplicate prompt name: {name}")
        prompts[name] = Path(path)
    return prompts


def build_matrix(prompts: list[str], ctx_sizes: list[int], max_tokens: list[int],
                 gpu_layers: list[int], threads: list[int]) -> list[Config]:
    return [Config(p, c, m, g, t)
            for p in prompts for c in ctx_sizes for m in max_tokens
            for g in gpu_layers for t in threads]


def _finite_non_negative(value, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Metric {name} is not a number: {value!r}")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Metric {name} is invalid: {value!r}")
    return value


def validate_metrics(document: dict, config: Config) -> None:
    """Reject anything that is not a well-formed real-inference record."""
    if document.get("schema") != SCHEMA:
        raise ValueError(f"Unexpected metrics schema: {document.get('schema')!r}")
    if not document.get("build", {}).get("llama_backend"):
        raise ValueError("Metrics come from a build without the real llama.cpp backend")
    cfg = document.get("config", {})
    expected = {"context_size": config.ctx_size, "max_tokens": config.max_tokens,
                "gpu_layers": config.gpu_layers, "threads": config.threads}
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"Metrics config {key}={cfg.get(key)!r}, expected {value!r}")
    if float(cfg.get("temperature", -1)) != 0.0:
        raise ValueError("Phase 3 benchmark requires greedy decoding (temperature 0)")
    for index, run in enumerate(document.get("runs", [])):
        if run.get("index") != index:
            raise ValueError("Run indices are not consecutive")
        if run.get("phase") != ("cold" if index == 0 else "warm"):
            raise ValueError(f"Run {index} has wrong phase {run.get('phase')!r}")
        stats = run.get("stats", {})
        if stats.get("stop_reason") not in ("eos", "length"):
            raise ValueError(f"Unsupported stop reason: {stats.get('stop_reason')!r}")
        if not isinstance(stats.get("prompt_tokens"), int) or stats["prompt_tokens"] <= 0:
            raise ValueError("No prompt tokens; not real model inference")
        generated = stats.get("generated_tokens")
        ids = run.get("token_ids")
        if not isinstance(generated, int) or not isinstance(ids, list):
            raise ValueError("Malformed generated token data")
        if generated != len(ids) or generated > config.max_tokens:
            raise ValueError("Generated token count disagrees with token IDs or max_tokens")
        for name in ("model_load_seconds", "tokenize_seconds", "context_setup_seconds",
                     "prefill_seconds", "ttft_seconds", "decode_seconds",
                     "request_seconds", "peak_rss_mib"):
            _finite_non_negative(stats.get(name), name)
        for name in ("peak_vram_mib", "avg_power_watts"):
            if stats.get(name) is not None:
                _finite_non_negative(stats.get(name), name)


def derive(stats: dict, ctx_size: int) -> dict:
    """Per-run derived metrics. None means "not measured", never zero."""
    def positive(name):
        value = stats.get(name)
        return float(value) if isinstance(value, (int, float)) and value > 0 else None

    generated = stats["generated_tokens"]
    prefill_s = stats["prefill_seconds"]
    decode_s = stats["decode_seconds"]
    power = positive("avg_power_watts")
    vram = positive("peak_vram_mib")
    decode_steps = generated - 1  # the first token is produced by prefill
    return {
        "model_load_s": stats["model_load_seconds"],
        "tokenize_s": stats["tokenize_seconds"],
        "context_setup_s": stats["context_setup_seconds"],
        "ttft_s": stats["ttft_seconds"],
        "prefill_tok_s": stats["prompt_tokens"] / prefill_s if prefill_s > 0 else None,
        "decode_tok_s": decode_steps / decode_s if decode_steps > 0 and decode_s > 0 else None,
        "ms_per_output_token": (1000.0 * decode_s / decode_steps
                                if decode_steps > 0 and decode_s > 0 else None),
        "request_s": stats["request_seconds"],
        "peak_rss_mib": positive("peak_rss_mib"),
        "peak_vram_mib": vram,
        "avg_power_watts": power,
        # The power sampler covers prefill + decode, i.e. ttft + decode.
        "energy_j_per_token": (power * (stats["ttft_seconds"] + decode_s) / generated
                               if power is not None and generated > 0 else None),
        # Upper bound: includes weights and compute buffers, not just KV cache.
        # See marginal_per_ctx_token() for the KV-attributable slope.
        "vram_mib_per_ctx_token": vram / ctx_size if vram is not None else None,
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------

def describe(values: list[float]) -> dict:
    if not values:
        raise ValueError("No values to summarize")
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return {
        "n": len(values),
        "median": statistics.median(values),
        "mean": mean,
        "stdev": stdev,
        "cv": stdev / mean if mean > 0 else None,
        "min": min(values),
        "max": max(values),
    }


def summarize_runs(runs: list[dict]) -> dict:
    """Summaries per phase. Cold and warm samples are never pooled."""
    summary: dict[str, dict] = {}
    for phase in ("cold", "warm"):
        selected = [r for r in runs if r["phase"] == phase]
        if not selected:
            continue
        summary[phase] = {}
        for metric in METRICS:
            values = [r["derived"][metric] for r in selected
                      if r["derived"][metric] is not None]
            if values:
                summary[phase][metric] = describe(values)
    return summary


def tokens_consistent(runs: list[dict]) -> bool:
    return len({tuple(r["token_ids"]) for r in runs}) <= 1


def linear_slope(points: list[tuple[float, float]]) -> float:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    mean_x, mean_y = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        raise ValueError("Need at least two distinct context sizes")
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / denominator


def marginal_per_ctx_token(results: list[dict], metric: str) -> list[dict]:
    """Least-squares slope of a memory metric over context size.

    Weights and fixed buffers do not scale with --ctx-size, so the slope
    isolates the memory cost of one additional context slot (dominated by
    the KV cache). Uses warm medians, falling back to cold if there are no
    warm runs. Groups need >= 2 distinct context sizes with measurements.
    """
    groups: dict[tuple, list[tuple[float, float]]] = {}
    for result in results:
        if result["status"] != "ok":
            continue
        summary = result["summary"]
        phase = summary.get("warm") or summary.get("cold") or {}
        if metric not in phase:
            continue
        cfg = result["config"]
        group = (cfg["prompt"], cfg["max_tokens"], cfg["gpu_layers"], cfg["threads"])
        groups.setdefault(group, []).append((float(cfg["ctx_size"]), phase[metric]["median"]))
    output = []
    for (prompt, max_tokens, gpu_layers, threads), points in sorted(groups.items()):
        if len({p[0] for p in points}) < 2:
            continue
        output.append({
            "metric": metric,
            "prompt": prompt, "max_tokens": max_tokens,
            "gpu_layers": gpu_layers, "threads": threads,
            "points": sorted(points),
            "mib_per_ctx_token": linear_slope(points),
        })
    return output


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def cpu_model() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    if sys.platform == "darwin":
        return optional_command(["sysctl", "-n", "machdep.cpu.brand_string"])
    return platform.processor() or None


def total_ram_mib() -> float | None:
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024.0
    return None


def cuda_driver_version() -> str | None:
    banner = optional_command(["nvidia-smi"])
    if banner:
        match = re.search(r"CUDA Version:\s*([0-9.]+)", banner)
        if match:
            return match.group(1)
    return None


def environment(apex: Path, model: Path, prompts: dict[str, str]) -> dict:
    root = Path(__file__).resolve().parents[1]
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "apex_executable": str(apex),
        "apex_binary_sha256": digest_file(apex),
        "repository_revision": optional_command(
            ["git", "-C", str(root), "describe", "--always", "--dirty", "--abbrev=40"]),
        "model": str(model),
        "model_size_bytes": model.stat().st_size,
        "model_sha256": digest_file(model),
        "prompts": {
            name: {"sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                   "bytes": len(text.encode("utf-8"))}
            for name, text in prompts.items()
        },
        "os": platform.platform(),
        "cpu": cpu_model(),
        "logical_cpus": os.cpu_count(),
        "total_ram_mib": total_ram_mib(),
        "python": platform.python_version(),
        "gpus": optional_command(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total",
                                  "--format=csv,noheader"]),
        "cuda_driver_version": cuda_driver_version(),
        "nvcc": optional_command(["nvcc", "--version"]),
        "env": {name: os.environ.get(name) for name in
                ("CUDA_VISIBLE_DEVICES", "GGML_CUDA_DISABLE_GRAPHS", "OMP_NUM_THREADS")},
    }


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------

def run_process(command: list[str], timeout: int, log_prefix: Path) -> subprocess.CompletedProcess:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Timed out after {timeout}s: {Path(command[0]).name}") from exc
    log_prefix.with_suffix(".stdout.txt").write_text(completed.stdout, encoding="utf-8")
    log_prefix.with_suffix(".stderr.log").write_text(completed.stderr, encoding="utf-8")
    return completed


def run_config(config: Config, args, apex: Path, model: Path, prompt_text: str,
               raw: Path) -> dict:
    result = {"key": config.key, "config": config.as_dict(), "status": "ok",
              "processes": [], "runs": []}
    for process_index in range(args.processes):
        stem = f"{config.key}-p{process_index + 1:02d}"
        metrics_path = raw / f"{stem}.metrics.json"
        command = [
            str(apex), "--model", str(model), "--prompt", prompt_text,
            "--ctx-size", str(config.ctx_size), "--max-tokens", str(config.max_tokens),
            "--gpu-layers", str(config.gpu_layers), "--threads", str(config.threads),
            "--temperature", "0", "--repeat", str(args.repeat),
            "--metrics-json", str(metrics_path),
        ]
        completed = run_process(command, args.timeout, raw / stem)
        if completed.returncode != 0:
            if CONTEXT_REJECTION in completed.stderr:
                result["status"] = "skipped-context"
                result["reason"] = "prompt tokens + max_tokens do not fit the context size"
                return result
            raise RuntimeError(f"{stem}: Apex exited {completed.returncode}; see {stem}.stderr.log")
        document = json.loads(metrics_path.read_text(encoding="utf-8"))
        validate_metrics(document, config)
        if len(document["runs"]) != args.repeat:
            raise ValueError(f"{stem}: expected {args.repeat} runs, got {len(document['runs'])}")
        result["build"] = document["build"]
        result["processes"].append({"process": process_index + 1,
                                    "metrics_file": metrics_path.name})
        for run in document["runs"]:
            result["runs"].append({
                "process": process_index + 1,
                "index": run["index"],
                "phase": run["phase"],
                "stats": run["stats"],
                "derived": derive(run["stats"], config.ctx_size),
                "token_ids": run["token_ids"],
            })

    result["summary"] = summarize_runs(result["runs"])
    correctness = {"deterministic": tokens_consistent(result["runs"])}
    if args.reference:
        correctness["reference"] = reference_check(config, args, model, prompt_text, raw,
                                                   result["runs"][0])
    result["correctness"] = correctness
    if not correctness["deterministic"]:
        result["status"] = "failed"
        result["reason"] = "greedy token IDs differ between runs"
    elif args.reference and not correctness["reference"]["matches"]:
        result["status"] = "failed"
        result["reason"] = "token IDs differ from the pinned llama.cpp reference"
    return result


def reference_check(config: Config, args, model: Path, prompt_text: str, raw: Path,
                    apex_run: dict) -> dict:
    # The reference caller has no thread option; threads must not change
    # greedy token IDs, so its default thread count is acceptable here.
    stem = f"{config.key}-reference"
    command = [str(args.reference.resolve()), str(model), prompt_text,
               str(config.ctx_size), str(config.max_tokens), str(config.gpu_layers)]
    completed = run_process(command, args.timeout, raw / stem)
    if completed.returncode != 0:
        raise RuntimeError(f"{stem}: reference exited {completed.returncode}")
    prompt_ids = read_ids(completed.stdout, "REF_PROMPT_IDS=")
    reference_ids = read_ids(completed.stdout, "REF_TOKEN_IDS=")
    return {
        "matches": (reference_ids == apex_run["token_ids"]
                    and len(prompt_ids) == apex_run["stats"]["prompt_tokens"]),
        "reference_tokens": len(reference_ids),
        "reference_prompt_tokens": len(prompt_ids),
    }


# --------------------------------------------------------------------------
# Reports
# --------------------------------------------------------------------------

def write_runs_csv(path: Path, results: list[dict]) -> None:
    fields = ["config", "prompt", "ctx_size", "max_tokens", "gpu_layers", "threads",
              "process", "index", "phase", "prompt_tokens", "generated_tokens",
              "stop_reason", *METRICS]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for run in result["runs"]:
                row = {"config": result["key"], **result["config"],
                       "process": run["process"], "index": run["index"],
                       "phase": run["phase"],
                       "prompt_tokens": run["stats"]["prompt_tokens"],
                       "generated_tokens": run["stats"]["generated_tokens"],
                       "stop_reason": run["stats"]["stop_reason"]}
                row.update({m: ("" if run["derived"][m] is None else run["derived"][m])
                            for m in METRICS})
                writer.writerow(row)


def write_summary_csv(path: Path, results: list[dict]) -> None:
    fields = ["config", "status", "phase", "metric", "n", "median", "mean", "stdev",
              "cv", "min", "max"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            for phase, metrics in result.get("summary", {}).items():
                for metric, stats in metrics.items():
                    writer.writerow({"config": result["key"], "status": result["status"],
                                     "phase": phase, "metric": metric,
                                     **{k: ("" if v is None else v) for k, v in stats.items()}})


def _fmt(stats: dict | None, digits: int = 2) -> str:
    if not stats:
        return "–"
    text = f"{stats['median']:.{digits}f}"
    if stats["n"] > 1:
        text += f" ± {stats['stdev']:.{digits}f}"
    return text


def render_markdown(report: dict) -> str:
    env = report["environment"]
    build = report.get("build") or {}
    lines = [
        f"# Apex Phase 3 benchmark — {report['label']}",
        "",
        f"- Time (UTC): {env['timestamp_utc']}",
        f"- Apex: {build.get('apex_git_revision', '?')} ({build.get('build_type', '?')}, "
        f"{build.get('compiler', '?')}); llama.cpp {build.get('llama_cpp_revision', '?')}; "
        f"CUDA backend {build.get('cuda_backend')}, CUDA graphs {build.get('cuda_graphs')}",
        f"- Model: `{Path(env['model']).name}` sha256 `{env['model_sha256']}`",
        f"- Host: {env['cpu']} ({env['logical_cpus']} logical CPUs), {env['os']}",
        f"- GPU: {env['gpus'] or 'none detected'}; CUDA driver {env['cuda_driver_version'] or '–'}",
        f"- Protocol: {report['protocol']['processes']} process(es) × "
        f"{report['protocol']['repeat']} in-process run(s) per configuration; "
        "greedy decoding; values are median ± stdev",
        "",
        "| Config | Status | Phase | n | TTFT s | Prefill tok/s | Decode tok/s | ms/token "
        "| Peak RSS MiB | Peak VRAM MiB | J/token | Tokens match |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in report["results"]:
        correctness = result.get("correctness", {})
        match = "–"
        if correctness:
            match = "yes" if correctness["deterministic"] else "NO"
            if "reference" in correctness:
                match += " / ref " + ("yes" if correctness["reference"]["matches"] else "NO")
        summary = result.get("summary", {})
        if not summary:
            lines.append(f"| `{result['key']}` | {result['status']} | – | 0 | – | – | – | – "
                         f"| – | – | – | {match} |")
            continue
        for phase, metrics in summary.items():
            n = metrics["request_s"]["n"] if "request_s" in metrics else 0
            lines.append(
                f"| `{result['key']}` | {result['status']} | {phase} | {n} "
                f"| {_fmt(metrics.get('ttft_s'), 4)} | {_fmt(metrics.get('prefill_tok_s'))} "
                f"| {_fmt(metrics.get('decode_tok_s'))} "
                f"| {_fmt(metrics.get('ms_per_output_token'), 3)} "
                f"| {_fmt(metrics.get('peak_rss_mib'), 1)} | {_fmt(metrics.get('peak_vram_mib'), 1)} "
                f"| {_fmt(metrics.get('energy_j_per_token'), 4)} | {match} |")
    marginal = report.get("marginal_per_ctx_token", [])
    if marginal:
        lines += ["", "## Memory per additional context token (least-squares slope)", "",
                  "| Metric | Prompt | Out | GPU layers | Threads | MiB per ctx token | Points |",
                  "|---|---|---:|---:|---:|---:|---|"]
        for item in marginal:
            points = ", ".join(f"{int(x)}→{y:.1f}" for x, y in item["points"])
            lines.append(f"| {item['metric']} | {item['prompt']} | {item['max_tokens']} "
                         f"| {item['gpu_layers']} | {item['threads']} "
                         f"| {item['mib_per_ctx_token']:.5f} | {points} |")
    lines += ["", "Caveats: `–` means not measured (never zero). VRAM and power come from "
              "coarse `nvidia-smi` sampling of GPU 0. Peak RSS is the process high-water "
              "mark. Cold = first generation after model load in a process; the OS page "
              "cache may still be warm from earlier processes. These are Apex diagnostics, "
              "not a llama.cpp comparison.", ""]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apex", type=Path, required=True, help="Real-backend Apex binary")
    parser.add_argument("--model", type=Path, required=True, help="Local GGUF file")
    parser.add_argument("--label", required=True, help="e.g. qwen05b-rtx5070")
    parser.add_argument("--prompt", action="append", metavar="NAME=PATH",
                        help="Prompt file (repeatable). Default: phase3_prompts/short+long")
    parser.add_argument("--ctx-sizes", default="1024,4096", help="Comma-separated")
    parser.add_argument("--max-tokens", default="128", help="Comma-separated")
    parser.add_argument("--gpu-layers", default="0", help="Comma-separated")
    parser.add_argument("--threads", default="0", help="Comma-separated; 0 = llama.cpp default")
    parser.add_argument("--processes", type=int, default=2,
                        help="Independent processes per configuration (each yields 1 cold run)")
    parser.add_argument("--repeat", type=int, default=5,
                        help="Generations per process (1 cold + N-1 warm)")
    parser.add_argument("--reference", type=Path,
                        help="apex_llama_reference binary for exact token-ID parity")
    parser.add_argument("--timeout", type=int, default=1800, help="Seconds per process")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--dry-run", action="store_true", help="Print the matrix and exit")
    args = parser.parse_args(argv)

    try:
        ctx_sizes = parse_int_list(args.ctx_sizes, "--ctx-sizes", 16)
        max_tokens = parse_int_list(args.max_tokens, "--max-tokens", 1)
        gpu_layers = parse_int_list(args.gpu_layers, "--gpu-layers", 0)
        threads = parse_int_list(args.threads, "--threads", 0)
        prompt_paths = parse_prompt_specs(args.prompt)
    except ValueError as exc:
        parser.error(str(exc))
    if args.processes < 1 or args.repeat < 1 or args.timeout < 1:
        parser.error("--processes, --repeat and --timeout must be >= 1")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("label must contain only letters, digits, _ or -")

    matrix = build_matrix(list(prompt_paths), ctx_sizes, max_tokens, gpu_layers, threads)
    if args.dry_run:
        for config in matrix:
            print(config.key)
        print(f"{len(matrix)} configurations × {args.processes} processes × "
              f"{args.repeat} runs")
        return 0

    apex, model = args.apex.resolve(), args.model.resolve()
    for path in (apex, model, *(args.reference and [args.reference.resolve()] or [])):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    prompts = {}
    for name, path in prompt_paths.items():
        if not path.is_file():
            parser.error(f"Prompt file not found: {path}")
        prompts[name] = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        if not prompts[name].strip():
            parser.error(f"Prompt {name} is empty")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out.resolve() / f"{timestamp}-{args.label}-{uuid4().hex[:8]}"
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=False)

    report = {
        "schema": REPORT_SCHEMA,
        "label": args.label,
        "environment": environment(apex, model, prompts),
        "protocol": {"processes": args.processes, "repeat": args.repeat,
                     "temperature": 0, "reference": bool(args.reference),
                     "matrix": [c.as_dict() for c in matrix]},
        "results": [],
    }
    exit_code = 0
    try:
        for number, config in enumerate(matrix, start=1):
            print(f"[{number}/{len(matrix)}] {config.key}", flush=True)
            result = run_config(config, args, apex, model, prompts[config.prompt_name], raw)
            report["results"].append(result)
            report.setdefault("build", result.get("build"))
            warm = result.get("summary", {}).get("warm", {}).get("decode_tok_s")
            print(f"    {result['status']}"
                  + (f"; warm decode {warm['median']:.2f} tok/s" if warm else ""), flush=True)
            if result["status"] == "failed":
                exit_code = 1
    except (RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        report["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
        exit_code = 1
    finally:
        ok = [r for r in report["results"] if r["status"] == "ok"]
        if not ok and "error" not in report:
            report["error"] = "No configuration produced measurements"
            exit_code = 1
        report["marginal_per_ctx_token"] = (
            marginal_per_ctx_token(report["results"], "peak_rss_mib")
            + marginal_per_ctx_token(report["results"], "peak_vram_mib"))
        (out / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        write_runs_csv(out / "runs.csv", report["results"])
        write_summary_csv(out / "summary.csv", report["results"])
        (out / "summary.md").write_text(render_markdown(report), encoding="utf-8")
        print(f"Report: {out}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
