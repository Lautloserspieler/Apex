"""Phase 3 benchmark harness tests. No model weights or Apex build needed."""
import csv
import json
import os
import stat
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3_bench import (  # noqa: E402
    Config, build_matrix, derive, describe, linear_slope, main,
    marginal_per_ctx_token, parse_int_list, parse_prompt_specs, summarize_runs,
    tokens_consistent, validate_metrics,
)

CONFIG = Config("short", 256, 4, 0, 2)


def stats(**overrides):
    base = {
        "prompt_tokens": 5, "generated_tokens": 4, "stop_reason": "length",
        "model_load_seconds": 0.5, "tokenize_seconds": 0.001,
        "context_setup_seconds": 0.01, "prefill_seconds": 0.05,
        "ttft_seconds": 0.06, "decode_seconds": 0.3, "request_seconds": 0.4,
        "peak_rss_mib": 100.0, "peak_vram_mib": 0.0, "avg_power_watts": 0.0,
    }
    base.update(overrides)
    return base


def document(runs=2, **stat_overrides):
    return {
        "schema": "apex.metrics/1",
        "build": {"llama_backend": True, "apex_git_revision": "abc"},
        "config": {"context_size": 256, "max_tokens": 4, "gpu_layers": 0,
                   "threads": 2, "temperature": 0},
        "runs": [{"index": i, "phase": "cold" if i == 0 else "warm",
                  "stats": stats(**stat_overrides), "token_ids": [1, 2, 3, 4]}
                 for i in range(runs)],
    }


class ParsingTests(unittest.TestCase):
    def test_int_lists(self):
        self.assertEqual(parse_int_list("512, 2048", "x", 16), [512, 2048])
        for bad in ("", "a", "8", "512,512"):
            with self.assertRaises(ValueError):
                parse_int_list(bad, "x", 16)

    def test_prompt_specs(self):
        self.assertEqual(set(parse_prompt_specs(None)), {"short", "long"})
        self.assertEqual(parse_prompt_specs(["a=x.txt"]), {"a": Path("x.txt")})
        for bad in (["nopath"], ["bad name=x"], ["a=x", "a=y"]):
            with self.assertRaises(ValueError):
                parse_prompt_specs(bad)

    def test_default_prompt_files_exist(self):
        for path in parse_prompt_specs(None).values():
            self.assertTrue(path.is_file(), path)
            self.assertTrue(path.read_text(encoding="utf-8").strip())

    def test_matrix_is_full_cross_product(self):
        matrix = build_matrix(["s", "l"], [256, 1024], [16], [0, 99], [0])
        self.assertEqual(len(matrix), 8)
        self.assertEqual(len({c.key for c in matrix}), 8)


class ValidationTests(unittest.TestCase):
    def test_valid_document(self):
        validate_metrics(document(), CONFIG)

    def test_rejects_build_without_real_backend(self):
        doc = document()
        doc["build"]["llama_backend"] = False
        with self.assertRaisesRegex(ValueError, "real llama.cpp backend"):
            validate_metrics(doc, CONFIG)

    def test_rejects_config_mismatch(self):
        with self.assertRaisesRegex(ValueError, "threads"):
            validate_metrics(document(), Config("short", 256, 4, 0, 8))

    def test_rejects_sampled_decoding(self):
        doc = document()
        doc["config"]["temperature"] = 0.7
        with self.assertRaisesRegex(ValueError, "greedy"):
            validate_metrics(doc, CONFIG)

    def test_rejects_wrong_phase_and_counts(self):
        doc = document()
        doc["runs"][1]["phase"] = "cold"
        with self.assertRaisesRegex(ValueError, "phase"):
            validate_metrics(doc, CONFIG)
        doc = document()
        doc["runs"][0]["token_ids"] = [1]
        with self.assertRaisesRegex(ValueError, "disagrees"):
            validate_metrics(doc, CONFIG)

    def test_rejects_mock_and_invalid_timings(self):
        with self.assertRaisesRegex(ValueError, "stop reason"):
            validate_metrics(document(stop_reason="mock"), CONFIG)
        with self.assertRaisesRegex(ValueError, "decode_seconds"):
            validate_metrics(document(decode_seconds=-1.0), CONFIG)

    def test_null_gpu_metrics_are_allowed(self):
        validate_metrics(document(peak_vram_mib=None, avg_power_watts=None), CONFIG)


class DerivedMetricTests(unittest.TestCase):
    def test_cpu_run_reports_gpu_metrics_as_not_measured(self):
        derived = derive(stats(), 256)
        self.assertAlmostEqual(derived["prefill_tok_s"], 100.0)
        # 4 generated tokens -> 3 decode steps (first comes from prefill).
        self.assertAlmostEqual(derived["decode_tok_s"], 10.0)
        self.assertAlmostEqual(derived["ms_per_output_token"], 100.0)
        for name in ("peak_vram_mib", "avg_power_watts", "energy_j_per_token",
                     "vram_mib_per_ctx_token"):
            self.assertIsNone(derived[name], name)

    def test_gpu_energy_and_vram_per_token(self):
        derived = derive(stats(peak_vram_mib=1024.0, avg_power_watts=200.0), 256)
        # 200 W over ttft (0.06 s) + decode (0.3 s) = 72 J for 4 tokens.
        self.assertAlmostEqual(derived["energy_j_per_token"], 18.0)
        self.assertAlmostEqual(derived["vram_mib_per_ctx_token"], 4.0)

    def test_single_token_has_no_decode_rate(self):
        derived = derive(stats(generated_tokens=1, decode_seconds=0.0), 256)
        self.assertIsNone(derived["decode_tok_s"])
        self.assertIsNone(derived["ms_per_output_token"])


class StatisticsTests(unittest.TestCase):
    def run_rows(self, phase, speeds):
        return [{"phase": phase,
                 "derived": derive(stats(decode_seconds=3 / s), 256)}
                for s in speeds]

    def test_describe(self):
        result = describe([1.0, 2.0, 3.0])
        self.assertEqual(result["median"], 2.0)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["stdev"], 1.0)
        self.assertEqual(describe([5.0])["stdev"], 0.0)

    def test_cold_and_warm_are_never_pooled(self):
        rows = self.run_rows("cold", [10.0]) + self.run_rows("warm", [100.0, 120.0])
        summary = summarize_runs(rows)
        self.assertEqual(summary["cold"]["decode_tok_s"]["median"], 10.0)
        self.assertEqual(summary["warm"]["decode_tok_s"]["median"], 110.0)
        self.assertNotIn("peak_vram_mib", summary["warm"])

    def test_token_consistency(self):
        self.assertTrue(tokens_consistent([{"token_ids": [1, 2]}, {"token_ids": [1, 2]}]))
        self.assertFalse(tokens_consistent([{"token_ids": [1, 2]}, {"token_ids": [1, 3]}]))

    def test_linear_slope(self):
        self.assertAlmostEqual(linear_slope([(1024, 600.0), (2048, 700.0), (4096, 900.0)]),
                               100.0 / 1024.0)  # exactly collinear points
        with self.assertRaises(ValueError):
            linear_slope([(1024, 1.0), (1024, 2.0)])

    def test_marginal_vram_slope_groups_by_everything_but_ctx(self):
        def result(ctx, vram, gpu=99, status="ok"):
            return {"status": status,
                    "config": {"prompt": "short", "ctx_size": ctx, "max_tokens": 16,
                               "gpu_layers": gpu, "threads": 0},
                    "summary": {"warm": {"peak_vram_mib": {"median": vram}}}}
        results = [result(1024, 500.0), result(4096, 800.0),
                   result(2048, 1.0, status="skipped-context"),
                   result(1024, 10.0, gpu=0)]  # single point -> no slope
        slopes = marginal_per_ctx_token(results, "peak_vram_mib")
        self.assertEqual(len(slopes), 1)
        self.assertAlmostEqual(slopes[0]["mib_per_ctx_token"], 300.0 / 3072.0)


FAKE_APEX = textwrap.dedent("""\
    #!{python}
    import json, sys
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    ctx, out = int(args["--ctx-size"]), int(args["--max-tokens"])
    prompt_tokens = len(args["--prompt"].split())
    if prompt_tokens + out - 1 >= ctx:
        print("[Apex error] Prompt + max_tokens exceeds --ctx-size", file=sys.stderr)
        sys.exit(1)
    ids = list(range(100, 100 + out))
    if "DIVERGE" in args["--prompt"]:
        ids = None
    runs = []
    for i in range(int(args["--repeat"])):
        run_ids = ids if ids is not None else list(range(i, i + out))
        runs.append({{"index": i, "phase": "cold" if i == 0 else "warm",
            "stats": {{"prompt_tokens": prompt_tokens, "generated_tokens": out,
                "stop_reason": "length", "model_load_seconds": 0.1,
                "tokenize_seconds": 0.001, "context_setup_seconds": 0.002,
                "prefill_seconds": 0.01, "ttft_seconds": 0.011,
                "decode_seconds": 0.02 * (out - 1), "request_seconds": 0.5,
                "peak_rss_mib": 50.0 + ctx / 1024, "peak_vram_mib": 0.0,
                "avg_power_watts": 0.0}},
            "token_ids": run_ids}})
    doc = {{"schema": "apex.metrics/1",
        "build": {{"llama_backend": True, "apex_git_revision": "fake",
                   "build_type": "Release", "compiler": "fake",
                   "llama_cpp_revision": "fake", "cuda_backend": False,
                   "cuda_graphs": False}},
        "config": {{"context_size": ctx, "max_tokens": out,
            "gpu_layers": int(args["--gpu-layers"]), "threads": int(args["--threads"]),
            "temperature": float(args["--temperature"])}},
        "runs": runs}}
    with open(args["--metrics-json"], "w", encoding="utf-8") as f:
        json.dump(doc, f)
    print("fake completion")
""")


@unittest.skipIf(os.name == "nt", "fake executable relies on a POSIX shebang")
class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.apex = self.dir / "apex"
        self.apex.write_text(FAKE_APEX.format(python=sys.executable), encoding="utf-8")
        self.apex.chmod(self.apex.stat().st_mode | stat.S_IXUSR)
        self.model = self.dir / "model.gguf"
        self.model.write_bytes(b"GGUF fake")
        self.short = self.dir / "short.txt"
        self.short.write_text("one two three", encoding="utf-8")
        self.long = self.dir / "long.txt"
        self.long.write_text(" ".join(["word"] * 40), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def run_bench(self, *extra):
        out = self.dir / "results"
        code = main(["--apex", str(self.apex), "--model", str(self.model),
                     "--label", "unit", "--prompt", f"short={self.short}",
                     "--ctx-sizes", "32,64", "--max-tokens", "4",
                     "--processes", "2", "--repeat", "3", "--out", str(out), *extra])
        report_dir = next(out.iterdir())
        return code, report_dir, json.loads((report_dir / "report.json").read_text())

    def test_full_matrix_writes_all_outputs(self):
        code, directory, report = self.run_bench("--prompt", f"long={self.long}")
        self.assertEqual(code, 0)
        statuses = {r["key"]: r["status"] for r in report["results"]}
        self.assertEqual(statuses["short-ctx32-out4-gpu0-t0"], "ok")
        self.assertEqual(statuses["long-ctx32-out4-gpu0-t0"], "skipped-context")
        self.assertEqual(statuses["long-ctx64-out4-gpu0-t0"], "ok")
        ok = next(r for r in report["results"] if r["key"] == "short-ctx32-out4-gpu0-t0")
        self.assertEqual(ok["summary"]["cold"]["request_s"]["n"], 2)
        self.assertEqual(ok["summary"]["warm"]["request_s"]["n"], 4)
        self.assertTrue(ok["correctness"]["deterministic"])
        self.assertEqual(report["build"]["apex_git_revision"], "fake")
        self.assertEqual(len(report["environment"]["model_sha256"]), 64)
        slopes = [m for m in report["marginal_per_ctx_token"]
                  if m["metric"] == "peak_rss_mib" and m["prompt"] == "short"]
        self.assertAlmostEqual(slopes[0]["mib_per_ctx_token"], 1 / 1024)
        with (directory / "runs.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        # short: 2 ctx x 2 processes x 3 runs; long: 1 ctx x 2 x 3.
        self.assertEqual(len(rows), 18)
        self.assertTrue((directory / "summary.csv").is_file())
        markdown = (directory / "summary.md").read_text(encoding="utf-8")
        self.assertIn("skipped-context", markdown)
        self.assertIn("| yes |", markdown)
        self.assertEqual(len(list((directory / "raw").glob("*.metrics.json"))), 6)

    def test_nondeterministic_tokens_fail_the_run(self):
        self.short.write_text("one DIVERGE three", encoding="utf-8")
        code, _, report = self.run_bench()
        self.assertEqual(code, 1)
        self.assertTrue(all(r["status"] == "failed" for r in report["results"]))
        self.assertFalse(report["results"][0]["correctness"]["deterministic"])

    def test_all_skipped_is_an_error(self):
        self.short.write_text(" ".join(["w"] * 100), encoding="utf-8")
        code, _, report = self.run_bench()
        self.assertEqual(code, 1)
        self.assertIn("No configuration", report["error"])

    def test_dry_run_needs_no_files(self):
        self.assertEqual(main(["--apex", "missing", "--model", "missing", "--label", "x",
                               "--dry-run", "--ctx-sizes", "64"]), 0)


if __name__ == "__main__":
    unittest.main()
