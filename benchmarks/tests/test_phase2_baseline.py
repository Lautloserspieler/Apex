"""Parser and summary tests; no real model, CUDA or third-party packages required."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from phase2_baseline import parse_stats, summarize


EXAMPLE = ("llama noise\n[Apex] prompt_tokens=74 output_tokens=256 "
           "stop=length load_s=0.769 ttft_s=0.07 "
           "prefill_tok_s=1081.59 decode_tok_s=122.766\n")


class BaselineTests(unittest.TestCase):
    def test_extract_real_stats(self):
        data = parse_stats(EXAMPLE)
        self.assertEqual(data["prompt_tokens"], 74)
        self.assertEqual(data["output_tokens"], 256)
        self.assertEqual(data["stop"], "length")
        self.assertAlmostEqual(data["decode_tok_s"], 122.766)

    def test_missing_metrics_rejected(self):
        with self.assertRaisesRegex(ValueError, "No \\[Apex\\] metrics"):
            parse_stats("MOCK OUTPUT - NOT LLM INFERENCE\n")

    def test_invalid_timing_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid Apex"):
            parse_stats(EXAMPLE.replace("decode_tok_s=122.766", "decode_tok_s=nan"))

    def test_unsupported_stop_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            parse_stats(EXAMPLE.replace("stop=length", "stop=mock"))

    def test_median_excludes_warmup_at_call_site(self):
        values = []
        for speed in (120.0, 100.0, 140.0):
            values.append({"metrics": {
                "load_s": .5, "ttft_s": .2, "prefill_tok_s": 70.0,
                "decode_tok_s": speed, "prompt_tokens": 30,
                "output_tokens": 64,
            }})
        stats = summarize(values)
        self.assertEqual(stats["decode_tok_s"],
                         {"median": 120.0, "min": 100.0, "max": 140.0})


if __name__ == "__main__":
    unittest.main()
