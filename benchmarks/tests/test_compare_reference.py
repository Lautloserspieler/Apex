"""Reference output parser tests; no GPU or GGUF file is required."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from compare_reference import read_ids


class TokenIdParsingTests(unittest.TestCase):
    def test_expected_ids(self):
        self.assertEqual(read_ids("other stdout\nREF_TOKEN_IDS=12,345,6\n",
                                  "REF_TOKEN_IDS="), [12, 345, 6])

    def test_empty_result_when_eos_first(self):
        self.assertEqual(read_ids("REF_TOKEN_IDS=\n", "REF_TOKEN_IDS="), [])

    def test_no_record_is_failure(self):
        with self.assertRaisesRegex(ValueError, "Expected one"):
            read_ids("no ids present\n", "REF_TOKEN_IDS=")

    def test_duplicate_record_is_failure(self):
        with self.assertRaisesRegex(ValueError, "found 2"):
            read_ids("REF_TOKEN_IDS=1\nREF_TOKEN_IDS=2\n",
                     "REF_TOKEN_IDS=")

    def test_negative_and_malformed_ids_rejected(self):
        for bad in ("REF_TOKEN_IDS=-1\n", "REF_TOKEN_IDS=1,\n",
                    "REF_TOKEN_IDS=1,x\n", "REF_TOKEN_IDS=1 2\n"):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ValueError, "Malformed"):
                    read_ids(bad, "REF_TOKEN_IDS=")


if __name__ == "__main__":
    unittest.main()
