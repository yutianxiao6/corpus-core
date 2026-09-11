from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from offline_rag.cli import _parse_filters, build_parser, main


class CliTests(unittest.TestCase):
    def test_query_override_flags_and_filter_parsing(self) -> None:
        args = build_parser().parse_args(
            [
                "query",
                "refund",
                "--profile",
                "balanced",
                "--organizer",
                "debug",
                "--final-k",
                "3",
                "--score-threshold",
                "0.4",
                "--filter",
                "department=support",
            ]
        )
        self.assertEqual(args.organizer, "debug")
        self.assertEqual(args.final_k, 3)
        self.assertEqual(_parse_filters(["department=support", "acl=[1, 2]"])["acl"], (1, 2))

    def test_evaluate_parser_has_metric_gates(self) -> None:
        args = build_parser().parse_args(
            ["evaluate", "eval.jsonl", "--k", "8", "--min-recall", "0.8", "--json"]
        )
        self.assertEqual(args.fixture, Path("eval.jsonl"))
        self.assertEqual(args.k, 8)
        self.assertEqual(args.min_recall, 0.8)
        self.assertTrue(args.json)

    def test_empty_invocation_prints_help(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("rag-index", output.getvalue())

    def test_init_creates_config_and_expected_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "project"
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(["init", str(target)])

            self.assertEqual(exit_code, 0)
            self.assertTrue((target / "rag.yaml").is_file())
            self.assertTrue((target / "data").is_dir())
            self.assertTrue((target / "documents").is_dir())
            self.assertTrue((target / "models").is_dir())
            self.assertIn("created", output.getvalue())

    def test_preview_outputs_machine_readable_counts_without_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "guide.md").write_text("# 安装\n\nInstall locally.", encoding="utf-8")
            (root / "notes.txt").write_text("离线说明", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                exit_code = main(["preview", str(root), "--json"])
            data = json.loads(output.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(data["source_count"], 2)
        self.assertEqual(data["successful_count"], 2)
        self.assertGreaterEqual(data["chunk_count"], 2)

    def test_doctor_fails_when_local_model_is_missing(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main(["doctor"])
        data = json.loads(output.getvalue())

        self.assertEqual(exit_code, 1)
        self.assertFalse(data["ok"])


if __name__ == "__main__":
    unittest.main()
