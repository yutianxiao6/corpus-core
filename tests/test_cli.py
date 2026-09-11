from __future__ import annotations

import contextlib
import io
import unittest

from offline_rag.cli import main


class CliTests(unittest.TestCase):
    def test_empty_invocation_prints_help(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = main([])

        self.assertEqual(exit_code, 0)
        self.assertIn("rag-index", output.getvalue())


if __name__ == "__main__":
    unittest.main()
