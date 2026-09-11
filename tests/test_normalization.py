from __future__ import annotations

import unittest

from offline_rag.normalization import normalize_text


class NormalizationTests(unittest.TestCase):
    def test_normalizes_unicode_line_endings_controls_and_prose_spacing(self) -> None:
        source = "Cafe\u0301\r\n含\x00有   空格\r\n\r\n\r\nEnd"
        self.assertEqual(normalize_text(source), "Café\n含有 空格\n\nEnd")

    def test_preserve_layout_keeps_code_whitespace(self) -> None:
        source = "def run():\r\n    return  1\r\n"
        self.assertEqual(
            normalize_text(source, preserve_layout=True), "def run():\n    return  1\n"
        )


if __name__ == "__main__":
    unittest.main()
