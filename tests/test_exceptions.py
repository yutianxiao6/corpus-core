from __future__ import annotations

import unittest

from corpuscore.exceptions import ConfigurationError, CorpusCoreError


class ExceptionTests(unittest.TestCase):
    def test_public_exception_has_stable_code_and_frozen_details(self) -> None:
        error = ConfigurationError("invalid config", details={"fields": ["unknown"]})

        self.assertIsInstance(error, CorpusCoreError)
        self.assertEqual(error.code, "configuration_error")
        self.assertEqual(str(error), "invalid config")
        self.assertEqual(error.details["fields"], ("unknown",))


if __name__ == "__main__":
    unittest.main()
