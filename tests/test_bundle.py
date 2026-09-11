from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class OfflineBundleTests(unittest.TestCase):
    def test_bundle_contains_models_wheels_manifest_and_valid_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = root / "demo.whl"
            wheel.write_bytes(b"wheel")
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            output = root / "bundle"
            subprocess.run(
                [
                    sys.executable,
                    "scripts/build_offline_bundle.py",
                    str(output),
                    "--wheel",
                    str(wheel),
                    "--model",
                    str(model),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            checksums = (output / "checksums.sha256").read_text(encoding="utf-8").splitlines()

        self.assertEqual(manifest["format_version"], 1)
        self.assertEqual(manifest["wheels"], ["wheelhouse/demo.whl"])
        self.assertEqual(manifest["models"], ["models/model"])
        self.assertTrue(any("wheelhouse/demo.whl" in line for line in checksums))
        self.assertTrue(any("models/model/config.json" in line for line in checksums))


if __name__ == "__main__":
    unittest.main()
