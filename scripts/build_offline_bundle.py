#!/usr/bin/env python3
"""Assemble an offline wheelhouse/model bundle and deterministic checksums."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--wheel", action="append", type=Path, default=[])
    parser.add_argument("--model", action="append", type=Path, default=[])
    parser.add_argument("--force", action="store_true", help="replace an existing output directory")
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    sources = [
        *(path.expanduser().resolve() for path in args.wheel),
        *(path.expanduser().resolve() for path in args.model),
    ]
    if any(
        source == output or source in output.parents or output in source.parents
        for source in sources
    ):
        raise SystemExit("output directory must not overlap a wheel or model input")
    if output.exists() and any(output.iterdir()):
        if not args.force:
            raise SystemExit(f"output is not empty; pass --force to replace: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    wheelhouse = output / "wheelhouse"
    models = output / "models"
    wheelhouse.mkdir()
    models.mkdir()
    copied_wheels: list[str] = []
    copied_models: list[str] = []
    for source in (path.expanduser().resolve() for path in args.wheel):
        if not source.is_file() or source.suffix not in (".whl", ".tar.gz", ".zip"):
            raise SystemExit(f"wheelhouse input must be a wheel/archive file: {source}")
        target = wheelhouse / source.name
        shutil.copy2(source, target)
        copied_wheels.append(target.relative_to(output).as_posix())
    for source in (path.expanduser().resolve() for path in args.model):
        if not source.is_dir():
            raise SystemExit(f"model input must be a directory: {source}")
        target = models / source.name
        if target.exists():
            raise SystemExit(f"duplicate model name in bundle: {target.name}")
        shutil.copytree(source, target)
        copied_models.append(target.relative_to(output).as_posix())
    files = sorted(path for path in output.rglob("*") if path.is_file())
    checksums = "".join(
        f"{_sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in files
    )
    (output / "checksums.sha256").write_text(checksums, encoding="utf-8")
    manifest = {
        "format_version": 1,
        "wheels": copied_wheels,
        "models": copied_models,
        "checksum_file": "checksums.sha256",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
