"""Portable, checksum-verified backup helpers for Qdrant Local deployments."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import mkdtemp
from uuid import uuid4

from corpuscore.exceptions import IndexCompatibilityError, VectorStoreError

BACKUP_FORMAT_VERSION = 1


def create_local_backup(
    qdrant_path: str | Path,
    destination: str | Path,
    *,
    collection_name: str,
    index_fingerprint: str | None = None,
) -> dict[str, object]:
    """Create a gzip tar archive while the caller holds the vector-store lock."""

    source = Path(qdrant_path).expanduser().resolve()
    target = Path(destination).expanduser().resolve()
    if not source.is_dir():
        raise VectorStoreError(f"Qdrant Local path does not exist: {source}")
    if target == source or source in target.parents:
        raise ValueError("backup destination must be outside the Qdrant Local directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "format_version": BACKUP_FORMAT_VERSION,
        "mode": "local",
        "collection_name": collection_name,
        "index_fingerprint": index_fingerprint,
        "source_sha256": _directory_sha256(source),
        "created_at": time.time(),
    }
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(source, arcname="qdrant")
            payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")
            info = tarfile.TarInfo("manifest.json")
            info.size = len(payload)
            archive.addfile(info, fileobj=_BytesReader(payload))
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    manifest["archive_sha256"] = _sha256(target)
    return manifest


def restore_local_backup(
    archive_path: str | Path,
    target_path: str | Path,
    *,
    collection_name: str,
    expected_index_fingerprint: str | None = None,
    replace_existing: bool = False,
) -> dict[str, object]:
    """Restore an archive into a Qdrant directory with atomic directory replacement."""

    archive = Path(archive_path).expanduser().resolve()
    target = Path(target_path).expanduser().resolve()
    if not archive.is_file():
        raise VectorStoreError(f"backup archive does not exist: {archive}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(mkdtemp(prefix=f"{target.name}.restore-", dir=target.parent))
    old_path: Path | None = None
    try:
        with tarfile.open(archive, "r:gz") as stream:
            member = stream.extractfile("manifest.json")
            if member is None:
                raise VectorStoreError("backup archive is missing manifest.json")
            manifest_raw = json.loads(member.read().decode("utf-8"))
            if not isinstance(manifest_raw, Mapping):
                raise VectorStoreError("backup manifest must be an object")
            _validate_manifest(manifest_raw, collection_name, expected_index_fingerprint)
            stream.extractall(temporary, filter="data")
        restored = temporary / "qdrant"
        if not restored.is_dir():
            raise VectorStoreError("backup archive is missing qdrant directory")
        expected_checksum = manifest_raw.get("source_sha256")
        if isinstance(expected_checksum, str) and _directory_sha256(restored) != expected_checksum:
            raise VectorStoreError("backup checksum verification failed")
        if target.exists() and not replace_existing:
            raise VectorStoreError("restore target exists; pass replace_existing=True explicitly")
        if target.exists():
            old_path = target.with_name(f"{target.name}.before-restore-{int(time.time())}")
            target.replace(old_path)
        restored.replace(target)
        return dict(manifest_raw)
    except Exception:
        if old_path is not None and not target.exists() and old_path.exists():
            old_path.replace(target)
        raise
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def read_server_backup(
    archive_path: str | Path,
    *,
    collection_name: str,
    expected_index_fingerprint: str | None = None,
) -> dict[str, object]:
    """Read and validate a Server snapshot descriptor."""

    path = Path(archive_path).expanduser().resolve()
    try:
        manifest_raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VectorStoreError("cannot read Qdrant Server snapshot descriptor") from exc
    if not isinstance(manifest_raw, Mapping):
        raise VectorStoreError("server backup manifest must be an object")
    if manifest_raw.get("format_version") != BACKUP_FORMAT_VERSION:
        raise VectorStoreError("unsupported backup format version")
    if manifest_raw.get("mode") != "server":
        raise VectorStoreError("backup is not a Qdrant Server descriptor")
    if manifest_raw.get("collection_name") != collection_name:
        raise VectorStoreError("backup collection name does not match configuration")
    if not isinstance(manifest_raw.get("snapshot_name"), str):
        raise VectorStoreError("server backup is missing snapshot_name")
    actual = manifest_raw.get("index_fingerprint")
    if expected_index_fingerprint is not None and actual != expected_index_fingerprint:
        raise IndexCompatibilityError(
            "backup index fingerprint does not match configuration",
            details={"expected": expected_index_fingerprint, "actual": actual},
        )
    return dict(manifest_raw)


def _validate_manifest(
    manifest: Mapping[str, object],
    collection_name: str,
    expected_index_fingerprint: str | None,
) -> None:
    if manifest.get("format_version") != BACKUP_FORMAT_VERSION:
        raise VectorStoreError("unsupported backup format version")
    if manifest.get("mode") != "local":
        raise VectorStoreError("backup is not a Qdrant Local archive")
    if manifest.get("collection_name") != collection_name:
        raise VectorStoreError("backup collection name does not match configuration")
    actual = manifest.get("index_fingerprint")
    if expected_index_fingerprint is not None and actual != expected_index_fingerprint:
        raise IndexCompatibilityError(
            "backup index fingerprint does not match configuration",
            details={"expected": expected_index_fingerprint, "actual": actual},
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        with file_path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


class _BytesReader:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self._offset = 0

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._payload) - self._offset
        value = self._payload[self._offset : self._offset + size]
        self._offset += len(value)
        return value
