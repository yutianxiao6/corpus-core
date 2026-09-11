"""Deterministic and symlink-safe local filesystem discovery."""

from __future__ import annotations

import fnmatch
import glob
import hashlib
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from offline_rag.contracts.common import JSONValue
from offline_rag.contracts.documents import SourceDescriptor
from offline_rag.exceptions import SourceDiscoveryError

_DEFAULT_IGNORES = ("**/~$*", "**/*.tmp", "**/.git/**")


@dataclass(frozen=True, slots=True)
class DiscoveryOptions:
    recursive: bool = True
    follow_symlinks: bool = False
    allow_external_symlinks: bool = False
    include_hidden: bool = False
    include_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = _DEFAULT_IGNORES
    allowed_extensions: tuple[str, ...] = ()
    max_file_size_bytes: int | None = None
    max_total_size_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.max_file_size_bytes is not None and self.max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")
        if self.max_total_size_bytes is not None and self.max_total_size_bytes <= 0:
            raise ValueError("max_total_size_bytes must be positive")
        normalized_extensions = tuple(
            extension.lower() if extension.startswith(".") else f".{extension.lower()}"
            for extension in self.allowed_extensions
        )
        object.__setattr__(self, "allowed_extensions", normalized_extensions)


class FileSystemSourceProvider:
    """Discover local files and produce stable descriptors.

    Source IDs depend on a caller-controlled namespace and normalized relative path,
    never the absolute deployment directory. Content is hashed while discovering so
    later loaders can detect a file changed between discovery and ingestion.
    """

    def __init__(
        self,
        inputs: Sequence[str | Path],
        *,
        root: str | Path | None = None,
        namespace: str = "default",
        options: DiscoveryOptions | None = None,
        metadata: Mapping[str, JSONValue] | None = None,
    ) -> None:
        if not inputs:
            raise ValueError("inputs must contain at least one path or glob")
        if not namespace.strip():
            raise ValueError("namespace must not be empty")
        self._inputs = tuple(str(item) for item in inputs)
        self._explicit_root = Path(root).expanduser().resolve() if root is not None else None
        self._namespace = namespace.strip()
        self._options = options or DiscoveryOptions()
        self._metadata = dict(metadata or {})

    def discover(self) -> Iterable[SourceDescriptor]:
        candidates = self._expand_inputs()
        root = self._root_for(candidates)
        seen_paths: set[Path] = set()
        total_size = 0
        for path in self._iter_files(candidates, root):
            canonical = path.resolve()
            if canonical in seen_paths:
                continue
            seen_paths.add(canonical)
            relative_path = self._relative_path(path, root)
            if not self._is_included(relative_path, path):
                continue
            descriptor = self._describe(path, relative_path)
            total_size += descriptor.size_bytes or 0
            maximum = self._options.max_total_size_bytes
            if maximum is not None and total_size > maximum:
                raise SourceDiscoveryError(
                    "discovered files exceed max_total_size_bytes",
                    details={"limit": maximum, "observed": total_size},
                )
            yield descriptor

    def _expand_inputs(self) -> tuple[Path, ...]:
        expanded: list[Path] = []
        for item in self._inputs:
            candidate = str(Path(item).expanduser())
            if glob.has_magic(candidate):
                matches = sorted(glob.glob(candidate, recursive=True, include_hidden=True))
                if not matches:
                    raise SourceDiscoveryError(
                        f"input pattern did not match any paths: {item}",
                        details={"pattern": item},
                    )
                expanded.extend(Path(match) for match in matches)
            else:
                path = Path(candidate)
                if not path.exists():
                    raise SourceDiscoveryError(
                        f"input path does not exist: {item}",
                        details={"path": item},
                    )
                expanded.append(path)
        return tuple(expanded)

    def _root_for(self, candidates: Sequence[Path]) -> Path:
        if self._explicit_root is not None:
            return self._explicit_root
        if len(self._inputs) == 1 and not glob.has_magic(self._inputs[0]):
            only = candidates[0].absolute()
            return only.resolve() if only.is_dir() else only.parent.resolve()
        return Path.cwd().resolve()

    def _iter_files(self, candidates: Sequence[Path], root: Path) -> Iterator[Path]:
        visited_directories: set[tuple[int, int]] = set()
        for candidate in candidates:
            absolute = candidate.absolute()
            if absolute.is_symlink() and not self._options.follow_symlinks:
                continue
            if absolute.is_file():
                if self._symlink_target_allowed(absolute, root):
                    yield absolute
                continue
            if absolute.is_dir():
                yield from self._walk_directory(absolute, root, visited_directories)

    def _walk_directory(
        self,
        directory: Path,
        root: Path,
        visited: set[tuple[int, int]],
    ) -> Iterator[Path]:
        try:
            stat = directory.stat(follow_symlinks=self._options.follow_symlinks)
            identity = (stat.st_dev, stat.st_ino)
            if identity in visited:
                return
            visited.add(identity)
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise SourceDiscoveryError(
                f"cannot scan directory: {directory}", details={"path": str(directory)}
            ) from exc

        for entry in entries:
            path = Path(entry.path)
            relative_path = self._relative_path(path, root)
            if self._is_hidden(relative_path) and not self._options.include_hidden:
                continue
            if self._matches_any(relative_path, self._options.ignore_patterns):
                continue
            if entry.is_symlink() and not self._options.follow_symlinks:
                continue
            try:
                if entry.is_dir(follow_symlinks=self._options.follow_symlinks):
                    if self._options.recursive and self._symlink_target_allowed(path, root):
                        yield from self._walk_directory(path, root, visited)
                elif entry.is_file(
                    follow_symlinks=self._options.follow_symlinks
                ) and self._symlink_target_allowed(path, root):
                    yield path
            except OSError as exc:
                raise SourceDiscoveryError(
                    f"cannot inspect path: {path}", details={"path": str(path)}
                ) from exc

    def _symlink_target_allowed(self, path: Path, root: Path) -> bool:
        if not path.is_symlink() or not self._options.follow_symlinks:
            return True
        if self._options.allow_external_symlinks:
            return True
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError):
            return False
        return True

    def _relative_path(self, path: Path, root: Path) -> str:
        try:
            relative = path.absolute().relative_to(root)
        except ValueError:
            relative = Path(path.name)
        return PurePosixPath(relative).as_posix()

    def _is_included(self, relative_path: str, path: Path) -> bool:
        if self._is_hidden(relative_path) and not self._options.include_hidden:
            return False
        if self._matches_any(relative_path, self._options.ignore_patterns):
            return False
        if self._options.include_patterns and not self._matches_any(
            relative_path, self._options.include_patterns
        ):
            return False
        extensions = self._options.allowed_extensions
        return not extensions or path.suffix.lower() in extensions

    @staticmethod
    def _is_hidden(relative_path: str) -> bool:
        return any(part.startswith(".") for part in PurePosixPath(relative_path).parts)

    @staticmethod
    def _matches_any(relative_path: str, patterns: Sequence[str]) -> bool:
        path = PurePosixPath(relative_path)
        for raw_pattern in patterns:
            pattern = raw_pattern.replace("\\", "/")
            if fnmatch.fnmatchcase(relative_path, pattern) or path.match(pattern):
                return True
            if pattern.startswith("**/") and fnmatch.fnmatchcase(relative_path, pattern[3:]):
                return True
        return False

    def _describe(self, path: Path, relative_path: str) -> SourceDescriptor:
        try:
            before = path.stat()
            maximum = self._options.max_file_size_bytes
            if maximum is not None and before.st_size > maximum:
                raise SourceDiscoveryError(
                    f"file exceeds max_file_size_bytes: {relative_path}",
                    details={"path": relative_path, "size": before.st_size, "limit": maximum},
                )
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            after = path.stat()
        except SourceDiscoveryError:
            raise
        except OSError as exc:
            raise SourceDiscoveryError(
                f"cannot read discovered file: {relative_path}",
                details={"path": relative_path},
            ) from exc
        signature_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        signature_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if signature_before != signature_after:
            raise SourceDiscoveryError(
                f"file changed while being scanned: {relative_path}",
                details={"path": relative_path},
            )

        source_key = f"{self._namespace}\0{relative_path}".encode()
        source_id = hashlib.sha256(source_key).hexdigest()
        source_metadata: dict[str, JSONValue] = dict(self._metadata)
        source_metadata.update({"filename": path.name, "extension": path.suffix.lower()})
        return SourceDescriptor(
            source_id=source_id,
            uri=path.absolute().as_uri(),
            relative_path=relative_path,
            media_type=None,
            size_bytes=after.st_size,
            modified_at=datetime.fromtimestamp(after.st_mtime, tz=UTC),
            content_hash=digest.hexdigest(),
            metadata=source_metadata,
        )
