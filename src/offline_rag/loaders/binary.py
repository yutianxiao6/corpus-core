"""Verified local binary file loader for container document formats."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from offline_rag.contracts.documents import LoadedContent, SourceDescriptor
from offline_rag.exceptions import DocumentLoadError


class BinaryFileLoader:
    def __init__(
        self,
        *,
        extensions: tuple[str, ...],
        media_types: tuple[str, ...] = (),
        max_file_size_bytes: int | None = None,
    ) -> None:
        if not extensions and not media_types:
            raise ValueError("at least one extension or media type is required")
        if max_file_size_bytes is not None and max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")
        self._extensions = tuple(
            item.lower() if item.startswith(".") else f".{item.lower()}" for item in extensions
        )
        self._media_types = tuple(item.lower() for item in media_types)
        self._max_file_size_bytes = max_file_size_bytes

    def supports(self, source: SourceDescriptor) -> bool:
        media_type = (source.media_type or "").partition(";")[0].strip().lower()
        suffix = Path(urlparse(source.uri).path).suffix.lower()
        return media_type in self._media_types or suffix in self._extensions

    def load(self, source: SourceDescriptor) -> LoadedContent:
        if not self.supports(source):
            raise DocumentLoadError(
                "binary loader does not support this source",
                details={"source_id": source.source_id},
            )
        path = self._local_path(source.uri)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DocumentLoadError(
                "cannot read binary source", details={"source_id": source.source_id}
            ) from exc
        maximum = self._max_file_size_bytes
        if maximum is not None and len(data) > maximum:
            raise DocumentLoadError(
                "binary source exceeds configured size limit",
                details={"source_id": source.source_id, "size": len(data), "limit": maximum},
            )
        if hashlib.sha256(data).hexdigest() != source.content_hash:
            raise DocumentLoadError(
                "source changed after discovery", details={"source_id": source.source_id}
            )
        return LoadedContent(source=source, binary=data)

    @staticmethod
    def _local_path(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme not in ("", "file"):
            raise DocumentLoadError("binary loader only accepts local file URIs")
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            raise DocumentLoadError("remote file URI authorities are not allowed")
        raw_path = unquote(parsed.path) if parsed.scheme else uri
        return Path(url2pathname(raw_path))


class PdfLoader(BinaryFileLoader):
    def __init__(self, *, max_file_size_bytes: int | None = None) -> None:
        super().__init__(
            extensions=(".pdf",),
            media_types=("application/pdf",),
            max_file_size_bytes=max_file_size_bytes,
        )


class DocxLoader(BinaryFileLoader):
    def __init__(self, *, max_file_size_bytes: int | None = None) -> None:
        super().__init__(
            extensions=(".docx",),
            media_types=(
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            max_file_size_bytes=max_file_size_bytes,
        )
