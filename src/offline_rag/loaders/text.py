"""Plain-text loader with deterministic local-only decoding."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import url2pathname

from charset_normalizer import from_bytes

from offline_rag.contracts.documents import LoadedContent, SourceDescriptor
from offline_rag.exceptions import DocumentLoadError


class TextLoader:
    def __init__(
        self,
        *,
        extensions: tuple[str, ...] = (".txt",),
        media_types: tuple[str, ...] = ("text/plain",),
        max_file_size_bytes: int | None = None,
    ) -> None:
        if max_file_size_bytes is not None and max_file_size_bytes <= 0:
            raise ValueError("max_file_size_bytes must be positive")
        self._extensions = tuple(
            item.lower() if item.startswith(".") else f".{item.lower()}" for item in extensions
        )
        self._media_types = tuple(item.lower() for item in media_types)
        self._max_file_size_bytes = max_file_size_bytes

    def supports(self, source: SourceDescriptor) -> bool:
        media_type = (source.media_type or "").partition(";")[0].strip().lower()
        if media_type in self._media_types:
            return True
        suffix = Path(urlparse(source.uri).path).suffix.lower()
        return suffix in self._extensions

    def load(self, source: SourceDescriptor) -> LoadedContent:
        if not self.supports(source):
            raise DocumentLoadError(
                "text loader does not support this source",
                details={"source_id": source.source_id},
            )
        path = self._local_path(source.uri)
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise DocumentLoadError(
                "cannot read text source", details={"source_id": source.source_id}
            ) from exc
        maximum = self._max_file_size_bytes
        if maximum is not None and len(data) > maximum:
            raise DocumentLoadError(
                "text source exceeds configured size limit",
                details={"source_id": source.source_id, "size": len(data), "limit": maximum},
            )
        observed_hash = hashlib.sha256(data).hexdigest()
        if observed_hash != source.content_hash:
            raise DocumentLoadError(
                "source changed after discovery",
                details={"source_id": source.source_id},
            )
        text, encoding = self._decode(data, source.source_id)
        return LoadedContent(source=source, text=text, metadata={"encoding": encoding})

    @staticmethod
    def _local_path(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme not in ("", "file"):
            raise DocumentLoadError("text loader only accepts local file URIs")
        if parsed.scheme == "file" and parsed.netloc not in ("", "localhost"):
            raise DocumentLoadError("remote file URI authorities are not allowed")
        raw_path = unquote(parsed.path) if parsed.scheme else uri
        return Path(url2pathname(raw_path))

    @staticmethod
    def _decode(data: bytes, source_id: str) -> tuple[str, str]:
        if data.startswith(b"\xef\xbb\xbf"):
            return data.decode("utf-8-sig"), "utf-8-sig"
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            return data.decode("utf-16"), "utf-16"
        try:
            return data.decode("utf-8"), "utf-8"
        except UnicodeDecodeError:
            match = from_bytes(data).best()
            if match is None or match.encoding is None:
                raise DocumentLoadError(
                    "cannot determine text encoding", details={"source_id": source_id}
                )
            try:
                return str(match), match.encoding.lower()
            except (LookupError, UnicodeError) as exc:
                raise DocumentLoadError(
                    "cannot decode text source", details={"source_id": source_id}
                ) from exc
