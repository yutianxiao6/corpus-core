"""Markdown text loader."""

from offline_rag.loaders.text import TextLoader


class MarkdownLoader(TextLoader):
    def __init__(self, *, max_file_size_bytes: int | None = None) -> None:
        super().__init__(
            extensions=(".md", ".markdown"),
            media_types=("text/markdown", "text/x-markdown"),
            max_file_size_bytes=max_file_size_bytes,
        )
