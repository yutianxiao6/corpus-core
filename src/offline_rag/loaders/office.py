"""Local-only loaders for Office Open XML documents."""

from __future__ import annotations

from offline_rag.loaders.binary import BinaryFileLoader


class ExcelLoader(BinaryFileLoader):
    def __init__(self, *, max_file_size_bytes: int | None = None) -> None:
        super().__init__(
            extensions=(".xlsx", ".xlsm"),
            media_types=(
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                "application/vnd.ms-excel.sheet.macroenabled.12",
            ),
            max_file_size_bytes=max_file_size_bytes,
        )


class PowerPointLoader(BinaryFileLoader):
    def __init__(self, *, max_file_size_bytes: int | None = None) -> None:
        super().__init__(
            extensions=(".pptx", ".pptm"),
            media_types=(
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                "application/vnd.ms-powerpoint.presentation.macroenabled.12",
            ),
            max_file_size_bytes=max_file_size_bytes,
        )
