"""Opt-in OCR parser for image-only PDF pages."""

from __future__ import annotations

import hashlib

from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
)
from corpuscore.exceptions import DocumentParseError
from corpuscore.normalization import normalize_text


class OcrPdfParser:
    name = "pdf_ocr"
    version = "1"

    def __init__(self, *, languages: str = "eng+chi_sim", dpi: int = 180) -> None:
        if not languages.strip():
            raise ValueError("languages must not be empty")
        if dpi <= 0:
            raise ValueError("dpi must be positive")
        self.languages = languages
        self.dpi = dpi

    def parse(self, loaded: LoadedContent) -> ParsedDocument:
        if loaded.binary is None:
            raise DocumentParseError("OCR PDF parser requires binary content")
        try:
            import fitz  # type: ignore[import-untyped]
            import pytesseract  # type: ignore[import-untyped]
            from PIL import Image

            document = fitz.open(stream=loaded.binary, filetype="pdf")
            blocks: list[ContentBlock] = []
            try:
                for page_number, page in enumerate(document, start=1):
                    pixmap = page.get_pixmap(dpi=self.dpi, alpha=False)
                    image = Image.frombytes("RGB", (pixmap.width, pixmap.height), pixmap.samples)
                    content = normalize_text(
                        pytesseract.image_to_string(image, lang=self.languages)
                    )
                    if content:
                        blocks.append(
                            ContentBlock(
                                ContentBlockType.IMAGE_TEXT,
                                content,
                                page=page_number,
                                metadata={"ocr_languages": self.languages, "ocr_dpi": self.dpi},
                            )
                        )
            finally:
                document.close()
        except ImportError as exc:
            raise DocumentParseError(
                "OCR support requires the optional 'ocr' extra and a local Tesseract binary",
                details={"source_id": loaded.source.source_id},
            ) from exc
        except Exception as exc:
            raise DocumentParseError(
                "cannot OCR PDF source", details={"source_id": loaded.source.source_id}
            ) from exc
        if not blocks:
            raise DocumentParseError(
                "OCR produced no indexable text", details={"source_id": loaded.source.source_id}
            )
        material = (
            f"{loaded.source.source_id}\0{loaded.source.content_hash}\0{self.name}\0{self.version}"
        )
        return ParsedDocument(
            document_id=hashlib.sha256(material.encode()).hexdigest(),
            source=loaded.source,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"format": "pdf", "ocr": True},
        )
