"""Built-in structure-aware document parsers."""

from offline_rag.parsers.docx import DocxParser
from offline_rag.parsers.html import HtmlParser
from offline_rag.parsers.markdown import MarkdownParser
from offline_rag.parsers.ocr import OcrPdfParser
from offline_rag.parsers.pdf import PdfParser
from offline_rag.parsers.pptx import PptxParser
from offline_rag.parsers.structured import StructuredTextParser
from offline_rag.parsers.text import TextParser
from offline_rag.parsers.xlsx import XlsxParser

__all__ = [
    "DocxParser",
    "HtmlParser",
    "MarkdownParser",
    "OcrPdfParser",
    "PdfParser",
    "PptxParser",
    "StructuredTextParser",
    "TextParser",
    "XlsxParser",
]
