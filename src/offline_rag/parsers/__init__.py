"""Built-in structure-aware document parsers."""

from offline_rag.parsers.docx import DocxParser
from offline_rag.parsers.html import HtmlParser
from offline_rag.parsers.markdown import MarkdownParser
from offline_rag.parsers.pdf import PdfParser
from offline_rag.parsers.structured import StructuredTextParser
from offline_rag.parsers.text import TextParser

__all__ = [
    "DocxParser",
    "HtmlParser",
    "MarkdownParser",
    "PdfParser",
    "StructuredTextParser",
    "TextParser",
]
