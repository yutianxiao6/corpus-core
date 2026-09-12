"""Built-in structure-aware document parsers."""

from corpuscore.parsers.code import CODE_EXTENSIONS, CODE_LANGUAGES, CodeParser
from corpuscore.parsers.docx import DocxParser
from corpuscore.parsers.html import HtmlParser
from corpuscore.parsers.markdown import MarkdownParser
from corpuscore.parsers.ocr import OcrPdfParser
from corpuscore.parsers.pdf import PdfParser
from corpuscore.parsers.pptx import PptxParser
from corpuscore.parsers.structured import StructuredTextParser
from corpuscore.parsers.text import TextParser
from corpuscore.parsers.xlsx import XlsxParser

__all__ = [
    "CODE_EXTENSIONS",
    "CODE_LANGUAGES",
    "CodeParser",
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
