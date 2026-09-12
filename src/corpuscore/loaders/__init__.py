"""Built-in document loaders."""

from corpuscore.loaders.binary import BinaryFileLoader, DocxLoader, PdfLoader
from corpuscore.loaders.markdown import MarkdownLoader
from corpuscore.loaders.office import ExcelLoader, PowerPointLoader
from corpuscore.loaders.text import TextLoader

__all__ = [
    "BinaryFileLoader",
    "DocxLoader",
    "ExcelLoader",
    "MarkdownLoader",
    "PdfLoader",
    "PowerPointLoader",
    "TextLoader",
]
