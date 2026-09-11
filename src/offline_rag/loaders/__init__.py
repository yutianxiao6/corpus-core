"""Built-in document loaders."""

from offline_rag.loaders.binary import BinaryFileLoader, DocxLoader, PdfLoader
from offline_rag.loaders.markdown import MarkdownLoader
from offline_rag.loaders.office import ExcelLoader, PowerPointLoader
from offline_rag.loaders.text import TextLoader

__all__ = [
    "BinaryFileLoader",
    "DocxLoader",
    "ExcelLoader",
    "MarkdownLoader",
    "PdfLoader",
    "PowerPointLoader",
    "TextLoader",
]
