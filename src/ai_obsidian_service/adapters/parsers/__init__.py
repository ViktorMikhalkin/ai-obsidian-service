from .epub_parser import EpubParser
from .md_parser import MarkdownParser
from .pdf_parser import PdfParser


def default_parsers():
    return [MarkdownParser(), PdfParser(), EpubParser()]
