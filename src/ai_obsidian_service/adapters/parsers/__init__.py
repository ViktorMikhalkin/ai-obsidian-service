from .md_parser import MarkdownParser
from .pdf_parser import PdfParser
from .epub_parser import EpubParser

def default_parsers():
    return [MarkdownParser(), PdfParser(), EpubParser()]
