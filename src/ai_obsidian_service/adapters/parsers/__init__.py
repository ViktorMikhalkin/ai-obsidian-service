from __future__ import annotations
from typing import Type, List, Protocol, runtime_checkable

# Architectural note:
# All formats are equal. No "default vs non-default" distinction.
# We use a unified registry of parser classes and build_parsers() factory.
# Selection is based only on backend availability (available()), not on "status".
#
# For backward compatibility we export default_parsers = build_parsers().

@runtime_checkable
class DocumentParser(Protocol):
    def accepts(self, path: str) -> bool: ...
    def parse(self, text: str): ...
    def parse_file(self, path: str): ...

# Import parser classes (each parser knows about its own availability)
from .md_parser import MarkdownParser  # always available

try:
    from .pdf_parser import PdfParser
except Exception:  # pragma: no cover - import is optional
    PdfParser = None  # type: ignore[assignment]

try:
    from .epub_parser import EpubParser
except Exception:  # pragma: no cover - import is optional
    EpubParser = None  # type: ignore[assignment]

# Unified registry of parser classes (all equal)
PARSER_CLASSES: list[type] = [MarkdownParser]

if PdfParser is not None:
    PARSER_CLASSES.append(PdfParser)  # type: ignore[arg-type]

if EpubParser is not None:
    PARSER_CLASSES.append(EpubParser)  # type: ignore[arg-type]

def register_parser(cls: type) -> None:
    """Register a new parser class (plugin-like model)."""
    if cls not in PARSER_CLASSES:
        PARSER_CLASSES.append(cls)

def build_parsers(*, require_available: bool = True) -> List[DocumentParser]:
    """Instantiate all registered parsers.
    If the class has a classmethod available() -> bool, and require_available=True,
    we create an instance only when it's actually available (dependencies installed).
    Otherwise - create without checking.
    """
    instances: list[DocumentParser] = []
    for cls in PARSER_CLASSES:
        try:
            if require_available and hasattr(cls, "available"):
                if not bool(getattr(cls, "available")()):  # type: ignore[misc]
                    continue
            instances.append(cls())  # type: ignore[call-arg, misc]
        except Exception:
            # don't block building other parsers
            continue
    return instances

# Back-compat export: historically code pulled default_parsers.
# Now it's just "parsers = build_parsers()".
default_parsers = build_parsers()

__all__ = [
    "DocumentParser",
    "MarkdownParser",
    "PdfParser",
    "EpubParser",
    "register_parser",
    "build_parsers",
    "default_parsers",
]