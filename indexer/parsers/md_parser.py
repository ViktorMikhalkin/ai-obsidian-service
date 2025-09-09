from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import frontmatter, re

CODE_BLOCK_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)
INLINE_CODE_RE = re.compile(r"`[^`]*`")
LINK_RE = re.compile(r"\[([^\]]+)\]\([^\)]+\)")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^\)]+\)")
HTML_TAG_RE = re.compile(r"<[^>]+>")
HEADER_HASH_RE = re.compile(r"^\s{0,3}#+\s*", re.MULTILINE)

@dataclass
class ParsedNote:
    path: str
    meta: Dict
    text: str

def strip_markdown(md: str) -> str:
    md = CODE_BLOCK_RE.sub(" ", md)
    md = INLINE_CODE_RE.sub(" ", md)
    md = IMAGE_RE.sub(" ", md)
    md = LINK_RE.sub(r"\\1", md)
    md = HEADER_HASH_RE.sub("", md)
    md = re.sub(r"^[>|*-]+\\s*", "", md, flags=re.MULTILINE)
    md = HTML_TAG_RE.sub(" ", md)
    md = re.sub(r"[*_#>`~]+", " ", md)
    md = re.sub(r"\\s+", " ", md)
    return md.strip()

def parse_markdown(path: Path):
    try:
        post = frontmatter.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            post = frontmatter.loads(path.read_text(errors="ignore"))
        except Exception:
            return None
    meta = post.metadata or {}
    text = strip_markdown(post.content or "")
    return ParsedNote(path=str(path), meta=meta, text=text)
