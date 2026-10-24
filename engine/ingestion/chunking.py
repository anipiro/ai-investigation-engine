"""Two chunking strategies, compared in the evaluation.

fixed     - sliding window of N words with M words of overlap. Structure-blind, predictable size.
recursive - structure-aware: split by markdown section -> paragraph -> sentence, then greedily
            pack units up to N words. Chunks never cut a sentence in half, carry the heading
            path of their section, and overlap by whole trailing sentences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    index: int
    text: str
    section: str | None = None
    page: int | None = None

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def split_sentences(text: str) -> list[str]:
    out = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            out.extend(s.strip() for s in _SENT_SPLIT.split(line) if s.strip())
    return out


def _pages(text: str) -> list[tuple[int | None, str]]:
    """Split on form feeds inserted by the PDF parser; non-PDF text is a single 'page'."""
    if "\f" not in text:
        return [(None, text)]
    return [(i + 1, p) for i, p in enumerate(text.split("\f"))]


def chunk_fixed(text: str, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")
    chunks: list[Chunk] = []
    for page, page_text in _pages(text):
        words = page_text.split()
        step = max_words - overlap
        start = 0
        while start < len(words):
            window = words[start : start + max_words]
            chunks.append(Chunk(index=len(chunks), text=" ".join(window), page=page))
            if start + max_words >= len(words):
                break
            start += step
    return chunks


def _sections(page_text: str) -> list[tuple[str | None, str]]:
    """Group lines under their markdown heading path ("A > B")."""
    path: list[tuple[int, str]] = []
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in page_text.split("\n"):
        m = _HEADING.match(line.strip())
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            path = [p for p in path if p[0] < level] + [(level, title)]
            sections.append((" > ".join(t for _, t in path), []))
        else:
            sections[-1][1].append(line)
    return [(name, "\n".join(lines).strip()) for name, lines in sections if "\n".join(lines).strip()]


def chunk_recursive(text: str, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page, page_text in _pages(text):
        for section, body in _sections(page_text):
            # Units are paragraphs if they fit, otherwise their sentences, otherwise word windows.
            units: list[str] = []
            for para in re.split(r"\n\s*\n", body):
                para = " ".join(para.split())
                if not para:
                    continue
                if len(para.split()) <= max_words:
                    units.append(para)
                    continue
                for sent in split_sentences(para):
                    if len(sent.split()) <= max_words:
                        units.append(sent)
                    else:
                        units.extend(c.text for c in chunk_fixed(sent, max_words, overlap))
            buf: list[str] = []
            buf_words = 0

            def flush() -> None:
                if buf:
                    chunks.append(Chunk(index=len(chunks), text=" ".join(buf), section=section, page=page))

            for unit in units:
                n = len(unit.split())
                if buf and buf_words + n > max_words:
                    flush()
                    # Overlap: carry trailing sentences of the previous chunk (whole sentences only).
                    carry: list[str] = []
                    carried = 0
                    for sent in reversed(split_sentences(" ".join(buf))):
                        k = len(sent.split())
                        if carried + k > overlap:
                            break
                        carry.insert(0, sent)
                        carried += k
                    buf, buf_words = carry, carried
                    if buf_words + n > max_words:
                        buf, buf_words = [], 0
                buf.append(unit)
                buf_words += n
            flush()
    return chunks


STRATEGIES = {"fixed": chunk_fixed, "recursive": chunk_recursive}


def chunk_text(text: str, strategy: str = "recursive", max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    try:
        fn = STRATEGIES[strategy]
    except KeyError:
        raise ValueError(f"unknown chunking strategy '{strategy}'") from None
    return fn(text, max_words=max_words, overlap=overlap)
