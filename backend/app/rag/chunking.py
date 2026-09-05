"""Simple, dependency-free document chunker: splits on paragraphs, then
packs sentences into ~chunk_size-char windows with overlap."""
import re


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 100) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # try to break on a sentence boundary near the end
        window = text[start:end]
        last_period = window.rfind(". ")
        if last_period > chunk_size * 0.5 and end < len(text):
            end = start + last_period + 1
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end) if end - overlap <= start else end - overlap
        if end >= len(text):
            break
    return [c for c in chunks if c]
