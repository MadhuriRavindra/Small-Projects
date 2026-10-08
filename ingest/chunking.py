import re


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, max_chars: int = 900, overlap: int = 120) -> list[str]:
    """Paragraph-first chunking; long paragraphs are split on sentence boundaries; small overlap."""
    text = clean(text)
    if not text:
        return []
    pieces = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chars:
            pieces.append(para)
        else:
            cur = ""
            for sent in re.split(r"(?<=[.!?])\s+", para):
                while len(sent) > max_chars:       # no punctuation at all: hard split
                    if cur:
                        pieces.append(cur); cur = ""
                    pieces.append(sent[:max_chars]); sent = sent[max_chars:]
                if len(cur) + len(sent) + 1 > max_chars and cur:
                    pieces.append(cur); cur = sent
                else:
                    cur = f"{cur} {sent}".strip()
            if cur:
                pieces.append(cur)
    chunks, cur = [], ""
    for p in pieces:
        if cur and len(cur) + len(p) + 2 > max_chars:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            cur = (tail + "\n" + p).strip() if tail else p
        else:
            cur = f"{cur}\n\n{p}".strip() if cur else p
    if cur:
        chunks.append(cur)
    return chunks
