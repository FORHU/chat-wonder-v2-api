import re


def normalize_text(value) -> str:
    if not value:
        return ""
    text = re.sub(r"\s+", " ", value).strip()
    return text


def chunk_legal_text(
    text: str,
    max_chars: int = 1600,
    overlap_chars: int = 250,
    max_chunks: int | None = 2048,
) -> list[str]:
    source = normalize_text(text)
    if len(source) < 60:
        return []

    def at_chunk_cap() -> bool:
        return max_chunks is not None and len(chunks) >= max_chunks

    # Sentence-aware chunking to avoid breaking citations and legal references.
    sentences = re.split(r"(?<=[\.\?\!;])\s+", source)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if at_chunk_cap():
            break
        if not sentence:
            continue
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
            continue

        if current:
            chunks.append(current)
            if at_chunk_cap():
                break
        if len(sentence) > max_chars:
            start = 0
            while start < len(sentence):
                if at_chunk_cap():
                    break
                end = min(len(sentence), start + max_chars)
                piece = sentence[start:end].strip()
                if len(piece) >= 60:
                    chunks.append(piece)
                    if at_chunk_cap():
                        break
                start = max(0, end - overlap_chars)
            current = ""
        else:
            current = sentence

    if current and len(current) >= 60 and not at_chunk_cap():
        chunks.append(current)

    return chunks

