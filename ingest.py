"""
Document ingestion and chunking for The Unofficial Guide.

Loads the .txt files in documents/, separates the metadata header from the
body, cleans the text, and splits it into chunks.

Chunking strategy (from planning.md):
  - ~500 character chunks, ~100 character overlap
  - split on paragraph boundaries first, then sentences, so we never cut
    mid-sentence
  - each chunk keeps its source filename so answers can be cited later

Run directly to inspect the output:
    python ingest.py
"""

import html
import random
import re
from pathlib import Path

DOCUMENTS_DIR = Path(__file__).parent / "documents"
CHUNK_SIZE = 500      # target characters per chunk
CHUNK_OVERLAP = 100   # characters of overlap carried into the next chunk


def load_documents(documents_dir=DOCUMENTS_DIR):
    """Read every .txt file and split off its metadata header.

    Each file starts with a small header block (SOURCE/URL/TYPE/TOPIC...),
    one blank line, then the body. Since the header has no blank lines inside
    it, splitting on the first blank line cleanly separates the two.

    Returns a list of dicts: {source, source_title, url, body}.
    """
    docs = []
    for path in sorted(documents_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        parts = raw.split("\n\n", 1)
        header_block = parts[0]
        body = parts[1] if len(parts) > 1 else ""

        # parse the "KEY: value" header lines
        meta = {}
        for line in header_block.splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip().upper()] = value.strip()

        docs.append({
            "source": path.name,                       # filename = canonical id for citations
            "source_title": meta.get("SOURCE", path.name),
            "url": meta.get("URL", ""),
            "body": body,
        })
    return docs


def clean_text(text):
    """Light cleanup: decode HTML entities and normalize whitespace.

    The text was already extracted to plain text, so this mostly guards
    against leftover entities (&amp;, &#39;) and collapses stray spacing
    while preserving paragraph breaks (blank lines).
    """
    text = html.unescape(text)
    # normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # collapse runs of spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)
    # strip trailing spaces on each line
    text = re.sub(r" *\n", "\n", text)
    # collapse 3+ newlines into a single blank line (one paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_heading(paragraph):
    """A short line with no sentence-ending punctuation is a section heading
    (e.g. "Lease, Utilities, and Tenant Rights"), not a standalone fact."""
    return len(paragraph) <= 60 and not re.search(r"[.!?]", paragraph)


def _to_sentences(text):
    """Break text into sentences, respecting paragraph breaks.

    Splits on blank lines first (paragraphs), then on sentence-ending
    punctuation. Heading-like paragraphs are merged into the paragraph that
    follows them so a bare heading never becomes its own chunk. Any sentence
    longer than CHUNK_SIZE is hard-split on spaces so no piece is oversized.
    """
    # collapse each paragraph to a single line
    paragraphs = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s*\n\s*", " ", para.strip())
        if para:
            paragraphs.append(para)

    # attach heading-like paragraphs to the next real paragraph
    merged, carry = [], ""
    for para in paragraphs:
        if _is_heading(para):
            carry = f"{carry} {para}".strip() if carry else para
            continue
        merged.append(f"{carry} — {para}" if carry else para)
        carry = ""
    if carry:  # a trailing heading with nothing after it
        merged.append(carry)

    sentences = []
    for para in merged:
        for sent in re.split(r"(?<=[.!?])\s+", para):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= CHUNK_SIZE:
                sentences.append(sent)
            else:
                sentences.extend(_hard_split(sent, CHUNK_SIZE))
    return sentences


def _hard_split(text, size):
    """Split an overly long string on spaces into <= size pieces."""
    words = text.split(" ")
    pieces, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > size and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Pack sentences into ~chunk_size chunks with ~overlap characters of
    sentence-level overlap between neighbors."""
    sentences = _to_sentences(text)
    chunks = []
    start = 0
    while start < len(sentences):
        # grow the chunk one sentence at a time until it would exceed chunk_size
        end, length = start, 0
        while end < len(sentences):
            piece = len(sentences[end]) + (1 if end > start else 0)
            if length + piece > chunk_size and end > start:
                break
            length += piece
            end += 1

        chunk = " ".join(sentences[start:end]).strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(sentences):
            break

        # step back to create the overlap, then make sure we still move forward
        ov_start, ov_len = end, 0
        while ov_start > start and ov_len < overlap:
            ov_start -= 1
            ov_len += len(sentences[ov_start]) + 1
        start = max(ov_start, start + 1)

    return chunks


def load_and_chunk(documents_dir=DOCUMENTS_DIR):
    """Full pipeline: load -> clean -> chunk.

    Returns a flat list of chunk dicts:
        {text, source, source_title, url, chunk_index}
    """
    records = []
    for doc in load_documents(documents_dir):
        cleaned = clean_text(doc["body"])
        for i, chunk in enumerate(chunk_text(cleaned)):
            records.append({
                "text": chunk,
                "source": doc["source"],
                "source_title": doc["source_title"],
                "url": doc["url"],
                "chunk_index": i,
            })
    return records


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents from {DOCUMENTS_DIR}\n")

    records = load_and_chunk()
    print(f"Total chunks: {len(records)}\n")

    # chunks per document
    per_doc = {}
    for r in records:
        per_doc[r["source"]] = per_doc.get(r["source"], 0) + 1
    print("Chunks per document:")
    for source, count in per_doc.items():
        print(f"  {count:3d}  {source}")

    # sanity checks
    lengths = [len(r["text"]) for r in records]
    empty = sum(1 for n in lengths if n == 0)
    oversized = sum(1 for n in lengths if n > CHUNK_SIZE * 1.5)
    print(
        f"\nChunk length: min={min(lengths)}  avg={sum(lengths)//len(lengths)}  "
        f"max={max(lengths)}"
    )
    print(f"Empty chunks: {empty}   Oversized (>{int(CHUNK_SIZE*1.5)}): {oversized}")

    # inspect 5 random chunks
    print("\n" + "=" * 70)
    print("5 RANDOM CHUNKS")
    print("=" * 70)
    for r in random.sample(records, 5):
        print(f"\n[{r['source']} #{r['chunk_index']}]  ({len(r['text'])} chars)")
        print(r["text"])
