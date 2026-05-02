"""Core retrieval-augmented generation pipeline.

Pipeline: PDF -> text -> overlapping chunks -> sentence embeddings -> FAISS index.
At query time the question is embedded, the top-k chunks are retrieved, and an
answer is produced either by an LLM (if an API key is configured) or by an
extractive fallback that returns the best-matching sentences.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")


@dataclass
class Chunk:
    text: str
    source: str
    page: int


def extract_pages(path: str) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int = 800, overlap: int = 150) -> list[str]:
    """Split text into overlapping windows, preferring sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks


@dataclass
class DocumentIndex:
    model: SentenceTransformer = field(default_factory=lambda: SentenceTransformer(EMBED_MODEL))
    chunks: list[Chunk] = field(default_factory=list)
    index: faiss.IndexFlatIP | None = None

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return vecs.astype("float32")

    def add_pdf(self, path: str, source_name: str | None = None) -> int:
        source = source_name or os.path.basename(path)
        new_chunks = [
            Chunk(text=c, source=source, page=page_no)
            for page_no, page_text in extract_pages(path)
            for c in chunk_text(page_text)
        ]
        if not new_chunks:
            return 0
        vecs = self._embed([c.text for c in new_chunks])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim on normalized vectors
        self.index.add(vecs)
        self.chunks.extend(new_chunks)
        return len(new_chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        q = self._embed([query])
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]


def build_prompt(question: str, hits: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[{n}] ({c.source}, p.{c.page}) {c.text}" for n, (c, _) in enumerate(hits, start=1)
    )
    return (
        "Answer the question using only the context below. Cite sources like [1]. "
        "If the answer is not in the context, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


def extractive_answer(question: str, hits: list[tuple[Chunk, float]], model) -> str:
    """Fallback without an LLM: return the sentences most similar to the question."""
    sentences = []
    for n, (chunk, _) in enumerate(hits, start=1):
        for s in re.split(r"(?<=[.!?])\s+", chunk.text):
            if len(s) > 30:
                sentences.append((s, n))
    if not sentences:
        return "I couldn't find anything relevant in the uploaded documents."
    q = model.encode([question], normalize_embeddings=True)
    s_vecs = model.encode([s for s, _ in sentences], normalize_embeddings=True)
    top = np.argsort(-(s_vecs @ q.T).ravel())[:3]
    return " ".join(f"{sentences[i][0]} [{sentences[i][1]}]" for i in sorted(top))


def generate_answer(question: str, hits: list[tuple[Chunk, float]], model) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return extractive_answer(question, hits, model)
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=500,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    return msg.content[0].text
