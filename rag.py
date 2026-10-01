"""Core retrieval-augmented generation pipeline.

Pipeline: PDF -> text -> overlapping chunks -> sentence embeddings -> FAISS index.
At query time the question is embedded, the top-k chunks are retrieved, and an
answer is produced either by an LLM (if an API key is configured) or by an
extractive fallback that returns the best-matching sentences.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterator

import faiss
import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

load_dotenv()  # read API keys from a local .env file, if present

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
        text = re.sub(r"(\w)-\s*\n\s*([a-z])", r"\1\2", text)  # rejoin words hyphenated across lines
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append((i, text))
    return pages


def load_chunks(path: str, source_name: str | None = None) -> list[Chunk]:
    source = source_name or os.path.basename(path)
    return [
        Chunk(text=c, source=source, page=page_no)
        for page_no, page_text in extract_pages(path)
        for c in chunk_text(page_text)
    ]


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
        new_chunks = load_chunks(path, source_name)
        if not new_chunks:
            return 0
        self.add_chunks(new_chunks, self._embed([c.text for c in new_chunks]))
        return len(new_chunks)

    def add_chunks(self, new_chunks: list[Chunk], vecs: np.ndarray) -> None:
        """Add chunks with precomputed embeddings (e.g. computed in another process)."""
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])  # cosine sim on normalized vectors
        self.index.add(vecs)
        self.chunks.extend(new_chunks)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []
        if is_overview(query):
            return self.overview(k)
        q = self._embed([query])
        scores, ids = self.index.search(q, min(k, len(self.chunks)))
        return [(self.chunks[i], float(s)) for s, i in zip(scores[0], ids[0]) if i != -1]

    def overview(self, k: int = 4) -> list[tuple[Chunk, float]]:
        """Most representative chunks for summary questions: closest to their
        document's centroid, with a boost for the opening (abstract/intro)."""
        vecs = self.index.reconstruct_n(0, self.index.ntotal)
        scored = []
        for doc in dict.fromkeys(c.source for c in self.chunks):
            ids = [i for i, c in enumerate(self.chunks) if c.source == doc]
            centroid = vecs[ids].mean(axis=0)
            centroid /= np.linalg.norm(centroid) or 1.0
            for pos, i in enumerate(ids):
                scored.append((float(vecs[i] @ centroid) + (0.15 if pos < 2 else 0.0), i))
        scored.sort(reverse=True)
        return [(self.chunks[i], s) for s, i in scored[:k]]

    def clear(self) -> None:
        self.chunks = []
        self.index = None


OVERVIEW_RE = re.compile(
    r"\b(summar\w*|overview|tl;?dr|gist|main (idea|point|topic)s?|key (points|takeaways)"
    r"|what (is|'s) (this|the) (doc|document|paper|file|pdf)\w*( about)?|abstract)\b",
    re.IGNORECASE,
)


def is_overview(question: str) -> bool:
    return bool(OVERVIEW_RE.search(question))


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
    sentences, seen = [], set()
    for n, (chunk, _) in enumerate(hits, start=1):
        for s in re.split(r"(?<=[.!?])\s+|\s*[•▪●]\s*", chunk.text):
            s = s.strip(" -–—")
            if _is_prose(s) and s.lower() not in seen:
                seen.add(s.lower())
                sentences.append((s, n))
    if not sentences:
        return "I couldn't find anything relevant in the uploaded documents."
    s_vecs = model.encode([s for s, _ in sentences], normalize_embeddings=True)
    if is_overview(question):  # most representative sentences rather than closest to the wording
        q = s_vecs.mean(axis=0, keepdims=True)
    else:
        q = model.encode([question], normalize_embeddings=True)
    top = np.argsort(-(s_vecs @ q.T).ravel())[:3]
    return "\n".join(f"- {sentences[i][0]} [{sentences[i][1]}]" for i in sorted(top))


def _is_prose(s: str) -> bool:
    """Skip tables, table-of-contents lines, captions and mid-sentence fragments."""
    if not 40 <= len(s) <= 350 or len(s.split()) < 6:
        return False
    if not s[0].isupper() or re.search(r"\.{4,}|…{2,}|_{4,}|±", s):
        return False
    if re.match(r"(table|fig(ure)?|algorithm)\b", s, re.IGNORECASE):
        return False
    letters = [ch for ch in s if ch.isalpha()]
    digits = sum(ch.isdigit() for ch in s)
    upper = sum(ch.isupper() for ch in letters)
    return len(letters) / len(s) > 0.65 and digits / len(s) < 0.1 and upper / max(len(letters), 1) < 0.3


def answer_mode() -> str:
    """Name of the backend that will write answers, based on configured keys."""
    if os.getenv("ANTHROPIC_API_KEY"):
        return "Claude Haiku"
    if os.getenv("HF_TOKEN"):
        return os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct").split("/")[-1]
    return "Offline extractive"


def stream_answer(question: str, hits: list[tuple[Chunk, float]], model) -> Iterator[str]:
    """Yield the answer in pieces. Uses Claude if ANTHROPIC_API_KEY is set, else a
    free Hugging Face model if HF_TOKEN is set, else the offline extractive fallback."""
    prompt = build_prompt(question, hits)
    if os.getenv("ANTHROPIC_API_KEY"):
        import anthropic

        client = anthropic.Anthropic()
        with client.messages.stream(
            model=os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001"),
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream
        return
    if os.getenv("HF_TOKEN"):
        from huggingface_hub import InferenceClient

        client = InferenceClient(token=os.getenv("HF_TOKEN"))
        started = False
        try:
            for event in client.chat_completion(
                model=os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct"),
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                stream=True,
            ):
                piece = event.choices[0].delta.content if event.choices else None
                if piece:
                    started = True
                    yield piece
            return
        except Exception as e:  # quota used up or model unavailable
            print(f"[rag] Hugging Face inference failed, using offline answer: {e}", file=sys.stderr)
            if started:
                return
    yield extractive_answer(question, hits, model)


def generate_answer(question: str, hits: list[tuple[Chunk, float]], model) -> str:
    return "".join(stream_answer(question, hits, model))
