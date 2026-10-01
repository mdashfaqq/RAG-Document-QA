"""FastAPI server for the RAG document Q&A system.

Endpoints:
  POST /upload   - upload a PDF; it is chunked, embedded and indexed
  POST /ask      - ask a question; returns an answer plus the source chunks
  GET  /documents- list indexed documents and chunk counts
  GET  /         - web UI (static/index.html)
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from rag import DocumentIndex, generate_answer

app = FastAPI(title="RAG Document Q&A")
store = DocumentIndex()
INDEX_HTML = Path(__file__).parent / "static" / "index.html"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3)
    top_k: int = Field(4, ge=1, le=10)


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        added = store.add_pdf(path, source_name=file.filename)
    finally:
        os.remove(path)
    if added == 0:
        raise HTTPException(422, "No extractable text found (scanned PDF?)")
    return {"document": file.filename, "chunks_indexed": added}


@app.post("/ask")
def ask(req: AskRequest):
    hits = store.search(req.question, k=req.top_k)
    if not hits:
        raise HTTPException(400, "No documents indexed yet. Upload a PDF first.")
    answer = generate_answer(req.question, hits, store.model)
    return {
        "answer": answer,
        "sources": [
            {"id": n, "source": c.source, "page": c.page, "score": round(s, 3), "text": c.text[:300]}
            for n, (c, s) in enumerate(hits, start=1)
        ],
    }


@app.get("/documents")
def documents():
    counts = Counter(c.source for c in store.chunks)
    return [{"document": d, "chunks": n} for d, n in counts.items()]


@app.get("/", include_in_schema=False)
def ui():
    return FileResponse(INDEX_HTML)
