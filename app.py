"""FastAPI server for the RAG document Q&A system.

Endpoints:
  POST /upload   - upload a PDF; it is chunked, embedded and indexed
  POST /ask      - ask a question; returns an answer plus the source chunks
  GET  /documents- list indexed documents and chunk counts
  GET  /         - minimal web UI
"""

from __future__ import annotations

import os
import tempfile
from collections import Counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from rag import DocumentIndex, generate_answer

app = FastAPI(title="RAG Document Q&A")
store = DocumentIndex()


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


@app.get("/", response_class=HTMLResponse)
def ui():
    return """<!doctype html><html><head><meta charset="utf-8"><title>RAG Q&A</title>
<style>body{font-family:system-ui;max-width:760px;margin:40px auto;padding:0 16px}
input,button{padding:8px;margin:4px 0}#q{width:70%}.src{font-size:13px;color:#555;border-left:3px solid #ccc;padding-left:8px;margin:6px 0}</style></head>
<body><h2>Ask your documents</h2>
<input type="file" id="f" accept=".pdf"><button onclick="up()">Upload</button><div id="us"></div>
<hr><input id="q" placeholder="Ask a question..."><button onclick="ask()">Ask</button>
<div id="a"></div>
<script>
async function up(){const f=document.getElementById('f').files[0];if(!f)return;
const d=new FormData();d.append('file',f);const r=await fetch('/upload',{method:'POST',body:d});
document.getElementById('us').textContent=JSON.stringify(await r.json());}
async function ask(){const q=document.getElementById('q').value;
const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
const j=await r.json();const a=document.getElementById('a');
if(!r.ok){a.textContent=j.detail;return;}
a.innerHTML='<p><b>'+j.answer.replace(/</g,'&lt;')+'</b></p>'+j.sources.map(s=>
'<div class="src">['+s.id+'] '+s.source+' p.'+s.page+' (score '+s.score+')<br>'+s.text.replace(/</g,'&lt;')+'...</div>').join('');}
</script></body></html>"""
