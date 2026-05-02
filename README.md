# RAG Document Q&A

Ask natural-language questions over your own PDF documents. The app uses **retrieval-augmented generation (RAG)**: documents are split into chunks, embedded with a sentence-transformer model, and stored in a FAISS vector index. Each question retrieves the most relevant chunks, and an answer is generated from them with source citations.

## Features

- Upload multiple PDFs through a REST API or the built-in web UI
- Sentence-aware chunking with overlap so answers aren't cut off at chunk boundaries
- Cosine-similarity search using normalized embeddings in a FAISS `IndexFlatIP`
- LLM answers with citations when an API key is set, and an extractive fallback (best-matching sentences) when it isn't, so the app works fully offline
- Every answer returns its sources: document name, page number and similarity score

## Architecture

```
PDF ──► pypdf text extraction ──► chunker (800 chars, 150 overlap)
                                       │
                                       ▼
                      sentence-transformers (all-MiniLM-L6-v2)
                                       │
                                       ▼
                              FAISS vector index
                                       ▲
question ──► embed ──► top-k search ───┘──► LLM / extractive answer + citations
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Open http://127.0.0.1:8000 for the UI, or http://127.0.0.1:8000/docs for the interactive API docs.

To use an LLM for generated answers, set an API key before starting:

```bash
export ANTHROPIC_API_KEY=your_key_here
```

## API

| Method | Endpoint     | Description                               |
|--------|--------------|-------------------------------------------|
| POST   | `/upload`    | Upload a PDF (multipart `file`)           |
| POST   | `/ask`       | `{"question": "...", "top_k": 4}`         |
| GET    | `/documents` | List indexed documents and chunk counts   |

Example:

```bash
curl -F "file=@paper.pdf" http://127.0.0.1:8000/upload
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What dataset did the authors use?"}'
```

## Limitations and next steps

- The index lives in memory, so it resets on restart. Persisting it with `faiss.write_index` is a natural next step.
- Scanned PDFs have no extractable text and would need OCR.
- Adding a cross-encoder re-ranker would improve retrieval precision.

## Tech stack

Python, FastAPI, sentence-transformers, FAISS, pypdf, Anthropic API (optional)
