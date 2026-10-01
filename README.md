---
title: RAG Document QA
emoji: 📄
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.35.0
python_version: "3.10"
app_file: space_app.py
pinned: false
---

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

To use an LLM for generated answers, set an API key before starting. Either a free Hugging Face token (`HF_TOKEN`, from huggingface.co/settings/tokens) or an Anthropic key works; Anthropic is used if both are set:

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

## Deployment

The repo is set up for [Hugging Face Spaces](https://huggingface.co/spaces) on the free tier. The Space runs `space_app.py`, a Gradio UI over the same pipeline as the FastAPI server.

1. Create a new Space with the **Gradio** SDK (Blank template) and **ZeroGPU** hardware.
2. Push this repo to it:
   ```bash
   git remote add space https://huggingface.co/spaces/mdashfaqq/RAG_QA
   git push space main
   ```
3. Optional: add `ANTHROPIC_API_KEY` under **Settings → Variables and secrets** for LLM answers.

A `Dockerfile` is also included for running the FastAPI server on Render, Railway or Fly.io. Serverless platforms like Vercel aren't a good fit: PyTorch exceeds their size limits and the in-memory index doesn't persist between requests.

## Limitations and next steps

- The index lives in memory, so it resets on restart. Persisting it with `faiss.write_index` is a natural next step.
- Scanned PDFs have no extractable text and would need OCR.
- Adding a cross-encoder re-ranker would improve retrieval precision.

## Tech stack

Python, FastAPI, sentence-transformers, FAISS, pypdf, Anthropic API (optional)
