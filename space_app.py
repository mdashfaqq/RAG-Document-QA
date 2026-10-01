"""Gradio UI for the RAG document Q&A system, used for the Hugging Face Space.

Wraps the same pipeline as the FastAPI server in app.py: upload PDFs, then ask
questions in a chat and get streamed answers with cited source chunks.
"""

from __future__ import annotations

import html
import os
import time

try:  # ZeroGPU hardware; must be imported before torch
    import spaces

    gpu = spaces.GPU
except ImportError:  # running locally
    def gpu(fn):
        return fn

import gradio as gr

from rag import DocumentIndex, answer_mode, load_chunks, stream_answer

store = DocumentIndex()
last_latency_ms: float | None = None

EXAMPLES = [
    "Summarize this document in 3 bullet points",
    "What are the key skills or methods?",
    "What are the main conclusions?",
]

THEME = gr.themes.Base(
    primary_hue="violet",
    secondary_hue="fuchsia",
    neutral_hue="slate",
    radius_size="lg",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
).set(
    body_background_fill_dark="#07070d",
    background_fill_primary_dark="rgba(255,255,255,0.03)",
    background_fill_secondary_dark="rgba(255,255,255,0.05)",
    block_background_fill_dark="rgba(255,255,255,0.035)",
    block_border_color_dark="rgba(255,255,255,0.08)",
    border_color_primary_dark="rgba(255,255,255,0.08)",
    block_label_background_fill_dark="transparent",
    input_background_fill_dark="rgba(255,255,255,0.05)",
    input_border_color_dark="rgba(255,255,255,0.10)",
    input_border_color_focus_dark="*primary_400",
    button_primary_background_fill_dark="linear-gradient(135deg, *primary_500, *secondary_500)",
    button_primary_background_fill_hover_dark="linear-gradient(135deg, *primary_400, *secondary_400)",
    button_primary_text_color_dark="white",
    button_secondary_background_fill_dark="rgba(255,255,255,0.06)",
    button_secondary_background_fill_hover_dark="rgba(255,255,255,0.10)",
    button_secondary_border_color_dark="rgba(255,255,255,0.10)",
    block_shadow_dark="0 10px 40px -12px rgba(0,0,0,0.6)",
)

CSS = """
body { background:
  radial-gradient(900px 500px at 10% -10%, rgba(139,92,246,.22), transparent 60%),
  radial-gradient(800px 500px at 100% 0%, rgba(217,70,239,.16), transparent 60%),
  radial-gradient(700px 600px at 50% 120%, rgba(56,189,248,.10), transparent 60%),
  #07070d !important; }
.gradio-container { max-width: 1240px !important; margin: 0 auto !important; background: transparent !important; }
footer { display: none !important; }

#nav { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 6px 4px 2px; }
#nav .brand { display: flex; align-items: center; gap: 12px; }
#nav .logo { width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center; font-size: 19px; color: white;
  background: linear-gradient(135deg, #8b5cf6, #d946ef); box-shadow: 0 8px 24px -6px rgba(168,85,247,.7); }
#nav .title { font-weight: 700; font-size: 1.15rem; letter-spacing: -.01em; }
#nav .tag { font-size: .78rem; opacity: .55; }
#nav .badge { display: flex; align-items: center; gap: 8px; font-size: .8rem; padding: 7px 12px; border-radius: 999px;
  background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10); white-space: nowrap; }
#nav .dot { width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 10px #22c55e; }
#nav .dot.off { background: #f59e0b; box-shadow: 0 0 10px #f59e0b; }

#hero { padding: 26px 4px 6px; }
#hero h1 { margin: 0; font-size: clamp(1.8rem, 4vw, 2.7rem); font-weight: 800; letter-spacing: -.035em; line-height: 1.1;
  background: linear-gradient(90deg, #fff 0%, #e9d5ff 40%, #f0abfc 100%); -webkit-background-clip: text; background-clip: text; color: transparent; }
#hero p { margin: 10px 0 0; opacity: .6; font-size: 1rem; max-width: 640px; }

#stats { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 14px 0 4px; }
@media (max-width: 720px) { #stats { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
#stats .stat { padding: 14px 16px; border-radius: 14px; background: rgba(255,255,255,.035); border: 1px solid rgba(255,255,255,.08); }
#stats .label { font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; opacity: .5; }
#stats .value { font-size: 1.5rem; font-weight: 700; margin-top: 4px; font-variant-numeric: tabular-nums; }
#stats .value small { font-size: .8rem; opacity: .5; font-weight: 500; margin-left: 2px; }

.glass { backdrop-filter: blur(14px); border-radius: 18px !important; }
.section-title { font-weight: 600; font-size: .95rem; margin: 2px 0 2px; }
.section-sub { opacity: .5; font-size: .82rem; }

#library-list { display: flex; flex-direction: column; gap: 8px; }
#library-list .doc { display: flex; align-items: center; gap: 10px; padding: 10px 12px; border-radius: 12px;
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); font-size: .88rem; }
#library-list .ico { width: 30px; height: 30px; border-radius: 8px; display: grid; place-items: center; flex: none;
  background: linear-gradient(135deg, rgba(139,92,246,.35), rgba(217,70,239,.25)); font-size: 14px; }
#library-list .name { flex: 1; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; font-weight: 500; }
#library-list .meta { opacity: .5; font-size: .76rem; white-space: nowrap; }
#library-list .empty { opacity: .45; font-size: .86rem; padding: 14px; text-align: center;
  border: 1px dashed rgba(255,255,255,.12); border-radius: 12px; }
#library-list .doc.pending { border-color: rgba(168,85,247,.45); background: rgba(139,92,246,.08); }
#library-list .doc.pending .meta { opacity: .8; color: #c4b5fd; }
#library-list .spin { width: 16px; height: 16px; border-radius: 50%; border: 2px solid rgba(196,181,253,.35);
  border-top-color: #c4b5fd; animation: docspin .7s linear infinite; }
@keyframes docspin { to { transform: rotate(360deg); } }

#chat { height: 560px !important; }
#chat .message { font-size: .95rem; line-height: 1.6; }
#credits { text-align: center; opacity: .35; font-size: .78rem; margin: 10px 0 4px; }
"""

FORCE_DARK = """
() => {
  if (!document.body.classList.contains('dark')) document.body.classList.add('dark');
}
"""


def nav_html() -> str:
    mode = answer_mode()
    offline = mode == "Offline extractive"
    return f"""
<div id="nav">
  <div class="brand">
    <div class="logo">✦</div>
    <div><div class="title">DocMind</div><div class="tag">Retrieval-augmented document Q&amp;A</div></div>
  </div>
  <div class="badge"><span class="dot{' off' if offline else ''}"></span>{html.escape(mode)}</div>
</div>
<div id="hero">
  <h1>Chat with your documents.</h1>
  <p>Drop in PDFs and ask anything. Answers stream in real time, grounded in your files, with page-level citations you can verify.</p>
</div>"""


def stats_html() -> str:
    docs = {c.source for c in store.chunks}
    pages = {(c.source, c.page) for c in store.chunks}
    latency = f"{last_latency_ms:.0f}<small>ms</small>" if last_latency_ms is not None else "—"
    tiles = [("Documents", len(docs)), ("Pages", len(pages)), ("Chunks", len(store.chunks)), ("Last search", latency)]
    cells = "".join(f'<div class="stat"><div class="label">{k}</div><div class="value">{v}</div></div>' for k, v in tiles)
    return f'<div id="stats">{cells}</div>'


def library_html(pending: list[str] | None = None) -> str:
    by_doc: dict[str, tuple[int, set[int]]] = {}
    for c in store.chunks:
        n, pages = by_doc.get(c.source, (0, set()))
        pages.add(c.page)
        by_doc[c.source] = (n + 1, pages)
    pending_rows = "".join(
        f'<div class="doc pending"><div class="ico"><div class="spin"></div></div>'
        f'<div class="name" title="{html.escape(name)}">{html.escape(name)}</div><div class="meta">Indexing…</div></div>'
        for name in pending or []
    )
    if not by_doc and not pending_rows:
        return '<div id="library-list"><div class="empty">Your library is empty.<br>Upload a PDF to begin.</div></div>'
    rows = pending_rows + "".join(
        f'<div class="doc"><div class="ico">📄</div><div class="name" title="{html.escape(name)}">{html.escape(name)}</div>'
        f'<div class="meta">{len(pages)} pg · {n} chunks</div></div>'
        for name, (n, pages) in by_doc.items()
    )
    return f'<div id="library-list">{rows}</div>'


def show_pending(files: list[str] | None):
    return library_html(pending=[os.path.basename(p) for p in files or []])


@gpu
def embed(texts: list[str]):
    # On ZeroGPU this runs in a separate process, so it must only compute and return;
    # changes to `store` made here would be lost.
    return store._embed(texts)


def index_files(files: list[str] | None):
    for path in files or []:
        name = os.path.basename(path)
        chunks = load_chunks(path, name)
        if chunks:
            store.add_chunks(chunks, embed([c.text for c in chunks]))
            gr.Info(f"Indexed {name}")
        else:
            gr.Warning(f"{name}: no extractable text (scanned PDF?)")
    return library_html(), stats_html(), None


def clear_library():
    global last_latency_ms
    store.clear()
    last_latency_ms = None
    return library_html(), stats_html(), []


def respond(question: str, history: list[dict], top_k: int):
    global last_latency_ms
    question = (question or "").strip()
    if len(question) < 3:
        yield "", history, stats_html()
        return
    history = history + [{"role": "user", "content": question}]
    if not store.chunks:
        history.append({"role": "assistant", "content": "Your library is empty. Upload a PDF on the left and it will be indexed automatically."})
        yield "", history, stats_html()
        return

    start = time.perf_counter()
    hits = store.search(question, k=int(top_k))
    last_latency_ms = (time.perf_counter() - start) * 1000

    sources = "\n\n".join(
        f"**[{n}] {c.source} · p.{c.page}** — relevance `{s:.2f}`\n\n> {c.text[:260].strip()}…"
        for n, (c, s) in enumerate(hits, start=1)
    )
    history.append({
        "role": "assistant",
        "content": sources,
        "metadata": {"title": f"Retrieved {len(hits)} passages in {last_latency_ms:.0f} ms", "status": "done"},
    })
    history.append({"role": "assistant", "content": ""})
    yield "", history, stats_html()

    answer = ""
    for piece in stream_answer(question, hits, store.model):
        answer += piece
        history[-1] = {"role": "assistant", "content": answer}
        yield "", history, stats_html()
    if answer_mode() == "Offline extractive":
        history[-1]["content"] += (
            "\n\n<sub>Offline mode: these are the most relevant sentences, not a written answer. "
            "Set `HF_TOKEN` for AI-generated answers.</sub>"
        )
        yield "", history, stats_html()


with gr.Blocks(title="DocMind · Document Q&A", theme=THEME, css=CSS, js=FORCE_DARK) as demo:
    gr.HTML(nav_html())
    stats = gr.HTML(stats_html())

    with gr.Row(equal_height=False):
        with gr.Column(scale=4, min_width=300):
            with gr.Group(elem_classes="glass"):
                gr.HTML('<div style="padding:14px 14px 4px"><div class="section-title">Library</div>'
                        '<div class="section-sub">PDFs are indexed the moment you upload them.</div></div>')
                files = gr.File(label="Drop PDFs here", file_types=[".pdf"], file_count="multiple", type="filepath", height=150)
            library = gr.HTML(library_html())
            with gr.Accordion("Retrieval settings", open=False):
                top_k = gr.Slider(1, 10, value=4, step=1, label="Passages per answer",
                                  info="More passages give the model more context but can add noise.")
            clear_lib = gr.Button("Clear library", variant="secondary", size="sm")

        with gr.Column(scale=8):
            chat = gr.Chatbot(
                elem_id="chat",
                elem_classes="glass",
                type="messages",
                show_label=False,
                show_copy_button=True,
                placeholder=(
                    "<div style='text-align:center;opacity:.55;padding:40px 10px'>"
                    "<div style='font-size:2.2rem'>✦</div>"
                    "<h3 style='margin:8px 0 4px'>Ask anything about your documents</h3>"
                    "Every answer cites the document and page it came from.</div>"
                ),
            )
            with gr.Row():
                question = gr.Textbox(placeholder="Ask a question…", show_label=False, container=False, scale=9, autofocus=True, lines=1, max_lines=4)
                send = gr.Button("Send ↑", variant="primary", scale=1, min_width=96)
            gr.Examples(EXAMPLES, inputs=question, label="Suggestions")
            clear_chat = gr.Button("New chat", variant="secondary", size="sm")

    gr.HTML('<div id="credits">sentence-transformers · FAISS · Gradio</div>')

    files.upload(show_pending, files, library, show_progress="hidden").then(
        index_files, files, [library, stats, files], show_progress="hidden"
    )
    clear_lib.click(clear_library, None, [library, stats, chat])
    send.click(respond, [question, chat, top_k], [question, chat, stats])
    question.submit(respond, [question, chat, top_k], [question, chat, stats])
    clear_chat.click(lambda: [], None, chat)

if __name__ == "__main__":
    demo.launch()
