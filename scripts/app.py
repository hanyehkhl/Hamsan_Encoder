"""رابط وب برای امتحان همسان در مرورگر، کنار مدل پایهٔ e5.

    uv run python scripts/app.py        # بعد http://127.0.0.1:7860 را باز کنید

دو بخش دارد:
  - جست‌وجو: چند پاراگراف بدهید و سؤال بپرسید؛ نتیجهٔ همسان و e5 کنار هم می‌آید.
  - شباهت: دو جمله بدهید و امتیاز شباهتشان را ببینید.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import re
from pathlib import Path

import gradio as gr
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
MODELS = {
    "همسان": str(ROOT / "models" / "hamsan-v6-merge-0.7"),
    "e5 پایه": "intfloat/multilingual-e5-base",
}
QUERY, PASSAGE = "query: ", "passage: "
RTL_CSS = "* { direction: rtl; text-align: right; }"
EXAMPLE_DOCS = (ROOT / "data" / "examples" / "faq.txt").read_text(encoding="utf-8")

models = {name: SentenceTransformer(path) for name, path in MODELS.items()}


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def search(docs_text: str, query: str, top_k: int) -> tuple[str, str]:
    docs = split_paragraphs(docs_text)
    if not docs or not query.strip():
        return "متن و سؤال را وارد کنید.", ""
    outputs = []
    for model in models.values():
        q = model.encode([QUERY + query], normalize_embeddings=True)
        d = model.encode([PASSAGE + p for p in docs], normalize_embeddings=True)
        scores = model.similarity(q, d)[0]
        ranked = scores.argsort(descending=True)[: int(top_k)]
        outputs.append("\n\n".join(
            f"**{rank}.** ({scores[i]:.2f}) {docs[i]}" for rank, i in enumerate(ranked.tolist(), 1)
        ))
    return outputs[0], outputs[1]


def similarity(a: str, b: str) -> str:
    if not a.strip() or not b.strip():
        return "دو جمله را وارد کنید."
    rows = []
    for name, model in models.items():
        emb = model.encode([QUERY + a, QUERY + b], normalize_embeddings=True)
        rows.append(f"| {name} | {float(model.similarity(emb[:1], emb[1:])[0][0]):.3f} |")
    return "| مدل | شباهت |\n|---|---|\n" + "\n".join(rows)


with gr.Blocks(title="همسان") as demo:
    gr.Markdown("# همسان\nمقایسهٔ مدل همسان با مدل پایهٔ e5")
    with gr.Tab("جست‌وجو"):
        docs_box = gr.Textbox(label="متن‌ها (پاراگراف‌ها را با یک خط خالی جدا کنید)",
                              value=EXAMPLE_DOCS, lines=12)
        query_box = gr.Textbox(label="سؤال", value="پولم کی برمی‌گرده؟")
        top_k = gr.Slider(1, 10, value=3, step=1, label="تعداد نتیجه")
        search_btn = gr.Button("جست‌وجو", variant="primary")
        with gr.Row():
            with gr.Column():
                gr.Markdown("### همسان")
                hamsan_out = gr.Markdown()
            with gr.Column():
                gr.Markdown("### e5 پایه")
                e5_out = gr.Markdown()
        search_btn.click(search, [docs_box, query_box, top_k], [hamsan_out, e5_out])
        query_box.submit(search, [docs_box, query_box, top_k], [hamsan_out, e5_out])
    with gr.Tab("شباهت دو جمله"):
        a_box = gr.Textbox(label="جملهٔ اول", value="هوا امروز خیلی گرمه.")
        b_box = gr.Textbox(label="جملهٔ دوم", value="دمای امروز بالاست.")
        sim_btn = gr.Button("مقایسه", variant="primary")
        sim_out = gr.Markdown()
        sim_btn.click(similarity, [a_box, b_box], sim_out)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=RTL_CSS)
