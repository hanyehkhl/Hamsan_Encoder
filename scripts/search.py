"""جستجوی معنایی در یک فایل متنی با همسان.

    uv run python scripts/search.py data/examples/faq.txt                  # حالت پرسش‌وپاسخ
    uv run python scripts/search.py data/examples/faq.txt -q "پولم کی برمی‌گرده؟"
    uv run python scripts/search.py notes.txt --model models/hamsan-v4/final --top 5

فایل ورودی: هر پاراگراف با یک خط خالی از بعدی جدا می‌شود.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stdin.reconfigure(encoding="utf-8")

import argparse
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hamsan-v6-merge-0.7"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    parser.add_argument("-q", "--query", nargs="+", help="بدون این، سؤال‌ها را یکی‌یکی می‌پرسد")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()

    text = Path(args.file).read_text(encoding="utf-8")
    docs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    model = SentenceTransformer(args.model)
    # متن‌ها یک بار به بردار تبدیل می‌شوند؛ هر سؤال فقط با آن‌ها مقایسه می‌شود
    doc_emb = model.encode(docs, prompt_name="document", normalize_embeddings=True)
    print(f"{len(docs)} پاراگراف بارگذاری شد.")

    def answer(query: str) -> None:
        q_emb = model.encode([query], prompt_name="query", normalize_embeddings=True)
        scores = model.similarity(q_emb, doc_emb)[0]
        print(f"\n❓ {query}")
        for rank, i in enumerate(scores.argsort(descending=True)[:args.top].tolist(), 1):
            print(f"  {rank}. ({scores[i]:.2f}) {docs[i]}")

    if args.query:
        for q in args.query:
            answer(q)
        return
    print("سؤالتان را بنویسید (خط خالی = خروج):")
    while query := input("\n> ").strip():
        answer(query)


if __name__ == "__main__":
    main()
