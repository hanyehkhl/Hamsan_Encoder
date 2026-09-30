"""پرسیدن سؤال از پیکرهٔ کوتلاس و مقایسهٔ مدل پایه با مدل آموزش‌دیده.

    uv run python scripts/ask.py                                  # سؤال‌های آزمون تازهٔ پایین
    uv run python scripts/ask.py "چه کسی هیولا را ساخت؟"         # سؤال دلخواه

سؤال‌های TEST_QUESTIONS با کلمه‌های تازه نوشته شده‌اند و در cuttlas_qa.jsonl نیستند.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse

from hazm import Normalizer
from sentence_transformers import SentenceTransformer

from evaluate import ROOT, load_corpus, prefixes

MODELS = ["intfloat/multilingual-e5-base", str(ROOT / "models" / "hamsan-v1" / "final")]
TOP_K = 3

# (سؤال، سندهای درست)
TEST_QUESTIONS = [
    ("کوتلاس اول داستان کجا کار می‌کرد؟", ["sum_p0"]),
    ("قاتل برای به دست آوردن کدام فیلم آدم کشت؟", ["sum_p0"]),
    ("مقتول چه کسی بود؟", ["sum_p0"]),
    ("جک در کدام خیابان خانه دارد؟", ["sum_p1"]),
    ("جک فکر می‌کند مل در زندگی قبلی چه نسبتی با او داشته؟", ["sum_p1"]),
    ("چرا کوتلاس به خانهٔ جک رفت؟", ["sum_p1"]),
    ("اسم موجودی که جک ساخت چیست؟", ["sum_p2"]),
    ("رعد و برق در ساختن هیولا چه نقشی داشت؟", ["sum_p2"]),
    ("جک چه آزمایشی انجام داد؟", ["sum_p2"]),
    ("جک مل را دزدید تا چه کار کند؟", ["sum_p3"]),
    ("با آتش کدام هیولا را می‌شود شکست داد؟", ["sum_p3"]),
    ("کوتلاس از چه چیزی برای جنگیدن با هیولا کمک گرفت؟", ["sum_p3"]),
    ("میخ چوبی به درد چه می‌خورد؟", ["sum_p3"]),
    ("سرانجام جک ویلسون چه شد؟", ["sum_p4"]),
    ("آیا کوتلاس توانست هیولا را شکست دهد؟", ["sum_p4"]),
]


def search(model: SentenceTransformer, name: str, corpus: dict[str, str],
           questions: list[str]) -> list[list[tuple[str, float]]]:
    q_pre, p_pre = prefixes(model, name)
    norm = Normalizer().normalize
    ids = list(corpus)
    doc_emb = model.encode([p_pre + norm(corpus[i]) for i in ids],
                           convert_to_tensor=True, normalize_embeddings=True)
    q_emb = model.encode([q_pre + norm(q) for q in questions],
                         convert_to_tensor=True, normalize_embeddings=True)
    scores = q_emb @ doc_emb.T
    ranked = scores.argsort(dim=1, descending=True)
    return [[(ids[j], scores[i, j].item()) for j in ranked[i].tolist()]
            for i in range(len(questions))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("questions", nargs="*")
    args = parser.parse_args()

    tests = [(q, []) for q in args.questions] or TEST_QUESTIONS
    questions = [q for q, _ in tests]
    corpus = load_corpus()

    results = {}
    for name in MODELS:
        results[name] = search(SentenceTransformer(name), name, corpus, questions)

    hits = {name: 0 for name in MODELS}
    for i, (q, gold) in enumerate(tests):
        print(f"\n❓ {q}" + (f"   (درست: {', '.join(gold)})" if gold else ""))
        for name in MODELS:
            ranking = results[name][i]
            if gold:
                rank = next(r for r, (d, _) in enumerate(ranking, 1) if d in gold)
                hits[name] += rank == 1
                mark = "✅" if rank == 1 else f"❌ رتبهٔ درست={rank}"
            else:
                mark = ""
            top = "  ".join(f"{d}({s:.2f})" for d, s in ranking[:TOP_K])
            print(f"   {name.replace(chr(92), '/').split('/')[-2 if name.endswith('final') else -1]:<24} {mark}  {top}")
        if not gold:
            best = results[MODELS[-1]][i][0][0]
            print(f"   ↳ {corpus[best][:200]}")

    if all(gold for _, gold in tests):
        print("\nرتبهٔ اول درست:")
        for name in MODELS:
            print(f"   {name.replace(chr(92), '/').split('/')[-2 if name.endswith('final') else -1]:<24} {hits[name]}/{len(tests)}")


if __name__ == "__main__":
    main()
