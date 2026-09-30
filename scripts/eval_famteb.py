"""سنجش روی FaMTEB (نسخهٔ فارسی MTEB؛ همان جدولی که در هاگینگ‌فیس دیده می‌شود).

    uv run python scripts/eval_famteb.py intfloat/multilingual-e5-base
    uv run python scripts/eval_famteb.py models/hamsan-v3/final
    uv run python scripts/eval_famteb.py models/hamsan-v3/final --full        # همهٔ Retrieval و STS
    uv run python scripts/eval_famteb.py models/hamsan-v3/final --tasks Farsick SynPerSTS
    uv run python scripts/eval_famteb.py --compare intfloat/multilingual-e5-base models/hamsan-v3/final

پیش‌فرض زیرمجموعهٔ سبک LITE_TASKS است (۸ Retrieval + ۲ STS)؛ کل بنچمارک روی لپ‌تاپ
چند ساعت برای هر مدل طول می‌کشد.
نتیجهٔ هر مدل در results/famteb/<مدل>/ ذخیره می‌شود و اجرای دوباره فقط دیتاست‌های
ناتمام را اجرا می‌کند.

توجه: بخش train از synthetic-persian-sts در آموزش بوده؛ SynPerSTS بخش test همان را
می‌سنجد، پس عددش کمی خوش‌بینانه است. (ParsiNLU از v4 به بعد در آموزش نیست.)
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import sys

sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import logging
import warnings
from pathlib import Path

# mteb برای تسک‌های ویدئویی/صوتی که ربطی به ما ندارند ده‌ها هشدار «beta» چاپ می‌کند
warnings.filterwarnings("ignore", message=".*currently in beta.*")
logging.getLogger("mteb").setLevel(logging.ERROR)

import mteb
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "results" / "famteb"
BENCHMARK = "MTEB(fas, v2)"

# پیشوندهای E5 برای هر دو مدل یکسان: سند = passage، بقیه (سؤال، STS، دسته‌بندی...) = query
PROMPTS = {
    "query": "query: ",
    "document": "passage: ",
    "STS": "query: ",
    "PairClassification": "query: ",
    "Classification": "query: ",
    "Clustering": "query: ",
}


# زیرمجموعه‌ای که روی لپ‌تاپ (RAM ۱۶ گیگ، GPU شش‌گیگی) برای هر مدل حدود نیم ساعت طول می‌کشد.
# دیتاست‌های چندصدهزارتایی (SynPerQA، PersianWebDocument، MIRACL، WebFAQ، Quora، TRECCOVID،
# HotpotQA/NQ/FEVER) کنار مانده‌اند: SynPerQA به‌تنهایی ۸+ گیگ RAM گرفت و سیستم گیر کرد.
# آن‌ها را با --full یا یکی‌یکی با --tasks اجرا کنید.
LITE_TASKS = [
    "SynPerChatbotRAGFAQRetrieval",     # ۱۰ هزار
    "WikipediaRetrievalMultilingual",   # ۱۵ هزار (بخش فارسی)
    "NeuCLIR2023RetrievalHardNegatives",  # ۱۶ هزار، خبر فارسی
    "MSMARCO-FaHardNegatives",          # ۹ هزار
    "ArguAna-Fa.v2",                    # ۱۰ هزار
    "FiQA2018-Fa.v2",                   # ۵۸ هزار
    "SCIDOCS-Fa.v2",                    # ۲۷ هزار
    "SciFact-Fa.v2",                    # ۵ هزار
    "Farsick",                          # STS
    "SynPerSTS",                        # STS
]


def slug(model: str) -> str:
    return model.replace("\\", "/").strip("/").replace("/", "__")


def pick_tasks(types: list[str] | None, names: list[str] | None) -> list:
    tasks = mteb.get_benchmark(BENCHMARK).tasks
    if names:
        return [t for t in tasks if t.metadata.name in names]
    if types:
        return [t for t in tasks if t.metadata.type in types]
    return [t for t in tasks if t.metadata.name in LITE_TASKS]


def run(model_name: str, tasks: list, batch_size: int) -> dict[str, float]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st = SentenceTransformer(model_name, device=device)
    st.max_seq_length = 512
    model = mteb.SentenceTransformerEncoderWrapper(st, model_prompts=PROMPTS)
    results = mteb.evaluate(
        model, tasks,
        cache=mteb.ResultCache(OUT_DIR / slug(model_name)),
        encode_kwargs={"batch_size": batch_size},
        raise_error=False,
    )
    scores = {r.task_name: r.main_score for r in results}
    (OUT_DIR / f"{slug(model_name)}.json").write_text(
        json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
    return scores


def load_scores(model_name: str) -> dict[str, float]:
    path = OUT_DIR / f"{slug(model_name)}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def print_table(models: list[str], tasks: list) -> None:
    scores = {m: load_scores(m) for m in models}
    names = [m.replace("\\", "/").rstrip("/").split("/")[-2 if m.rstrip("/\\").endswith("final") else -1]
             for m in models]
    by_type: dict[str, list[str]] = {}
    for t in tasks:
        by_type.setdefault(t.metadata.type, []).append(t.metadata.name)

    print(f"\n{'دیتاست':40s}" + "".join(f"{n:>22s}" for n in names))
    for task_type, task_names in by_type.items():
        print(f"--- {task_type}")
        for name in task_names:
            row = [scores[m].get(name) for m in models]
            print(f"{name:40s}" + "".join(f"{v:>22.4f}" if v is not None else f"{'-':>22s}" for v in row))
        # میانگین فقط روی دیتاست‌هایی که همهٔ مدل‌ها دارند، تا مقایسه منصفانه باشد
        common = [n for n in task_names if all(n in scores[m] for m in models)]
        if common:
            means = [sum(scores[m][n] for n in common) / len(common) for m in models]
            print(f"{'میانگین (' + str(len(common)) + ' دیتاست)':40s}" + "".join(f"{v:>22.4f}" for v in means))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="+")
    parser.add_argument("--full", action="store_true",
                        help="همهٔ دیتاست‌های --types به‌جای زیرمجموعهٔ سبک (چند ساعت و RAM زیاد)")
    parser.add_argument("--types", nargs="+", default=["Retrieval", "STS"],
                        help="با --full: Retrieval STS PairClassification Reranking Classification Clustering ...")
    parser.add_argument("--tasks", nargs="+", help="نام دیتاست‌ها؛ بر بقیه اولویت دارد")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--compare", action="store_true", help="فقط جدول نتایج ذخیره‌شده، بدون اجرا")
    args = parser.parse_args()

    tasks = pick_tasks(args.types if args.full else None, args.tasks)
    print(f"{len(tasks)} دیتاست از {BENCHMARK}: {', '.join(t.metadata.name for t in tasks)}")
    if not args.compare:
        for m in args.models:
            print(f"\n=== {m}")
            run(m, tasks, args.batch_size)
    print_table(args.models, tasks)


if __name__ == "__main__":
    main()
