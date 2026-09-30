"""فاز ۳: Fine-tune کردن multilingual-e5-base روی جفت‌جمله‌های فارسی.

    uv run python scripts/train.py --max-pairs 5000 --epochs 1     # دور آزمایشی کوتاه (چند دقیقه)
    uv run python scripts/train.py                                  # آموزش کامل
    uv run python scripts/train.py --no-mined                       # بدون منفی سخت (برای مقایسه)

اگر data/train/negatives.jsonl باشد (scripts/mine_negatives.py)، جفت‌های سؤال → پاراگراف
با منفی سخت‌شان به سه‌تایی تبدیل می‌شوند.

تنظیم‌ها برای GPU شش‌گیگی (RTX 4050):
  - CachedMultipleNegativesRankingLoss: batch بزرگ (۱۲۸ = ۱۲۷ منفی برای هر جفت)
    ولی حافظه فقط به اندازهٔ mini-batch ۱۶تایی مصرف می‌شود.
  - MatryoshkaLoss: بردارهای کوتاه‌شده (۵۱۲/۲۵۶/۱۲۸ بُعد) هم قابل‌استفاده می‌مانند.
  - bf16 و طول ۲۵۶ توکن.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.training_args import BatchSamplers, MultiDatasetBatchSamplers

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train"
CUTTLAS_DIR = ROOT / "data" / "cuttlas"
EVAL_DIR = ROOT / "data" / "eval"

BASE_MODEL = "intfloat/multilingual-e5-base"
QUERY, PASSAGE = "query: ", "passage: "
MATRYOSHKA_DIMS = [768, 512, 256, 128]
SEED = 37
# مجوز غیرتجاری (CC BY-NC-SA). build_pairs.py دیگر آن را نمی‌سازد؛ این فیلتر برای pairs.jsonl
# قدیمی است و بعد از جدا کردن dev اعمال می‌شود تا dev با مدل‌های قبلی یکی بماند
EXCLUDED_SOURCES = {"parsinlu"}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def add_prefixes(row: dict) -> dict:
    """E5 با پیشوند آموزش دیده: سؤال/عنوان = query، متن = passage؛ جفت متقارن = هر دو query."""
    doc = PASSAGE if row["kind"] == "qa" else QUERY
    out = {"anchor": QUERY + row["anchor"], "positive": doc + row["positive"]}
    if "negative" in row:
        out["negative"] = doc + row["negative"]
    return out


def load_split(name: str, max_rows: int | None, dev_size: int) -> tuple[list[dict], list[dict]]:
    rows = read_jsonl(TRAIN_DIR / name)
    random.Random(SEED).shuffle(rows)
    if max_rows:
        rows = rows[:max_rows]
    return rows[dev_size:], rows[:dev_size]


def attach_mined_negatives(pairs: list[dict]) -> tuple[list[dict], list[dict]]:
    """جفت‌هایی که در negatives.jsonl منفی سخت دارند سه‌تایی می‌شوند (scripts/mine_negatives.py)."""
    path = TRAIN_DIR / "negatives.jsonl"
    if not path.exists():
        print("negatives.jsonl نیست؛ بدون منفی سخت آموزش می‌بیند")
        return pairs, []
    negatives = {(r["anchor"], r["positive"]): r["negative"] for r in read_jsonl(path)}
    rest, triplets = [], []
    for r in pairs:
        neg = negatives.get((r["anchor"], r["positive"]))
        if neg:
            triplets.append({**r, "negative": neg})
        else:
            rest.append(r)
    return rest, triplets


def group_batches(pairs: list[dict], triplets: list[dict], mixed: bool) -> dict[str, list[dict]]:
    """هر منبع یک دیتاست جدا ← هر batch فقط از یک منبع است (روش E5/BGE).

    در batch مخلوط، منفی‌های درون batch خیلی آسان‌اند (جواب پزشکی در برابر متن خبری) و مدل
    فقط «حوزه» را تشخیص می‌دهد؛ در batch هم‌منبع، ۱۲۷ منفی هم‌حوزه و سخت‌اند.
    """
    if mixed:
        return {k: v for k, v in [("pairs", pairs), ("triplets", triplets)] if v}
    groups: dict[str, list[dict]] = {}
    for kind, rows in [("pairs", pairs), ("triplets", triplets)]:
        for r in rows:
            groups.setdefault(f"{kind}-{r['source']}", []).append(r)
    return groups


def cuttlas_evaluator() -> InformationRetrievalEvaluator:
    """همان ارزیابی طلایی scripts/evaluate.py، برای دیدن پیشرفت در طول آموزش."""
    corpus = {r["scene_id"]: r["text"] for r in read_jsonl(CUTTLAS_DIR / "scenes.jsonl")}
    for r in read_jsonl(CUTTLAS_DIR / "summary_chunks.jsonl"):
        if r["level"] == "paragraph":
            corpus[r["chunk_id"]] = r["text"]
    qa = read_jsonl(EVAL_DIR / "cuttlas_qa.jsonl")
    return InformationRetrievalEvaluator(
        queries={r["qid"]: r["query"] for r in qa},
        corpus=corpus,
        relevant_docs={r["qid"]: set(r["doc_ids"]) for r in qa},
        query_prompt=QUERY, corpus_prompt=PASSAGE,
        name="cuttlas", show_progress_bar=False,
    )


def dev_evaluator(dev_rows: list[dict]) -> InformationRetrievalEvaluator:
    """جفت‌های کنارگذاشته از دادهٔ عمومی: هر anchor باید positive خودش را پیدا کند."""
    qa_rows = [r for r in dev_rows if r["kind"] == "qa"]
    corpus, queries, relevant = {}, {}, {}
    passage_ids: dict[str, str] = {}
    for i, r in enumerate(qa_rows):
        pid = passage_ids.setdefault(r["positive"], f"d{len(passage_ids)}")
        corpus[pid] = r["positive"]
        queries[f"q{i}"] = r["anchor"]
        relevant[f"q{i}"] = {pid}
    return InformationRetrievalEvaluator(
        queries=queries, corpus=corpus, relevant_docs=relevant,
        query_prompt=QUERY, corpus_prompt=PASSAGE,
        name="dev", show_progress_bar=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=None, help="برای دور آزمایشی")
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--mini-batch-size", type=int, default=16, help="اگر حافظه کم آمد کمترش کنید")
    parser.add_argument("--lr", type=float, default=7e-6,
                        help="کمتر از ۱e-5 تا مدل پایه کمتر فراموش کند")
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--output", default="models/hamsan-v6")
    parser.add_argument("--mixed-batches", action="store_true",
                        help="رفتار قدیمی: منابع مختلف در یک batch")
    parser.add_argument("--no-mined", action="store_true", help="بدون منفی‌های سخت استخراج‌شده")
    parser.add_argument("--resume", action="store_true",
                        help="ادامهٔ آموزش از آخرین checkpoint در پوشهٔ --output")
    args = parser.parse_args()

    pairs, dev_pairs = load_split("pairs.jsonl", args.max_pairs, dev_size=1000)
    triplets, _ = load_split("triplets.jsonl", args.max_pairs, dev_size=0)
    pairs = [r for r in pairs if r["source"] not in EXCLUDED_SOURCES]
    dev_pairs = [r for r in dev_pairs if r["source"] not in EXCLUDED_SOURCES]
    triplets = [r for r in triplets if r["source"] not in EXCLUDED_SOURCES]
    if not args.no_mined:
        pairs, mined = attach_mined_negatives(pairs)
        triplets += mined
        print(f"منفی سخت استخراج‌شده: {len(mined)} سه‌تایی")
    train = DatasetDict({
        name: Dataset.from_list([add_prefixes(r) for r in rows])
        for name, rows in group_batches(pairs, triplets, args.mixed_batches).items()
    })
    print(f"آموزش: {len(pairs)} جفت + {len(triplets)} سه‌تایی | dev: {len(dev_pairs)}")

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = args.max_seq_length
    # کاربر نهایی می‌تواند بنویسد: model.encode(text, prompt_name="query")
    # «document» همان passage است، با نامی که mteb (و لیدربورد هاگینگ‌فیس) دنبالش می‌گردد
    model.prompts = {"query": QUERY, "passage": PASSAGE, "document": PASSAGE}
    # جدول واژگان ۲۵۰هزارتایی ۷۰٪ وزن‌های مدل است؛ فریزش ~۲.۳ گیگ حافظهٔ GPU آزاد می‌کند
    # و ۱۲ لایهٔ ترنسفورمر همچنان آموزش می‌بینند
    model[0].auto_model.embeddings.word_embeddings.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"وزن‌های قابل آموزش: {trainable / 1e6:.0f}M از {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    inner = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=args.mini_batch_size)
    loss = losses.MatryoshkaLoss(model, inner, matryoshka_dims=MATRYOSHKA_DIMS)

    evaluator = SequentialEvaluator([dev_evaluator(dev_pairs), cuttlas_evaluator()])
    print("قبل از آموزش:", {k: round(v, 3) for k, v in evaluator(model).items()
                            if k.endswith(("ndcg@10", "accuracy@1"))})

    out = ROOT / args.output
    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(out),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        # هر batch فقط از یک دیتاست (group_batches) و به نسبت اندازه‌شان
        multi_dataset_batch_sampler=MultiDatasetBatchSamplers.PROPORTIONAL,
        eval_strategy="steps",
        eval_steps=100,
        save_steps=100,
        save_total_limit=2,
        logging_steps=25,
        # بهترین نسخه با dev عمومی انتخاب می‌شود، نه با آزمون کوتلاس، تا آن عدد واقعی بماند
        load_best_model_at_end=True,
        metric_for_best_model="eval_dev_cosine_ndcg@10",
        dataloader_num_workers=0,
        seed=SEED,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=train, loss=loss, evaluator=evaluator,
    )
    # «ارزیابی قبل از آموزش» بالا در ادامه هم اجرا می‌شود ولی روی وزن‌ها اثری ندارد
    trainer.train(resume_from_checkpoint=True if args.resume else None)

    final = out / "final"
    model.save_pretrained(str(final))
    print("بعد از آموزش:", {k: round(v, 3) for k, v in evaluator(model).items()
                            if k.endswith(("ndcg@10", "accuracy@1"))})
    print(f"مدل ذخیره شد: {final}")


if __name__ == "__main__":
    main()
