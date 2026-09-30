"""سنجش یک مدل Embedding روی مجموعهٔ ارزیابی طلایی کوتلاس.

    python scripts/evaluate.py intfloat/multilingual-e5-base
    python scripts/evaluate.py models/hamsan-v1/final

دو آزمون:
  1. بازیابی: هر سؤال باید صحنه/پاراگراف درست را از میان کل پیکره پیدا کند.
  2. شباهت: جفت‌های هم‌معنا باید شباهت بیشتری از جفت‌های بی‌ربط بگیرند.
"""

from __future__ import annotations

import os

# دانلود Xet هاگینگ‌فیس روی این شبکه گیر می‌کند؛ دانلود مستقیم کار می‌کند
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import json
from pathlib import Path

import torch
from hazm import Normalizer
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
normalizer = Normalizer()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prefixes(model: SentenceTransformer, model_name: str) -> tuple[str, str]:
    """پیشوند query/passage: اول از تنظیمات خود مدل (مدل آموزش‌دیدهٔ ما آن را ذخیره می‌کند)،
    وگرنه برای خانوادهٔ E5 پیش‌فرض آن‌ها."""
    if "query" in model.prompts and "passage" in model.prompts:
        return model.prompts["query"], model.prompts["passage"]
    return ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")


def load_corpus() -> dict[str, str]:
    corpus = {r["scene_id"]: r["text"] for r in read_jsonl(DATA / "cuttlas" / "scenes.jsonl")}
    for r in read_jsonl(DATA / "cuttlas" / "summary_chunks.jsonl"):
        if r["level"] == "paragraph":
            corpus[r["chunk_id"]] = r["text"]
    return corpus


def eval_retrieval(model: SentenceTransformer, q_pre: str, p_pre: str) -> dict:
    corpus = load_corpus()
    ids = list(corpus)
    qa = read_jsonl(DATA / "eval" / "cuttlas_qa.jsonl")

    doc_emb = model.encode([p_pre + normalizer.normalize(corpus[i]) for i in ids],
                           convert_to_tensor=True, normalize_embeddings=True)
    q_emb = model.encode([q_pre + normalizer.normalize(r["query"]) for r in qa],
                         convert_to_tensor=True, normalize_embeddings=True)
    ranking = torch.argsort(q_emb @ doc_emb.T, dim=1, descending=True).cpu()

    hits1 = hits5 = mrr = 0.0
    failures = []
    for row, order in zip(qa, ranking):
        ranked_ids = [ids[j] for j in order[:10]]
        # یک سؤال می‌تواند چند جواب درست داشته باشد (مثلاً ترانه‌ای که در دو صحنه تکرار شده)
        ranks = [ranked_ids.index(d) + 1 for d in row["doc_ids"] if d in ranked_ids]
        rank = min(ranks) if ranks else None
        hits1 += rank == 1
        hits5 += rank is not None and rank <= 5
        mrr += 1 / rank if rank else 0
        if rank != 1:
            failures.append(f"  [{rank or '>10'}] {row['query']}  → آمد: {ranked_ids[0]}، درست: {'/'.join(row['doc_ids'])}")

    n = len(qa)
    return {"recall@1": hits1 / n, "recall@5": hits5 / n, "mrr@10": mrr / n,
            "corpus_size": len(ids), "failures": failures}


def eval_sts(model: SentenceTransformer, q_pre: str) -> dict:
    rows = read_jsonl(DATA / "eval" / "cuttlas_sts.jsonl")
    a = model.encode([q_pre + normalizer.normalize(r["sentence1"]) for r in rows],
                     convert_to_tensor=True, normalize_embeddings=True)
    b = model.encode([q_pre + normalizer.normalize(r["sentence2"]) for r in rows],
                     convert_to_tensor=True, normalize_embeddings=True)
    sims = (a * b).sum(dim=1).tolist()
    pos = [s for s, r in zip(sims, rows) if r["label"] == 1]
    neg = [s for s, r in zip(sims, rows) if r["label"] == 0]
    # چند درصد از همهٔ (مثبت، منفی)ها درست مرتب شده‌اند
    ordered = sum(p > q for p in pos for q in neg) / (len(pos) * len(neg))
    return {"pos_mean": sum(pos) / len(pos), "neg_mean": sum(neg) / len(neg),
            "gap": sum(pos) / len(pos) - sum(neg) / len(neg), "pair_accuracy": ordered}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--show-failures", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    q_pre, p_pre = prefixes(model, args.model)

    ret = eval_retrieval(model, q_pre, p_pre)
    sts = eval_sts(model, q_pre)

    print(f"\nمدل: {args.model}   (device={device})")
    print(f"بازیابی روی {ret['corpus_size']} سند:  "
          f"R@1={ret['recall@1']:.3f}  R@5={ret['recall@5']:.3f}  MRR@10={ret['mrr@10']:.3f}")
    print(f"شباهت:  مثبت={sts['pos_mean']:.3f}  منفی={sts['neg_mean']:.3f}  "
          f"فاصله={sts['gap']:.3f}  دقت جفتی={sts['pair_accuracy']:.3f}")
    if args.show_failures:
        print("\nسؤال‌هایی که رتبهٔ اول نشدند:")
        print("\n".join(ret["failures"]))

    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    name = args.model.replace("/", "__")
    ret.pop("failures")
    (out / f"{name}.json").write_text(
        json.dumps({"model": args.model, "retrieval": ret, "sts": sts}, ensure_ascii=False, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()
