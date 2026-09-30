"""فاز ۲ب: استخراج منفی سخت برای جفت‌های سؤال → پاراگراف با مدل پایه.

    uv run python scripts/mine_negatives.py

برای هر سؤال، پاراگرافی پیدا می‌شود که به سؤال شبیه است ولی جوابش نیست (معمولاً پاراگراف
دیگری از همان مقاله). train.py این‌ها را به‌عنوان negative کنار جفت‌ها می‌گذارد.

خروجی: data/train/negatives.jsonl   {anchor, positive, negative, source}

- فقط جفت‌های آموزشی استخراج می‌شوند؛ جفت‌های dev (همان تقسیم train.py) کنار می‌مانند.
- هر منبع جدا استخراج می‌شود: پاراگراف‌های PQuAD از ویکی‌پدیا آمده‌اند و ممکن است عین
  پاراگراف اول همان مقاله در دادهٔ ویکی‌پدیا باشند؛ آن‌وقت «منفی» در واقع جواب درست است.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import json
from collections import defaultdict

from datasets import Dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import mine_hard_negatives

from train import BASE_MODEL, PASSAGE, QUERY, TRAIN_DIR, load_split

# med_qa عمداً نیست: جواب‌هایش کوتاه و بسیار شبیه هم‌اند و «منفی سخت» اغلب جواب درست دیگری است
SOURCES = ["pquad", "persian_qa", "wikipedia", "news-title", "news-summary"]
OUT = TRAIN_DIR / "negatives.jsonl"


def main() -> None:
    train_rows, _ = load_split("pairs.jsonl", None, dev_size=1000)
    by_source = defaultdict(list)
    for r in train_rows:
        if r["source"] in SOURCES:
            by_source[r["source"]].append(r)

    model = SentenceTransformer(BASE_MODEL)
    model.max_seq_length = 256   # همان طولی که در آموزش دیده می‌شود

    with OUT.open("w", encoding="utf-8") as f:
        for source in SOURCES:
            rows = by_source[source]
            print(f"\n→ {source}: {len(rows)} جفت", flush=True)
            ds = Dataset.from_dict({"anchor": [r["anchor"] for r in rows],
                                    "positive": [r["positive"] for r in rows]})
            mined = mine_hard_negatives(
                ds, model,
                query_prompt=QUERY, corpus_prompt=PASSAGE,
                # تنظیم مقالهٔ NV-Retriever: منفی باید دست‌کم ۵٪ کم‌امتیازتر از مثبت باشد؛
                # نامزدهای خیلی نزدیک به مثبت احتمالاً خودشان جواب‌اند
                relative_margin=0.05,
                range_max=30,
                num_negatives=1,
                sampling_strategy="top",
                batch_size=64,
            )
            for r in mined:
                f.write(json.dumps({**r, "source": source}, ensure_ascii=False) + "\n")

    print(f"\nذخیره شد: {OUT}")


if __name__ == "__main__":
    main()
