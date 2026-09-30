"""فاز ۲: ساخت دادهٔ آموزشی (جفت‌جمله‌های هم‌معنا) از دیتاست‌های عمومی فارسی + کوتلاس.

    uv run python scripts/build_pairs.py

خروجی در data/train/:
    pairs.jsonl     {anchor, positive, kind, source}
    triplets.jsonl  {anchor, positive, negative, kind, source}   ← با منفی سخت
    report.txt

kind:
    "qa"  = نامتقارن (سؤال/عنوان → پاراگراف)؛ در آموزش E5 پیشوند query:/passage: می‌گیرد
    "sym" = متقارن (دو جملهٔ هم‌معنا)؛ هر دو طرف query: می‌گیرند

اندازه‌ها برای لپ‌تاپ با GPU شش‌گیگی انتخاب شده‌اند (حدود ۱۵۰ تا ۲۰۰ هزار جفت).
"""

from __future__ import annotations

import os

# دانلود Xet هاگینگ‌فیس روی این شبکه گیر می‌کند؛ دانلود مستقیم کار می‌کند
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import ast
import csv
import io
import json
import random
import re
import zipfile
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from hazm import Normalizer
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "train"
CUTTLAS_DIR = ROOT / "data" / "cuttlas"
EVAL_QA = ROOT / "data" / "eval" / "cuttlas_qa.jsonl"

SEED = 37
WIKI_ARTICLES = 40_000       # تعداد مقالهٔ ویکی‌پدیا (عنوان → پاراگراف اول)
MAX_PASSAGE_CHARS = 1200     # متن‌های بلندتر بریده می‌شوند؛ مدل در آموزش بیش از ~۲۵۶ توکن نمی‌بیند
MIN_PASSAGE_CHARS = 80
STS_POS_MIN = 4              # امتیاز ۴ و ۵ از ۵ = هم‌معنا
STS_NEG_MAX = 1              # امتیاز ۰ و ۱ = منفی سخت برای همان جمله
NEWS_PER_CATEGORY = 3_000    # pn_summary: سقف هر موضوع تا نفت‌وانرژی (۱۶ هزار خبر) غالب نشود
# v4 نشان داد ۹۰٪ دادهٔ هم‌حوزهٔ ویکی‌پدیا مدل را فقط در همان حوزه بهتر می‌کند و در
# FaMTEB (مالی، علمی، FAQ، وب) عقب می‌برد؛ سقف منابع ویکی‌پدیایی جا را به حوزه‌های دیگر می‌دهد
SOURCE_CAPS = {"wikipedia": 15_000, "pquad": 15_000, "med_qa": 15_000}

normalizer = Normalizer()
random.seed(SEED)


def parquet(repo: str, config: str, split: str = "train"):
    """دیتاست‌هایی که اسکریپت بارگذاری قدیمی دارند؛ نسخهٔ Parquet تبدیل‌شدهٔ خود هاب را می‌خوانیم."""
    files = f"hf://datasets/{repo}@refs/convert/parquet/{config}/{split}/*.parquet"
    return load_dataset("parquet", data_files=files, split="train")


def norm(text: str, max_chars: int | None = None) -> str:
    text = normalizer.normalize(" ".join(str(text).split()))
    if max_chars and len(text) > max_chars:
        # بریدن در آخرین مرز جمله قبل از سقف
        cut = text[:max_chars]
        end = max(cut.rfind("."), cut.rfind("؟"), cut.rfind("!"))
        text = cut[: end + 1] if end > max_chars // 2 else cut
    return text


def has_answer(answers) -> bool:
    if isinstance(answers, str):
        answers = ast.literal_eval(answers)
    return bool(answers and answers.get("text"))


# ---------------------------------------------------------------- منابع

def from_squad_like(ds, source: str) -> list[dict]:
    """PQuAD و PersianQA: سؤال → پاراگرافی که جوابش را دارد."""
    rows = []
    for r in ds:
        if str(r.get("is_impossible", "False")) == "True" or not has_answer(r["answers"]):
            continue
        rows.append({"anchor": norm(r["question"]),
                     "positive": norm(r["context"], MAX_PASSAGE_CHARS),
                     "kind": "qa", "source": source})
    return rows


def from_sts() -> tuple[list[dict], list[dict]]:
    """synthetic-persian-sts: امتیاز بالا = جفت مثبت؛ امتیاز پایینِ همان جمله = منفی سخت."""
    ds = load_dataset("MCINext/synthetic-persian-sts", split="train")
    pos, neg = defaultdict(list), defaultdict(list)
    for r in ds:
        score = float(r["score"])
        s1, s2 = norm(r["sentence1"]), norm(r["sentence2"])
        if score >= STS_POS_MIN:
            pos[s1].append(s2)
        elif score <= STS_NEG_MAX:
            neg[s1].append(s2)
    pairs, triplets = [], []
    for s1, positives in pos.items():
        for s2 in positives:
            row = {"anchor": s1, "positive": s2, "kind": "sym", "source": "synthetic-sts"}
            if neg[s1]:
                triplets.append({**row, "negative": random.choice(neg[s1])})
            else:
                pairs.append(row)
    return pairs, triplets


def from_wikipedia() -> list[dict]:
    """ویکی‌پدیای فارسی: عنوان مقاله → پاراگراف اول. به‌صورت stream، فقط به اندازهٔ لازم دانلود می‌شود."""
    ds = load_dataset("wikimedia/wikipedia", "20231101.fa", split="train", streaming=True)
    rows = []
    for r in ds:
        first = next((p for p in r["text"].split("\n")
                      if len(p.strip()) >= MIN_PASSAGE_CHARS), None)
        if not first:
            continue
        rows.append({"anchor": norm(r["title"]),
                     "positive": norm(first, MAX_PASSAGE_CHARS),
                     "kind": "qa", "source": "wikipedia"})
        if len(rows) >= WIKI_ARTICLES:
            break
    return rows


def from_news() -> list[dict]:
    """pn_summary (MIT): خبرهای فارسی در ۱۸ موضوع؛ نیمی تیتر → متن، نیمی خلاصه → متن.

    اسکریپت بارگذاری این دیتاست با datasets جدید اجرا نمی‌شود؛ خود zip خوانده می‌شود.
    """
    path = hf_hub_download("HooshvareLab/pn_summary", "data/pn_summary.zip", repo_type="dataset")
    text = zipfile.ZipFile(path).read("pn_summary/train.csv").decode("utf-8")
    csv.field_size_limit(10**9)
    by_category = defaultdict(list)
    for r in csv.DictReader(io.StringIO(text), delimiter="\t", quoting=csv.QUOTE_NONE):
        by_category[r["category"]].append(r)
    rows = []
    for articles in by_category.values():
        random.shuffle(articles)
        for i, r in enumerate(articles[:NEWS_PER_CATEGORY]):
            # [n] در این دیتاست جای شکست پاراگراف است
            article = norm(r["article"].replace("[n]", " "), MAX_PASSAGE_CHARS)
            anchor, source = (r["title"], "news-title") if i % 2 else (r["summary"], "news-summary")
            if len(article) >= MIN_PASSAGE_CHARS and anchor.strip():
                rows.append({"anchor": norm(anchor), "positive": article,
                             "kind": "qa", "source": source})
    return rows


def from_med_qa() -> list[dict]:
    """persian-med-qa (Apache-2.0): سؤال پزشکی → جواب کوتاه؛ نزدیک به FAQ و چت‌بات.

    سؤال‌های تکراری (مثل «چگونه از بیماری‌های قلبی پیشگیری کرد؟» با ۴۳۹ جواب متفاوت) کنار
    می‌روند؛ در یک batch، جواب‌های دیگرِ همان سؤال منفیِ غلط می‌شدند.
    """
    path = hf_hub_download("aictsharif/persian-med-qa", "train.csv", repo_type="dataset")
    with open(path, encoding="utf-8") as f:
        data = list(csv.DictReader(f))
    counts = defaultdict(int)
    for r in data:
        counts[r["question"]] += 1
    rows = []
    for r in data:
        answer = re.sub(r"\s*\(منبع:[^)]*\)\s*$", "", r["answer"])
        if counts[r["question"]] == 1 and len(answer) >= 30:
            rows.append({"anchor": norm(r["question"]), "positive": norm(answer, MAX_PASSAGE_CHARS),
                         "kind": "qa", "source": "med_qa"})
    return rows


def from_cuttlas() -> list[dict]:
    """کوتلاس: هر جمله → بقیهٔ صحنهٔ خودش (آشنایی با نام‌ها و فضای داستان).

    صحنه‌هایی که جواب سؤال‌های ارزیابی‌اند و خلاصهٔ داستان عمداً کنار گذاشته می‌شوند
    تا عدد ارزیابی واقعی بماند.
    """
    eval_docs = {d for l in EVAL_QA.open(encoding="utf-8") for d in json.loads(l)["doc_ids"]}
    by_scene = defaultdict(list)
    for line in (CUTTLAS_DIR / "sentences.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["scene_id"] not in eval_docs:
            by_scene[r["scene_id"]].append(r["text"])
    rows = []
    for sents in by_scene.values():
        if len(sents) < 3:
            continue
        for i, s in enumerate(sents):
            rest = " ".join(sents[:i] + sents[i + 1:])
            rows.append({"anchor": s, "positive": rest[:MAX_PASSAGE_CHARS],
                         "kind": "qa", "source": "cuttlas"})

    # سؤال‌هایی که مدل مولد برای هر صحنه نوشته (data/cuttlas/generated_qa.py)
    scenes = {json.loads(l)["scene_id"]: json.loads(l)["text"]
              for l in (CUTTLAS_DIR / "scenes.jsonl").open(encoding="utf-8")}
    for scene_id, queries in load_generated_qa().items():
        assert scene_id in scenes, f"صحنهٔ ناموجود: {scene_id}"
        assert scene_id not in eval_docs, f"صحنهٔ ارزیابی نباید در آموزش باشد: {scene_id}"
        for q in queries:
            rows.append({"anchor": norm(q), "positive": scenes[scene_id][:MAX_PASSAGE_CHARS],
                         "kind": "qa", "source": "cuttlas-generated"})
    return rows


def load_generated_qa() -> dict[str, list[str]]:
    namespace: dict = {}
    exec((CUTTLAS_DIR / "generated_qa.py").read_text(encoding="utf-8"), namespace)
    return namespace["QA"]


# ---------------------------------------------------------------- جمع‌بندی

def dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        key = (r["anchor"], r["positive"])
        if r["anchor"] and r["positive"] and r["anchor"] != r["positive"] and key not in seen:
            seen.add(key)
            out.append(r)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pairs, triplets = [], []

    # ParsiNLU عمداً نیست: مجوزش CC BY-NC-SA (غیرتجاری) است و مدل را هم غیرتجاری می‌کرد
    steps = [
        ("PQuAD", lambda: (from_squad_like(load_dataset("Z-Jafari/PQuAD", split="train"), "pquad"), [])),
        ("PersianQA", lambda: (from_squad_like(parquet("SajjadAyoubi/persian_qa", "persian_qa"), "persian_qa"), [])),
        ("synthetic-persian-sts", from_sts),
        ("Wikipedia fa", lambda: (from_wikipedia(), [])),
        ("Cuttlas", lambda: (from_cuttlas(), [])),
        ("pn_summary", lambda: (from_news(), [])),
        ("persian-med-qa", lambda: (from_med_qa(), [])),
    ]
    cache_dir = OUT_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    for name, fn in steps:
        print(f"→ {name} ...", flush=True)
        # هر منبع جدا ذخیره می‌شود تا اگر منبع بعدی خطا داد، کار قبلی از دست نرود
        cache = cache_dir / (name.replace(" ", "_") + ".json")
        if cache.exists():
            p, t = json.loads(cache.read_text(encoding="utf-8"))
        else:
            p, t = fn()
            cache.write_text(json.dumps([p, t], ensure_ascii=False), encoding="utf-8")
        print(f"   {len(p)} جفت، {len(t)} سه‌تایی", flush=True)
        # سقف بعد از کش اعمال می‌شود تا کش ویکی‌پدیا (۴۰ هزار) بدون دانلود دوباره به کار بیاید
        cap = SOURCE_CAPS.get(p[0]["source"]) if p else None
        if cap and len(p) > cap:
            p = random.Random(SEED).sample(p, cap)
            print(f"   سقف {cap} جفت", flush=True)
        pairs += p
        triplets += t

    pairs, triplets = dedupe(pairs), dedupe(triplets)
    random.shuffle(pairs)
    random.shuffle(triplets)
    write_jsonl(OUT_DIR / "pairs.jsonl", pairs)
    write_jsonl(OUT_DIR / "triplets.jsonl", triplets)

    counts = defaultdict(lambda: [0, 0])
    for r in pairs:
        counts[r["source"]][0] += 1
    for r in triplets:
        counts[r["source"]][1] += 1
    lines = [f"{src:15s} جفت={p:>7}  سه‌تایی={t:>6}" for src, (p, t) in sorted(counts.items())]
    lines.append(f"{'جمع':15s} جفت={len(pairs):>7}  سه‌تایی={len(triplets):>6}")
    report = "\n".join(lines)
    (OUT_DIR / "report.txt").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)


if __name__ == "__main__":
    main()
