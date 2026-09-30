"""ترکیب وزن‌های مدل آموزش‌دیده با مدل پایه (linear merge).

    uv run python scripts/merge.py models/hamsan-v4/final --alphas 0.5 0.7

وزن جدید = α × مدل آموزش‌دیده + (1-α) × مدل پایه
α=1 همان مدل آموزش‌دیده است و α=0 همان e5 پایه.

fine-tune در حوزهٔ داده‌های آموزش (ویکی‌پدیا، سؤال‌وجواب) کمک می‌کند ولی در حوزه‌های دیگر
کمی از توان عمومی مدل پایه را از بین می‌برد؛ ترکیب معمولاً بخش بزرگی از سود را نگه می‌دارد
و بیشتر آن افت را برمی‌گرداند. خروجی مثل هر مدل دیگری با evaluate.py و eval_famteb.py
سنجیده می‌شود.

خروجی: models/<نام مدل>-merge-<α>/   (مثلاً models/hamsan-v4-merge-0.5)
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
BASE_MODEL = "intfloat/multilingual-e5-base"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="مدل آموزش‌دیده، مثلاً models/hamsan-v4/final")
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.7])
    parser.add_argument("--base", default=BASE_MODEL)
    args = parser.parse_args()

    base = SentenceTransformer(args.base, device="cpu").state_dict()
    model_dir = Path(args.model)
    # models/hamsan-v4/final → hamsan-v4
    name = model_dir.parent.name if model_dir.name == "final" else model_dir.name

    for alpha in args.alphas:
        # هر بار از نو بارگذاری می‌شود تا ترکیب‌ها روی هم انباشته نشوند؛
        # پیکربندی و پیشوندهای query/passage/document از مدل آموزش‌دیده می‌آیند
        model = SentenceTransformer(str(model_dir), device="cpu")
        tuned = model.state_dict()
        assert tuned.keys() == base.keys(), "معماری دو مدل یکی نیست"
        merged = {
            k: alpha * v + (1 - alpha) * base[k] if torch.is_floating_point(v) else v
            for k, v in tuned.items()
        }
        model.load_state_dict(merged)
        out = ROOT / "models" / f"{name}-merge-{alpha:g}"
        model.save_pretrained(str(out))
        print(f"α={alpha:g} → {out}")


if __name__ == "__main__":
    main()
