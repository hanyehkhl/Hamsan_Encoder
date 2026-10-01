<div align="center">

# Hamsan (همسان)

**Persian sentence embeddings for semantic search, RAG and similarity**

[![Model on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Model-hanyeh00%2Fhamsan-yellow)](https://huggingface.co/hanyeh00/hamsan)
[![Model license: CC BY-SA 4.0](https://img.shields.io/badge/model%20license-CC%20BY--SA%204.0-blue)](https://creativecommons.org/licenses/by-sa/4.0/)
[![Code license: MIT](https://img.shields.io/badge/code%20license-MIT-green)](LICENSE)
![Language: Persian](https://img.shields.io/badge/language-Persian%20(fa)-orange)

</div>

همسان یک مدل embedding فارسی است که جمله‌ها و متن‌ها را به بردار تبدیل می‌کند تا بتوانید
جست‌وجوی معنایی، RAG و سنجش شباهت انجام دهید.

Hamsan is a Persian text-embedding model fine-tuned from
[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base).
This repository holds the full training and evaluation pipeline; the model itself lives on
[Hugging Face](https://huggingface.co/hanyeh00/hamsan).

## Highlights

- **+1.9 points retrieval over e5-base** on a lite FaMTEB subset (0.473 vs 0.454 nDCG@10).
- **Same size and speed as e5-base:** 278M parameters, runs on a laptop.
- **Open license:** CC BY-SA 4.0 for the model.
- **Matryoshka vectors:** truncate to 512 / 256 / 128 dims for a smaller index.
- **Built-in prompts:** `query` and `passage` prefixes are saved in the model.

## Quick start

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("hanyeh00/hamsan")

queries = model.encode(["پایتخت ایران کجاست؟"], prompt_name="query")
docs = model.encode([
    "تهران پایتخت و پرجمعیت‌ترین شهر ایران است.",
    "اصفهان یکی از شهرهای تاریخی ایران است.",
], prompt_name="passage")

print(model.similarity(queries, docs))
```

Use `prompt_name="query"` for questions and search queries, and `prompt_name="passage"` for
the documents you search over. For sentence-to-sentence similarity, use `"query"` for both.

## Results

Lite [FaMTEB](https://huggingface.co/spaces/mteb/leaderboard) subset. Retrieval: nDCG@10; STS: Spearman.

| Task | e5-base | **Hamsan** |
|---|---|---|
| SynPerChatbotRAGFAQRetrieval | 0.285 | **0.287** |
| WikipediaRetrievalMultilingual | **0.881** | 0.861 |
| MSMARCO-FaHardNegatives | **0.667** | 0.628 |
| ArguAna-Fa.v2 | 0.338 | **0.494** |
| FiQA2018-Fa.v2 | **0.230** | 0.227 |
| SCIDOCS-Fa.v2 | 0.118 | **0.141** |
| SciFact-Fa.v2 | 0.582 | 0.582 |
| NeuCLIR2023RetrievalHardNegatives | 0.530 | **0.561** |
| **Retrieval average** | 0.454 | **0.473** |
| Farsick | 0.686 | **0.707** |
| SynPerSTS | **0.869** | 0.853 |
| **STS average** | 0.778 | **0.780** |

Larger models such as BGE-M3 or [Hakim](https://huggingface.co/MCINext/Hakim) score higher on FaMTEB.

## Try it locally

A small web demo compares Hamsan with e5-base side by side:

```bash
uv sync
uv run python scripts/app.py     # open http://127.0.0.1:7860
```

## Reproduce

The pipeline uses [uv](https://docs.astral.sh/uv/) and was run on a 6 GB laptop GPU (RTX 4050).

| Step | Command |
|---|---|
| 1. Build training pairs | `uv run python scripts/build_pairs.py` |
| 2. Mine hard negatives | `uv run python scripts/mine_negatives.py` |
| 3. Fine-tune | `uv run python scripts/train.py --output models/hamsan-v6` |
| 4. Merge with e5-base | `uv run python scripts/merge.py models/hamsan-v6/final --alphas 0.7` |
| 5. Evaluate (lite FaMTEB) | `uv run python scripts/eval_famteb.py models/hamsan-v6-merge-0.7` |

Training recipe: 1 epoch on ~115k Persian pairs and triplets (Wikipedia, PQuAD, PersianQA,
pn_summary news, persian-med-qa, synthetic STS), `MatryoshkaLoss` over
`CachedMultipleNegativesRankingLoss`, batch 128 with every batch drawn from a single source,
lr 7e-6, then a linear merge `0.7 × fine-tuned + 0.3 × e5-base`.
See [model_card.md](model_card.md) for details.

## Repository layout

```
scripts/
  build_pairs.py       training pairs from public Persian datasets
  mine_negatives.py    hard-negative mining with the base model
  train.py             fine-tuning (sentence-transformers)
  merge.py             linear merge with e5-base
  eval_famteb.py       lite FaMTEB evaluation
  evaluate.py          in-house retrieval test
  search.py, ask.py    command-line search
  app.py               web demo (Gradio)
data/                  evaluation sets and training logs
model_card.md          Hugging Face model card
```

## License

- **Code:** [MIT](LICENSE)
- **Model weights:** [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

## Citation

```bibtex
@misc{hamsan2026,
  title  = {Hamsan: Persian Sentence Embeddings},
  author = {Hanyeh},
  year   = {2026},
  url    = {https://huggingface.co/hanyeh00/hamsan}
}
```

Hamsan is built on Multilingual E5
([Wang et al., 2024](https://arxiv.org/abs/2402.05672)).
