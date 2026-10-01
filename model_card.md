---
language:
- fa
license: cc-by-sa-4.0
base_model: intfloat/multilingual-e5-base
library_name: sentence-transformers
pipeline_tag: sentence-similarity
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- persian
- farsi
- matryoshka
widget:
- source_sentence: 'query: پایتخت ایران کجاست؟'
  sentences:
  - 'passage: تهران پایتخت و پرجمعیت‌ترین شهر ایران است.'
  - 'passage: اصفهان یکی از شهرهای تاریخی ایران است.'
  - 'passage: دریای خزر بزرگ‌ترین دریاچهٔ جهان است.'
---

# Hamsan (همسان) — Persian sentence embeddings

**Hamsan** is a Persian text-embedding model built on
[`intfloat/multilingual-e5-base`](https://huggingface.co/intfloat/multilingual-e5-base).
It turns Persian sentences and passages into 768-dimensional vectors for semantic search,
RAG, clustering, and similarity.

همسان یک مدل embedding فارسی است برای جست‌وجوی معنایی، RAG و سنجش شباهت جمله‌ها.

- **Same size and speed as e5-base:** 278M parameters, runs on a laptop CPU or a small GPU.
- **Open license:** CC BY-SA 4.0. No non-commercial training data.
- **Short vectors:** trained with Matryoshka loss, so vectors can be cut to 512/256/128 dims.
- **Built-in prompts:** `query` / `passage` prefixes are saved in the model.

## Usage

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

- Questions and search queries: `prompt_name="query"`.
- Documents and passages to search over: `prompt_name="passage"`.
- Comparing two sentences with each other: use `"query"` for both.

Shorter vectors (smaller index, faster search):

```python
model = SentenceTransformer("hanyeh00/hamsan", truncate_dim=256)
```

## Evaluation

Lite [FaMTEB](https://huggingface.co/spaces/mteb/leaderboard) subset (8 retrieval + 2 STS tasks).
Retrieval: nDCG@10. STS: Spearman.

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
| Farsick (STS) | 0.686 | **0.707** |
| SynPerSTS (STS) | **0.869** | 0.853 |
| **STS average** | 0.778 | **0.780** |

Hamsan improves the retrieval average over e5-base by about 2 points, with the largest gain on
argument retrieval (ArguAna, +0.16) and gains on scientific (SCIDOCS) and news (NeuCLIR)
retrieval. It is slightly weaker on web search (MSMARCO) and Wikipedia retrieval, and on par
for sentence similarity. Larger models such as BGE-M3 or
[Hakim](https://huggingface.co/MCINext/Hakim) score higher on FaMTEB.

## Training

1. **Fine-tuning** e5-base for 1 epoch on ~115k Persian pairs and triplets from several domains:
   - Persian Wikipedia (title → paragraph), PQuAD and PersianQA (question → paragraph)
   - [pn_summary](https://huggingface.co/datasets/HooshvareLab/pn_summary) news
     (title / summary → article)
   - [persian-med-qa](https://huggingface.co/datasets/aictsharif/persian-med-qa)
     (medical question → answer)
   - synthetic STS triplets

   Hard negatives were mined with the base model. Every batch comes from a single source,
   so in-batch negatives are from the same domain.
   Loss: `MatryoshkaLoss(CachedMultipleNegativesRankingLoss)`, batch 128, lr 7e-6,
   max length 256, token embeddings frozen.
2. **Merging** with the base model to keep its general knowledge:
   `weights = 0.7 × fine-tuned + 0.3 × e5-base`.

## Limitations

- Only Persian was evaluated; other languages may be weaker than in e5-base.
- Not a source of medical advice, even though medical Q&A was part of the training data.
- Inputs longer than 256 tokens are truncated.

## Citation

If you use this model, please cite the base model:

```bibtex
@article{wang2024multilingual,
  title={Multilingual E5 Text Embeddings: A Technical Report},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2402.05672},
  year={2024}
}
```
