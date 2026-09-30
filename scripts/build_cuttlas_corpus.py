"""فاز ۱: ساخت پیکرهٔ تمیز کوتلاس از زیرنویس‌ها و خلاصه.

ورودی:  datasets_cutlass/*.srt  و  datasets_cutlass/summary.txt
خروجی: data/cuttlas/sentences.jsonl, scenes.jsonl, summary_chunks.jsonl, report.txt

اجرا:
    python scripts/build_cuttlas_corpus.py
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from hazm import Normalizer, sent_tokenize

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "datasets_cutlass"
OUT_DIR = ROOT / "data" / "cuttlas"

SCENE_GAP_SEC = 3.0        # مکث بیشتر از این = صحنهٔ جدید
SCENE_MAX_CHARS = 450      # صحنه‌های خیلی بلند شکسته می‌شوند
MIN_SENTENCE_WORDS = 3     # جمله‌های کوتاه‌تر (آه، بله...) کنار گذاشته می‌شوند
MAX_NGRAM = 8              # بلندترین عبارتی که تکرارش جمع می‌شود

normalizer = Normalizer()

PERSIAN_CHAR = re.compile(r"[؀-ۿ]")
BRACKET_TAG = re.compile(r"\[[^\]]*\]|\([^)]*\)|\[[^\]]*$|[\[\]()]")
ANY_LATIN = re.compile(r"[A-Za-z]")
MAX_SENTENCE_WORDS = 40    # جمله‌های بلندتر معمولاً چند دیالوگِ به‌هم‌چسبیده یا حلقه‌اند
MIN_UNIQUE_RATIO = 0.6     # نسبت کلمات یکتا؛ کمتر از این = متن تکراری/خراب
LATIN_TOKEN = re.compile(r"^[A-Za-z]+[.,!?]*$")
EPISODE_RE = re.compile(r"Series\s+(\d+)")
TIME_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class Stats:
    cues: int = 0
    tag_only: int = 0
    repetition_fixed: int = 0
    words_before: int = 0
    words_after: int = 0
    sentences: int = 0
    dropped_short: int = 0
    dropped_bad: int = 0
    scenes: int = 0
    notes: list[str] = field(default_factory=list)


def to_seconds(ts: str) -> float:
    h, m, s, ms = map(int, TIME_RE.search(ts).groups())
    return h * 3600 + m * 60 + s + ms / 1000


def parse_srt(path: Path) -> list[Cue]:
    raw = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n")
    cues = []
    for block in re.split(r"\n\s*\n", raw.strip()):
        lines = block.strip().split("\n")
        idx = next((i for i, l in enumerate(lines) if "-->" in l), None)
        if idx is None:
            continue
        start, end = lines[idx].split("-->")
        text = " ".join(lines[idx + 1:]).strip()
        cues.append(Cue(to_seconds(start), to_seconds(end), text))
    return cues


def collapse_repeats(words: list[str]) -> tuple[list[str], bool]:
    """عبارت‌هایی که پشت‌سرهم تکرار شده‌اند را به یک بار کاهش می‌دهد.

    مقایسه بدون علائم نگارشی است («بار،» = «بار») ولی کلمهٔ اصلی حفظ می‌شود.
    """
    changed = False
    for n in range(MAX_NGRAM, 0, -1):
        keys = [strip_punct(w) for w in words]
        out, i = [], 0
        while i < len(words):
            chunk = keys[i:i + n]
            j = i + n
            while len(chunk) == n and keys[j:j + n] == chunk:
                j += n
            if j > i + n:
                changed = True
                out.extend(words[i:i + n])
                i = j
            else:
                out.append(words[i])
                i += 1
        words = out
    return words, changed


def strip_punct(w: str) -> str:
    return w.strip("،,.!?؟:;«»\"'-")


def clean_cue(text: str, stats: Stats) -> str:
    text = BRACKET_TAG.sub(" ", text)
    words = text.split()
    stats.words_before += len(words)
    # حذف توکن‌های لاتین و تک‌حرفی‌های سرگردان (مثل "cu" یا "گ")؛ «و» می‌ماند
    words = [
        w for w in words
        if not LATIN_TOKEN.match(w)
        and not (len(strip_punct(w)) == 1 and strip_punct(w) != "و")
    ]
    words, changed = collapse_repeats(words)
    if changed:
        stats.repetition_fixed += 1
    stats.words_after += len(words)
    return " ".join(words)


def is_bad_sentence(sent: str) -> bool:
    words = [strip_punct(w) for w in sent.split()]
    return (
        bool(ANY_LATIN.search(sent))
        or len(words) > MAX_SENTENCE_WORDS
        or len(set(words)) / len(words) < MIN_UNIQUE_RATIO
    )


def split_scenes(cues: list[Cue]) -> list[list[Cue]]:
    scenes, current, length = [], [], 0
    for cue in cues:
        gap = cue.start - current[-1].end if current else 0
        if current and (gap > SCENE_GAP_SEC or length > SCENE_MAX_CHARS):
            scenes.append(current)
            current, length = [], 0
        current.append(cue)
        length += len(cue.text)
    if current:
        scenes.append(current)
    return scenes


def process_episode(path: Path, episode: str) -> tuple[list[dict], list[dict], Stats]:
    stats = Stats()
    cues = parse_srt(path)
    stats.cues = len(cues)

    cleaned = []
    for cue in cues:
        text = clean_cue(cue.text, stats)
        if not PERSIAN_CHAR.search(text):
            stats.tag_only += 1
            continue
        cleaned.append(Cue(cue.start, cue.end, text))

    sentences, scenes = [], []
    for s_idx, scene_cues in enumerate(split_scenes(cleaned)):
        # چسباندن زیرنویس‌ها، سپس شکستن دوباره به جملهٔ کامل
        joined = normalizer.normalize(" ".join(c.text for c in scene_cues))
        kept = []
        for sent in sent_tokenize(joined):
            if len(sent.split()) < MIN_SENTENCE_WORDS:
                stats.dropped_short += 1
                continue
            if is_bad_sentence(sent):
                stats.dropped_bad += 1
                continue
            kept.append(sent)
        if not kept:
            continue
        scene_id = f"ep{episode}_s{s_idx:03d}"
        for k, sent in enumerate(kept):
            sentences.append({"episode": episode, "scene_id": scene_id,
                              "sent_id": f"{scene_id}_{k:02d}", "text": sent})
        scenes.append({"episode": episode, "scene_id": scene_id,
                       "start": scene_cues[0].start, "end": scene_cues[-1].end,
                       "text": " ".join(kept)})

    stats.sentences = len(sentences)
    stats.scenes = len(scenes)
    return sentences, scenes, stats


def pick_files() -> dict[str, Path]:
    """برای هر قسمت فقط یک فایل: آن که متن فارسی بیشتری دارد."""
    best: dict[str, tuple[int, Path]] = {}
    for path in sorted(RAW_DIR.glob("*.srt")):
        match = EPISODE_RE.search(path.name)
        if not match:
            continue
        ep = match.group(1).zfill(2)
        size = len(PERSIAN_CHAR.findall(path.read_text(encoding="utf-8-sig")))
        if ep not in best or size > best[ep][0]:
            best[ep] = (size, path)
    return {ep: p for ep, (_, p) in sorted(best.items())}


def process_summary() -> list[dict]:
    path = RAW_DIR / "summary.txt"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8-sig")
    title_match = re.search(r"^#+\s*(.+)$", text, re.M)
    title = normalizer.normalize(title_match.group(1)) if title_match else ""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text)
                  if p.strip() and not p.strip().startswith("#")]
    chunks = []
    for p_idx, para in enumerate(paragraphs):
        para = normalizer.normalize(para)
        chunks.append({"source": "summary", "title": title,
                       "chunk_id": f"sum_p{p_idx}", "level": "paragraph",
                       "text": para})
        for s_idx, sent in enumerate(sent_tokenize(para)):
            chunks.append({"source": "summary", "title": title,
                           "chunk_id": f"sum_p{p_idx}_s{s_idx}",
                           "parent": f"sum_p{p_idx}", "level": "sentence",
                           "text": sent})
    return chunks


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_sentences, all_scenes, report = [], [], []
    files = pick_files()
    skipped = sorted(set(RAW_DIR.glob("*.srt")) - set(files.values()))

    seen = set()
    for ep, path in files.items():
        sentences, scenes, st = process_episode(path, ep)
        # حذف جمله‌های تکراری در کل پیکره
        unique = [s for s in sentences if not (s["text"] in seen or seen.add(s["text"]))]
        all_sentences.extend(unique)
        all_scenes.extend(scenes)
        kept_ratio = st.words_after / st.words_before if st.words_before else 0
        report.append(
            f"قسمت {ep}  ({path.name})\n"
            f"  زیرنویس‌ها: {st.cues}  | فقط برچسب/بی‌متن: {st.tag_only}"
            f"  | دارای تکرار حلقه‌ای: {st.repetition_fixed}\n"
            f"  کلمات: {st.words_before} → {st.words_after}  ({kept_ratio:.0%} باقی ماند)\n"
            f"  جمله‌ها: {st.sentences} (یکتا: {len(unique)})"
            f"  | حذف‌شده: کوتاه {st.dropped_short}، خراب {st.dropped_bad}  | صحنه‌ها: {st.scenes}\n"
        )

    summary_chunks = process_summary()
    write_jsonl(OUT_DIR / "sentences.jsonl", all_sentences)
    write_jsonl(OUT_DIR / "scenes.jsonl", all_scenes)
    write_jsonl(OUT_DIR / "summary_chunks.jsonl", summary_chunks)

    total = (
        "=== جمع کل ===\n"
        f"قسمت‌ها: {len(files)}  | جمله‌های یکتا: {len(all_sentences)}"
        f"  | صحنه‌ها: {len(all_scenes)}\n"
        f"تکه‌های خلاصه: {sum(c['level'] == 'paragraph' for c in summary_chunks)} پاراگراف،"
        f" {sum(c['level'] == 'sentence' for c in summary_chunks)} جمله\n"
    )
    if skipped:
        total += "فایل‌های تکراری کنار گذاشته‌شده:\n" + "".join(f"  - {p.name}\n" for p in skipped)
    text = "\n".join(report) + "\n" + total
    (OUT_DIR / "report.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
