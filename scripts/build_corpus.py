"""Build the reproducible seed corpus (JSONL) from the raw downloads.

  data/raw/hotpotqa/distractor_validation.parquet   (hotpotqa/hotpot_qa, distractor, validation split)
  data/raw/postmortems/README.md                    (github.com/danluu/post-mortems)

Outputs data/corpus/{hotpotqa,postmortems}.jsonl, data/corpus/hotpot_questions.jsonl and
data/corpus/build_report.json (counts, duplicate statistics).

Why this subset: the distractor *validation* split has gold supporting-fact titles for every
question (needed for Recall@K / MRR), and its 7,405 questions x 10 paragraphs give ~74k paragraph
records — after de-duplication still well above the 50k requirement — while staying small enough
to embed on a laptop CPU and fit Qdrant Cloud's free 1 GB cluster. Every question's gold and
distractor paragraphs are in the index, so retrieval is a real needle-in-66k-haystack problem.

Usage: uv run python scripts/build_corpus.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.datasets import hotpot_doc_id, postmortem_doc_id  # noqa: E402
from engine.ingestion.dedup import content_hash, hamming, simhash64  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "corpus"


def build_hotpot(report: dict) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(RAW / "hotpotqa" / "distractor_validation.parquet")
    rows = table.to_pylist()
    by_title: dict[str, str] = {}
    hashes: dict[str, str] = {}  # content hash -> canonical doc id
    alias: dict[str, str] = {}  # title whose text duplicates another title -> canonical doc id
    stats = Counter()
    with open(OUT / "hotpotqa.jsonl", "w", encoding="utf-8") as f_docs, open(OUT / "hotpot_questions.jsonl", "w", encoding="utf-8") as f_q:
        for r in rows:
            titles, sents = r["context"]["title"], r["context"]["sentences"]
            for title, sentences in zip(titles, sents):
                stats["paragraphs_seen"] += 1
                text = "".join(sentences).strip()
                if not text:
                    stats["empty"] += 1
                    continue
                if title in by_title:
                    stats["duplicate_title_same_text" if by_title[title] == text else "duplicate_title_different_text"] += 1
                    continue
                h = content_hash(text)
                if h in hashes:
                    stats["duplicate_text_other_title"] += 1
                    alias[title] = hashes[h]
                    continue
                by_title[title] = text
                hashes[h] = hotpot_doc_id(title)
                f_docs.write(json.dumps({"doc_id": hotpot_doc_id(title), "title": title, "text": text, "source": "hotpotqa",
                                         "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}, ensure_ascii=False) + "\n")
            gold = list(dict.fromkeys(r["supporting_facts"]["title"]))
            f_q.write(json.dumps({"id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"],
                                  "level": r["level"], "gold_titles": gold, "gold_doc_ids": [alias.get(t, hotpot_doc_id(t)) for t in gold]},
                                 ensure_ascii=False) + "\n")
    stats["unique_documents"] = len(by_title)
    stats["questions"] = len(rows)
    report["hotpotqa"] = dict(stats)


ENTRY = re.compile(r"^\[([^\]]+)\]\((https?://[^)]+)\)\.?\s*(.+)$")


def build_postmortems(report: dict) -> None:
    lines = (RAW / "postmortems" / "README.md").read_text(encoding="utf-8").splitlines()
    category = None
    stats = Counter()
    per_company: Counter = Counter()
    seen: dict[str, int] = {}
    docs = []
    for line in lines:
        line = line.strip()
        if line.startswith("## "):
            category = line[3:].strip()
            continue
        if category in (None, "Table of Contents", "Other lists of postmortems", "Analysis", "Contributors"):
            continue
        m = ENTRY.match(line)
        if not m:
            continue
        company, url, desc = m.group(1), m.group(2), m.group(3).strip()
        stats["entries_seen"] += 1
        h = content_hash(desc)
        if h in seen:
            stats["duplicate_entries"] += 1
            continue
        seen[h] = 1
        per_company[company] += 1
        first = re.split(r"(?<=[.!?])\s", desc, maxsplit=1)[0]
        docs.append({"doc_id": postmortem_doc_id(company, per_company[company]),
                     "title": f"{company} post-mortem: {first[:90]}",
                     "text": f"{company} incident ({category}). {desc}", "source": "postmortems", "url": url,
                     "metadata": {"company": company, "category": category}})
    # Near-duplicate report (same incident linked twice with slightly different wording).
    sims = [(d["doc_id"], simhash64(d["text"])) for d in docs]
    near = sum(1 for i in range(len(sims)) for j in range(i + 1, len(sims)) if hamming(sims[i][1], sims[j][1]) <= 3)
    with open(OUT / "postmortems.jsonl", "w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    stats["unique_documents"] = len(docs)
    stats["near_duplicate_pairs"] = near
    report["postmortems"] = dict(stats)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    build_hotpot(report)
    build_postmortems(report)
    report["total_documents"] = report["hotpotqa"]["unique_documents"] + report["postmortems"]["unique_documents"]
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
