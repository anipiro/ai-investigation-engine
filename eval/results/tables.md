### Retrieval (document level)

343 questions with gold documents (49 evaluation questions with gold + 300 extra HotpotQA questions).

| Retriever | Chunking | Recall@5 | Recall@10 | All gold @10 | MRR | nDCG@10 | ms/query |
|---|---|---|---|---|---|---|---|
| dense | recursive | 72.8% | 78.4% | 62.7% | 0.859 | 0.745 | 22 |
| bm25 | recursive | 70.5% | 81.4% | 63.8% | 0.883 | 0.746 | 15 |
| hybrid_rrf | recursive | 79.3% | 85.6% | 71.4% | 0.892 | 0.791 | 38 |
| hybrid_rrf+title | recursive | 81.4% | 86.4% | 73.2% | 0.895 | 0.800 | 32 |
| hybrid+rerank | recursive | 84.0% | 88.6% | 77.5% | 0.938 | 0.844 | 1016 |
| hybrid+rerank+mmr (default) | recursive | 83.2% | 88.5% | 77.3% | 0.938 | 0.843 | 1520 |
| dense | fixed | 72.3% | 78.0% | 63.0% | 0.847 | 0.739 | 36 |
| hybrid_rrf+title | fixed | 81.5% | 86.6% | 73.8% | 0.893 | 0.802 | 55 |
| hybrid+rerank+mmr (default) | fixed | 83.4% | 88.8% | 77.8% | 0.942 | 0.845 | 1669 |

Default retriever by question category:

| Category | n | Recall@10 | MRR |
|---|---|---|---|
| hotpot_bridge | 260 | 85.0% | 0.932 |
| hotpot_comparison | 64 | 99.2% | 0.984 |
| injection | 2 | 100.0% | 0.750 |
| northwind_conflict | 3 | 100.0% | 1.000 |
| northwind_multi | 2 | 100.0% | 1.000 |
| northwind_multihop | 2 | 100.0% | 0.292 |
| northwind_simple | 2 | 100.0% | 0.667 |
| postmortem | 6 | 100.0% | 1.000 |
| postmortem_multi | 1 | 100.0% | 1.000 |
| unanswerable | 1 | 100.0% | 1.000 |

### End-to-end answers

| Metric | agentic | simple |
|---|---|---|
| Answer correctness (LLM judge) | 91.8% | 91.8% |
| Key facts present | 90.7% | 90.7% |
| Exact match (short answer) | 38.8% | 36.7% |
| Token F1 (short answer) | 57.6% | 55.0% |
| Evidence recall (gold docs retrieved) | 97.7% | 97.7% |
| Citation precision | 92.5% | 92.3% |
| Citation recall | 92.1% | 94.4% |
| Correct abstention (unanswerable) | 100.0% | 100.0% |
| False abstention (answerable) | 7.1% | 7.1% |
| Conflict detected | 100.0% | 100.0% |
| Injection resisted | 100.0% | 100.0% |
| Mean latency (ms) | 9,254 | 2,528 |
| LLM calls / question | 3.5 | 1.0 |
| Tokens / question | 5,475 | 1,950 |

Answer correctness (LLM judge) by category:

| Category | n | agentic | simple |
|---|---|---|---|
| hotpot_bridge | 13 | 76.9% | 76.9% |
| hotpot_comparison | 11 | 90.9% | 90.9% |
| injection | 2 | 100.0% | 100.0% |
| northwind_conflict | 3 | 100.0% | 100.0% |
| northwind_multi | 2 | 100.0% | 100.0% |
| northwind_multihop | 2 | 100.0% | 100.0% |
| northwind_simple | 2 | 100.0% | 100.0% |
| postmortem | 6 | 100.0% | 100.0% |
| postmortem_multi | 1 | 100.0% | 100.0% |
| unanswerable | 7 | 100.0% | 100.0% |
