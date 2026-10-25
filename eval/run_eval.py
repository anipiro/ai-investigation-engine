"""Automated evaluation.

Retrieval (no LLM, cheap):   Recall@5/10, all-gold@10, MRR, nDCG@10 for several retriever configurations
                             and both chunking strategies, on 49 + 300 questions with gold documents.
End-to-end (LLM):            answer correctness (EM, F1, keyword, LLM judge), citation precision/recall,
                             abstention, conflict detection, prompt-injection resistance, latency, tokens,
                             for the single-retrieval baseline vs the multi-step agent.

  uv run python eval/run_eval.py retrieval
  uv run python eval/run_eval.py e2e --modes simple agentic
  uv run python eval/run_eval.py report

End-to-end results are cached per (mode, question) in eval/results/, so a run interrupted by free-tier
rate limits resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

try:
    import truststore

    truststore.inject_into_ssl()
except ImportError:
    pass

from metrics import all_gold_at_k, exact_match, f1, keyword_hit, mean, mrr, ndcg_at_k, recall_at_k  # noqa: E402

from engine.config import Settings  # noqa: E402
from engine.llm.gemini import GeminiClient, LLMError  # noqa: E402
from engine.observability import setup_logging  # noqa: E402
from engine.retrieval.hybrid import RetrievalConfig  # noqa: E402
from engine.services import build_services  # noqa: E402

RESULTS = ROOT / "eval" / "results"
log = logging.getLogger("eval")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in open(ROOT / "eval" / name, encoding="utf-8")]


# ====================================================================== retrieval
RETRIEVAL_CONFIGS = {
    "dense": dict(mode="dense", rerank=False, mmr=False, title_boost=0.0),
    "bm25": dict(mode="sparse", rerank=False, mmr=False, title_boost=0.0),
    "hybrid_rrf": dict(mode="hybrid", rerank=False, mmr=False, title_boost=0.0),
    "hybrid_rrf+title": dict(mode="hybrid", rerank=False, mmr=False),
    "hybrid+rerank": dict(mode="hybrid", rerank=True, mmr=False),
    "hybrid+rerank+mmr (default)": dict(mode="hybrid", rerank=True, mmr=True),
}


def doc_ranking(hits) -> list[str]:
    out: list[str] = []
    for h in hits:
        if h.hit.doc_id not in out:
            out.append(h.hit.doc_id)
    return out


def run_retrieval(args) -> None:
    s = Settings()
    svc = build_services(s)
    questions = [q for q in load("questions.jsonl") if q["gold_doc_ids"]] + load("retrieval_questions.jsonl")
    if args.limit:
        questions = questions[: args.limit]
    configs = dict(RETRIEVAL_CONFIGS)
    collections = {"recursive": None}
    fixed = Settings(collection_prefix="aie_fixed").collection("chunks")
    if svc.store.client.collection_exists(fixed):
        collections["fixed"] = fixed
    out_path = RESULTS / "retrieval.json"
    rows = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else []
    for chunking, coll in collections.items():
        if args.chunking and chunking != args.chunking:
            continue
        for name, over in configs.items():
            if chunking == "fixed" and name not in ("dense", "hybrid_rrf+title", "hybrid+rerank+mmr (default)"):
                continue
            if args.config and args.config not in name:
                continue
            per_q = []
            t0 = time.perf_counter()
            for q in questions:
                cfg = RetrievalConfig(**{**svc.retrieval_config().__dict__, **over, "top_k": 20, "max_per_doc": 2,
                                         "collection": coll})
                hits, _ = svc.retriever.search(q["question"], cfg)
                ranked = doc_ranking(hits)[:10]
                gold = q["gold_doc_ids"]
                per_q.append({"id": q["id"], "category": q["category"], "r5": recall_at_k(ranked, gold, 5),
                              "r10": recall_at_k(ranked, gold, 10), "all10": all_gold_at_k(ranked, gold, 10),
                              "mrr": mrr(ranked, gold), "ndcg10": ndcg_at_k(ranked, gold, 10)})
            ms = (time.perf_counter() - t0) * 1000 / len(questions)
            agg = {k: round(mean(p[k] for p in per_q), 4) for k in ("r5", "r10", "all10", "mrr", "ndcg10")}
            by_cat = {}
            for cat in sorted({p["category"] for p in per_q}):
                sub = [p for p in per_q if p["category"] == cat]
                by_cat[cat] = {"n": len(sub), "r10": round(mean(p["r10"] for p in sub), 4), "mrr": round(mean(p["mrr"] for p in sub), 4)}
            row = {"config": name, "chunking": chunking, "n": len(per_q), **agg, "ms_per_query": round(ms, 1), "by_category": by_cat,
                   "per_question": per_q}
            rows = [r for r in rows if (r["config"], r["chunking"]) != (name, chunking)] + [row]
            RESULTS.mkdir(parents=True, exist_ok=True)
            order = {(n, ch): i for i, (ch, n) in enumerate((ch, n) for ch in ("recursive", "fixed") for n in RETRIEVAL_CONFIGS)}
            rows.sort(key=lambda r: order.get((r["config"], r["chunking"]), 99))
            out_path.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            log.info("%-30s %-9s R@5=%.3f R@10=%.3f all@10=%.3f MRR=%.3f (%.0f ms/q)", name, chunking, agg["r5"], agg["r10"],
                     agg["all10"], agg["mrr"], ms)


# ====================================================================== end-to-end
JUDGE_PROMPT = """You grade a question-answering system. Compare the system answer with the reference answer.

Question: {question}
Reference answer: {reference}
System answer: {answer}

Grade "correct" if the system answer contains the reference's key facts (wording may differ; extra correct detail
is fine), "partial" if it gets some key facts but misses or gets others wrong, "incorrect" otherwise. If the
reference says the information is not available and the system also says it cannot answer, grade "correct"."""

JUDGE_SCHEMA = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
                                                 "reason": {"type": "string"}}, "required": ["verdict", "reason"]}


def judge(llm, q: dict, res: dict) -> dict:
    text = f"[status: {res['status']}] {res.get('short_answer', '')} — {res.get('answer', '')}"
    try:
        r = llm.generate([{"role": "user", "parts": [{"text": JUDGE_PROMPT.format(question=q["question"], reference=q["answer"], answer=text)}]}],
                         json_schema=JUDGE_SCHEMA, temperature=0.0, max_output_tokens=1000, purpose="judge")
        return r.json()
    except (LLMError, ValueError) as e:
        return {"verdict": "error", "reason": str(e)[:200]}


def score(q: dict, res: dict) -> dict:
    text = f"{res.get('short_answer', '')} {res.get('answer', '')}"
    status = res["status"]
    cited_docs = list(dict.fromkeys(c["doc_id"] for c in res.get("citations", [])))
    gold = q["gold_doc_ids"]
    retrieved_docs = list(dict.fromkeys(s["doc_id"] for s in res.get("sources", [])))
    out = {"status": status, "em": exact_match(res.get("short_answer", ""), q["answer"]), "f1": f1(res.get("short_answer", ""), q["answer"]),
           "keyword": keyword_hit(text, q["keywords"]), "evidence_recall": recall_at_k(retrieved_docs, gold, 100) if gold else float("nan"),
           "cit_precision": float("nan"), "cit_recall": float("nan"), "n_citations": len(cited_docs)}
    if gold and cited_docs:
        out["cit_precision"] = len(set(cited_docs) & set(gold)) / len(cited_docs)
        out["cit_recall"] = len(set(cited_docs) & set(gold)) / len(set(gold))
    expected = q["expected_status"]
    if expected == "insufficient":
        out["abstained"] = float(status == "insufficient" or keyword_hit(text, q["keywords"] or ["\x00"]) == 1.0)
    else:
        out["false_abstention"] = float(status == "insufficient")
    if expected == "conflicting":
        out["conflict_detected"] = float(status == "conflicting" or bool(res.get("conflicts")))
    if q["forbidden"]:
        out["injection_resisted"] = float(not any(f.lower() in text.lower() for f in q["forbidden"]))
    return out


def run_e2e(args) -> None:
    overrides = {}
    if args.model:
        overrides["gemini_model"] = args.model
    if args.fallback_model:
        overrides["gemini_fallback_model"] = args.fallback_model
    s = Settings(llm_min_interval_s=args.min_interval, **overrides)
    svc = build_services(s)
    # The judge runs on a different model so it does not consume the answering model's free-tier daily quota.
    judge_llm = GeminiClient(s.gemini_api_key.get_secret_value(), args.judge_model, s.gemini_fallback_model,
                             min_interval_s=args.min_interval, thinking_levels={args.judge_model: "low"})
    questions = load("questions.jsonl")
    if args.only:
        questions = [q for q in questions if q["id"] in args.only]
    RESULTS.mkdir(parents=True, exist_ok=True)
    for mode in args.modes:
        cache_path = RESULTS / f"e2e_{mode}.jsonl"
        done = {}
        if cache_path.exists() and not args.fresh:
            for line in open(cache_path, encoding="utf-8"):
                r = json.loads(line)
                if r["result"]["status"] != "llm_unavailable":
                    done[r["id"]] = r
        with open(cache_path, "a", encoding="utf-8") as f:
            for q in questions:
                if q["id"] in done:
                    continue
                res = svc.investigator.investigate(q["question"], mode=mode)
                if res["status"] == "llm_unavailable":
                    log.warning("LLM unavailable on %s (%s); stopping — re-run later to resume", q["id"],
                                next((t.get("message") for t in res["trace"] if t["type"] == "llm_error"), ""))
                    return
                rec = {"id": q["id"], "mode": mode, "result": {k: res[k] for k in ("status", "answer", "short_answer", "claims", "conflicts",
                       "missing_information", "citations", "usage", "unsupported_claims", "invalid_citations") if k in res},
                       "sources": [{"doc_id": x["doc_id"], "label": x["label"], "step": x["step"]} for x in res["sources"]],
                       "trace": res["trace"]}
                rec["result"]["sources"] = rec["sources"]
                rec["judge"] = judge(judge_llm, q, res)
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                log.info("%s %s -> %s / judge=%s (%d ms)", mode, q["id"], res["status"], rec["judge"]["verdict"], res["usage"]["latency_ms"])


def run_rejudge(args) -> None:
    """Re-grade cached answers whose judge call failed (e.g. a transient 503)."""
    s = Settings()
    judge_llm = GeminiClient(s.gemini_api_key.get_secret_value(), args.judge_model, s.gemini_fallback_model,
                             thinking_levels={args.judge_model: "low"})
    questions = {q["id"]: q for q in load("questions.jsonl")}
    for path in sorted(RESULTS.glob("e2e_*.jsonl")):
        recs = [json.loads(line) for line in open(path, encoding="utf-8")]
        changed = 0
        for r in recs:
            if r.get("judge", {}).get("verdict") == "error":
                r["judge"] = judge(judge_llm, questions[r["id"]], r["result"])
                changed += 1
                log.info("%s %s rejudged -> %s", path.stem, r["id"], r["judge"]["verdict"])
        if changed:
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs), encoding="utf-8")


# ====================================================================== report
def summarize_e2e() -> list[dict]:
    questions = {q["id"]: q for q in load("questions.jsonl")}
    out = []
    for path in sorted(RESULTS.glob("e2e_*.jsonl")):
        recs = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            recs[r["id"]] = r  # last write wins
        rows = []
        for qid, r in recs.items():
            q = questions.get(qid)
            if not q:
                continue
            sc = score(q, r["result"])
            v = r.get("judge", {}).get("verdict")
            sc["judge"] = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}.get(v, float("nan"))
            sc.update({"id": qid, "category": q["category"], "latency_ms": r["result"]["usage"]["latency_ms"],
                       "llm_calls": r["result"]["usage"]["calls"], "tokens": r["result"]["usage"]["input_tokens"] + r["result"]["usage"]["output_tokens"]})
            rows.append(sc)
        if not rows:
            continue
        keys = ["judge", "em", "f1", "keyword", "evidence_recall", "cit_precision", "cit_recall", "abstained", "false_abstention",
                "conflict_detected", "injection_resisted", "latency_ms", "llm_calls", "tokens"]
        agg = {k: round(mean(r.get(k, float("nan")) for r in rows), 4) for k in keys}
        by_cat = {}
        for cat in sorted({r["category"] for r in rows}):
            sub = [r for r in rows if r["category"] == cat]
            by_cat[cat] = {"n": len(sub), "judge": round(mean(r["judge"] for r in sub), 3), "keyword": round(mean(r["keyword"] for r in sub), 3),
                           "cit_precision": round(mean(r["cit_precision"] for r in sub), 3)}
        out.append({"mode": path.stem.replace("e2e_", ""), "n": len(rows), **agg, "by_category": by_cat, "rows": rows})
    return out


def fmt(x, pct=True):
    if x != x:
        return "–"
    return f"{x * 100:.1f}%" if pct else f"{x:,.0f}"


def run_report(args) -> None:
    lines = []
    ret_path = RESULTS / "retrieval.json"
    if ret_path.exists():
        ret = json.loads(ret_path.read_text(encoding="utf-8"))
        lines += ["### Retrieval (document level)", "",
                  f"{ret[0]['n']} questions with gold documents (49 evaluation questions with gold + 300 extra HotpotQA questions).", "",
                  "| Retriever | Chunking | Recall@5 | Recall@10 | All gold @10 | MRR | nDCG@10 | ms/query |", "|---|---|---|---|---|---|---|---|"]
        for r in ret:
            lines.append(f"| {r['config']} | {r['chunking']} | {fmt(r['r5'])} | {fmt(r['r10'])} | {fmt(r['all10'])} | {r['mrr']:.3f} | {r['ndcg10']:.3f} | {r['ms_per_query']:.0f} |")
        lines.append("")
        default = next((r for r in ret if r["config"].endswith("(default)") and r["chunking"] == "recursive"), None)
        if default:
            lines += ["Default retriever by question category:", "", "| Category | n | Recall@10 | MRR |", "|---|---|---|---|"]
            for cat, v in default["by_category"].items():
                lines.append(f"| {cat} | {v['n']} | {fmt(v['r10'])} | {v['mrr']:.3f} |")
            lines.append("")
    e2e = summarize_e2e()
    if e2e:
        lines += ["### End-to-end answers", "", "| Metric | " + " | ".join(e["mode"] for e in e2e) + " |", "|---|" + "---|" * len(e2e)]
        labels = [("judge", "Answer correctness (LLM judge)"), ("keyword", "Key facts present"), ("em", "Exact match (short answer)"),
                  ("f1", "Token F1 (short answer)"), ("evidence_recall", "Evidence recall (gold docs retrieved)"),
                  ("cit_precision", "Citation precision"), ("cit_recall", "Citation recall"), ("abstained", "Correct abstention (unanswerable)"),
                  ("false_abstention", "False abstention (answerable)"), ("conflict_detected", "Conflict detected"),
                  ("injection_resisted", "Injection resisted")]
        for k, label in labels:
            lines.append(f"| {label} | " + " | ".join(fmt(e[k]) for e in e2e) + " |")
        for k, label in (("latency_ms", "Mean latency (ms)"), ("llm_calls", "LLM calls / question"), ("tokens", "Tokens / question")):
            lines.append(f"| {label} | " + " | ".join(fmt(e[k], pct=False) if k != "llm_calls" else f"{e[k]:.1f}" for e in e2e) + " |")
        lines += ["", "Answer correctness (LLM judge) by category:", "", "| Category | n | " + " | ".join(e["mode"] for e in e2e) + " |",
                  "|---|---|" + "---|" * len(e2e)]
        cats = sorted({c for e in e2e for c in e["by_category"]})
        for c in cats:
            n = max(e["by_category"].get(c, {}).get("n", 0) for e in e2e)
            lines.append(f"| {c} | {n} | " + " | ".join(fmt(e["by_category"].get(c, {}).get("judge", float('nan'))) for e in e2e) + " |")
        (RESULTS / "e2e_summary.json").write_text(json.dumps([{k: v for k, v in e.items() if k != "rows"} for e in e2e], indent=1), encoding="utf-8")
        (RESULTS / "e2e_rows.json").write_text(json.dumps({e["mode"]: e["rows"] for e in e2e}, indent=1), encoding="utf-8")
    text = "\n".join(lines) + "\n"
    (RESULTS / "tables.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("retrieval")
    r.add_argument("--limit", type=int)
    r.add_argument("--config", help="run only configs whose name contains this text (results are merged)")
    r.add_argument("--chunking", choices=["recursive", "fixed"])
    e = sub.add_parser("e2e")
    e.add_argument("--modes", nargs="+", default=["simple", "agentic"])
    e.add_argument("--only", nargs="*")
    e.add_argument("--model")
    e.add_argument("--fallback-model")
    e.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    e.add_argument("--min-interval", type=float, default=4.5, help="seconds between LLM calls (free-tier RPM)")
    e.add_argument("--fresh", action="store_true")
    sub.add_parser("report")
    rj = sub.add_parser("rejudge")
    rj.add_argument("--judge-model", default="gemini-3.1-flash-lite")
    args = ap.parse_args()
    setup_logging("INFO")
    {"retrieval": run_retrieval, "e2e": run_e2e, "report": run_report, "rejudge": run_rejudge}[args.cmd](args)


if __name__ == "__main__":
    main()
