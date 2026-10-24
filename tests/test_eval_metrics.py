import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from metrics import all_gold_at_k, exact_match, f1, keyword_hit, mean, mrr, ndcg_at_k, recall_at_k  # noqa: E402


def test_retrieval_metrics():
    ranked = ["x", "a", "y", "b"]
    assert recall_at_k(ranked, ["a", "b"], 2) == 0.5
    assert all_gold_at_k(ranked, ["a", "b"], 4) == 1.0
    assert mrr(ranked, ["a", "b"]) == 0.5
    assert mrr(ranked, ["z"]) == 0.0
    assert 0 < ndcg_at_k(ranked, ["a", "b"], 4) < 1
    assert math.isnan(recall_at_k(ranked, [], 5))


def test_answer_metrics():
    assert exact_match("The Beatles!", "beatles") == 1.0
    assert f1("Tim Burton directed it", "Tim Burton") == 2 * 0.5 * 1 / 1.5
    assert keyword_hit("CHG-1187 cut the pool to 80", ["chg-1187", "80", "200"]) == 2 / 3
    assert mean([1.0, float("nan"), 0.0]) == 0.5
