"""Post-hoc verification of the model's citations against the evidence it was shown.

The LLM may cite ids that do not exist, or attach a citation to a claim the passage does not
support. We (1) drop unknown ids, (2) score each claim's lexical support against its cited
passages, and (3) downgrade an "answered" response with no valid citations to "insufficient".
"""

from __future__ import annotations

import re
from typing import Any

from engine.retrieval.sparse import tokenize

CITE_RX = re.compile(r"\[(S\d+)\]")


def support_score(statement: str, passages: list[str]) -> float:
    """Fraction of the statement's content terms that occur in the cited passages."""
    st = set(tokenize(statement))
    if not st:
        return 1.0
    pt: set[str] = set()
    for p in passages:
        pt |= set(tokenize(p))
    return len(st & pt) / len(st)


def verify(answer: dict[str, Any], evidence: dict[str, str], weak_threshold: float = 0.35) -> dict[str, Any]:
    """evidence maps label (S#) -> passage text. Returns a cleaned copy of answer plus diagnostics."""
    out = dict(answer)
    valid_labels = set(evidence)
    invalid: set[str] = set()

    claims = []
    for c in answer.get("claims", []) or []:
        cites = [x for x in c.get("citations", []) if x in valid_labels]
        invalid |= {x for x in c.get("citations", []) if x not in valid_labels}
        score = support_score(c.get("statement", ""), [evidence[x] for x in cites]) if cites else 0.0
        claims.append({**c, "citations": cites, "support": round(score, 2), "supported": bool(cites) and score >= weak_threshold})
    out["claims"] = claims

    conflicts = []
    for c in answer.get("conflicts", []) or []:
        cites = [x for x in c.get("citations", []) if x in valid_labels]
        invalid |= {x for x in c.get("citations", []) if x not in valid_labels}
        conflicts.append({**c, "citations": cites})
    out["conflicts"] = conflicts

    text_cites = CITE_RX.findall(answer.get("answer", ""))
    invalid |= {x for x in text_cites if x not in valid_labels}
    for bad in invalid:
        out["answer"] = out.get("answer", "").replace(f"[{bad}]", "")

    used = sorted({x for x in text_cites if x in valid_labels} | {x for c in claims for x in c["citations"]},
                  key=lambda s: int(s[1:]))
    out["cited_labels"] = used
    out["invalid_citations"] = sorted(invalid)
    out["unsupported_claims"] = sum(1 for c in claims if not c["supported"])
    if out.get("status") == "answered" and not used:
        out["status"] = "insufficient"
        out["verification_note"] = "answer had no valid citations; downgraded to insufficient"
    if out.get("status") == "answered" and conflicts and any(len(c["citations"]) >= 2 for c in conflicts):
        out["has_conflicts"] = True
    return out
