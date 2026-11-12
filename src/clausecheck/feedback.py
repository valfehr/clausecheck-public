"""Phase 7.3: route lawyer feedback to where it improves the system, and refuse labels from rubber-stampers.

Inputs (SQLite; RDS on AWS): `feedback` (verdict decisions, written by the graph's review step), `redline_feedback`
(fix decisions, written by the API), `seeds` (known-answer items planted in review queues), `consent` (tenants that
allow their contracts to be used for training).

Outputs (JSONL in an output folder; versioned S3 prefixes on AWS):
  eval_candidates      severity changes that are corroborated, from reviewers in good standing  → classification evals
  retrieval_candidates "wrong passage" corrections, same rule                                    → retrieval evals
  training_pool        approved, unflagged verdicts from good reviewers of consenting tenants   → future tuning/few-shot
  gold_redlines        accepted/edited fixes (the lawyer's final wording)                       → drafter examples
  negative_redlines    rejected fixes with a reason                                             → drafter/verifier evals
  audit_queue          uncorroborated changes, reviewer disagreements, untrusted reviewers      → a senior lawyer
  rule_backlog         "rule gap" reports                                                       → the legal team
Eval candidates are tagged split="feedback": they never enter the release holdout.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

GOOD_STANDING = {"min_seeds_caught": 2, "max_approval_rate": 0.95}


def ensure_tables(con: sqlite3.Connection) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS feedback (id INTEGER PRIMARY KEY AUTOINCREMENT, contract_hash TEXT,
                   doc_key TEXT, rule_id TEXT, prompt_version TEXT, model TEXT, model_status TEXT,
                   needs_human INTEGER, reviewer TEXT, action TEXT, new_status TEXT, reason_code TEXT,
                   comment TEXT, created REAL)""")
    con.execute("CREATE TABLE IF NOT EXISTS seeds (contract_hash TEXT, rule_id TEXT, true_status TEXT, shown_status TEXT, "
                "PRIMARY KEY (contract_hash, rule_id))")
    con.execute("CREATE TABLE IF NOT EXISTS consent (tenant TEXT PRIMARY KEY, training INTEGER)")
    con.execute("""CREATE TABLE IF NOT EXISTS redline_feedback (fix_id TEXT PRIMARY KEY, tenant TEXT, reviewer TEXT,
                   rule_id TEXT, contract_hash TEXT, proposed TEXT, final TEXT, action TEXT, reason_code TEXT,
                   comment TEXT, edit_ratio REAL, created REAL)""")


def _rows(con, q, *a) -> list[dict]:
    con.row_factory = sqlite3.Row
    return [dict(r) for r in con.execute(q, a).fetchall()]


def standing(con: sqlite3.Connection) -> dict[str, dict]:
    """Per reviewer: seeds caught, approval rate on real (non-seed) items, good standing."""
    seeds = {(s["contract_hash"], s["rule_id"]): s for s in _rows(con, "SELECT * FROM seeds")}
    out: dict[str, dict] = defaultdict(lambda: {"seeds_seen": 0, "seeds_caught": 0, "items": 0, "approved": 0})
    for f in _rows(con, "SELECT * FROM feedback"):
        r = out[f["reviewer"]]
        s = seeds.get((f["contract_hash"], f["rule_id"]))
        if s:
            r["seeds_seen"] += 1
            r["seeds_caught"] += f["action"] == "change_severity" and f["new_status"] == s["true_status"]
        else:
            r["items"] += 1
            r["approved"] += f["action"] == "approve"
    for r in out.values():
        r["approval_rate"] = round(r["approved"] / max(r["items"], 1), 3)
        r["good_standing"] = (r["seeds_caught"] >= GOOD_STANDING["min_seeds_caught"]
                              and r["approval_rate"] < GOOD_STANDING["max_approval_rate"])
    return dict(out)


def route(con: sqlite3.Connection, out_dir: Path, auditor: Callable[[dict], str | None] | None = None,
          tenant_of: Callable[[dict], str] = lambda f: "") -> dict[str, list[dict]]:
    """auditor(feedback_row) -> the auditor's own status for that item (blind to the reviewer's answer), or None."""
    ensure_tables(con)
    st = standing(con)
    seeds = {(s["contract_hash"], s["rule_id"]) for s in _rows(con, "SELECT * FROM seeds")}
    consent = {r["tenant"]: bool(r["training"]) for r in _rows(con, "SELECT * FROM consent")}
    fb = [f for f in _rows(con, "SELECT * FROM feedback") if (f["contract_hash"], f["rule_id"]) not in seeds]
    out: dict[str, list[dict]] = {k: [] for k in ("eval_candidates", "retrieval_candidates", "training_pool",
                                                  "gold_redlines", "negative_redlines", "audit_queue", "rule_backlog")}
    good = {r for r, s in st.items() if s["good_standing"]}
    final = lambda f: f["new_status"] if f["action"] == "change_severity" else f["model_status"]  # noqa: E731
    by_item: dict[tuple, list[dict]] = defaultdict(list)
    for f in fb:
        by_item[(f["contract_hash"], f["rule_id"])].append(f)

    for key, fs in by_item.items():
        trusted = [f for f in fs if f["reviewer"] in good]
        labels = Counter(final(f) for f in trusted)
        if len(labels) > 1:
            # trusted reviewers disagree: a blind audit may settle it; otherwise a senior lawyer decides
            a = auditor(trusted[0]) if auditor else None
            if a in labels and a != trusted[0]["model_status"]:
                f0 = trusted[0]
                out["eval_candidates"].append({"contract_hash": f0["contract_hash"], "doc_key": f0["doc_key"],
                                               "rule_id": f0["rule_id"], "model_status": f0["model_status"],
                                               "label": a, "split": "feedback",
                                               "corroborated_by": "blind audit settled a disagreement"})
            elif a not in labels:
                out["audit_queue"].append({"item": key, "why": "reviewers disagree", "labels": dict(labels)})
            continue  # if the audit sides with the model's own verdict, nothing new to learn
        for f in fs:
            base = {"contract_hash": f["contract_hash"], "doc_key": f["doc_key"], "rule_id": f["rule_id"],
                    "reviewer": f["reviewer"], "model_status": f["model_status"]}
            if f["reviewer"] not in good:
                if f["action"] != "approve":
                    out["audit_queue"].append({**base, "why": "reviewer not in good standing", "proposed": final(f)})
                continue  # an untrusted approval is not a label
            if f["action"] == "rule_gap" or f["reason_code"] == "rule_gap":
                out["rule_backlog"].append({**base, "comment": f["comment"]})
            elif f["action"] in ("change_severity", "wrong_passage"):
                others = [o for o in trusted if o is not f]
                corroborated = any(final(o) == final(f) for o in others)
                how = "second reviewer"
                if not corroborated and auditor:
                    corroborated, how = auditor(f) == final(f), "blind audit"
                kind = "eval_candidates" if f["action"] == "change_severity" else "retrieval_candidates"
                if corroborated:
                    out[kind].append({**base, "label": final(f), "split": "feedback", "corroborated_by": how})
                else:
                    out["audit_queue"].append({**base, "why": "uncorroborated change", "proposed": final(f)})
            elif f["action"] == "approve" and not f["needs_human"] and consent.get(tenant_of(f), False):
                out["training_pool"].append({**base, "label": f["model_status"]})

    for r in _rows(con, "SELECT * FROM redline_feedback"):
        if r["action"] in ("accept", "edit"):
            out["gold_redlines"].append({k: r[k] for k in ("fix_id", "rule_id", "final", "edit_ratio", "reason_code")})
        else:
            out["negative_redlines"].append({k: r[k] for k in ("fix_id", "rule_id", "proposed", "reason_code", "comment")})

    # dedupe eval candidates: two corroborating reviewers yield one case
    seen, uniq = set(), []
    for c in out["eval_candidates"]:
        k = (c["contract_hash"], c["rule_id"], c["label"])
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    out["eval_candidates"] = uniq
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, rows in out.items():
        (out_dir / f"{k}.jsonl").write_text("\n".join(json.dumps(r, default=str) for r in rows), encoding="utf-8")
    (out_dir / "standing.json").write_text(json.dumps(st, indent=1), encoding="utf-8")
    return out
