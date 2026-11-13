# Full contract vs RAG top-10 (Claude Haiku 5.5, re-ask-3x, 3 fresh runs per split)

| | Full contract (test) | RAG top-10 (test) | Full contract (holdout) | RAG top-10 (holdout) | Full contract (long) | RAG top-10 (long) |
|---|---|---|---|---|---|---|
| Cases per run | 66 | 66 | 44 | 44 | 28 | 28 |
| **Red recall** (≥ 90%), pooled over 3 runs | 100% of 63 | 100% of 63 | 94% of 36 | 100% of 36 | 91% of 45 | 87% of 45 |
| **Critical misses** red → green (0) | 0 | 0 | 0 | 0 | 0 | 0 |
| Clause genuinely absent → found red (retrieval suite) | 21/21 | 21/21 | 3/3 | 3/3 | 0/0 | 0/0 |
| **Citations verified** (≥ 90%) | 100% | 100% | 99% | 100% | 100% | 100% |
| Severity accuracy (std + hard) | 96% of 126 | 92% of 126 | 93% of 87 | 97% of 87 | 87% of 84 | 89% of 84 |
| Cases whose verdict changed across runs | 2 (3%) | 2 (3%) | 2 (5%) | 3 (7%) | 3 (11%) | 0 (0%) |
| **RED ↔ GREEN flips** (0) | 0 | 0 | 0 | 0 | 0 | 0 |
| **Odd answers shown without a flag** (0) | 0 | 0 | 1 | 2 | 0 | 0 |
| **Checks flagged for a human** (≤ 15%) | 5% | 5% | 6% | 10% | 6% | 4% |
| Input tokens per check (median) | 12,638 | 3,151 | 16,854 | 3,176 | 56,366 | 3,126 |
| $ per check, eval (no prompt caching) | $0.00304 | $0.00095 | $0.00361 | $0.00118 | $0.01609 | $0.00102 |
| p50 latency per check | 2.7s | 1.9s | 3.0s | 2.1s | 4.0s | 1.8s |
| **Gate** | ✅ PASS | ✅ PASS | ❌ BLOCK | ❌ BLOCK | ✅ PASS | ❌ BLOCK |
