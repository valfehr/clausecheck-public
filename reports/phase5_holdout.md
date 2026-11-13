# Model swap eval (HOLDOUT split, RAG top-10, re-ask-3x; only the model changes)

| | Claude Haiku 5.5 (incumbent) | llama-3.1-8b | gemma-3-4b | gemma-3-4b-tuned | gemma-3-4b-tuned-e2 | gemma-3-4b-tuned-parties |
|---|---|---|---|---|---|---|
| runs | 3 | 3 | 3 | 3 | 3 | 3 |
| **Red recall** (gate ≥ 90%) | 100% of 36 | 92% of 36 | 39% of 36 | 75% of 36 | 89% of 36 | 86% of 36 |
| **Critical misses red→green** (gate 0) | 0 | 2 | 3 | 3 | 1 | 4 |
| **Citations verified** (gate ≥ 90%) | 100% | 77% | 28% | 95% | 98% | 99% |
| **Flip rate over 3 runs** (gate ≤ 5%) | 7% | 30% | 25% | 27% | 27% | 25% |
| **RED↔GREEN flips** (gate 0) | 0 | 2 | 1 | 2 | 1 | 5 |
| Severity accuracy (std + hard) | 97% of 87 | 56% of 87 | 44% of 87 | 63% of 87 | 68% of 87 | 70% of 87 |
| Flagged for human review | 10% | 28% | 72% | 13% | 8% | 8% |
| Errors (schema/server) | 0 | 0 | 0 | 0 | 0 | 0 |
| p50 latency per call | 2.1s | 4.7s | 5.2s | 6.1s | 6.0s | 6.5s |
| **Gate** | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK |

## Cost per correct verdict

| | $ per correct verdict | how |
|---|---|---|
| Claude Haiku 5.5 | $0.00181 | API bill, one run, all 44 cases |
| llama-3.1-8b | $0.00252 | 2.4 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b | $0.00445 | 3.3 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned | $0.00277 | 3.0 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned-e2 | $0.00238 | 2.8 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned-parties | $0.00273 | 3.3 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |

## Accuracy per rule (severity cases, pooled over runs)

| rule | Claude Haiku 5.5 (incumbent) | llama-3.1-8b | gemma-3-4b | gemma-3-4b-tuned | gemma-3-4b-tuned-e2 | gemma-3-4b-tuned-parties |
|---|---|---|---|---|---|---|
| anti_assignment | 12/12 (100%) | 8/12 (67%) | 2/12 (17%) | 12/12 (100%) | 11/12 (92%) | 12/12 (100%) |
| audit_rights | 15/15 (100%) | 9/15 (60%) | 8/15 (53%) | 10/15 (67%) | 8/15 (53%) | 11/15 (73%) |
| cap_on_liability | 12/12 (100%) | 9/12 (75%) | 8/12 (67%) | 9/12 (75%) | 10/12 (83%) | 9/12 (75%) |
| change_of_control | 18/18 (100%) | 8/18 (44%) | 8/18 (44%) | 5/18 (28%) | 9/18 (50%) | 8/18 (44%) |
| governing_law | 9/9 (100%) | 3/9 (33%) | 3/9 (33%) | 7/9 (78%) | 7/9 (78%) | 6/9 (67%) |
| insurance | 9/9 (100%) | 7/9 (78%) | 8/9 (89%) | 9/9 (100%) | 8/9 (89%) | 9/9 (100%) |
| ip_ownership | 6/9 (67%) | 2/9 (22%) | 1/9 (11%) | 3/9 (33%) | 3/9 (33%) | 3/9 (33%) |
| termination_for_convenience | 3/3 (100%) | 3/3 (100%) | 0/3 (0%) | 0/3 (0%) | 3/3 (100%) | 3/3 (100%) |

## Error types (expected → predicted, pooled)

- **Claude Haiku 5.5 (incumbent)**: green→red ×2, green→yellow ×1
- **llama-3.1-8b**: yellow→green ×20, yellow→red ×9, green→red ×5, red→green ×2, red→yellow ×1, green→yellow ×1
- **gemma-3-4b**: red→yellow ×19, yellow→green ×18, yellow→red ×5, red→green ×3, green→red ×3, green→yellow ×1
- **gemma-3-4b-tuned**: yellow→red ×8, red→yellow ×6, yellow→green ×6, green→red ×5, green→yellow ×4, red→green ×3
- **gemma-3-4b-tuned-e2**: yellow→green ×10, yellow→red ×7, green→red ×4, green→yellow ×3, red→yellow ×3, red→green ×1
- **gemma-3-4b-tuned-parties**: yellow→green ×8, yellow→red ×7, red→green ×4, green→red ×4, green→yellow ×2, red→yellow ×1
