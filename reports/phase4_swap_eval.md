# Phase 4 — model swap eval (test split, RAG top-10, re-ask-3x; only the model changes)

| | Claude Haiku 5.5 (incumbent) | llama-3.1-8b | gemma-3-4b | gemma-3-4b-tuned | gemma-3-4b-tuned-greedy | gemma-3-4b-tuned-e2 | gemma-3-4b-tuned-e2-greedy | gemma-3-4b-tuned-parties | gemma-3-4b-tuned-ptyline |
|---|---|---|---|---|---|---|---|---|---|
| runs | 1 | 3 | 3 | 3 | 3 | 3 | 3 | 3 | 3 |
| **Red recall** (gate ≥ 90%) | 100% of 21 | 84% of 63 | 24% of 63 | 90% of 63 | 90% of 63 | 92% of 63 | 87% of 63 | 89% of 63 | 86% of 63 |
| **Critical misses red→green** (gate 0) | 0 | 3 | 10 | 1 | 2 | 2 | 3 | 5 | 1 |
| **Citations verified** (gate ≥ 90%) | 100% | 73% | 25% | 95% | 97% | 99% | 97% | 98% | 97% |
| **Flip rate over 3 runs** (gate ≤ 5%) | n/a (1 run) | 32% | 20% | 27% | 17% | 23% | 20% | 29% | 32% |
| **RED↔GREEN flips** (gate 0) | n/a | 7 | 1 | 0 | 0 | 3 | 0 | 2 | 1 |
| Severity accuracy (std + hard) | 95% of 42 | 63% of 126 | 43% of 126 | 75% of 126 | 70% of 126 | 69% of 126 | 71% of 126 | 67% of 126 | 63% of 126 |
| Flagged for human review | 3% | 33% | 75% | 11% | 3% | 10% | 3% | 10% | 12% |
| Errors (schema/server) | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| p50 latency per call | 1.8s | 6.4s | 5.4s | 6.1s | 6.1s | 6.1s | 5.9s | 5.9s | 5.8s |
| **Gate** | ✅ PASS | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK | ❌ BLOCK |

## Cost per correct verdict

| | $ per correct verdict | how |
|---|---|---|
| Claude Haiku 5.5 | $0.00154 | API bill, all 66 cases |
| llama-3.1-8b | $0.00327 | 5.1 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b | $0.00508 | 5.4 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned | $0.00220 | 4.1 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned-greedy | $0.00174 | 3.0 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned-e2 | $0.00220 | 3.8 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned-e2-greedy | $0.00166 | 2.9 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned-parties | $0.00215 | 3.6 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |
| gemma-3-4b-tuned-ptyline | $0.00235 | 3.7 min/run of GPU time × $1.01/h, only if the GPU is busy 100% of the time |

## Accuracy per rule (severity cases, pooled over runs)

| rule | Claude Haiku 5.5 (incumbent) | llama-3.1-8b | gemma-3-4b | gemma-3-4b-tuned | gemma-3-4b-tuned-greedy | gemma-3-4b-tuned-e2 | gemma-3-4b-tuned-e2-greedy | gemma-3-4b-tuned-parties | gemma-3-4b-tuned-ptyline |
|---|---|---|---|---|---|---|---|---|---|
| anti_assignment | 6/6 (100%) | 12/18 (67%) | 9/18 (50%) | 17/18 (94%) | 15/18 (83%) | 14/18 (78%) | 15/18 (83%) | 15/18 (83%) | 15/18 (83%) |
| audit_rights | 4/4 (100%) | 8/12 (67%) | 3/12 (25%) | 8/12 (67%) | 9/12 (75%) | 5/12 (42%) | 11/12 (92%) | 8/12 (67%) | 5/12 (42%) |
| cap_on_liability | 4/4 (100%) | 9/12 (75%) | 6/12 (50%) | 9/12 (75%) | 7/12 (58%) | 11/12 (92%) | 9/12 (75%) | 7/12 (58%) | 6/12 (50%) |
| change_of_control | 2/2 (100%) | 4/6 (67%) | 0/6 (0%) | 3/6 (50%) | 4/6 (67%) | 4/6 (67%) | 3/6 (50%) | 2/6 (33%) | 3/6 (50%) |
| governing_law | 4/5 (80%) | 12/15 (80%) | 11/15 (73%) | 12/15 (80%) | 12/15 (80%) | 12/15 (80%) | 12/15 (80%) | 12/15 (80%) | 12/15 (80%) |
| insurance | 6/6 (100%) | 11/18 (61%) | 6/18 (33%) | 14/18 (78%) | 10/18 (56%) | 11/18 (61%) | 12/18 (67%) | 11/18 (61%) | 11/18 (61%) |
| ip_ownership | 6/7 (86%) | 6/21 (29%) | 9/21 (43%) | 12/21 (57%) | 11/21 (52%) | 9/21 (43%) | 12/21 (57%) | 8/21 (38%) | 9/21 (43%) |
| termination_for_convenience | 8/8 (100%) | 17/24 (71%) | 10/24 (42%) | 19/24 (79%) | 20/24 (83%) | 21/24 (88%) | 15/24 (62%) | 21/24 (88%) | 18/24 (75%) |

## Error types (expected → predicted, pooled)

- **Claude Haiku 5.5 (incumbent)**: yellow→green ×1, green→yellow ×1
- **llama-3.1-8b**: yellow→green ×32, yellow→red ×7, red→yellow ×7, red→green ×3, green→yellow ×1
- **gemma-3-4b**: red→yellow ×38, yellow→green ×32, red→green ×10, green→yellow ×7, yellow→red ×1, green→red ×1
- **gemma-3-4b-tuned**: green→yellow ×11, yellow→green ×10, red→yellow ×5, green→red ×3, yellow→red ×2, red→green ×1
- **gemma-3-4b-tuned-greedy**: yellow→green ×16, green→yellow ×15, red→yellow ×4, yellow→red ×2, red→green ×2
- **gemma-3-4b-tuned-e2**: yellow→green ×12, green→yellow ×12, yellow→red ×10, red→yellow ×3, red→green ×2
- **gemma-3-4b-tuned-e2-greedy**: yellow→green ×13, green→yellow ×10, yellow→red ×6, red→yellow ×5, red→green ×3
- **gemma-3-4b-tuned-parties**: yellow→green ×18, green→yellow ×10, red→green ×5, yellow→red ×5, green→red ×2, red→yellow ×2
- **gemma-3-4b-tuned-ptyline**: green→yellow ×15, yellow→green ×13, yellow→red ×12, red→yellow ×8, red→green ×1
