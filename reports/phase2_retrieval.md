# Phase 2 — retrieval recall (Elasticsearch, per contract, per rule)

108 (contract, rule) cases with a known relevant clause; dev + test only.

| method | split | hit@1 | hit@3 | hit@5 | hit@10 | all@5 (multi-span) | MRR | ms/query |
|---|---|---|---|---|---|---|---|---|
| bm25 | dev | 89% | 94% | 96% | 96% | 11/13 | 0.92 | 8 |
| bm25 | test | 93% | 96% | 98% | 98% | 15/17 | 0.95 | 7 |
| dense | dev | 76% | 91% | 94% | 100% | 9/13 | 0.84 | 31 |
| dense | test | 78% | 93% | 93% | 98% | 11/17 | 0.85 | 29 |
| hybrid | dev | 91% | 96% | 96% | 98% | 12/13 | 0.93 | 35 |
| hybrid | test | 91% | 94% | 96% | 100% | 14/17 | 0.93 | 37 |
| hybrid+rerank | dev | 81% | 91% | 93% | 96% | 9/13 | 0.87 | 611 |
| hybrid+rerank | test | 81% | 93% | 94% | 96% | 11/17 | 0.88 | 523 |

## hit@5 by rule (test)

| method | governing_law | anti_assignment | cap_on_liability | audit_rights | termination_for_convenience | insurance | ip_ownership | change_of_control |
|---|---|---|---|---|---|---|---|---|
| bm25 | 5/5 | 9/9 | 7/7 | 5/6 | 9/9 | 6/6 | 9/9 | 3/3 |
| dense | 5/5 | 8/9 | 6/7 | 6/6 | 9/9 | 6/6 | 7/9 | 3/3 |
| hybrid | 5/5 | 9/9 | 7/7 | 5/6 | 9/9 | 6/6 | 8/9 | 3/3 |
| hybrid+rerank | 5/5 | 9/9 | 6/7 | 6/6 | 9/9 | 6/6 | 7/9 | 3/3 |
