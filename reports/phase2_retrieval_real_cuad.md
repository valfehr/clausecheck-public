# Phase 2 — retrieval recall (Elasticsearch, per contract, per rule)

34 (contract, rule) cases with a known relevant clause; dev + test only — **real CUAD lawyer spans only**.

| method | split | hit@1 | hit@3 | hit@5 | hit@10 | all@5 (multi-span) | MRR | ms/query |
|---|---|---|---|---|---|---|---|---|
| bm25 | dev | 71% | 82% | 88% | 88% | 7/9 | 0.78 | 4 |
| bm25 | test | 76% | 88% | 94% | 94% | 9/11 | 0.84 | 5 |
| dense | dev | 71% | 82% | 88% | 100% | 6/9 | 0.80 | 31 |
| dense | test | 59% | 76% | 76% | 94% | 6/11 | 0.68 | 31 |
| hybrid | dev | 82% | 88% | 88% | 94% | 8/9 | 0.86 | 34 |
| hybrid | test | 76% | 82% | 88% | 100% | 8/11 | 0.81 | 33 |
| hybrid+rerank | dev | 71% | 88% | 88% | 88% | 6/9 | 0.79 | 628 |
| hybrid+rerank | test | 59% | 76% | 82% | 88% | 5/11 | 0.70 | 541 |

## hit@5 by rule (test)

| method | governing_law | anti_assignment | cap_on_liability | audit_rights | termination_for_convenience | insurance | ip_ownership | change_of_control |
|---|---|---|---|---|---|---|---|---|
| bm25 | 1/1 | 3/3 | 3/3 | 2/3 | 2/2 | 1/1 | 3/3 | 1/1 |
| dense | 1/1 | 2/3 | 2/3 | 3/3 | 2/2 | 1/1 | 1/3 | 1/1 |
| hybrid | 1/1 | 3/3 | 3/3 | 2/3 | 2/2 | 1/1 | 2/3 | 1/1 |
| hybrid+rerank | 1/1 | 3/3 | 2/3 | 3/3 | 2/2 | 1/1 | 1/3 | 1/1 |
