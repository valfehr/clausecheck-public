# Phase 1 — experiments (test split; one change at a time vs E0)

| metric | SC-run1 | RAG10-sc | RAG5-sc |
|---|---|---|---|
| severity acc (std + hard) | 93% (39/42) | 95% (40/42) | 95% (40/42) |
| hard-case acc | 93% (14/15) | 100% (15/15) | 100% (15/15) |
| red recall (all reds) | 100% (21/21) | 100% (21/21) | 100% (21/21) |
| critical miss red→green | 0% (0/21) | 0% (0/21) | 0% (0/21) |
| citation hit | 96% (52/54) | 98% (53/54) | 94% (51/54) |
| citation verified | 100% (66/66) | 100% (66/66) | 98% (65/66) |
| $ per case | $0.00313 | $0.00094 | $0.00071 |
| $ per correct verdict | $0.00324 | $0.00074 | $0.00058 |
| p50 latency | 2.7s | 1.8s | 1.8s |
| cases / errors | 66 / 0 | 66 / 0 | 66 / 0 |
