# Phase 1 — experiments (test split; one change at a time vs E0)

| metric | E0-haiku-low | E1-haiku-medium | E2-haiku-low-fewshot |
|---|---|---|---|
| severity acc (std + hard) | 95% (40/42) | 95% (40/42) | 88% (37/42) |
| hard-case acc | 100% (15/15) | 100% (15/15) | 87% (13/15) |
| red recall (all reds) | 100% (21/21) | 100% (21/21) | 100% (21/21) |
| critical miss red→green | 0% (0/21) | 0% (0/21) | 0% (0/21) |
| citation hit | 93% (50/54) | 96% (52/54) | 94% (51/54) |
| citation verified | 95% (63/66) | 98% (65/66) | 98% (65/66) |
| $ per case | $0.00178 | $0.00188 | $0.00184 |
| $ per correct verdict | $0.00186 | $0.00197 | $0.00210 |
| p50 latency | 2.4s | 3.8s | 2.7s |
| cases / errors | 66 / 0 | 66 / 0 | 66 / 0 |
