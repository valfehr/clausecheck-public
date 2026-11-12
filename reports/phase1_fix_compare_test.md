# Phase 1 — experiments (test split; one change at a time vs E0)

| metric | E0-haiku-low | E0v11-haiku-low |
|---|---|---|
| severity acc (std + hard) | 98% (41/42) | 90% (38/42) |
| hard-case acc | 100% (15/15) | 93% (14/15) |
| red recall (all reds) | 100% (21/21) | 100% (21/21) |
| critical miss red→green | 0% (0/21) | 0% (0/21) |
| citation hit | 93% (50/54) | 98% (53/54) |
| citation verified | 95% (63/66) | 100% (66/66) |
| $ per case | $0.00178 | $0.00178 |
| $ per correct verdict | $0.00181 | $0.00192 |
| p50 latency | 2.4s | 2.6s |
| cases / errors | 66 / 0 | 66 / 0 |
