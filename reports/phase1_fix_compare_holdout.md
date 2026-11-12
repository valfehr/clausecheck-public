# Phase 1 — experiments (test split; one change at a time vs E0)

| metric | RELEASE-E0-holdout | RELEASE-E0v11-holdout |
|---|---|---|
| severity acc (std + hard) | 90% (26/29) | 86% (25/29) |
| hard-case acc | 100% (8/8) | 100% (8/8) |
| red recall (all reds) | 100% (12/12) | 92% (11/12) |
| critical miss red→green | 0% (0/12) | 0% (0/12) |
| citation hit | 94% (33/35) | 94% (33/35) |
| citation verified | 93% (41/44) | 100% (44/44) |
| $ per case | $0.00218 | $0.00213 |
| $ per correct verdict | $0.00231 | $0.00234 |
| p50 latency | 2.8s | 2.4s |
| cases / errors | 44 / 0 | 44 / 0 |
