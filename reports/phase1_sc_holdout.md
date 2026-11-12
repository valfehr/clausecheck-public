# Phase 1 — experiments (test split; one change at a time vs E0)

| metric | RELEASE-E0v11-holdout | RELEASE-SC-holdout |
|---|---|---|
| severity acc (std + hard) | 86% (25/29) | 86% (25/29) |
| hard-case acc | 100% (8/8) | 100% (8/8) |
| red recall (all reds) | 92% (11/12) | 92% (11/12) |
| critical miss red→green | 0% (0/12) | 0% (0/12) |
| citation hit | 94% (33/35) | 94% (33/35) |
| citation verified | 100% (44/44) | 100% (44/44) |
| $ per case | $0.00213 | $0.00162 |
| $ per correct verdict | $0.00234 | $0.00193 |
| p50 latency | 2.4s | 6.8s |
| cases / errors | 44 / 0 | 44 / 0 |
