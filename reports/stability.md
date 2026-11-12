# Stability eval: 3 identical runs, 66 cases (test split)

Runs: `SC-run1`, `SC-run2`, `SC-run3`

- **Flip rate** (runs disagree): **2/66 (3.0%)**
- **RED ↔ GREEN flips**: **0**
- Single-run accuracy: 46/49, 47/49, 46/49; **majority of 3**: 46/49
- Flips by least-confident stated confidence: medium 2/27, high 0/39
- Flips by rule: audit_rights 1, change_of_control 1
- Flagged for human review (any run): 8/66; avg $/case per run: 0.00304

**Gate (flip rate ≤ 5%, zero RED ↔ GREEN): PASSED**

## Flipped cases

- `hard-audit_rights-override_elsewhere`: green / yellow / green (expected yellow)
- `ret-change_of_control-p2`: yellow / green / green (expected n/a)
