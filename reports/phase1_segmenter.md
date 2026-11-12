# Phase 1 — segmenter vs CUAD lawyer spans (8 rule categories)

- Labelled spans checked: **3810**
- Span inside 1 segment: **95%**, within ≤2: **99%**, within ≤3: 100%
- Excess context (covering segment chars / span chars): median **2.2x**, p90 6.3x
- Segment length: median 581 chars, p90 912 chars, total segments 45,221

## After the line-start fix (2026-11-10)

The first version assumed single-line text (true for only 6/510 contracts) and missed section numbers at line starts.

| MAX_CHARS | spans in 1 segment | median excess | total segments |
|---|---|---|---|
| 1500 | 97% | 3.0x | 36,385 |
| **900 (kept)** | **95%** | **2.2x** | 45,221 |
| 600 | 93% | 1.8x | 54,557 |
