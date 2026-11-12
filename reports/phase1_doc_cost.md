# Phase 1 — real cost per document (8 golden rules, Haiku 5.5, effort low)

Call pattern: rule 1 writes the contract to the prompt cache, rules 2-8 run in parallel and read it.

- **Median $/doc: $0.0060** (mean $0.0079); without prompt caching: median $0.0169
- Caching saved **69%** of total cost; cache hits on rules 2-8: 70/70
- Output (incl. reasoning) is **25%** of the cached bill; most of the rest is the first full read of the contract (the cache write)
- Wall-clock for a full 8-rule review: median **7.1s**, max 11.9s
- Segmentation (local Python, once per doc): median 3 ms, $0
- At list price, 1,000 contracts/month ≈ **$7.94** with caching vs $25.33 without

| contract | family | tokens | cache hits | $/doc cached | $/doc no cache | wall s | verdicts (8 rules) |
|---|---|---|---|---|---|---|---|
| TcPipelinesLp_20160226_10-K_EX-99.12_945 | Transportation | 3,618 | 7/7 | $0.0018 | $0.0039 | 4.5 | YRRRRRRR |
| ON4COMMUNICATIONSINC_07_02_2009-EX-10.1- | Promotion | 6,267 | 7/7 | $0.0029 | $0.0066 | 5.5 | RYRRRRRR |
| SightLife Surgical, Inc. - STRATEGIC SAL | Marketing | 8,990 | 7/7 | $0.0032 | $0.0086 | 4.3 | YGRRYGGY |
| FerroglobePlc_20150624_F-4A_EX-10.20_915 | Outsourcing | 11,850 | 7/7 | $0.0038 | $0.0109 | 4.6 | RRRRYRRY |
| BNLFINANCIALCORP_03_30_2007-EX-10.8-OUTS | Outsourcing | 14,567 | 7/7 | $0.0055 | $0.0139 | 7.4 | YYYRYYRR |
| OTISWORLDWIDECORP_04_03_2020-EX-10.4-INT | Unknown | 22,256 | 7/7 | $0.0065 | $0.0198 | 9.5 | GYRRRRYR |
| PHLVARIABLEINSURANCECOCT_08_17_2009-EX-1 | Strategic Alliance | 35,126 | 7/7 | $0.0091 | $0.0303 | 6.9 | YYRRYRRG |
| MOELIS_CO_03_24_2014-EX-10.19-STRATEGIC  | Strategic Alliance | 42,174 | 7/7 | $0.0112 | $0.0366 | 8.4 | GGRYYYRR |
| OFGBANCORP_03_28_2007-EX-10.23-OUTSOURCI | Outsourcing | 52,814 | 7/7 | $0.0140 | $0.0450 | 11.9 | GRRYYGRR |
| Array BioPharma Inc. - LICENSE, DEVELOPM | Development | 93,272 | 7/7 | $0.0214 | $0.0777 | 9.8 | GYRYYRYR |

Actual spend for this benchmark: $0.0794