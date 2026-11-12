# Phase 0 — CUAD exploration (tokens = chars / 2.97, calibrated on the real tokenizer)

- Contracts over 100k tokens (Haiku 5.5 long-prompt pricing applies): **3**

- Contracts: **510**, families: **26** (unknown family: 2)
- Tokens p10 / p50 / p90 / p99: **2,149 / 11,159 / 41,116 / 91,412**, max 113,875
- Total corpus: ~9.0M tokens
- Contracts that fit in ~12k tokens (rough KV budget of a local 8B 4-bit model on 8 GB): **53%**

## Cost of one full long-context pass (input only, list prices, no caching)

| Model | $/M input | One read of the whole corpus |
|---|---|---|
| Haiku 5.5 | $0.10 | $0.90 |
| Sonnet 5.5 | $2.00 | $18.05 |
| Opus 5.5 | $4.00 | $36.10 |

## Families

| family                  |   n |   median_tokens |
|:------------------------|----:|----------------:|
| Maintenance             |  34 |         11083.5 |
| License                 |  33 |          8437   |
| Distributor             |  32 |         11908   |
| Strategic Alliance      |  32 |         14389   |
| Sponsorship             |  31 |          9007   |
| Service                 |  28 |          7250.5 |
| Development             |  28 |         15161   |
| Collaboration           |  26 |         16893.5 |
| Endorsement             |  24 |          7429.5 |
| Joint Venture           |  23 |          1337   |
| Co-Branding             |  22 |         15602.5 |
| Hosting                 |  20 |         12283   |
| Supply                  |  18 |         18791.5 |
| Outsourcing             |  18 |         16619.5 |
| Manufacturing           |  17 |         15436   |
| Ip                      |  17 |         19026   |
| Marketing               |  16 |         15500   |
| Franchise               |  15 |         39351   |
| Agency                  |  13 |         40928   |
| Transportation          |  13 |         16571   |
| Promotion               |  12 |         21553   |
| Reseller                |  12 |         13088   |
| Consulting              |  11 |          7005   |
| Affiliate               |  10 |          8943.5 |
| Non-Compete/Non-Solicit |   3 |          5522   |
| Unknown                 |   2 |         45442   |

## Clause prevalence (% of contracts where lawyers found the clause)

| category                           |   present_pct |   median_span_chars |
|:-----------------------------------|--------------:|--------------------:|
| Document Name                      |           100 |                  26 |
| Parties                            |           100 |                  80 |
| Agreement Date                     |            92 |                  16 |
| Governing Law                      |            86 |                 184 |
| Expiration Date                    |            81 |                 211 |
| Effective Date                     |            76 |                  22 |
| Anti-Assignment                    |            73 |                 369 |
| Cap On Liability                   |            54 |                 617 |
| License Grant                      |            50 |                 794 |
| Audit Rights                       |            42 |                 589 |
| Termination For Convenience        |            36 |                 209 |
| Post-Termination Services          |            36 |                 613 |
| Exclusivity                        |            35 |                 622 |
| Renewal Term                       |            35 |                 300 |
| Insurance                          |            33 |                 688 |
| Revenue/Profit Sharing             |            33 |                 614 |
| Minimum Commitment                 |            32 |                 555 |
| Non-Transferable License           |            27 |                 540 |
| Ip Ownership Assignment            |            24 |                 741 |
| Change Of Control                  |            24 |                 568 |
| Non-Compete                        |            23 |                 525 |
| Uncapped Liability                 |            22 |                 558 |
| Notice Period To Terminate Renewal |            22 |                 317 |
| Covenant Not To Sue                |            20 |                 475 |
| Rofr/Rofo/Rofn                     |            17 |                 888 |
| Volume Restriction                 |            16 |                 387 |
| Competitive Restriction Exception  |            15 |                 525 |
| Warranty Duration                  |            15 |                 502 |
| Irrevocable Or Perpetual License   |            14 |                 777 |
| Liquidated Damages                 |            12 |                 584 |
| Affiliate License-Licensee         |            12 |                 646 |
| No-Solicit Of Employees            |            12 |                 458 |
| Joint Ip Ownership                 |             9 |                 580 |
| Non-Disparagement                  |             7 |                 436 |
| No-Solicit Of Customers            |             7 |                 430 |
| Third Party Beneficiary            |             6 |                 209 |
| Most Favored Nation                |             5 |                 565 |
| Affiliate License-Licensor         |             5 |                1312 |
| Unlimited/All-You-Can-Eat-License  |             3 |                 560 |
| Price Restrictions                 |             3 |                 298 |
| Source Code Escrow                 |             3 |                1994 |