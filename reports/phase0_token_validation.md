# Phase 0 — token estimate vs real tokenizer (20 random contracts, free endpoint)

- Real tokens / (chars/4) estimate: **1.35x**
- Measured chars per token: **2.97**

| contract | family | est (chars/4) | real |
|---|---|---|---|
| GluMobileInc_20070319_S-1A_EX-10.09_436630_EX-10.0 | License | 28,198 | 38,950 |
| XLITECHNOLOGIES,INC_12_02_2015-EX-10.02-STRATEGIC  | Strategic Alliance | 1,428 | 2,173 |
| GluMobileInc_20070319_S-1A_EX-10.09_436630_EX-10.0 | License | 855 | 1,435 |
| TALCOTTRESOLUTIONLIFEINSURANCECO-SEPARATEACCOUNTTW | Service | 7,449 | 9,370 |
| QBIOMEDINC_04_08_2020-EX-99.1-JOINT FILING AGREEME | Joint Venture | 683 | 1,357 |
| INTRICONCORP_03_10_2009-EX-10.22-Strategic Allianc | Strategic Alliance | 13,241 | 17,443 |
| HUBEIMINKANGPHARMACEUTICALLTD_09_19_2006-EX-10.1-O | Outsourcing | 2,830 | 3,715 |
| ALCOSTORESINC_12_14_2005-EX-10.26-AGENCY AGREEMENT | Agency | 2,838 | 4,053 |
| LifewayFoodsInc_20160316_10-K_EX-10.24_9489766_EX- | Endorsement | 2,749 | 3,571 |
| GpaqAcquisitionHoldingsInc_20200123_S-4A_EX-10.8_1 | Service | 13,768 | 18,563 |
| RMRGROUPINC_01_22_2020-EX-99.1-JOINT FILING AGREEM | Joint Venture | 161 | 288 |
| GOOSEHEADINSURANCE,INC_04_02_2018-EX-10.6-Franchis | Franchise | 75,192 | 99,136 |
| BLUEHILLSBANCORP,INC_05_20_2014-EX-1.1-AGENCY AGRE | Agency | 32,685 | 41,008 |
| PROLONGINTERNATIONALCORP_03_23_1998-EX-10.16-SPONS | Sponsorship | 3,742 | 4,918 |
| NETGEAR,INC_04_21_2003-EX-10.16-DISTRIBUTOR AGREEM | Distributor | 10,375 | 13,865 |
| WARNERCHILCOTTPLC_12_31_2003-EX-4.36-DEVELOPMENT A | Development | 7,420 | 9,158 |
| GIGGLESN_HUGS,INC_06_23_2016-EX-10.2-STRATEGIC ALL | Strategic Alliance | 5,589 | 6,887 |
| HEMISPHERX - Sales, Marketing, Distribution, and S | Supply | 18,421 | 25,509 |
| AURASYSTEMSINC_06_16_2010-EX-10.25-STRATEGIC ALLIA | Strategic Alliance | 11,258 | 14,731 |
| ArmstrongFlooringInc_20190107_8-K_EX-10.2_11471795 | Ip | 14,214 | 24,329 |

## Smoke test (Haiku 5.5, structured output)
- Parsed: `governing_law_state='California'`; tokens in/out 327/15; cost $0.00004; latency 1.572s
- Second identical call served from disk cache: **True** (cost $0.0)

Total project spend so far: $0.0000