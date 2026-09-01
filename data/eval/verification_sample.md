# Verification sample

Labels in `queries.jsonl` are **generated**, not human-verified. Each entry
below shows a query and the exact passage labelled as its answer.

For each one, mark `[x] ok` if the passage genuinely answers the query, or
`[x] reject` with a short reason if it does not. The rejection rate is the
only evidence available about how trustworthy the generated labels are in
bulk, so a completed sample is worth more than a larger unverified set.

Sampled 40 of 586 queries, spread across the lexical-overlap range.

---

## 1. `q-68515ec4c774`

**Query:** In 2024, what amount did Coca Cola Co record as purchase obligations?

- lexical overlap: `0.17`
- document: `ko-10-k-2023-12-31`
- section: Part II > Item 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
- expected value: `13,701`

**Labelled passage:**

```
| Purchase obligations5 | 23,392 | 13,701 | 3,330 | 2,057 | 4,304 |
```

- [ ] ok
- [x] reject &mdash; reason:several figures under one label; unclear which is 2024

## 2. `q-a2c71352568c`

**Query:** What was Chevron Corp's international for 2023?

- lexical overlap: `0.25`
- document: `cvx-10-k-2023-12-31`
- section: Part III > Item 14. Principal Accountant Fees and Services > Note 18
- expected value: `183,996`

**Labelled passage:**

```
| International | 183,996 | 188,556 | 202,757 | 84,561 | 88,549 | 94,770 | 4,130 | 2,599 | 2,349 | 8,109 | 9,830 | 10,824 |
```

- [ ] ok
- [x] reject &mdash; reason: wrong question

## 3. `q-14bb7017d00d`

**Query:** In 2024, what amount did Cisco Systems, Inc. record as accumulated depreciation?

- lexical overlap: `0.29`
- document: `csco-10-k-2024-07-27`
- section: Part II > Item 8
- expected value: `(61)`

**Labelled passage:**

```
| Accumulated depreciation | (61) | (78) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 4. `q-1dc64d00b96b`

**Query:** Report Chevron Corp's dividend yield figure for the 2024 fiscal year.

- lexical overlap: `0.29`
- document: `cvx-10-k-2024-12-31`
- section: Part III > Item 14. Principal Accountant Fees and Services > Note 22
- expected value: `4.1`

**Labelled passage:**

```
| Dividend yield | 4.1 | % | 3.5 | % | 5.0 | % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 5. `q-c18a9de5829f`

**Query:** Report Intel Corp's discount rate figure for the 2025 fiscal year.

- lexical overlap: `0.29`
- document: `intc-10-k-2025-12-27`
- section: Note 17
- expected value: `4.8 %`

**Labelled passage:**

```
| Discount rate | 4.8 % | 4.6 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 6. `q-88eacdeedbbc`

**Query:** In 2022, what amount did Merck & Co., Inc. record as net income?

- lexical overlap: `0.33`
- document: `mrk-10-k-2022-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `$ 5.73`

**Labelled passage:**

```
| Net Income | $ 5.73 | $ 5.16 | $ 2.79 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 7. `q-1d3dbe696ef3`

**Query:** In 2025, what amount did Amgen Inc record as finished goods?

- lexical overlap: `0.33`
- document: `amgn-10-k-2025-12-31`
- section: Part IV > Item 16
- expected value: `1,885`

**Labelled passage:**

```
| Finished goods | 1,885 | 2,060 |
```

- [ ] ok
- [x] reject &mdash; reason:wrong question, i thing it should as what is the total sum amount of the finished goods

## 8. `q-76502a04414e`

**Query:** Blackrock Finance, Inc. multi-asset 2022

- lexical overlap: `0.33`
- document: `blk-10-k-2022-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `9`

**Labelled passage:**

```
| Multi-asset | 9 | % | 9 | % | 8 | % | 8 | % |
```

- [ ] ok
- [x] reject &mdash; reason:wrong question

## 9. `q-cd14f105d979`

**Query:** How much did Duke Energy Corp report for hedge funds in 2023?

- lexical overlap: `0.33`
- document: `duk-10-k-2024-12-31`
- section: Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
- expected value: `2 %`

**Labelled passage:**

```
| Hedge funds | 1 % | — % | 2 % |
```

- [ ] ok
- [x] reject &mdash; reason:several figures under one label; unclear which is 2023

## 10. `q-6bc728443601`

**Query:** Report American Express Co's actual tax rates figure for the 2024 fiscal year.

- lexical overlap: `0.38`
- document: `axp-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 19
- expected value: `21.5 %`

**Labelled passage:**

```
| Actual tax rates | 21.5 % | 20.3 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 11. `q-a65454159ba5`

**Query:** What was Goldman Sachs Group Inc's total capital ratio for 2022?

- lexical overlap: `0.38`
- document: `gs-10-k-2022-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `13.3 %`

**Labelled passage:**

```
| Total capital ratio | 13.3 % | 12.4 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 12. `q-43d8a28a33bf`

**Query:** What was Nvidia Corp's dividends paid for 2023?

- lexical overlap: `0.40`
- document: `nvda-10-k-2023-01-29`
- section: Part IV > Item 15. EXHIBIT AND FINANCIAL STATEMENT SCHEDULES
- expected value: `(398)`

**Labelled passage:**

```
| Dividends paid | (398) | (399) | (395) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 13. `q-e69449d66cb7`

**Query:** What was Nvidia Corp's accounts receivable for 2026?

- lexical overlap: `0.40`
- document: `nvda-10-k-2026-01-25`
- section: Part IV > Item 15. Exhibits and Financial Statement Schedules
- expected value: `(15,399)`

**Labelled passage:**

```
| Accounts receivable | (15,399) | (13,063) | (6,172) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 14. `q-64eb6479d438`

**Query:** What was At&T Inc.'s gain (loss) on repurchases for 2024?

- lexical overlap: `0.40`
- document: `t-10-k-2024-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 17 - SALES OF RECEIVABLES
- expected value: `$ (14)`

**Labelled passage:**

```
| Gain (loss) on repurchases1 | $ (14) |  | $ (16) |  | $ (21) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 15. `q-cbe5bc330dd7`

**Query:** Merck & Co., Inc. retained earnings 2024

- lexical overlap: `0.40`
- document: `mrk-10-k-2024-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `63,069`

**Labelled passage:**

```
| Retained earnings | 63,069 | 53,895 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 16. `q-fe40ac3b38ff`

**Query:** What was General Electric Co's other items for 2024?

- lexical overlap: `0.40`
- document: `ge-10-k-2024-12-31`
- section: Note 19 - OTHER INCOME (LOSS)
- expected value: `151`

**Labelled passage:**

```
| Other items | 151 | 92 | 74 |
```

- [ ] ok
- [x] reject &mdash; reason: wrong question

## 17. `q-ee30e5c38720`

**Query:** What was Exxon Mobil Corp's total consolidated subsidiaries for 2022?

- lexical overlap: `0.43`
- document: `xom-10-k-2022-12-31`
- section: Part I > Item 2. PROPERTIES
- expected value: `4`

**Labelled passage:**

```
| Total Consolidated Subsidiaries | 4 | 4 | 2 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 18. `q-94f3046d9845`

**Query:** In 2022, what amount did Amgen Inc record as earnings of foreign subsidiaries?

- lexical overlap: `0.43`
- document: `amgn-10-k-2022-12-31`
- section: Part IV > Item 16
- expected value: `192`

**Labelled passage:**

```
| Earnings of foreign subsidiaries | 192 | — |
```

- [x] ok
- [ ] reject &mdash; reason:

## 19. `q-1983dea0cb00`

**Query:** Report Nike, Inc.'s subpart f deferred tax benefit figure for the 2023 fiscal year.

- lexical overlap: `0.44`
- document: `nke-10-k-2023-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 7 - INCOME TAXES
- expected value: `0.0 %`

**Labelled passage:**

```
| Subpart F deferred tax benefit | 0.0 % | -4.7 % | 0.0 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 20. `q-96878bfe5e50`

**Query:** Report Nike, Inc.'s weighted-average discount rate figure for the 2025 fiscal year.

- lexical overlap: `0.44`
- document: `nke-10-k-2025-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 17 - LEASES
- expected value: `3.1 %`

**Labelled passage:**

```
| Weighted-average discount rate | 3.1 % | 2.9 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 21. `q-14ebb28195eb`

**Query:** In 2025, what amount did American Express Co record as remaining performance period (in years)?

- lexical overlap: `0.50`
- document: `axp-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 10
- expected value: `2.9`

**Labelled passage:**

```
| Remaining performance period (in years) | 2.9 | 2.9 | 2.9 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 22. `q-570240733a01`

**Query:** What was Unitedhealth Group Inc's total acquired intangible assets for 2023?

- lexical overlap: `0.50`
- document: `unh-10-k-2023-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
- expected value: `$ 2,174`

**Labelled passage:**

```
| Total acquired intangible assets | $ 2,174 | $ 5,814 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 23. `q-b799f5f6fd89`

**Query:** Apple Inc. other current assets 2025

- lexical overlap: `0.50`
- document: `aapl-10-k-2025-09-27`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `14,585`

**Labelled passage:**

```
| Other current assets | 14,585 |  | 14,287 |
```

- [ ] ok
- [x] reject &mdash; reason: i think this is a wrong question

## 24. `q-f1620d85d312`

**Query:** What was Nike, Inc.'s stock-based compensation for 2025?

- lexical overlap: `0.50`
- document: `nke-10-k-2025-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 7 - INCOME TAXES
- expected value: `1.5 %`

**Labelled passage:**

```
| Stock-based compensation | 1.5 % | -0.5 % | -1.1 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 25. `q-2721d9fba126`

**Query:** Blackrock, Inc. active multi-asset 2025

- lexical overlap: `0.50`
- document: `blk-10-k-2025-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `7`

**Labelled passage:**

```
| Active multi-asset | 7 | % | 8 | % | 9 | % | 9 | % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 26. `q-50a4fb24f648`

**Query:** What was At&T Inc.'s depreciation and amortization for 2024?

- lexical overlap: `0.50`
- document: `t-10-k-2024-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 13 - INCOME TAXES
- expected value: `$ 36,531`

**Labelled passage:**

```
| Depreciation and amortization | $ 36,531 | $ 37,931 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 27. `q-970bbd0ce1e3`

**Query:** How much did Duke Energy Corp report for natural gas and fuel oil(a) in 2024?

- lexical overlap: `0.50`
- document: `duk-10-k-2024-12-31`
- section: Item 1. BUSINESS
- expected value: `34.7 %`

**Labelled passage:**

```
| Natural gas and fuel oil(a) | 34.7 % | 33.3 % | 34.2 % | 3.39 | 3.81 | 6.35 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 28. `q-b5717601ae50`

**Query:** American Express Co unamortized underwriting fees 2025

- lexical overlap: `0.50`
- document: `axp-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 8
- expected value: `(135)`

**Labelled passage:**

```
| Unamortized Underwriting Fees | (135) | (96) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 29. `q-f441618e228a`

**Query:** How much did Coca Cola Co report for foreign currency contracts in 2025?

- lexical overlap: `0.50`
- document: `ko-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 5 - HEDGING TRANSACTIONS AND DERIVATIVE FINANCIAL INSTRUMENTS
- expected value: `$ 1,067`

**Labelled passage:**

```
| Foreign currency contracts | $ 1,067 | $ 59 | $ 8 | $ 19 | $ (6) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 30. `q-1abf4e6d79b9`

**Query:** In 2024, what amount did Walmart Inc. record as net impact of repatriated international earnings?

- lexical overlap: `0.56`
- document: `wmt-10-k-2024-01-31`
- section: Part II > Item 8 > Note 9 - Taxes
- expected value: `(0.4) %`

**Labelled passage:**

```
| Net impact of repatriated international earnings | (0.4) % | (0.4) % | (0.3) % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 31. `q-c671df566fac`

**Query:** How much did American Express Co report for tax credits and tax-exempt income (a) in 2022?

- lexical overlap: `0.57`
- document: `axp-10-k-2022-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 20
- expected value: `(0.9)`

**Labelled passage:**

```
| Tax credits and tax-exempt income (a) | (0.9) | (0.1) | (4.1) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 32. `q-4ff9e4d212e4`

**Query:** How much did Amgen Inc report for risk-free interest rate in 2024?

- lexical overlap: `0.57`
- document: `amgn-10-k-2024-12-31`
- section: Part IV > Item 16
- expected value: `4.4 %`

**Labelled passage:**

```
| Risk-free interest rate | 4.4 % | 3.4 % | 2.8 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 33. `q-f4a5ba2c1a33`

**Query:** In 2023, what amount did Boeing Co record as net actuarial loss/(gain)?

- lexical overlap: `0.57`
- document: `ba-10-k-2023-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 16 - Postretirement Plans
- expected value: `$18,175`

**Labelled passage:**

```
| Net actuarial loss/(gain) | $18,175 | $17,448 | ($1,852) | ($1,862) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 34. `q-dd272cf2b440`

**Query:** Report Boeing Co's weighted average remaining lease term (years) figure for the 2023 fiscal year.

- lexical overlap: `0.60`
- document: `ba-10-k-2023-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 12 - Leases
- expected value: `11`

**Labelled passage:**

```
| Weighted average remaining lease term (years) | 11 | 12 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 35. `q-decdaecc8339`

**Query:** How much did Johnson & Johnson report for risk-free rate in 2025?

- lexical overlap: `0.60`
- document: `jnj-10-k-2025-12-28`
- section: Part II > Item 8. Financial statements and supplementary data
- expected value: `4.33 %`

**Labelled passage:**

```
| Risk-free rate | 4.33 % | 4.15 % | 3.74 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 36. `q-51ef51e607cf`

**Query:** Apple Inc. statutory federal income tax rate 2022

- lexical overlap: `0.62`
- document: `aapl-10-k-2022-09-24`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `21 %`

**Labelled passage:**

```
| Statutory federal income tax rate | 21 % | 21 % | 21 % |
```

- [x] ok
- [ ] reject &mdash; reason:

## 37. `q-20e8ba9a8645`

**Query:** Nike, Inc. weighted-average remaining lease term (in years) 2025

- lexical overlap: `0.67`
- document: `nke-10-k-2025-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 17 - LEASES
- expected value: `6.6`

**Labelled passage:**

```
| Weighted-average remaining lease term (in years) | 6.6 | 6.9 |
```

- [x] ok
- [ ] reject &mdash; reason:

## 38. `q-3f010395ed29`

**Query:** Report Chevron Corp's net borrowings (repayments) of short-term obligations with three months or less maturity figure for the 2023 fiscal year.

- lexical overlap: `0.67`
- document: `cvx-10-k-2023-12-31`
- section: Part III > Item 14. Principal Accountant Fees and Services > Note 3
- expected value: `135`

**Labelled passage:**

```
| Net borrowings (repayments) of short-term obligations with three months or less maturity | 135 | 263 | (3,114) |
```

- [ ] ok
- [x] reject &mdash; reason: confusiong figures

## 39. `q-13972440f043`

**Query:** In 2023, what amount did Boeing Co record as inventory and long-term contract methods of income recognition?

- lexical overlap: `0.70`
- document: `ba-10-k-2023-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 4 - Income Taxes
- expected value: `($5,115)`

**Labelled passage:**

```
| Inventory and long-term contract methods of income recognition | ($5,115) | ($4,369) |
```

- [x] ok
- [ ] reject &mdash; reason:

## 40. `q-4f72a926cf6b`

**Query:** Jpmorgan Chase & Co settlement of first republic deposit and other related party transactions(c) 2023

- lexical overlap: `0.73`
- document: `jpm-10-k-2024-12-31`
- section: Part IV > Note 34 - Business combinations
- expected value: `5,447`

**Labelled passage:**

```
| Settlement of First Republic deposit and other related party transactions(c) | 5,447 |
```

- [x] ok
- [ ] reject &mdash; reason:
