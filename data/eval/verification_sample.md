# Verification sample

Labels in `queries.jsonl` are **generated**, not human-verified. Each entry
below shows a query and the exact passage labelled as its answer.

For each one, mark `[x] ok` if the passage genuinely answers the query, or
`[x] reject` with a short reason if it does not. The rejection rate is the
only evidence available about how trustworthy the generated labels are in
bulk, so a completed sample is worth more than a larger unverified set.

Sampled 40 of 216 queries, spread across the lexical-overlap range.

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
- [ ] reject &mdash; reason:

## 2. `q-9f33eda94a35`

**Query:** Report Goldman Sachs Group Inc's commodity prices figure for the 2021 fiscal year.

- lexical overlap: `0.22`
- document: `gs-10-k-2022-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `$ 45`

**Labelled passage:**

```
| Commodity prices | $ 82 $ 18 | $ 45 | $ | 14 |  |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 3. `q-78b1f110cec6`

**Query:** In 2022, what amount did Goldman Sachs Group Inc record as interest rates?

- lexical overlap: `0.25`
- document: `gs-10-k-2023-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `$ 137`

**Labelled passage:**

```
| Interest rates | $ 148 $ 70 | $ 137 | $ | 56 |  |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 4. `q-d3edb5a86f84`

**Query:** Chevron Corp international 2022

- lexical overlap: `0.25`
- document: `cvx-10-k-2022-12-31`
- section: Part III > Item 14. Principal Accountant Fees and Services
- expected value: `1,818`

**Labelled passage:**

```
| International | 1,818 |  | 1,960 |  | 2,025 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 5. `q-48ee43da658d`

**Query:** Report Nike, Inc.'s foreign earnings figure for the 2023 fiscal year.

- lexical overlap: `0.29`
- document: `nke-10-k-2023-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 7 - INCOME TAXES
- expected value: `1.7 %`

**Labelled passage:**

```
| Foreign earnings | 1.7 % | -1.8 % | 0.2 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 6. `q-7e369bb92298`

**Query:** How much did Goldman Sachs Group Inc report for interest rates in 2024?

- lexical overlap: `0.29`
- document: `gs-10-k-2025-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `$ 121`

**Labelled passage:**

```
| Interest rates | $ 92 $ 54 | $ 121 | $ | 57 |  |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 7. `q-009f233c4392`

**Query:** In 2022, what amount did Chevron Corp record as united states?

- lexical overlap: `0.33`
- document: `cvx-10-k-2022-12-31`
- section: Part III > Item 14. Principal Accountant Fees and Services > Note 14
- expected value: `$ 50,822`

**Labelled passage:**

```
| United States | $ 50,822 | $ 29,219 | $ 14,577 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 8. `q-50b3acd0c480`

**Query:** What was Blackrock Finance, Inc.'s liquid alternatives for 2023?

- lexical overlap: `0.33`
- document: `blk-10-k-2023-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `4`

**Labelled passage:**

```
| Liquid alternatives | 4 | % | 4 | % | 1 | % | 1 | % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 9. `q-976b53636c37`

**Query:** In 2025, what amount did Costco Wholesale Corp /New record as deferred membership fees?

- lexical overlap: `0.33`
- document: `cost-10-k-2025-08-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `2,854`

**Labelled passage:**

```
| Deferred membership fees | 2,854 | 2,501 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 10. `q-ed582dbd6bdc`

**Query:** In 2025, what amount did Goldman Sachs Group Inc record as cet1 capital ratio?

- lexical overlap: `0.33`
- document: `gs-10-k-2025-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `12.0 %`

**Labelled passage:**

```
| CET1 capital ratio | 12.0 % | 11.9 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 11. `q-3daae41313cf`

**Query:** Report Merck & Co., Inc.'s benefit obligation december figure for the 2024 fiscal year.

- lexical overlap: `0.38`
- document: `mrk-10-k-2024-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `$ 10,151`

**Labelled passage:**

```
| Benefit obligation December 31 | $ 10,151 | $ 10,446 | $ 8,274 | $ 9,042 | $ 1,136 | $ 1,104 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 12. `q-956d97d0649f`

**Query:** What was Goldman Sachs Group Inc's leverage ratio requirement for 2025?

- lexical overlap: `0.38`
- document: `gs-10-k-2025-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `3.7 %`

**Labelled passage:**

```
| Leverage ratio requirement | 3.7 % | 3.7 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 13. `q-21b34c85f0c4`

**Query:** Blackrock, Inc. cash management 2025

- lexical overlap: `0.40`
- document: `blk-10-k-2025-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `6`

**Labelled passage:**

```
| Cash management | 6 | % | 7 | % | 8 | % | 7 | % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 14. `q-3693270420fd`

**Query:** What was Blackrock, Inc.'s liquid alternatives for 2024?

- lexical overlap: `0.40`
- document: `blk-10-k-2024-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `4`

**Labelled passage:**

```
| Liquid alternatives | 4 | % | 4 | % | 1 | % | 1 | % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 15. `q-66bd501fa69e`

**Query:** What was Nvidia Corp's accounts receivable for 2025?

- lexical overlap: `0.40`
- document: `nvda-10-k-2025-01-26`
- section: Part IV > Item 15. Exhibits and Financial Statement Schedules
- expected value: `(13,063)`

**Labelled passage:**

```
| Accounts receivable | (13,063) | (6,172) | 822 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 16. `q-9e16afe1be62`

**Query:** What was Coca Cola Co's translation and other for 2025?

- lexical overlap: `0.40`
- document: `ko-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 9 - SUPPLY CHAIN FINANCE PROGRAM
- expected value: `(12)`

**Labelled passage:**

```
| Translation and other | (12) |  | — |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 17. `q-e1c368e4c3f4`

**Query:** General Electric Co other changes 2022

- lexical overlap: `0.40`
- document: `ge-10-k-2022-12-31`
- section: Note 24 - COMMITMENTS, GUARANTEES, PRODUCT WARRANTIES AND OTHER LOSS CONTINGENCIES
- expected value: `(90)`

**Labelled passage:**

```
| Other changes | (90) | (81) | 14 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 18. `q-51a4409bce55`

**Query:** Report Goldman Sachs Group Inc's settlement of employee share-based awards figure for the 2022 fiscal year.

- lexical overlap: `0.42`
- document: `gs-10-k-2022-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 24
- expected value: `(2.4)`

**Labelled passage:**

```
| Settlement of employee share-based awards | (2.4) | (0.7) | (1.0) |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 19. `q-38bdfcd52cc6`

**Query:** How much did Unitedhealth Group Inc report for expected dividend yield in 2022?

- lexical overlap: `0.43`
- document: `unh-10-k-2023-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
- expected value: `1.2%`

**Labelled passage:**

```
| Expected dividend yield | 1.3% - 1.5% | 1.2% | 1.3% - 1.5% |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 20. `q-7c06374309f9`

**Query:** In 2018, what amount did Walmart Inc. record as s&p 500 retailing index?

- lexical overlap: `0.43`
- document: `wmt-10-k-2023-01-31`
- section: Part II > Item 5
- expected value: `100.00`

**Labelled passage:**

```
| S&P 500 Retailing Index | 100.00 | 108.42 | 127.45 | 180.19 | 195.77 | 160.10 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 21. `q-f84c1e7a6542`

**Query:** In 2024, what amount did Nvidia Corp record as total current assets?

- lexical overlap: `0.43`
- document: `nvda-10-k-2024-01-28`
- section: Part IV > Item 15. Exhibit and Financial Statement Schedules
- expected value: `44,345`

**Labelled passage:**

```
| Total current assets | 44,345 | 23,073 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 22. `q-f57829a3c551`

**Query:** Costco Wholesale Corp /New interest income and other, net 2024

- lexical overlap: `0.44`
- document: `cost-10-k-2024-09-01`
- section: Part II > Item 6. Reserved
- expected value: `$ 624`

**Labelled passage:**

```
| Interest income and other, net | $ 624 | $ 533 | $ 205 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 23. `q-14ebb28195eb`

**Query:** In 2025, what amount did American Express Co record as remaining performance period (in years)?

- lexical overlap: `0.50`
- document: `axp-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 10
- expected value: `2.9`

**Labelled passage:**

```
| Remaining performance period (in years) | 2.9 | 2.9 | 2.9 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 24. `q-2e6b34a44cc2`

**Query:** Report American Express Co's u.s. statutory federal income tax rate figure for the 2024 fiscal year.

- lexical overlap: `0.50`
- document: `axp-10-k-2024-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 20
- expected value: `21.0 %`

**Labelled passage:**

```
| U.S. statutory federal income tax rate | 21.0 % | 21.0 % | 21.0 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 25. `q-4ba2ba599e0f`

**Query:** How much did Chevron Corp report for marketable securities sold in 2024?

- lexical overlap: `0.50`
- document: `cvx-10-k-2025-12-31`
- section: Part III > Item 14. Principal Accountant Fees and Services > Note 3
- expected value: `45`

**Labelled passage:**

```
| Marketable securities sold | — | 45 | 464 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 26. `q-784c39eb2ab9`

**Query:** What was Coca Cola Co's commercial paper borrowings for 2024?

- lexical overlap: `0.50`
- document: `ko-10-k-2023-12-31`
- section: Part II > Item 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
- expected value: `$ 4,209`

**Labelled passage:**

```
| Commercial paper borrowings | $ 4,209 | $ 4,209 | $ — | $ — | $ — |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 27. `q-839b5360416c`

**Query:** How much did Procter & Gamble Co report for total lease payments in 2024?

- lexical overlap: `0.50`
- document: `pg-10-k-2024-06-30`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 12
- expected value: `1,031`

**Labelled passage:**

```
| Total lease payments | 1,031 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 28. `q-9fe02507e5c0`

**Query:** Southern Co alabama power 2023

- lexical overlap: `0.50`
- document: `so-10-k-2025-12-31`
- section: Part II > Item 7A. QUANTITATIVE AND QUALITATIVE DISCLOSURES ABOUT MARKET RISK
- expected value: `40`

**Labelled passage:**

```
| Alabama Power | — | — | 40 | — | — | 5.5 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 29. `q-b941f09afa0a`

**Query:** Report Boeing Co's other postretirement benefit obligations figure for the 2025 fiscal year.

- lexical overlap: `0.50`
- document: `ba-10-k-2025-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 6 - Income Taxes
- expected value: `562`

**Labelled passage:**

```
| Other postretirement benefit obligations | 562 | 587 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 30. `q-da41dc7a3db7`

**Query:** How much did At&T Inc. report for operating income in 2025?

- lexical overlap: `0.50`
- document: `t-10-k-2025-12-31`
- section: Part II > Item 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
- expected value: `24,162`

**Labelled passage:**

```
| Operating Income | 24,162 | 19,049 | 23,461 | 26.8 | (18.8) |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 31. `q-ea2c041f071a`

**Query:** How much did Goldman Sachs Group Inc report for tax-exempt income, including dividends in 2024?

- lexical overlap: `0.50`
- document: `gs-10-k-2024-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 24
- expected value: `(0.6)`

**Labelled passage:**

```
| Tax-exempt income, including dividends | (0.6) | (1.0) | (2.2) |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 32. `q-fdaa35bd2eff`

**Query:** What was Merck & Co., Inc.'s short-term investments for 2022?

- lexical overlap: `0.50`
- document: `mrk-10-k-2022-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `498`

**Labelled passage:**

```
| Short-term investments | 498 | — |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 33. `q-0b93ebcaaa7e`

**Query:** Amgen Inc acquired in-process research and development 2021

- lexical overlap: `0.57`
- document: `amgn-10-k-2022-12-31`
- section: Part II > Item 7
- expected value: `$ 1,505`

**Labelled passage:**

```
| Acquired in-process research and development | $ — | NM | $ 1,505 | NM | $ — |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 34. `q-6cb8c10ffa53`

**Query:** What was Nike, Inc.'s interest (income) expense, net for 2025?

- lexical overlap: `0.57`
- document: `nke-10-k-2026-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 15 - SEGMENT INFORMATION
- expected value: `(107)`

**Labelled passage:**

```
| Interest (income) expense, net | (107) |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 35. `q-bbfb090e276b`

**Query:** Apple Inc. other non-current assets 2024

- lexical overlap: `0.57`
- document: `aapl-10-k-2024-09-28`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 6 - Consolidated Financial Statement Details
- expected value: `55,335`

**Labelled passage:**

```
| Other non-current assets | 55,335 | 46,906 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 36. `q-e5229ed51661`

**Query:** What was Apple Inc.'s accumulated other comprehensive loss for 2024?

- lexical overlap: `0.57`
- document: `aapl-10-k-2024-09-28`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `(7,172)`

**Labelled passage:**

```
| Accumulated other comprehensive loss | (7,172) |  | (11,452) |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 37. `q-2085c224efbd`

**Query:** Boeing Co estimated amortization expense 2026

- lexical overlap: `0.60`
- document: `ba-10-k-2025-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 4 - Goodwill and Acquired Intangibles
- expected value: `$197`

**Labelled passage:**

```
| Estimated amortization expense | $197 | $182 | $155 | $150 | $144 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 38. `q-bf1951123d90`

**Query:** How much did Boeing Co report for mortgage backed and asset backed in 2024?

- lexical overlap: `0.60`
- document: `ba-10-k-2025-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 18 - Postretirement Plans
- expected value: `161`

**Labelled passage:**

```
| Mortgage backed and asset backed | 161 | 2 | 5 | $4 | 172 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 39. `q-7e0e662a981a`

**Query:** Coca Cola Co benefit payments for other postretirement benefit plans 2026

- lexical overlap: `0.62`
- document: `ko-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 14 - PENSION AND OTHER POSTRETIREMENT BENEFIT PLANS
- expected value: `21`

**Labelled passage:**

```
| Benefit payments for other postretirement benefit plans | 21 | 18 | 17 | 15 | 14 | 65 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 40. `q-11fc9704c12b`

**Query:** What was General Electric Co's gains (losses) on retained and sold ownership interests for 2025?

- lexical overlap: `0.67`
- document: `ge-10-k-2025-12-31`
- section: Note 19 - OTHER INCOME (LOSS)
- expected value: `$ 21`

**Labelled passage:**

```
| Gains (losses) on retained and sold ownership interests | $ 21 | $ 518 | $ 5,778 |
```

- [ ] ok
- [ ] reject &mdash; reason:
