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
- [ ] reject &mdash; reason:

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
- [ ] reject &mdash; reason:

## 3. `q-1162074b5351`

**Query:** Costco Wholesale Corp /New receivables, net 2024

- lexical overlap: `0.29`
- document: `cost-10-k-2024-09-01`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `2,721`

**Labelled passage:**

```
| Receivables, net | 2,721 | 2,285 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 4. `q-09df5fd594cf`

**Query:** Report Nvidia Corp's direct customer b figure for the 2025 fiscal year.

- lexical overlap: `0.29`
- document: `nvda-10-k-2025-01-26`
- section: Part II > Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `11 %`

**Labelled passage:**

```
| Direct Customer B | 11 % | 13 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 5. `q-af076cc0cc00`

**Query:** What was Goldman Sachs Group Inc's tlac to rwas for 2025?

- lexical overlap: `0.29`
- document: `gs-10-k-2025-12-31`
- section: Part II > Item 7. Management’s Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `22.0 %`

**Labelled passage:**

```
| TLAC to RWAs | 22.0 % | 22.0 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 6. `q-6bfa3a36c975`

**Query:** In 2025, what amount did Blackrock, Inc. record as january 31, 2028?

- lexical overlap: `0.33`
- document: `blk-10-k-2025-12-31`
- section: Part IV > Item 16. Form 10-K Summary
- expected value: `221,825`

**Labelled passage:**

```
| January 31, 2028 | 221,825 | — | — |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 7. `q-08822516f63f`

**Query:** Report Exxon Mobil Corp's canada/other americas figure for the 2022 fiscal year.

- lexical overlap: `0.33`
- document: `xom-10-k-2022-12-31`
- section: Part I > Item 2. PROPERTIES
- expected value: `33`

**Labelled passage:**

```
| Canada/Other Americas | 33 | 28 | 36 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 8. `q-63e696bc7165`

**Query:** How much did Duke Energy Corp report for debt securities in 2022?

- lexical overlap: `0.33`
- document: `duk-10-k-2022-12-31`
- section: Part I > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
- expected value: `30 %`

**Labelled passage:**

```
| Debt securities | 35 % | 30 % | 62 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 9. `q-be6047eb6f60`

**Query:** In 2024, what amount did Costco Wholesale Corp /New record as total net sales?

- lexical overlap: `0.33`
- document: `cost-10-k-2024-09-01`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 11 - Segment Reporting
- expected value: `$ 249,625`

**Labelled passage:**

```
| Total net sales | $ 249,625 | $ 237,710 | $ 222,730 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 10. `q-255904090788`

**Query:** Report Nvidia Corp's prepaid supply agreements figure for the 2023 fiscal year.

- lexical overlap: `0.38`
- document: `nvda-10-k-2023-01-29`
- section: Part IV > Item 15. EXHIBIT AND FINANCIAL STATEMENT SCHEDULES > Note 10 - Balance Sheet Components
- expected value: `$ 2,989`

**Labelled passage:**

```
| Prepaid supply agreements | $ 2,989 |  | $ 1,747 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 11. `q-6229d7de2045`

**Query:** Report Nvidia Corp's acquisition termination cost figure for the 2023 fiscal year.

- lexical overlap: `0.38`
- document: `nvda-10-k-2024-01-28`
- section: Part II > Item 7. Management's Discussion and Analysis of Financial Condition and Results of Operations
- expected value: `5.0`

**Labelled passage:**

```
| Acquisition termination cost | — | 5.0 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 12. `q-21b34c85f0c4`

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

## 13. `q-8b7d35bc2bf3`

**Query:** How much did Chevron Corp report for affiliated companies in 2024?

- lexical overlap: `0.40`
- document: `cvx-10-k-2024-12-31`
- section: Part I > Item 1. Business
- expected value: `1,849`

**Labelled passage:**

```
| Affiliated Companies | 1,849 | 2,063 | 2,099 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 14. `q-0fb5b894ecc9`

**Query:** How much did Coca Cola Co report for lease obligations in 2023?

- lexical overlap: `0.40`
- document: `ko-10-k-2022-12-31`
- section: Part II > Item 7. MANAGEMENT’S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS
- expected value: `466`

**Labelled passage:**

```
| Lease obligations | 2,291 | 466 | 680 | 411 | 734 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 15. `q-95ad7f945b9c`

**Query:** How much did Pfizer Inc report for income taxes in 2025?

- lexical overlap: `0.40`
- document: `pfe-10-k-2025-12-31`
- section: Part II > Item 8
- expected value: `$ 4,688`

**Labelled passage:**

```
| Income taxes | $ 4,688 | $ 3,605 | $ 3,147 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 16. `q-cff202e73361`

**Query:** In 2025, what amount did Boeing Co record as commercial airplanes?

- lexical overlap: `0.40`
- document: `ba-10-k-2025-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 7 - Accounts Receivable, net
- expected value: `129`

**Labelled passage:**

```
| Commercial Airplanes | 129 | 48 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 17. `q-368c6d3eda50`

**Query:** Unitedhealth Group Inc total intangible assets 2024

- lexical overlap: `0.43`
- document: `unh-10-k-2024-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
- expected value: `$ 10,602`

**Labelled passage:**

```
| Total intangible assets | $ 10,602 | $ 2,174 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 18. `q-11db0ef44c04`

**Query:** In 2023, what amount did Amgen Inc record as gross product sales?

- lexical overlap: `0.43`
- document: `amgn-10-k-2023-12-31`
- section: Part IV > Item 16
- expected value: `$ 9,775`

**Labelled passage:**

```
| Gross product sales | $ 9,775 | $ 8,319 | $ 7,681 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 19. `q-ac40bbda1cb6`

**Query:** Cisco Systems, Inc. secure, agile networks 2022

- lexical overlap: `0.43`
- document: `csco-10-k-2022-07-30`
- section: Part II > Item 7
- expected value: `$ 23,829`

**Labelled passage:**

```
| Secure, Agile Networks | $ 23,829 | $ 22,722 | $ 23,265 | $ 1,107 | 5 % | $ (543) | (2) % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 20. `q-30ed474919fd`

**Query:** Report Nike, Inc.'s weighted-average discount rate figure for the 2026 fiscal year.

- lexical overlap: `0.44`
- document: `nke-10-k-2026-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 17 - LEASES
- expected value: `3.6 %`

**Labelled passage:**

```
| Weighted-average discount rate | 3.6 % | 3.1 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 21. `q-b0fa654ef2ef`

**Query:** Report Nvidia Corp's total long-lived assets figure for the 2025 fiscal year.

- lexical overlap: `0.44`
- document: `nvda-10-k-2025-01-26`
- section: Part IV > Item 15. Exhibits and Financial Statement Schedules > Note 16 - Segment Information
- expected value: `$ 6,283`

**Labelled passage:**

```
| Total long-lived assets | $ 6,283 |  | $ 3,914 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 22. `q-1bcea8b7d7d3`

**Query:** In 2022, what amount did Boeing Co record as other employee benefits?

- lexical overlap: `0.50`
- document: `ba-10-k-2022-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 4 - Income Taxes
- expected value: `1,095`

**Labelled passage:**

```
| Other employee benefits | 1,095 | 991 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 23. `q-784c39eb2ab9`

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

## 24. `q-b799f5f6fd89`

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
- [ ] reject &mdash; reason:

## 25. `q-f1620d85d312`

**Query:** What was Nike, Inc.'s stock-based compensation for 2025?

- lexical overlap: `0.50`
- document: `nke-10-k-2025-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 7 - INCOME TAXES
- expected value: `1.5 %`

**Labelled passage:**

```
| Stock-based compensation | 1.5 % | -0.5 % | -1.1 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 26. `q-270eea3e397e`

**Query:** How much did Johnson & Johnson report for interest income in 2025?

- lexical overlap: `0.50`
- document: `jnj-10-k-2025-12-28`
- section: Part II > Item 8. Financial statements and supplementary data
- expected value: `(1,056)`

**Labelled passage:**

```
| Interest income | (1,056) | (1,332) | (1,261) |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 27. `q-4c8cc1e973d1`

**Query:** How much did General Electric Co report for net investment hedges(b) in 2024?

- lexical overlap: `0.50`
- document: `ge-10-k-2024-12-31`
- section: Note 20 - RESTRUCTURING CHARGES AND SEPARATION COSTS
- expected value: `348`

**Labelled passage:**

```
| Net investment hedges(b) | 348 | (150) | — | — |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 28. `q-91f6355309b6`

**Query:** What was Jpmorgan Chase & Co's lending-related commitments for 2023?

- lexical overlap: `0.50`
- document: `jpm-10-k-2025-12-31`
- section: Part IV > Note 34 - Business combinations
- expected value: `2,614`

**Labelled passage:**

```
| Lending-related commitments | 2,614 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 29. `q-adc02940c064`

**Query:** American Express Co actual tax rates 2024

- lexical overlap: `0.50`
- document: `axp-10-k-2024-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 20
- expected value: `21.5 %`

**Labelled passage:**

```
| Actual tax rates | 21.5 % | 20.3 % | 21.6 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 30. `q-e6c296eb8121`

**Query:** Cisco Systems, Inc. maximum potential future payments 2025

- lexical overlap: `0.50`
- document: `csco-10-k-2025-07-26`
- section: Part II > Item 8
- expected value: `$ 123`

**Labelled passage:**

```
| Maximum potential future payments | $ 123 | $ 127 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 31. `q-5aeadc6d93cd`

**Query:** In 2024, what amount did Nike, Inc. record as weighted average expected life (in years)?

- lexical overlap: `0.56`
- document: `nke-10-k-2024-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 9 - COMMON STOCK AND STOCK-BASED COMPENSATION
- expected value: `5.8`

**Labelled passage:**

```
| Weighted average expected life (in years) | 5.8 | 5.8 | 5.8 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 32. `q-a3a6c992618b`

**Query:** How much did Procter & Gamble Co report for present value of lease liabilities in 2022?

- lexical overlap: `0.57`
- document: `pg-10-k-2022-06-30`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 12
- expected value: `$ 800`

**Labelled passage:**

```
| Present value of lease liabilities | $ 800 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 33. `q-32397c0ecee7`

**Query:** American Express Co tax credits and tax-exempt income 2024

- lexical overlap: `0.57`
- document: `axp-10-k-2024-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 20
- expected value: `(0.7)`

**Labelled passage:**

```
| Tax credits and tax-exempt income | (0.7) | (0.7) | (0.9) |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 34. `q-ad4679f177b8`

**Query:** Apple Inc. weighted-average diluted shares 2024

- lexical overlap: `0.57`
- document: `aapl-10-k-2024-09-28`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 3 - Earnings Per Share
- expected value: `15,408,095`

**Labelled passage:**

```
| Weighted-average diluted shares | 15,408,095 | 15,812,547 | 16,325,819 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 35. `q-2085c224efbd`

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

## 36. `q-b152f084106a`

**Query:** In 2022, what amount did Amgen Inc record as additions based on tax positions related to the current year?

- lexical overlap: `0.60`
- document: `amgn-10-k-2022-12-31`
- section: Part IV > Item 16
- expected value: `151`

**Labelled passage:**

```
| Additions based on tax positions related to the current year | 151 | 171 | 165 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 37. `q-db335a976fe4`

**Query:** American Express Co total non-interest-bearing liabilities 2025

- lexical overlap: `0.62`
- document: `axp-10-k-2025-12-31`
- section: Part IV > Item 16. FORM 10-K SUMMARY
- expected value: `56,018`

**Labelled passage:**

```
| Total non-interest-bearing liabilities | 56,018 | 52,228 | 49,335 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 38. `q-bb95cabcc92d`

**Query:** What was At&T Inc.'s discount rate in effect for determining interest cost for 2025?

- lexical overlap: `0.62`
- document: `t-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 14 - PENSION AND POSTRETIREMENT BENEFITS
- expected value: `5.40 %`

**Labelled passage:**

```
| Discount rate in effect for determining interest cost1 | 5.40 % |  | 4.90 % |  | 5.30 % |  | 5.30 % |  | 4.90 % |  | 5.10 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 39. `q-89a8339e0c42`

**Query:** How much did Walmart Inc. report for weighted-average discount rate - operating leases in 2026?

- lexical overlap: `0.67`
- document: `wmt-10-k-2026-01-31`
- section: Part II > Item 8 > Note 6 - Leases
- expected value: `6.7%`

**Labelled passage:**

```
| Weighted-average discount rate - operating leases | 6.7% | 6.5% |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 40. `q-57f01f934b58`

**Query:** Report Chevron Corp's net borrowings (repayments) of short-term obligations with three months or less maturity figure for the 2025 fiscal year.

- lexical overlap: `0.67`
- document: `cvx-10-k-2025-12-31`
- section: Part III > Item 14. Principal Accountant Fees and Services > Note 3
- expected value: `964`

**Labelled passage:**

```
| Net borrowings (repayments) of short-term obligations with three months or less maturity | 964 | 1,169 | 135 |
```

- [ ] ok
- [ ] reject &mdash; reason:
