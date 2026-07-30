# Verification sample

Labels in `queries.jsonl` are **generated**, not human-verified. Each entry
below shows a query and the exact passage labelled as its answer.

For each one, mark `[x] ok` if the passage genuinely answers the query, or
`[x] reject` with a short reason if it does not. The rejection rate is the
only evidence available about how trustworthy the generated labels are in
bulk, so a completed sample is worth more than a larger unverified set.

Sampled 40 of 220 queries, spread across the lexical-overlap range.

---

## 1. `q-50e83bc1b8a3`

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

## 2. `q-90222e290941`

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

## 3. `q-6bba39bacc18`

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

## 4. `q-e0b4b20aceba`

**Query:** Report Exxon Mobil Corp's united states figure for the 2024 fiscal year.

- lexical overlap: `0.25`
- document: `xom-10-k-2024-12-31`
- section: Part IV > Item 16. FORM 10-K SUMMARY
- expected value: `1,248`

**Labelled passage:**

```
| United States | 1,248 |  | 803 |  | 776 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 5. `q-6246985023da`

**Query:** Report American Express Co's valuation allowances figure for the 2023 fiscal year.

- lexical overlap: `0.29`
- document: `axp-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 19
- expected value: `0.1`

**Labelled passage:**

```
| Valuation allowances | — | 0.1 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 6. `q-7907960093a5`

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

## 7. `q-aa8544d40f2a`

**Query:** Report Procter & Gamble Co's over 5 years figure for the 2023 fiscal year.

- lexical overlap: `0.29`
- document: `pg-10-k-2023-06-30`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 12
- expected value: `196`

**Labelled passage:**

```
| Over 5 years | 196 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 8. `q-44c9c73a3009`

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

## 9. `q-7e2735aa091f`

**Query:** In 2025, what amount did Procter & Gamble Co record as translation and other?

- lexical overlap: `0.33`
- document: `pg-10-k-2025-06-30`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 14
- expected value: `98`

**Labelled passage:**

```
| Translation and other | 98 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 10. `q-b24cd6f713d1`

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

## 11. `q-070da32873bf`

**Query:** Report American Express Co's actual tax rates figure for the 2024 fiscal year.

- lexical overlap: `0.38`
- document: `axp-10-k-2025-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 19
- expected value: `21.5 %`

**Labelled passage:**

```
| Actual tax rates | 21.5 % | 20.3 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 12. `q-b10d8b0ae974`

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

## 13. `q-0476c17db3fd`

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

## 14. `q-50eb6659dc06`

**Query:** How much did American Express Co report for expected volatility in 2023?

- lexical overlap: `0.40`
- document: `axp-10-k-2023-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 10
- expected value: `45 %`

**Labelled passage:**

```
| Expected volatility | 45 % | 42 % | 41 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 15. `q-94d2bf3e7137`

**Query:** Caterpillar Inc commercial paper 2023

- lexical overlap: `0.40`
- document: `cat-10-k-2023-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `5.2 %`

**Labelled passage:**

```
| Commercial paper | 5.2 % | 4.2 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 16. `q-a5b86213a540`

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

## 17. `q-c6d7cfc794ac`

**Query:** In 2022, what amount did At&T Inc. record as private equity?

- lexical overlap: `0.40`
- document: `t-10-k-2022-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 14 - PENSION AND POSTRETIREMENT BENEFITS
- expected value: `14`

**Labelled passage:**

```
| Private equity | — % - 16 % | 14 | 12 | — % - 6 % | 1 | 1 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 18. `q-e2464ed4e444`

**Query:** How much did Blackrock, Inc. report for january 31, 2025 in 2022?

- lexical overlap: `0.40`
- document: `blk-10-k-2024-12-31`
- section: Part IV > Item 16. Form 10-K Summary
- expected value: `197,817`

**Labelled passage:**

```
| January 31, 2025 | — | — | 197,817 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 19. `q-09aa829ca4c1`

**Query:** What was Unitedhealth Group Inc's expected life in years for 2022?

- lexical overlap: `0.43`
- document: `unh-10-k-2022-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
- expected value: `4.7`

**Labelled passage:**

```
| Expected life in years | 4.7 | 4.8 | 5.1 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 20. `q-81a442ebda41`

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

## 21. `q-ceda45797b8d`

**Query:** Cisco Systems, Inc. deferred tax assets 2024

- lexical overlap: `0.43`
- document: `csco-10-k-2024-07-27`
- section: Part II > Item 8
- expected value: `6,262`

**Labelled passage:**

```
| Deferred tax assets | 6,262 | 6,576 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 22. `q-7bbb1d14e532`

**Query:** Report Procter & Gamble Co's present value of lease liabilities figure for the 2023 fiscal year.

- lexical overlap: `0.44`
- document: `pg-10-k-2023-06-30`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 12
- expected value: `$ 817`

**Labelled passage:**

```
| Present value of lease liabilities | $ 817 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 23. `q-023499a2ab9e`

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

## 24. `q-0da6479a6802`

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

## 25. `q-244f0fd024ce`

**Query:** Boeing Co valuation allowance 2023

- lexical overlap: `0.50`
- document: `ba-10-k-2023-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 4 - Income Taxes
- expected value: `(4,550)`

**Labelled passage:**

```
| Valuation allowance | (4,550) | (3,162) |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 26. `q-4e0e78b1eecd`

**Query:** What was Duke Energy Corp's natural gas and fuel oil(a) for 2022?

- lexical overlap: `0.50`
- document: `duk-10-k-2022-12-31`
- section: Item 1. BUSINESS
- expected value: `34.2 %`

**Labelled passage:**

```
| Natural gas and fuel oil(a) | 34.2 % | 31.8 % | 31.3 % | 6.35 | 3.89 | 2.55 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 27. `q-705a18b2a1b8`

**Query:** What was Unitedhealth Group Inc's total acquired intangible assets for 2023?

- lexical overlap: `0.50`
- document: `unh-10-k-2023-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
- expected value: `$ 2,174`

**Labelled passage:**

```
| Total acquired intangible assets | $ 2,174 | $ 5,814 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 28. `q-8b893f366341`

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

## 29. `q-b71b37b4d785`

**Query:** At&T Inc. guarantee obligation recorded 2024

- lexical overlap: `0.50`
- document: `t-10-k-2024-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 17 - SALES OF RECEIVABLES
- expected value: `930`

**Labelled passage:**

```
| Guarantee obligation recorded | 930 |  | 932 |  | 703 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 30. `q-befd6e335536`

**Query:** What was Walmart Inc.'s federal tax credits for 2024?

- lexical overlap: `0.50`
- document: `wmt-10-k-2024-01-31`
- section: Part II > Item 8 > Note 9 - Taxes
- expected value: `(1.5) %`

**Labelled passage:**

```
| Federal tax credits | (1.5) % | (1.3) % | (1.1) % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 31. `q-cd3b063dbe84`

**Query:** What was Amgen Inc's segment net income for 2025?

- lexical overlap: `0.50`
- document: `amgn-10-k-2025-12-31`
- section: Part IV > Item 16
- expected value: `7,711`

**Labelled passage:**

```
| Segment net income | 7,711 | 4,090 | 6,717 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 32. `q-e31c0cf148c2`

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

## 33. `q-495ca380153f`

**Query:** In 2025, what amount did Nike, Inc. record as income tax audits and contingency reserves?

- lexical overlap: `0.56`
- document: `nke-10-k-2025-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 7 - INCOME TAXES
- expected value: `2.7 %`

**Labelled passage:**

```
| Income tax audits and contingency reserves | 2.7 % | 1.8 % | 1.0 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 34. `q-26f1228288b9`

**Query:** Nike, Inc. federal income tax rate 2024

- lexical overlap: `0.57`
- document: `nke-10-k-2024-05-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 7 - INCOME TAXES
- expected value: `21.0 %`

**Labelled passage:**

```
| Federal income tax rate | 21.0 % | 21.0 % | 21.0 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 35. `q-5f97033e73b4`

**Query:** How much did Apple Inc. report for net deferred tax assets in 2024?

- lexical overlap: `0.57`
- document: `aapl-10-k-2024-09-28`
- section: Part II > Item 8. Financial Statements and Supplementary Data > Note 7 - Income Taxes
- expected value: `$ 19,202`

**Labelled passage:**

```
| Net deferred tax assets | $ 19,202 | $ 17,251 |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 36. `q-9529ce24a405`

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

## 37. `q-eba853295733`

**Query:** Report Merck & Co., Inc.'s acquisition of verona pharma plc, net of cash acquired figure for the 2025 fiscal year.

- lexical overlap: `0.58`
- document: `mrk-10-k-2025-12-31`
- section: Part II > Item 8. Financial Statements and Supplementary Data
- expected value: `(10,042)`

**Labelled passage:**

```
| Acquisition of Verona Pharma plc, net of cash acquired | (10,042) | — | — |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 38. `q-3c6776461a83`

**Query:** How much did Duke Energy Corp report for duke energy ohio in 2025?

- lexical overlap: `0.60`
- document: `duk-10-k-2025-12-31`
- section: Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA
- expected value: `2.9 %`

**Labelled passage:**

```
| Duke Energy Ohio | 2.9 % | 2.9 % | 2.8 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 39. `q-f455ef4f2e47`

**Query:** Report At&T Inc.'s long-term rate of return on plan assets figure for the 2024 fiscal year.

- lexical overlap: `0.60`
- document: `t-10-k-2024-12-31`
- section: Part II > Item 8. FINANCIAL STATEMENTS AND SUPPLEMENTARY DATA > Note 14 - PENSION AND POSTRETIREMENT BENEFITS
- expected value: `7.75 %`

**Labelled passage:**

```
| Long-term rate of return on plan assets | 7.75 % |  | 7.50 % |  | 6.75 % |  | 4.00 % |  | 6.50 % |  | 4.50 % |
```

- [ ] ok
- [ ] reject &mdash; reason:

## 40. `q-4a57f1fd036a`

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
