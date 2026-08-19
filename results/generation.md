# Generation and long-context evaluation

Model `gemini-3.6-flash`, 60 of 586 queries (seeded stratified sample across the lexical-overlap range), of which 11 also received a long-context answer.

The **long-context arm is handed the correct filing**, while the retrieval
arm must find it among 120. That makes long-context a deliberately
generous baseline: any retrieval win on cost or latency at comparable
accuracy holds despite the comparison being stacked against it.

> **Partial run.** still rate-limited after 8 attempts on /models/gemini-3.6-flash:generateContent. The daily free-tier quota is likely spent; re-run later. Cached work is preserved, so the run resumes rather than restarts.
> Numbers below cover only the answers that completed.

| arm | answered | refused | value acc (answered) | value acc (all) | citation prec | citation recall | faithfulness |
|---|---|---|---|---|---|---|---|
| `long_context` | 11 | 0 | 0.636 | 0.636 | not measured | not measured | not measured |
| `retrieval` | 23 | 23 | 0.609 | 0.304 | 0.246 | 0.348 | 1.000 |

## Retrieval versus long context

| | retrieval | long context | ratio |
|---|---|---|---|
| mean prompt tokens | 7,316 | 130,819 | 17.9x |
| cost per query (USD) | 0.011155 | 0.196508 | **17.6x** |
| p95 latency (s) | 63.149 | None | Nonex |

### On the brief's 1,250x claim

The project brief asserted retrieval is *roughly 1,250x cheaper per query*.
Measured here: **17.6x**.

1,250x is only reachable by assuming a full 1M-token context, an
~800-token retrieval prompt, and zero output cost. The brief's own draft
resume bullet says 1/40th, which is far closer to what this measures.

## API usage for this run

```json
{
  "live_calls": 80,
  "cached_calls": 50,
  "prompt_tokens": 1929950,
  "output_tokens": 1562,
  "thought_tokens": 32888,
  "rate_limited_responses": 10,
  "seconds_spent_waiting": 503.1,
  "p95_latency_seconds": 60.945
}
```

Costs above are the published paid-tier prices applied to the API's own
reported token counts. The run itself was on the free tier and cost $0.
