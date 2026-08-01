# Generation and long-context evaluation

Model `gemini-3.6-flash`, 12 of 216 queries (seeded stratified sample across the lexical-overlap range).

The **long-context arm is handed the correct filing**, while the retrieval
arm must find it among 120. That makes long-context a deliberately
generous baseline: any retrieval win on cost or latency at comparable
accuracy holds despite the comparison being stacked against it.

> **Partial run.** still rate-limited after 8 attempts on /models/gemini-3.6-flash:generateContent. The daily free-tier quota is likely spent; re-run later. Cached work is preserved, so the run resumes rather than restarts.
> Numbers below cover only the answers that completed.

| arm | answered | refused | value acc (answered) | value acc (all) | citation prec | citation recall | faithfulness |
|---|---|---|---|---|---|---|---|
| `long_context` | 9 | 0 | 0.556 | 0.556 | not measured | not measured | not measured |
| `retrieval` | 5 | 5 | 0.600 | 0.300 | 0.567 | 0.800 | not measured |

## Retrieval versus long context

| | retrieval | long context | ratio |
|---|---|---|---|
| mean prompt tokens | 7,345 | 130,701 | 17.8x |
| cost per query (USD) | 0.011224 | 0.196322 | **17.5x** |
| p95 latency (s) | 4.542 | 8.543 | 1.9x |

### On the brief's 1,250x claim

The project brief asserted retrieval is *roughly 1,250x cheaper per query*.
Measured here: **17.5x**.

1,250x is only reachable by assuming a full 1M-token context, an
~800-token retrieval prompt, and zero output cost. The brief's own draft
resume bullet says 1/40th, which is far closer to what this measures.

## API usage for this run

```json
{
  "live_calls": 19,
  "cached_calls": 19,
  "prompt_tokens": 1249763,
  "output_tokens": 600,
  "thought_tokens": 10846,
  "rate_limited_responses": 8,
  "seconds_spent_waiting": 391.5,
  "p95_latency_seconds": 8.543
}
```

Costs above are the published paid-tier prices applied to the API's own
reported token counts. The run itself was on the free tier and cost $0.
