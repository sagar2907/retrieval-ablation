# Generation and long-context evaluation

Model `gemini-3.6-flash`, 66 of 586 queries (seeded stratified sample across the lexical-overlap range), of which 11 also received a long-context answer.

The **long-context arm is handed the correct filing**, while the retrieval
arm must find it among 120. That makes long-context a deliberately
generous baseline: any retrieval win on cost or latency at comparable
accuracy holds despite the comparison being stacked against it.

> **Partial run.** still rate-limited after 8 attempts on /models/gemini-3.6-flash:generateContent. The server names the exceeded quota as GenerateRequestsPerDayPerProjectPerModel-FreeTier (limit 20). Re-run later; cached work is preserved, so the run resumes rather than restarts.
> Numbers below cover only the answers that completed.

| arm | answered | refused | value acc (answered) | value acc (all) | citation prec | citation recall | faithfulness |
|---|---|---|---|---|---|---|---|
| `long_context` | 9 | 2 | 0.667 | 0.545 | not measured | not measured | 1.000 |
| `retrieval` | 8 | 5 | 1.000 | 0.615 | 0.354 | 0.625 | 0.875 |

## Retrieval versus long context

| | retrieval | long context | ratio |
|---|---|---|---|
| mean prompt tokens | 5,815 | 131,596 | 22.6x |
| cost per query (USD) | 0.009018 | 0.197733 | **21.9x** |
| p95 latency (s) | 51.923 | 60.566 | 1.2x |

### On the brief's 1,250x claim

The project brief asserted retrieval is *roughly 1,250x cheaper per query*.
Measured here: **21.9x**.

1,250x is only reachable by assuming a full 1M-token context, an
~800-token retrieval prompt, and zero output cost. The brief's own draft
resume bullet says 1/40th, which is far closer to what this measures.

## API usage for this run

```json
{
  "live_calls": 41,
  "cached_calls": 4,
  "prompt_tokens": 2764954,
  "output_tokens": 1043,
  "thought_tokens": 16203,
  "rate_limited_responses": 26,
  "seconds_spent_waiting": 1359.8,
  "p95_latency_seconds": 50.189
}
```

Costs above are the published paid-tier prices applied to the API's own
reported token counts. The run itself was on the free tier and cost $0.
