# Generation and long-context evaluation

Model `gemini-3.6-flash`, 170 of 586 queries (seeded stratified sample across the lexical-overlap range), of which 11 also received a long-context answer.

The **long-context arm is handed the correct filing**, while the retrieval
arm must find it among 120. That makes long-context a deliberately
generous baseline: any retrieval win on cost or latency at comparable
accuracy holds despite the comparison being stacked against it.

> **Partial run.** still rate-limited after 8 attempts on /models/gemini-3.6-flash:generateContent. The server names the exceeded quota as GenerateRequestsPerDayPerProjectPerModel-FreeTier (limit 20). Re-run later; cached work is preserved, so the run resumes rather than restarts.
> Numbers below cover only the answers that completed.

| arm | answered | refused | value acc (answered) | value acc (all) | citation prec | citation recall | faithfulness |
|---|---|---|---|---|---|---|---|
| `long_context` | 11 | 0 | 0.636 | 0.636 | not measured | not measured | 0.955 |
| `retrieval` | 63 | 67 | 0.683 | 0.331 | 0.291 | 0.476 | 0.968 |

## Retrieval versus long context

| | retrieval | long context | ratio |
|---|---|---|---|
| mean prompt tokens | 6,916 | 130,819 | 18.9x |
| cost per query (USD) | 0.010552 | 0.196508 | **18.6x** |
| p95 latency (s) | 39.088 | None | Nonex |

### On the brief's 1,250x claim

The project brief asserted retrieval is *roughly 1,250x cheaper per query*.
Measured here: **18.6x**.

1,250x is only reachable by assuming a full 1M-token context, an
~800-token retrieval prompt, and zero output cost. The brief's own draft
resume bullet says 1/40th, which is far closer to what this measures.

## API usage for this run

```json
{
  "live_calls": 214,
  "cached_calls": 213,
  "prompt_tokens": 4178448,
  "output_tokens": 3654,
  "thought_tokens": 77892,
  "rate_limited_responses": 10,
  "seconds_spent_waiting": 475.1,
  "p95_latency_seconds": 54.18
}
```

Costs above are the published paid-tier prices applied to the API's own
reported token counts. The run itself was on the free tier and cost $0.
