# Generation and long-context evaluation

Model `gemini-3.6-flash`, 90 of 586 queries (seeded stratified sample across the lexical-overlap range), of which 11 also received a long-context answer.

The **long-context arm is handed the correct filing**, while the retrieval
arm must find it among 120. That makes long-context a deliberately
generous baseline: any retrieval win on cost or latency at comparable
accuracy holds despite the comparison being stacked against it.

> **Partial run.** still rate-limited after 8 attempts on /models/gemini-3.6-flash:generateContent. The daily free-tier quota is likely spent; re-run later. Cached work is preserved, so the run resumes rather than restarts.
> Numbers below cover only the answers that completed.

| arm | answered | refused | value acc (answered) | value acc (all) | citation prec | citation recall | faithfulness |
|---|---|---|---|---|---|---|---|
| `long_context` | 11 | 0 | 0.636 | 0.636 | not measured | not measured | not measured |
| `retrieval` | 32 | 34 | 0.656 | 0.318 | 0.266 | 0.438 | 1.000 |

## Retrieval versus long context

| | retrieval | long context | ratio |
|---|---|---|---|
| mean prompt tokens | 7,147 | 130,819 | 18.3x |
| cost per query (USD) | 0.010895 | 0.196508 | **18.0x** |
| p95 latency (s) | 15.489 | None | Nonex |

### On the brief's 1,250x claim

The project brief asserted retrieval is *roughly 1,250x cheaper per query*.
Measured here: **18.0x**.

1,250x is only reachable by assuming a full 1M-token context, an
~800-token retrieval prompt, and zero output cost. The brief's own draft
resume bullet says 1/40th, which is far closer to what this measures.

## API usage for this run

```json
{
  "live_calls": 109,
  "cached_calls": 80,
  "prompt_tokens": 2119586,
  "output_tokens": 2008,
  "thought_tokens": 42590,
  "rate_limited_responses": 10,
  "seconds_spent_waiting": 425.6,
  "p95_latency_seconds": 42.627
}
```

Costs above are the published paid-tier prices applied to the API's own
reported token counts. The run itself was on the free tier and cost $0.
