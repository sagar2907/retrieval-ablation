# Generation and long-context evaluation

Model `gemini-3.6-flash`, 27 of 586 queries (seeded stratified sample across the lexical-overlap range), of which 11 also received a long-context answer.

The **long-context arm is handed the correct filing**, while the retrieval
arm must find it among 120. That makes long-context a deliberately
generous baseline: any retrieval win on cost or latency at comparable
accuracy holds despite the comparison being stacked against it.

| arm | answered | refused | value acc (answered) | value acc (all) | citation prec | citation recall | faithfulness |
|---|---|---|---|---|---|---|---|
| `long_context` | 11 | 0 | 0.636 | 0.636 | not measured | not measured | not measured |
| `retrieval` | 12 | 15 | 0.583 | 0.259 | 0.403 | 0.500 | 1.000 |

## Retrieval versus long context

| | retrieval | long context | ratio |
|---|---|---|---|
| mean prompt tokens | 7,085 | 130,819 | 18.5x |
| cost per query (USD) | 0.010806 | 0.196508 | **18.2x** |
| p95 latency (s) | None | None | Nonex |

### On the brief's 1,250x claim

The project brief asserted retrieval is *roughly 1,250x cheaper per query*.
Measured here: **18.2x**.

1,250x is only reachable by assuming a full 1M-token context, an
~800-token retrieval prompt, and zero output cost. The brief's own draft
resume bullet says 1/40th, which is far closer to what this measures.

## API usage for this run

```json
{
  "live_calls": 50,
  "cached_calls": 50,
  "prompt_tokens": 1709179,
  "output_tokens": 1078,
  "thought_tokens": 23380,
  "rate_limited_responses": 0,
  "seconds_spent_waiting": 0.0,
  "p95_latency_seconds": 42.018
}
```

Costs above are the published paid-tier prices applied to the API's own
reported token counts. The run itself was on the free tier and cost $0.
