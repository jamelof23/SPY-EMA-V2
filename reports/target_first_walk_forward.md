# Does the rest of the chart add to the target and stop distances?

Three chronological checks compared the same tree model with and without daily and intraday chart features. Both versions knew the distances from the completed signal close to the prior daily EMA8 target and the prior close stop, normalized by recent five-minute candle range. Both also knew time of day. Each yearly model fitted only earlier years and every represented day had equal weight. A success means SPY reached the target first within the session.

| Validation year | Sessions | Decisions | Distances and time AUC | Plus chart AUC | Distances and time log loss | Plus chart log loss |
|---|---:|---:|---:|---:|---:|---:|
| 2022 | 93 | 579 | 0.862 | 0.870 | 0.407 | 0.409 |
| 2023 | 61 | 373 | 0.889 | 0.892 | 0.394 | 0.387 |
| 2024 | 51 | 326 | 0.911 | 0.901 | 0.368 | 0.386 |

Across 205 out-of-year trading sessions, adding chart features increased average log loss by 0.0034. The session bootstrap 95% interval ran from −0.0280 to +0.0356. Chart features helped in 2023 and failed to improve loss in 2022 and 2024.

This supports using distance to the target, distance to the stop and time remaining as the first benchmark for this setup. A nearby target is mechanically easier to touch. High target-first AUC alone does not establish positive expected returns. All comparisons use SPY prices and exclude options premiums and costs. No 2025 or 2026 observations were scored in this study.
