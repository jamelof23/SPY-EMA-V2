# Does predicting a target touch translate into payoff?

This exploratory study used the 2022, 2023 and 2024 out-of-year target probabilities from the distance-and-time model. The target was the prior daily EMA8 and the stop was the previous close. At :05 and :35, each filter could take the first qualifying entry of a session. An eligible next-bar open inside the two levels was required. Targets were capped at the planned level, adverse gaps exited at the bar open, and remaining positions closed that session. An assumed two basis points per completed round trip was applied to SPY. The eligible population covers 205 trading sessions and includes different candle colors.

| Fixed filter | Trades | Target first | Stop first | Neither | Net basis points per eligible day | 95% interval |
|---|---:|---:|---:|---:|---:|---:|
| First eligible observation | 205 | 53 | 131 | 21 | −2.44 | −9.32 to +4.66 |
| Estimated target probability ≥ 60% | 70 | 51 | 13 | 6 | +0.50 | −2.48 to +3.28 |
| Estimated reward minus risk > 2 basis points | 94 | 12 | 79 | 3 | +0.38 | −3.82 to +5.07 |
| Planned upside at least as large as downside | 183 | 23 | 137 | 23 | −4.77 | −11.00 to +2.20 |

Skipping a session contributes zero to the per-day result. The reward-minus-risk filter uses estimated target probability times planned upside minus the remaining probability times planned downside. Unresolved positions therefore enter this screening calculation as potential full stops. The reported realized prices use the actual end-of-session exit where applicable.

A high target-first rate can coexist with little payoff, especially if the target is close and the stop is distant. All four session-resampled intervals included zero. These are exploratory results for the underlying SPY price and do not estimate option-contract profit.
