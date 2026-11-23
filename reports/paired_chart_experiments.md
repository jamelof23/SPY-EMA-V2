# SPY overnight research update

The 2020–2025 dataset contains 117,184 paired decisions from 1,508 trading sessions. Each pair combines 42 daily candles through the previous close with 42 five-minute candles through the decision time. Volume, EMA8 and EMA21 appear in both views. The complete dataset has 97,792 development pairs through 2024 and 19,392 validation pairs in 2025. Eight missing five-minute intervals in March 2020 were left missing. No 2026 market data entered these experiments.

## Does daily context help predict the next 30 minutes?

The first image experiment fitted two small CNNs using 71,793 nonflat observations through 2023. The 17,936 eligible 2024 observations determined the best epoch by log loss. The selected epoch for both networks was 3. The same 17,806 eligible decisions in 2025 were scored once for this comparison. Targets use SPY from the next five-minute open to the close of the sixth subsequent bar. Ties, missing forward intervals and forecasts extending beyond the session close were excluded.

| 2025 model | Accuracy | Balanced accuracy | AUC | Log loss |
|---|---:|---:|---:|---:|
| Constant up probability | 52.40% | 50.00% | 0.5000 | 0.69202 |
| Five-minute image CNN | 52.40% | 50.00% | 0.5155 | 0.69189 |
| Daily and five-minute image CNN | 52.40% | 50.00% | 0.5172 | 0.69177 |
| Daily and intraday numerical tree model | 52.75% | 51.00% | 0.5223 | 0.69162 |

At a 0.50 decision threshold both CNNs predicted up on every 2025 observation. The paired model improved the ranking slightly. A bootstrap that resampled whole trading sessions put the paired-minus-intraday log-loss difference between −0.00038 and +0.00015. That interval includes zero. None of these results establishes an options trading edge.

### CPU image CNN epochs

The table shows 2020–2023 fit performance and 2024 development performance. Lower loss is better.

| Image inputs | Epoch | Fit loss | Fit accuracy | Development loss | Development accuracy | Development AUC |
|---|---:|---:|---:|---:|---:|---:|
| Five-minute | 1 | 0.69168 | 52.71% | 0.69251 | 52.34% | 0.5011 |
| Five-minute | 2 | 0.69161 | 52.76% | 0.69211 | 52.34% | 0.5022 |
| Five-minute | 3 | 0.69159 | 52.76% | 0.69209 | 52.34% | 0.5019 |
| Five-minute | 4 | 0.69158 | 52.76% | 0.69210 | 52.34% | 0.5028 |
| Daily + five-minute | 1 | 0.69164 | 52.72% | 0.69245 | 52.34% | 0.5007 |
| Daily + five-minute | 2 | 0.69153 | 52.76% | 0.69217 | 52.34% | 0.4976 |
| Daily + five-minute | 3 | 0.69148 | 52.76% | 0.69211 | 52.34% | 0.4974 |
| Daily + five-minute | 4 | 0.69144 | 52.76% | 0.69223 | 52.34% | 0.4972 |

## The green-day CALL example

For this specific interpretation, the previous daily candle closes green, the five-minute EMA8 crosses above EMA21 on a completed bar, and SPY must remain between the prior close and the prior daily EMA8 at the signal close and next open. The previous close is the stop and the previous daily EMA8 is the fixed target. A bar touching both levels has an unknown order unless its open establishes which level came first. Only the first eligible entry per session counts.

| Period | First eligible sessions | Target first | Stop first | Neither by close |
|---|---:|---:|---:|---:|
| 2020–2024 | 48 | 12 | 25 | 11 |
| 2025 | 7 | 5 | 2 | 0 |

On November 19, 2024 at 3:40 p.m. Eastern, the signal close and next open were $589.91. The prior close was $588.15 and the fixed daily EMA8 target was $590.29. SPY touched the target during the bar beginning at 3:45 p.m. That trade risked $1.76 in the underlying to reach a target roughly $0.38 away. This illustrates why a successful target touch alone says little about profitability.

## Can the model learn when SPY will reach the target before the stop?

This exploratory study used one observation every half hour at :05 or :35 when the next entry was bracketed by the prior close below and prior daily EMA8 above. It included different daily candle colors and intraday patterns. The successful outcome was touching the target first by the close. Stop-first and neither outcomes were negatives. A price gap outside the two levels at the next open cancelled the candidate. Ambiguous or incomplete price paths were excluded. Every represented day received equal total weight.

Training covered 1,581 observations from 246 sessions through 2023. The 2024 check covered 326 observations across 51 sessions with 84 target-first, 121 stop-first and 121 neither. The 2025 data were excluded from this exploratory comparison.

| 2024 model | Day-weighted balanced accuracy | Day-weighted AUC | Day-weighted log loss |
|---|---:|---:|---:|
| Constant probability | 50.00% | 0.500 | 0.647 |
| Distance to target, distance to stop and time | 75.88% | 0.893 | 0.407 |
| Chart features plus distances | 79.78% | 0.901 | 0.386 |

Knowing where SPY was between the target and stop explained most of the ranking. Adding chart features lowered log loss by 0.021 in 2024. The 95% interval from resampling whole sessions was −0.093 to +0.063, so the additional chart contribution remains uncertain. These metrics answer a narrower question than 30-minute direction and cannot be compared as like-for-like scores. The study measures movement in SPY and excludes option premiums, spreads and fees.

## Larger comparison

The full PyTorch experiment compares a five-minute CNN, a paired CNN, and a paired CNN that also receives six distances expressed in common price units. It runs on a standard runner after the requested free T4 was unavailable. The live job records each epoch and writes model weights with optimizer state. See the live run for its current status.

[Live full-size experiment](https://github.com/jamelof23/SPY-EMA-V2/actions/runs/45137557151)

