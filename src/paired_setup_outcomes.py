"""Annotate one proposed CALL setup in pre-2026 SPY chart pairs.

A missing rule trigger says nothing about other discretionary setups. Entry
eligibility and outcome columns are kept outside image inputs.
"""
import numpy as np
import pandas as pd


def call_candidates(daily, bars, pairs):
    import exchange_calendars as xc
    schedule = xc.get_calendar("XNYS", start="2020-01-01", end="2025-12-31").schedule
    closes = schedule.close.dt.tz_convert("America/New_York").dt.tz_localize(None)
    last = daily.iloc[pairs.daily_end]
    previous = bars.iloc[pairs.intraday_end - 1]
    current = bars.iloc[pairs.intraday_end]
    signal = ((last.close.to_numpy() > last.open.to_numpy()) &
              (previous.ema8.to_numpy() <= previous.ema21.to_numpy()) &
              (current.ema8.to_numpy() > current.ema21.to_numpy()) &
              (current.index - previous.index == pd.Timedelta(minutes=5)))
    out = pairs.loc[signal, ["split"]].copy()
    out["stop_previous_close"] = last.close.to_numpy()[signal]
    out["target_previous_daily_ema8"] = last.ema8.to_numpy()[signal]
    out["signal_close"] = current.close.to_numpy()[signal]
    out["levels_bracket_signal_close"] = ((out.signal_close > out.stop_previous_close) &
                                          (out.signal_close < out.target_previous_daily_ema8))
    entries, outcomes, exits = [], [], []
    for r in out.itertuples():
        t = r.Index
        end = closes.at[t.normalize()]
        entry = np.nan
        exit_time = pd.NaT
        if not r.levels_bracket_signal_close:
            result = "levels_do_not_bracket_signal"
        elif t >= end:
            result = "no_next_regular_bar"
        elif t not in bars.index:
            result = "missing_next_bar"
        else:
            entry = bars.at[t, "open"]
            if not r.stop_previous_close < entry < r.target_previous_daily_ema8:
                result = "gap_outside_levels"
            else:
                result = "neither_by_close"
                for at in pd.date_range(t, end, freq="5min", inclusive="left"):
                    if at not in bars.index or bars.at[at, "minute_count"] != 5:
                        result = "incomplete_path"
                        break
                    high, low, op = (bars.at[at, k] for k in ("high", "low", "open"))
                    if op >= r.target_previous_daily_ema8:
                        result = "target_first"
                    elif op <= r.stop_previous_close:
                        result = "stop_first"
                    elif high >= r.target_previous_daily_ema8 and low <= r.stop_previous_close:
                        result = "both_same_bar"
                    elif high >= r.target_previous_daily_ema8:
                        result = "target_first"
                    elif low <= r.stop_previous_close:
                        result = "stop_first"
                    if result != "neither_by_close":
                        exit_time = at
                        break
        entries.append(entry)
        outcomes.append(result)
        exits.append(exit_time)
    out["next_bar_open"] = entries
    out["outcome"] = outcomes
    out["event_bar_open_et"] = exits
    tradable = out.outcome.isin(["target_first", "stop_first", "neither_by_close",
                                "both_same_bar", "incomplete_path"])
    out["first_eligible_signal_of_session"] = False
    valid = out.loc[tradable]
    out.loc[valid.index[~valid.index.normalize().duplicated()], "first_eligible_signal_of_session"] = True
    return out
