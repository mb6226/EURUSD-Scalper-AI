# Step 10 economic calendar

`economic_calendar.csv` is the point-in-time input for the stateful EURUSD economic regime engine.

Required columns:

- `release_time`: UTC timestamp when the result became public
- `currency`: `USD` or `EUR`
- `impact`: `HIGH`, `MEDIUM`, or `LOW`
- `event`: event name
- `bias`: EURUSD directional impact, `+1` bullish EURUSD, `-1` bearish EURUSD, `0` neutral/unknown
- optional `actual`, `forecast`, `previous`

The regime engine never uses future events. For each month, the opening regime is the previous month's final regime. New events accumulate during the month; the regime changes only when the accumulated directional score reaches the flip threshold.

For the first month in the historical sample, use `monthly_regime_seed.csv` only if a prior-month regime is not present in the event history.

Do not fill historical `bias` from the subsequent price move. It must represent the fundamental interpretation available at release time; otherwise Step 10 would contain look-ahead bias.
