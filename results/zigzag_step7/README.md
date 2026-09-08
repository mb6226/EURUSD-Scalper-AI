# Step 7 — Previous Swing Fibonacci Dataset

Uses the selected Step 6 research baseline exactly:

- Depth: 350
- Deviation: 125 MT5 points
- Backstep: 40
- EURUSD point: 0.00001

For every ZigZag swing after the first, the immediately preceding swing is used as the Fibonacci reference. The dataset records the previous swing's endpoints and exact price levels for 23.6%, 38.2%, 50.0%, 61.8%, 78.6%, and 100.0% retracement.

## Outputs

- `swings_with_previous_fibo.csv` — complete swing-by-swing dataset, including the previous swing Fibonacci levels.
- `swing_previous_fibo_summary.csv` — Step 6-style min/max/median statistics plus Fibonacci-eligible swing count.
- `previous_fibo_zone_distribution.csv` — distribution of the current swing endpoint relative to the previous swing retracement zones.
- `previous_fibo_direction_stats.csv` — statistics split by previous swing direction.

The Fibonacci calculation is descriptive only. It does not optimize entries, TP/SL, or PnL and does not use future data beyond the swing endpoint when constructing the corresponding row.
