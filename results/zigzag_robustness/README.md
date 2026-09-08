# EURUSD M1 ZigZag Step 4 — Robustness / Sensitivity

Candidates are inherited from Step 3: `results/zigzag_quarterly/quarter_top20.csv`.
Baseline center: depth=350, deviation=150, backstep=40.
Depth is probed every 5 bars across +/-50 bars around the Step 3 center.
Deviation preserves the full Step 3 range; backstep preserves Step 3 values and adds the next neighboring value.
The robustness score measures retention of efficiency, swing size, swing duration, and pivot density versus the inherited baseline.
Plateau output is descriptive and is not a final trading parameter.
