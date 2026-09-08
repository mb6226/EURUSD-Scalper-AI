"""MT5-compatible ZigZag core for EURUSD OHLC data.

The implementation mirrors the supplied MetaQuotes ZigZag.mq5 logic:
Depth, Deviation (in points), and Backstep are preserved with the same
candidate-extreme and alternating-extreme state machine.
"""

from __future__ import annotations

import numpy as np


def _rolling_extreme(a: np.ndarray, depth: int, find_max: bool) -> np.ndarray:
    """Return the MT5 Lowest/Highest equivalent for every chronological bar.

    MT5's helper examines the current bar plus up to depth-1 preceding bars.
    """
    n = len(a)
    out = np.empty(n, dtype=a.dtype)
    out[:depth] = np.nan
    from collections import deque

    dq: deque[int] = deque()
    for i, x in enumerate(a):
        while dq and dq[0] <= i - depth:
            dq.popleft()
        if find_max:
            while dq and a[dq[-1]] <= x:
                dq.pop()
        else:
            while dq and a[dq[-1]] >= x:
                dq.pop()
        dq.append(i)
        if i >= depth - 1:
            out[i] = a[dq[0]]
    return out


def zigzag_mt5(
    high: np.ndarray,
    low: np.ndarray,
    depth: int,
    deviation_points: int,
    backstep: int,
) -> np.ndarray:
    """Calculate the ZigZag line using the supplied MT5 source semantics.

    Parameters
    ----------
    high, low:
        Chronological OHLC arrays (oldest -> newest).
    depth:
        MT5 InpDepth.
    deviation_points:
        MT5 InpDeviation, expressed in symbol points. The caller supplies
        the point size separately when converting this integer to price units.
    backstep:
        MT5 InpBackstep.

    Returns
    -------
    np.ndarray
        ZigZag price at pivots and NaN elsewhere.
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    if high.shape != low.shape:
        raise ValueError("high and low must have the same shape")
    n = len(high)
    if n < max(100, depth + backstep + 2):
        return np.full(n, np.nan)

    # The source uses deviation * _Point. EURUSD's point size is supplied by
    # the optimizer; this core receives the already-converted price threshold.
    # A scalar wrapper below avoids recomputing the rolling extrema.
    return _zigzag_from_candidates(high, low, depth, deviation_points, backstep)


def _zigzag_from_candidates(
    high: np.ndarray,
    low: np.ndarray,
    depth: int,
    deviation_price: float,
    backstep: int,
) -> np.ndarray:
    n = len(high)
    hi_ext = _rolling_extreme(high, depth, True)
    lo_ext = _rolling_extreme(low, depth, False)
    hi_map = np.zeros(n, dtype=np.float64)
    lo_map = np.zeros(n, dtype=np.float64)

    last_low = 0.0
    last_high = 0.0
    start = depth - 1

    # Candidate-extreme stage: direct translation of the supplied MQL5 loop.
    for shift in range(start, n):
        val = lo_ext[shift]
        if val == last_low:
            val = 0.0
        else:
            last_low = val
            if (low[shift] - val) > deviation_price:
                val = 0.0
            else:
                lo = max(start, shift - backstep)
                for j in range(shift - 1, lo - 1, -1):
                    res = lo_map[j]
                    if res != 0.0 and res > val:
                        lo_map[j] = 0.0
        if low[shift] == val:
            lo_map[shift] = val

        val = hi_ext[shift]
        if val == last_high:
            val = 0.0
        else:
            last_high = val
            if (val - high[shift]) > deviation_price:
                val = 0.0
            else:
                lo = max(start, shift - backstep)
                for j in range(shift - 1, lo - 1, -1):
                    res = hi_map[j]
                    if res != 0.0 and res < val:
                        hi_map[j] = 0.0
        if high[shift] == val:
            hi_map[shift] = val

    # Final alternating-extreme stage, following Extremum/Peak/Bottom.
    zz = np.full(n, np.nan, dtype=np.float64)
    search = 0  # 0=Extremum, 1=Peak, -1=Bottom
    last_low = 0.0
    last_high = 0.0
    last_low_pos = -1
    last_high_pos = -1

    for shift in range(start, n):
        if search == 0:
            if last_low == 0.0 and last_high == 0.0:
                if hi_map[shift] != 0.0:
                    last_high = high[shift]
                    last_high_pos = shift
                    search = -1
                    zz[shift] = last_high
                if lo_map[shift] != 0.0:
                    last_low = low[shift]
                    last_low_pos = shift
                    search = 1
                    zz[shift] = last_low
        elif search == 1:  # Peak: next high, while replacing lower lows.
            if lo_map[shift] != 0.0 and lo_map[shift] < last_low and hi_map[shift] == 0.0:
                if last_low_pos >= 0:
                    zz[last_low_pos] = np.nan
                last_low_pos = shift
                last_low = lo_map[shift]
                zz[shift] = last_low
            if hi_map[shift] != 0.0 and lo_map[shift] == 0.0:
                last_high = hi_map[shift]
                last_high_pos = shift
                zz[shift] = last_high
                search = -1
        else:  # Bottom: next low, while replacing higher highs.
            if hi_map[shift] != 0.0 and hi_map[shift] > last_high and lo_map[shift] == 0.0:
                if last_high_pos >= 0:
                    zz[last_high_pos] = np.nan
                last_high_pos = shift
                last_high = hi_map[shift]
                zz[shift] = last_high
            if lo_map[shift] != 0.0 and hi_map[shift] == 0.0:
                last_low = lo_map[shift]
                last_low_pos = shift
                zz[shift] = last_low
                search = 1

    return zz


def zigzag_mt5_with_point(
    high: np.ndarray,
    low: np.ndarray,
    depth: int,
    deviation_points: int,
    backstep: int,
    point: float,
) -> np.ndarray:
    """Convenience wrapper matching MQL5's deviation * _Point exactly."""
    return _zigzag_from_candidates(
        np.asarray(high, dtype=np.float64),
        np.asarray(low, dtype=np.float64),
        depth,
        float(deviation_points) * float(point),
        backstep,
    )
