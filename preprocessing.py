"""
preprocessing.py

Turns two independent raw streams (accelerometer, gyroscope - each sampled
at its own rate, on its own clock, with its own gaps) into clean, aligned,
fixed-rate windows the classifier can actually eat.

Steps, in order (this is the "neck" of the project's pipeline diagram):
    1. match_timestamps   - accel and gyro don't tick at the same instants,
                             so pair them up by nearest-neighbour timestamp
    2. resample_25hz       - re-grid the paired stream onto an even 25Hz clock
    3. handle_missing      - short gaps get interpolated, long gaps get
                             marked and dropped rather than faked
    4. make_windows        - slice into overlapping fixed-length windows and
                             turn each window into a feature vector
"""

import numpy as np
import pandas as pd

TARGET_HZ = 25
MATCH_TOLERANCE_MS = 30          # accel/gyro samples within 30ms count as "same instant"
MAX_GAP_FOR_INTERP_MS = 500      # gaps shorter than this get interpolated through
WINDOW_SECONDS = 2.0
WINDOW_OVERLAP = 0.5


def match_timestamps(accel: pd.DataFrame, gyro: pd.DataFrame) -> pd.DataFrame:
    """Nearest-neighbour join of accel and gyro on timestamp, within tolerance."""
    accel = accel.rename(columns={"x": "ax", "y": "ay", "z": "az"})
    gyro = gyro.rename(columns={"x": "gx", "y": "gy", "z": "gz"})

    merged = pd.merge_asof(
        accel.sort_values("timestamp_ms"),
        gyro.sort_values("timestamp_ms"),
        on="timestamp_ms",
        direction="nearest",
        tolerance=MATCH_TOLERANCE_MS,
        suffixes=("", "_gyro"),
    )
    return merged


def resample_25hz(merged: pd.DataFrame) -> pd.DataFrame:
    """Re-grid onto an even TARGET_HZ clock via linear interpolation."""
    if merged.empty:
        return merged

    t0, t1 = merged["timestamp_ms"].iloc[0], merged["timestamp_ms"].iloc[-1]
    step_ms = 1000.0 / TARGET_HZ
    grid = np.arange(t0, t1, step_ms)

    out = pd.DataFrame({"timestamp_ms": grid})
    value_cols = [c for c in merged.columns if c not in ("timestamp_ms", "activity") and not c.startswith("activity")]

    for col in value_cols:
        out[col] = np.interp(grid, merged["timestamp_ms"], merged[col])

    if "activity" in merged.columns:
        # nearest-neighbour label carry-over, just for evaluating the demo classifier
        idx = np.searchsorted(merged["timestamp_ms"], grid)
        idx = np.clip(idx, 0, len(merged) - 1)
        out["activity"] = merged["activity"].values[idx]
    elif "activity_gyro" in merged.columns:
        idx = np.searchsorted(merged["timestamp_ms"], grid)
        idx = np.clip(idx, 0, len(merged) - 1)
        out["activity"] = merged["activity_gyro"].values[idx]

    out["_gap_ms"] = None
    return out


def handle_missing(resampled: pd.DataFrame, raw_timestamps: pd.Series) -> pd.DataFrame:
    """
    Flags stretches of the resampled grid that fall inside a gap in the raw
    data bigger than MAX_GAP_FOR_INTERP_MS. Those rows are dropped rather
    than left as invented, interpolated-across-nothing values.
    """
    if resampled.empty:
        return resampled

    raw_sorted = np.sort(raw_timestamps.values)
    gaps_start, gaps_end = [], []
    diffs = np.diff(raw_sorted)
    for i, d in enumerate(diffs):
        if d > MAX_GAP_FOR_INTERP_MS:
            gaps_start.append(raw_sorted[i])
            gaps_end.append(raw_sorted[i + 1])

    if not gaps_start:
        return resampled.drop(columns=["_gap_ms"], errors="ignore")

    mask = np.zeros(len(resampled), dtype=bool)
    ts = resampled["timestamp_ms"].values
    for s, e in zip(gaps_start, gaps_end):
        mask |= (ts > s) & (ts < e)

    cleaned = resampled.loc[~mask].drop(columns=["_gap_ms"], errors="ignore").reset_index(drop=True)
    return cleaned


def _window_features(window: pd.DataFrame) -> dict:
    """One fixed-length slice of the 25Hz stream -> a small feature vector."""
    feats = {}
    for col in ("ax", "ay", "az", "gx", "gy", "gz"):
        if col not in window.columns:
            continue
        v = window[col].to_numpy()
        feats[f"{col}_mean"] = float(np.mean(v))
        feats[f"{col}_std"] = float(np.std(v))
        feats[f"{col}_min"] = float(np.min(v))
        feats[f"{col}_max"] = float(np.max(v))
        feats[f"{col}_energy"] = float(np.mean(v ** 2))

        # dominant frequency via FFT - this is what separates "walking" from
        # "standing and moving" more than any raw amplitude number does
        if len(v) >= 4:
            spectrum = np.abs(np.fft.rfft(v - np.mean(v)))
            freqs = np.fft.rfftfreq(len(v), d=1.0 / TARGET_HZ)
            if len(spectrum) > 1:
                dom_idx = 1 + np.argmax(spectrum[1:])  # skip DC component
                feats[f"{col}_dom_freq"] = float(freqs[dom_idx])
                feats[f"{col}_dom_power"] = float(spectrum[dom_idx])
            else:
                feats[f"{col}_dom_freq"] = 0.0
                feats[f"{col}_dom_power"] = 0.0

    accel_mag = np.sqrt(window["ax"] ** 2 + window["ay"] ** 2 + window["az"] ** 2) if "ax" in window else None
    if accel_mag is not None:
        feats["accel_mag_std"] = float(np.std(accel_mag))
        feats["accel_mag_mean"] = float(np.mean(accel_mag))

    return feats


def make_windows(cleaned: pd.DataFrame, window_seconds: float = WINDOW_SECONDS,
                  overlap: float = WINDOW_OVERLAP):
    """
    Slides a window_seconds-long window across the cleaned 25Hz stream with
    the given fractional overlap. Returns (feature_rows, window_meta) where
    window_meta carries the start/end timestamp (and majority ground-truth
    label, if present) for each window.
    """
    if cleaned.empty:
        return [], []

    win_len = int(window_seconds * TARGET_HZ)
    step = max(1, int(win_len * (1 - overlap)))

    feature_rows, meta = [], []
    n = len(cleaned)
    for start in range(0, n - win_len + 1, step):
        chunk = cleaned.iloc[start:start + win_len]

        # skip windows that straddle a dropped gap (timestamps not contiguous)
        expected_span = (win_len - 1) * (1000.0 / TARGET_HZ)
        actual_span = chunk["timestamp_ms"].iloc[-1] - chunk["timestamp_ms"].iloc[0]
        if actual_span > expected_span * 1.5:
            continue

        feats = _window_features(chunk)
        feature_rows.append(feats)

        entry = {
            "start_ms": int(chunk["timestamp_ms"].iloc[0]),
            "end_ms": int(chunk["timestamp_ms"].iloc[-1]),
        }
        if "activity" in chunk.columns:
            entry["true_activity"] = chunk["activity"].mode().iloc[0]
        meta.append(entry)

    return feature_rows, meta