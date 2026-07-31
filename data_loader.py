"""
data_loader.py

Reads raw accelerometer / gyroscope csvs. Expects columns:
    timestamp_ms, x, y, z   (an optional "activity" column is ignored at
                              load time - that's ground truth, kept around
                              only so we can grade our own classifier later)

This matches both the demo data from generate_sample_data.py and, if you
point it at the real ExtraSensory raw-measurement download, the real thing
after you've concatenated a user's per-timestamp files into one csv with
these four columns.
"""

import pandas as pd


REQUIRED_COLS = {"timestamp_ms", "x", "y", "z"}


def load_stream(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {missing}")
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def load_accel_gyro(accel_path: str, gyro_path: str):
    accel = load_stream(accel_path)
    gyro = load_stream(gyro_path)
    return accel, gyro