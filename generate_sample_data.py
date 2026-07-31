"""
generate_sample_data.py

The real ExtraSensory raw measurement files are enormous (accelerometer alone
is 6.1GB) and aren't bundled with this project. This script fakes a single
day of raw phone-accelerometer + phone-gyroscope readings for one made-up
user, in the same shape the real files come in:

    timestamp_ms, x, y, z

...one short burst of ~20 seconds every minute, sampled around 40Hz, with
gaps and clock jitter thrown in on purpose, because that's what the real
sensor logs look like too.

If you swap in real ExtraSensory raw csvs later, data_loader.py reads that
exact same shape, so nothing else in the pipeline needs to change.
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(7)

# the 7 "main activity" classes ExtraSensory records
ACTIVITIES = [
    "LYING_DOWN",
    "SITTING",
    "STANDING_IN_PLACE",
    "STANDING_AND_MOVING",
    "WALKING",
    "RUNNING",
    "BICYCLING",
]

# rough signal "personality" per activity: (accel noise std, accel periodic amplitude,
# step frequency in Hz, gyro noise std, gyro periodic amplitude)
ACTIVITY_PROFILE = {
    "LYING_DOWN":          dict(a_noise=0.02, a_amp=0.00, freq=0.0, g_noise=0.01, g_amp=0.00),
    "SITTING":             dict(a_noise=0.03, a_amp=0.00, freq=0.0, g_noise=0.02, g_amp=0.00),
    "STANDING_IN_PLACE":   dict(a_noise=0.05, a_amp=0.02, freq=0.3, g_noise=0.03, g_amp=0.02),
    "STANDING_AND_MOVING": dict(a_noise=0.08, a_amp=0.10, freq=0.6, g_noise=0.06, g_amp=0.08),
    "WALKING":             dict(a_noise=0.12, a_amp=0.55, freq=1.8, g_noise=0.15, g_amp=0.35),
    "RUNNING":             dict(a_noise=0.20, a_amp=1.30, freq=2.8, g_noise=0.30, g_amp=0.70),
    "BICYCLING":           dict(a_noise=0.10, a_amp=0.35, freq=1.3, g_noise=0.25, g_amp=0.55),
}

GRAVITY = 9.81


def _make_burst(activity: str, start_ms: int, duration_s: float = 20.0, hz: float = 40.0):
    """One ~20s recording-session burst, like ExtraSensory's per-minute sessions."""
    profile = ACTIVITY_PROFILE[activity]
    n = int(duration_s * hz)

    # jittery real-world sample timing instead of a perfect grid
    dt = 1000.0 / hz
    jitter = RNG.normal(0, dt * 0.05, n)
    t = start_ms + np.cumsum(np.full(n, dt) + jitter)

    tt = np.arange(n) / hz
    phase = RNG.uniform(0, 2 * np.pi)

    # accelerometer: gravity mostly on one axis + periodic "gait" wobble + noise
    ax = profile["a_amp"] * np.sin(2 * np.pi * profile["freq"] * tt + phase) + RNG.normal(0, profile["a_noise"], n)
    ay = profile["a_amp"] * 0.6 * np.cos(2 * np.pi * profile["freq"] * tt + phase * 0.5) + RNG.normal(0, profile["a_noise"], n)
    az = GRAVITY + profile["a_amp"] * 0.3 * np.sin(2 * np.pi * profile["freq"] * tt) + RNG.normal(0, profile["a_noise"], n)

    gx = profile["g_amp"] * np.sin(2 * np.pi * profile["freq"] * tt + phase) + RNG.normal(0, profile["g_noise"], n)
    gy = profile["g_amp"] * np.cos(2 * np.pi * profile["freq"] * tt) + RNG.normal(0, profile["g_noise"], n)
    gz = profile["g_amp"] * 0.4 * np.sin(2 * np.pi * profile["freq"] * tt * 0.5) + RNG.normal(0, profile["g_noise"], n)

    accel = pd.DataFrame({"timestamp_ms": t.astype(np.int64), "x": ax, "y": ay, "z": az})
    gyro = pd.DataFrame({"timestamp_ms": (t + RNG.normal(0, 4, n)).astype(np.int64), "x": gx, "y": gy, "z": gz})
    return accel, gyro


def generate_day(out_dir: str, uuid: str = "demo-user-01"):
    """
    Simulates a day: a rough activity schedule, one recording burst per minute
    (skipping some minutes entirely, and occasionally dropping the gyro burst
    only, to mimic sensors that weren't always available).
    """
    os.makedirs(out_dir, exist_ok=True)

    # a made up but plausible day
    schedule = [
        ("LYING_DOWN", 0, 6 * 60),           # midnight - 6am asleep
        ("SITTING", 6 * 60, 7 * 60),         # waking up / breakfast
        ("STANDING_AND_MOVING", 7 * 60, 7 * 60 + 15),
        ("WALKING", 7 * 60 + 15, 7 * 60 + 30),
        ("SITTING", 7 * 60 + 30, 11 * 60),   # class / desk work
        ("STANDING_IN_PLACE", 11 * 60, 11 * 60 + 10),
        ("WALKING", 11 * 60 + 10, 11 * 60 + 25),
        ("SITTING", 11 * 60 + 25, 13 * 60),  # lunch + more sitting
        ("BICYCLING", 13 * 60, 13 * 60 + 20),
        ("SITTING", 13 * 60 + 20, 17 * 60),  # afternoon desk work
        ("WALKING", 17 * 60, 17 * 60 + 15),
        ("RUNNING", 17 * 60 + 15, 17 * 60 + 45),
        ("STANDING_IN_PLACE", 17 * 60 + 45, 18 * 60),
        ("SITTING", 18 * 60, 22 * 60),       # evening
        ("LYING_DOWN", 22 * 60, 24 * 60),    # bed
    ]

    accel_rows, gyro_rows = [], []
    day_start_ms = 0

    for minute in range(24 * 60):
        # ~85% of minutes actually got a recording session, like real phones
        if RNG.random() > 0.85:
            continue

        activity = "SITTING"
        for act, start_min, end_min in schedule:
            if start_min <= minute < end_min:
                activity = act
                break

        start_ms = day_start_ms + minute * 60_000
        accel, gyro = _make_burst(activity, start_ms)
        accel["activity"] = activity
        gyro["activity"] = activity
        accel_rows.append(accel)

        # ~5% of the time the gyro just wasn't available for that burst
        if RNG.random() > 0.05:
            gyro_rows.append(gyro)

    accel_df = pd.concat(accel_rows, ignore_index=True)
    gyro_df = pd.concat(gyro_rows, ignore_index=True)

    accel_path = os.path.join(out_dir, f"{uuid}_accel_raw.csv")
    gyro_path = os.path.join(out_dir, f"{uuid}_gyro_raw.csv")
    accel_df.to_csv(accel_path, index=False)
    gyro_df.to_csv(gyro_path, index=False)

    print(f"wrote {len(accel_df):,} accel rows -> {accel_path}")
    print(f"wrote {len(gyro_df):,} gyro rows  -> {gyro_path}")
    return accel_path, gyro_path


if __name__ == "__main__":
    generate_day(os.path.join(os.path.dirname(__file__), "data", "raw"))