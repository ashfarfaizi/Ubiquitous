"""
pipeline.py

Wires the whole thing together, top to bottom of the diagram:

    raw accel + gyro
        -> match_timestamps
        -> resample_25hz
        -> handle_missing
        -> make_windows
        -> classify_windows
        -> collapse into an activity timeline

run_pipeline() is the one function everything else (the Flask app, the CLI)
calls.
"""

from datetime import datetime, timezone

import data_loader
import preprocessing as prep
import classifier


def _fmt(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%H:%M:%S")


def build_timeline(meta, preds, confidences):
    """Collapses consecutive windows with the same predicted label into
    single timeline segments, the way a human would describe their day
    ("sitting from 9 to 11", not fifty separate two-second entries)."""
    segments = []
    for m, pred, conf in zip(meta, preds, confidences):
        if segments and segments[-1]["activity"] == pred and m["start_ms"] - segments[-1]["end_ms"] < 5000:
            segments[-1]["end_ms"] = m["end_ms"]
            segments[-1]["confidences"].append(conf)
        else:
            segments.append({
                "activity": pred,
                "start_ms": m["start_ms"],
                "end_ms": m["end_ms"],
                "confidences": [conf],
            })

    timeline = []
    for s in segments:
        timeline.append({
            "activity": s["activity"],
            "start_ms": s["start_ms"],
            "end_ms": s["end_ms"],
            "start_time": _fmt(s["start_ms"]),
            "end_time": _fmt(s["end_ms"]),
            "duration_s": round((s["end_ms"] - s["start_ms"]) / 1000, 1),
            "avg_confidence": round(sum(s["confidences"]) / len(s["confidences"]), 3),
        })
    return timeline


def run_pipeline(accel_path: str, gyro_path: str):
    accel, gyro = data_loader.load_accel_gyro(accel_path, gyro_path)

    merged = prep.match_timestamps(accel, gyro)
    resampled = prep.resample_25hz(merged)
    cleaned = prep.handle_missing(resampled, accel["timestamp_ms"])
    feature_rows, meta = prep.make_windows(cleaned)

    model, columns = classifier.load_model()
    preds, confidences = classifier.classify_windows(feature_rows, model, columns)

    timeline = build_timeline(meta, preds, confidences)

    stats = {
        "raw_accel_samples": len(accel),
        "raw_gyro_samples": len(gyro),
        "matched_samples": len(merged),
        "resampled_samples": len(resampled),
        "clean_samples_after_gap_removal": len(cleaned),
        "windows_classified": len(feature_rows),
        "timeline_segments": len(timeline),
    }
    return timeline, stats


if __name__ == "__main__":
    import os
    accel_p = os.path.join(os.path.dirname(__file__), "data", "raw", "demo-user-01_accel_raw.csv")
    gyro_p = os.path.join(os.path.dirname(__file__), "data", "raw", "demo-user-01_gyro_raw.csv")
    tl, stats = run_pipeline(accel_p, gyro_p)
    print(stats)
    for seg in tl[:10]:
        print(seg)